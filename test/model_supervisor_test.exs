defmodule LlamaCppEx.ModelSupervisorTest do
  # async: false — the manager is a node-wide singleton keyed on a named ETS
  # table, and the Registry/DynamicSupervisor names are global too.
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog

  alias LlamaCppEx.{ModelManager, ModelSupervisor}
  alias LlamaCppEx.ModelManager.ModelIO

  # `ModelSupervisor` is the documented entry point for the multi-model manager
  # and had no test at all: every manager test starts `ModelManager` directly, so
  # nothing checked that the three children exist, that they start in the order
  # the manager depends on, that the restart strategy encodes that dependency, or
  # that the manager's options are the only ones forwarded.

  # A backend that records what it was asked for and never touches a GGUF file.
  # `:fake_bytes` drives the budget check.
  defmodule RecordingIO do
    @behaviour LlamaCppEx.ModelManager.Backend

    @impl true
    def resolve_source(source, opts) do
      send(Keyword.fetch!(opts, :test_pid), {:resolved, source})
      {:ok, "fake.gguf", Keyword.get(opts, :fake_bytes, 1_000)}
    end

    @impl true
    def load_model(_path, _opts), do: {:ok, %LlamaCppEx.Model{ref: make_ref()}}

    @impl true
    def start_server(_id, _path, _opts), do: {:error, :not_used}

    @impl true
    def stop_server(_pid), do: :ok
  end

  defp start_sup(opts \\ []), do: start_supervised!({ModelSupervisor, opts})

  defp autoload(id, opts) do
    [{id, {:path, "#{id}.gguf"}, [mode: :direct, test_pid: self()] ++ opts}]
  end

  defp await_status(id, status, attempts \\ 200)

  defp await_status(id, status, 0),
    do: flunk("#{id} never reached #{status}: #{inspect(ModelManager.info(id))}")

  defp await_status(id, status, attempts) do
    case ModelManager.info(id) do
      {:ok, %{status: ^status} = info} ->
        info

      _ ->
        Process.sleep(5)
        await_status(id, status, attempts - 1)
    end
  end

  # A :sys.get_state/1 call is serialized behind the manager's mailbox, so once
  # its in-flight load map is empty the autoload has definitely been finalized —
  # no sleeping on a guess.
  defp await_idle(attempts \\ 200)
  defp await_idle(0), do: flunk("the manager still has loads in flight")

  defp await_idle(attempts) do
    if :sys.get_state(ModelManager).loads == %{} do
      :ok
    else
      Process.sleep(5)
      await_idle(attempts - 1)
    end
  end

  # init/1 is the entire module. Reading the spec it returns is deterministic and
  # covers the two things a live restart test can only approximate: the child
  # order and the strategy that encodes the dependency between them.
  describe "init/1" do
    test "declares the three children in dependency order" do
      assert {:ok, {_flags, children}} = ModelSupervisor.init([])

      assert Enum.map(children, & &1.id) == [
               LlamaCppEx.ModelRegistry,
               LlamaCppEx.ModelDynSup,
               ModelManager
             ]

      assert [registry, dynsup, manager] = children

      assert {Registry, :start_link, [[keys: :unique, name: LlamaCppEx.ModelRegistry]]} =
               registry.start

      assert {DynamicSupervisor, :start_link,
              [[strategy: :one_for_one, name: LlamaCppEx.ModelDynSup]]} = dynsup.start

      assert {ModelManager, :start_link, [_opts]} = manager.start
    end

    test "uses :rest_for_one, because the manager depends on both children" do
      # The manager looks models up via the Registry and starts servers under the
      # DynamicSupervisor, so if either restarts the manager must too. :one_for_one
      # would leave it holding references into a dead Registry.
      assert {:ok, {%{strategy: :rest_for_one}, _}} = ModelSupervisor.init([])
    end

    test "forwards only the manager's own options" do
      assert {:ok, {_flags, children}} =
               ModelSupervisor.init(
                 memory_budget: 4_096,
                 models: [{"a", {:path, "a.gguf"}}],
                 io: RecordingIO,
                 name: :some_supervisor,
                 unrelated: :ignored
               )

      manager = List.last(children)
      assert {ModelManager, :start_link, [opts]} = manager.start

      assert Enum.sort(Keyword.keys(opts)) == [:io, :memory_budget, :models]
      assert opts[:memory_budget] == 4_096
      assert opts[:io] == RecordingIO
    end
  end

  describe "start_link/1" do
    test "brings up the Registry, the DynamicSupervisor and the manager" do
      sup = start_sup()

      assert is_pid(Process.whereis(ModelIO.registry()))
      assert is_pid(Process.whereis(ModelIO.dynamic_supervisor()))
      assert is_pid(Process.whereis(ModelManager))
      assert Supervisor.count_children(sup).active == 3

      # And the manager is usable through its public API, not merely alive.
      assert ModelManager.list() == []
    end

    test ":name renames this supervisor, not the manager" do
      pid = start_supervised!({ModelSupervisor, [name: :my_model_sup]})

      assert Process.whereis(:my_model_sup) == pid
      # The manager stays a node-wide singleton under its module name, which is
      # where the client API looks for it.
      assert is_pid(Process.whereis(ModelManager))
      refute Process.whereis(ModelSupervisor)
    end

    test ":models and :io reach the manager and the autoload runs" do
      start_sup(io: RecordingIO, models: autoload("chat", []), unrelated_option: :ignored)

      assert_receive {:resolved, {:path, "chat.gguf"}}, 2_000
      assert %{id: "chat", mode: :direct} = await_status("chat", :ready)
      assert ModelManager.list() |> Enum.map(& &1.id) == ["chat"]
    end

    test ":memory_budget reaches the manager" do
      # The budget is only observable through its effect. The refusal names the
      # exact limit, so this pins the value that was forwarded, not merely that
      # *some* budget arrived — the default is :infinity, under which this load
      # succeeds.
      log =
        capture_log(fn ->
          start_sup(
            memory_budget: 500,
            io: RecordingIO,
            models: autoload("big", fake_bytes: 5_000)
          )

          assert_receive {:resolved, {:path, "big.gguf"}}, 2_000
          await_idle()
        end)

      assert log =~ ~s(auto-load of "big" failed)
      assert log =~ "insufficient_memory"
      assert log =~ "available: 500"
      assert ModelManager.info("big") == {:error, :not_loaded}
    end

    test "the same model loads when the budget fits" do
      # The control for the test above: same backend, same size, bigger budget.
      start_sup(
        memory_budget: 50_000,
        io: RecordingIO,
        models: autoload("big", fake_bytes: 5_000)
      )

      assert %{status: :ready} = await_status("big", :ready)
    end

    test "no :models means no autoload" do
      start_sup(io: RecordingIO)

      assert ModelManager.list() == []
      refute_receive {:resolved, _}, 100
    end
  end
end
