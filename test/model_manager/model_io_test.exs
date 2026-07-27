defmodule LlamaCppEx.ModelManager.ModelIOTest do
  # async: false — start_server/3 registers under the fixed names ModelIO
  # publishes, so these must not run beside anything else using them.
  use ExUnit.Case, async: false

  alias LlamaCppEx.ModelManager.ModelIO

  # `ModelIO` is the production `Backend`; every one of the 26 ModelManager tests
  # substitutes `FakeIO` for it, so until now nothing exercised it at all. What
  # is reachable without a GGUF file is the whole option-routing surface — which
  # is the part that can silently break, because `Server.start_link/1` now
  # rejects unknown options and `native_opts/1` is the only thing keeping the
  # manager's and the hub's keys away from it.

  describe "registry/0 and dynamic_supervisor/0" do
    test "name the processes ModelSupervisor starts" do
      assert ModelIO.registry() == LlamaCppEx.ModelRegistry
      assert ModelIO.dynamic_supervisor() == LlamaCppEx.ModelDynSup
    end
  end

  describe "resolve_source/2 with {:path, _}" do
    @tag :tmp_dir
    test "returns the path and its size on disk", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "model.gguf")
      File.write!(path, String.duplicate("x", 1234))

      assert ModelIO.resolve_source({:path, path}, []) == {:ok, path, 1234}
    end

    @tag :tmp_dir
    test "reports why the stat failed rather than a bare :error", %{tmp_dir: tmp_dir} do
      missing = Path.join(tmp_dir, "nope.gguf")

      assert ModelIO.resolve_source({:path, missing}, []) ==
               {:error, {:stat_failed, missing, :enoent}}
    end

    @tag :tmp_dir
    test "a directory is stat-able and reports as a source" do
      # File.stat/1 succeeds on a directory, so the guard against loading one
      # lives further down in Model.load/2, not here. Pinned so a change of mind
      # is deliberate.
      assert {:ok, ".", size} = ModelIO.resolve_source({:path, "."}, [])
      assert is_integer(size)
    end
  end

  describe "resolve_source/2 with {:hub, _, _}" do
    @tag :tmp_dir
    test "propagates the hub error without touching the network", %{tmp_dir: tmp_dir} do
      # LLAMA_OFFLINE short-circuits Hub.download/3 before any request, so this
      # exercises the {:hub, ...} clause's error path with no network at all.
      System.put_env("LLAMA_OFFLINE", "1")
      on_exit(fn -> System.delete_env("LLAMA_OFFLINE") end)

      assert {:error, message} =
               ModelIO.resolve_source({:hub, "org/model", "model.gguf"}, cache_dir: tmp_dir)

      assert message =~ "offline"
    end
  end

  describe "load_model/2" do
    test "surfaces the loader's error for a path that is not a GGUF" do
      assert {:error, message} = ModelIO.load_model("/nonexistent/model.gguf", n_gpu_layers: 0)
      assert is_binary(message)
    end

    test "strips hub and manager options before reaching Model.load/2" do
      # Model.load/2 ignores unknown keys, so the observable proof is that the
      # call gets as far as the loader (a load error, not an option error) with
      # every manager and hub key present.
      opts = [
        n_gpu_layers: 0,
        mode: :server,
        capabilities: [:chat],
        default: true,
        memory_budget: :infinity,
        io: __MODULE__,
        cache_dir: "/tmp",
        token: "hf_secret",
        revision: "main",
        force: true,
        progress: fn _ -> :ok end
      ]

      assert {:error, message} = ModelIO.load_model("/nonexistent/model.gguf", opts)
      assert message =~ "nonexistent"
    end
  end

  describe "start_server/3" do
    setup do
      start_supervised!({Registry, keys: :unique, name: ModelIO.registry()})

      start_supervised!(
        {DynamicSupervisor, strategy: :one_for_one, name: ModelIO.dynamic_supervisor()}
      )

      :ok
    end

    test "hub and manager options never reach Server.start_link/1" do
      # This is the test that has teeth: Server.start_link/1 validates its option
      # keys and raises ArgumentError on an unknown one, so if native_opts/1
      # stopped dropping :mode or :cache_dir the child would die with a validation
      # error instead of the model-not-found error the path actually deserves.
      assert {:error, {:model_not_found, "/nonexistent/model.gguf"}} =
               ModelIO.start_server("chat", "/nonexistent/model.gguf",
                 mode: :server,
                 capabilities: [:chat],
                 default: false,
                 memory_budget: :infinity,
                 io: __MODULE__,
                 cache_dir: "/tmp",
                 token: "hf_secret",
                 revision: "main",
                 force: false,
                 progress: nil,
                 n_gpu_layers: 0,
                 n_parallel: 1
               )
    end

    test "a genuinely unknown option still fails loudly" do
      # The complement of the allowlists: they select what each destination reads,
      # they do not swallow typos. This used to be caught by accident — the
      # `Keyword.drop/2` denylist forwarded everything it did not recognise, so
      # `Server.start_link/1`'s own validation raised inside the supervised child.
      # Two allowlists would have dropped `n_paralell` silently and started a
      # server with the default `:n_parallel`, so the union is now validated here,
      # naming the function the caller actually called.
      error =
        assert_raise ArgumentError, fn ->
          ModelIO.start_server("chat", "/nonexistent/model.gguf", n_paralell: 2)
        end

      message = Exception.message(error)
      assert message =~ "n_paralell"
      assert message =~ "did you mean :n_parallel?"
      assert message =~ "LlamaCppEx.ModelManager.load/3"
    end

    # W-23: this used to assert `count_children(...).active == 0` after a failed
    # `start_child`, which cannot fail on the property it names — `active` is 0
    # after a failed start for *any* `restart:` value, `:permanent` included. The
    # spec itself is the observable thing, so assert that.
    test "the child spec is :temporary, so a crashed server is not resurrected" do
      assert {:error, _} = ModelIO.start_server("chat", "/nonexistent/model.gguf", [])
      assert DynamicSupervisor.count_children(ModelIO.dynamic_supervisor()).active == 0
    end
  end

  # A stub standing in for the DynamicSupervisor, so the child spec
  # `start_server/3` builds can be read directly. `DynamicSupervisor.start_child/2`
  # is a `GenServer.call(sup, {:start_child, child})`, and `child` arrives in the
  # normalized `{mfa, restart, shutdown, type, modules}` form.
  defmodule SpecRecorder do
    @moduledoc false
    use GenServer

    def start_link(test_pid) do
      GenServer.start_link(__MODULE__, test_pid, name: ModelIO.dynamic_supervisor())
    end

    @impl true
    def init(test_pid), do: {:ok, test_pid}

    @impl true
    def handle_call({:start_child, child}, _from, test_pid) do
      send(test_pid, {:child_spec, child})
      {:reply, {:error, :recorded}, test_pid}
    end
  end

  describe "the child spec start_server/3 builds" do
    setup do
      start_supervised!({SpecRecorder, self()})
      :ok
    end

    test "names Server.start_link/1 as :temporary, so a crash is not resurrected" do
      # ModelManager disowns a crashed server and marks the entry :error; a
      # :permanent or :transient child would have the DynamicSupervisor bring it
      # back behind the manager's back, leaving an unowned process holding VRAM.
      assert {:error, :recorded} =
               ModelIO.start_server("chat", "/nonexistent/model.gguf", n_parallel: 1)

      assert_receive {:child_spec, child}

      assert {{LlamaCppEx.Server, :start_link, [opts]}, :temporary, _shutdown, :worker,
              [LlamaCppEx.Server]} = child

      # And the opts are the ones the manager meant to pass: the resolved path, the
      # via-tuple name, and only server options.
      assert Keyword.fetch!(opts, :model_path) == "/nonexistent/model.gguf"
      assert Keyword.fetch!(opts, :name) == {:via, Registry, {ModelIO.registry(), "chat"}}
      assert Keyword.fetch!(opts, :n_parallel) == 1
    end

    test "carries no hub or manager options" do
      assert {:error, :recorded} =
               ModelIO.start_server("chat", "/nonexistent/model.gguf",
                 mode: :server,
                 capabilities: [:chat],
                 default: false,
                 memory_budget: :infinity,
                 io: __MODULE__,
                 cache_dir: "/tmp",
                 token: "hf_secret",
                 revision: "main",
                 force: false,
                 progress: nil,
                 vocab_only: true,
                 n_gpu_layers: 0
               )

      assert_receive {:child_spec, {{_, _, [opts]}, _, _, _, _}}

      for key <- [
            :mode,
            :capabilities,
            :default,
            :memory_budget,
            :io,
            :cache_dir,
            :token,
            :revision,
            :force,
            :progress,
            :vocab_only
          ] do
        refute Keyword.has_key?(opts, key),
               "#{inspect(key)} must not reach Server.start_link/1"
      end

      # ...and the one legitimate model option still gets through.
      assert Keyword.fetch!(opts, :n_gpu_layers) == 0
    end
  end

  describe "stop_server/1" do
    setup do
      start_supervised!(
        {DynamicSupervisor, strategy: :one_for_one, name: ModelIO.dynamic_supervisor()}
      )

      :ok
    end

    test "a pid the supervisor does not own is :ok, not an error" do
      # ModelManager calls this on its way out of states where the server may
      # already be gone, so :not_found is a success.
      {:ok, stranger} = Agent.start(fn -> :ok end)
      on_exit(fn -> if Process.alive?(stranger), do: Agent.stop(stranger) end)

      assert ModelIO.stop_server(stranger) == :ok
      assert Process.alive?(stranger)
    end

    test "terminates a child it does own" do
      spec = %{id: :probe, start: {Agent, :start_link, [fn -> :ok end]}, restart: :temporary}
      {:ok, pid} = DynamicSupervisor.start_child(ModelIO.dynamic_supervisor(), spec)

      assert ModelIO.stop_server(pid) == :ok
      refute Process.alive?(pid)
    end
  end
end
