defmodule LlamaCppEx.ModelManagerTest do
  # async: false — the manager registers under a fixed name and owns named ETS
  # tables, so tests must not run concurrently.
  use ExUnit.Case, async: false

  alias LlamaCppEx.ModelManager

  # A stub standing in for LlamaCppEx.Server. It answers the raw protocol
  # messages a routed request produces: {:generate_tokens, ...} and
  # :get_model. NOTE: full-path ModelManager.generate/3 can't be unit-tested
  # against a stub anymore — Server.generate tokenizes in the CALLER via
  # get_model/1, which needs a real model NIF resource; routing tests assert
  # via route/1 + the raw tokens call instead (real-path coverage lives in
  # the smoke tests). Started unlinked so killing it (DOWN test) doesn't take
  # down the manager.
  defmodule StubServer do
    use GenServer

    def start(reply), do: GenServer.start(__MODULE__, reply)

    @impl true
    def init(reply), do: {:ok, reply}

    @impl true
    def handle_call({:generate_tokens, _tokens, _max, _req_opts}, _from, reply),
      do: {:reply, reply, reply}

    def handle_call(:get_model, _from, reply), do: {:reply, fake_model(), reply}

    defp fake_model, do: %LlamaCppEx.Model{ref: make_ref()}
  end

  # Routes id through the manager and performs the stubbed raw call — the
  # unit-testable equivalent of ModelManager.generate/3 for :server entries.
  defp generate_via_route(id) do
    with {:ok, {:server, pid, _entry}} <- ModelManager.route(id) do
      GenServer.call(pid, {:generate_tokens, [1, 2, 3], 256, []})
    end
  end

  # Fake Backend: no real GGUF files, no native loads. Behaviour is driven by
  # keys threaded through the load opts (`fake_bytes`, `fake_*_error`, ...).
  defmodule FakeIO do
    @behaviour LlamaCppEx.ModelManager.Backend

    @impl true
    def resolve_source(source, opts) do
      case Keyword.get(opts, :fake_resolve_error) do
        nil ->
          maybe_block(opts)
          {:ok, resolve_path(source), Keyword.get(opts, :fake_bytes, 1_000)}

        reason ->
          {:error, reason}
      end
    end

    defp resolve_path({:path, p}), do: p
    defp resolve_path({:hub, _repo, file}), do: file

    # Optional rendezvous: when `:fake_block` is a pid, notify it and park here
    # until released, so a test can hold a load mid-resolve and prove the manager
    # stays responsive to other calls.
    defp maybe_block(opts) do
      case Keyword.get(opts, :fake_block) do
        nil ->
          :ok

        notify ->
          send(notify, {:resolving, self()})

          receive do
            :release -> :ok
          end
      end
    end

    @impl true
    def load_model(_path, opts) do
      case Keyword.get(opts, :fake_load_error) do
        nil -> {:ok, %LlamaCppEx.Model{ref: make_ref()}}
        reason -> {:error, reason}
      end
    end

    @impl true
    def start_server(_id, _path, opts) do
      case Keyword.get(opts, :fake_start_error) do
        nil -> StubServer.start(Keyword.get(opts, :fake_reply, {:ok, "stub"}))
        reason -> {:error, reason}
      end
    end

    @impl true
    def stop_server(pid) do
      if Process.alive?(pid), do: GenServer.stop(pid, :normal, 1_000)
      :ok
    end
  end

  defp start_manager(opts \\ []) do
    start_supervised!({ModelManager, Keyword.merge([io: FakeIO], opts)})
    :ok
  end

  describe "load/3 and lifecycle" do
    setup do
      start_manager()
      :ok
    end

    test "loads a server-backed model and reports it ready" do
      assert {:ok, "chat"} = ModelManager.load("chat", {:path, "chat.gguf"})
      assert ModelManager.loaded?("chat")
      assert {:ok, %{id: "chat", status: :ready, mode: :server}} = ModelManager.info("chat")
    end

    test "list/0 returns sanitized views without raw refs" do
      {:ok, _} = ModelManager.load("chat", {:path, "chat.gguf"})
      [entry] = ModelManager.list()

      assert entry.id == "chat"
      assert entry.status == :ready
      refute Map.has_key?(entry, :model)
      refute Map.has_key?(entry, :server_pid)
    end

    test "embedding capability forces :direct mode" do
      {:ok, _} = ModelManager.load("emb", {:path, "emb.gguf"}, capabilities: [:embed])
      assert {:ok, %{mode: :direct, capabilities: [:embed]}} = ModelManager.info("emb")
    end

    test "explicit mode override is honored" do
      {:ok, _} = ModelManager.load("d", {:path, "d.gguf"}, mode: :direct)
      assert {:ok, %{mode: :direct}} = ModelManager.info("d")
    end

    test "refuses to load the same id twice" do
      {:ok, _} = ModelManager.load("chat", {:path, "chat.gguf"})
      assert {:error, :already_loaded} = ModelManager.load("chat", {:path, "chat.gguf"})
    end

    test "unload removes the model and frees the backing server" do
      {:ok, _} = ModelManager.load("chat", {:path, "chat.gguf"})
      {:ok, {:server, pid, _}} = ModelManager.route("chat")

      assert :ok = ModelManager.unload("chat")
      refute ModelManager.loaded?("chat")
      assert {:error, :not_loaded} = ModelManager.info("chat")
      refute Process.alive?(pid)
    end

    test "unload of an unknown model errors" do
      assert {:error, :not_loaded} = ModelManager.unload("nope")
    end

    test "propagates resolve and start errors" do
      assert {:error, :boom} =
               ModelManager.load("x", {:path, "x.gguf"}, fake_resolve_error: :boom)

      refute ModelManager.loaded?("x")

      assert {:error, :no_start} =
               ModelManager.load("y", {:path, "y.gguf"}, fake_start_error: :no_start)

      refute ModelManager.loaded?("y")
    end
  end

  describe "routing" do
    setup do
      start_manager()
      :ok
    end

    test "generate routes to the server-backed model" do
      {:ok, _} = ModelManager.load("chat", {:path, "chat.gguf"}, fake_reply: {:ok, "hello"})
      assert {:ok, "hello"} = generate_via_route("chat")
    end

    test "generate on a missing model returns :not_loaded" do
      assert {:error, :not_loaded} = ModelManager.generate("ghost", "hi")
    end

    test "route reports the dispatch target per mode" do
      {:ok, _} = ModelManager.load("s", {:path, "s.gguf"}, mode: :server)
      {:ok, _} = ModelManager.load("d", {:path, "d.gguf"}, mode: :direct)

      assert {:ok, {:server, pid, _}} = ModelManager.route("s")
      assert is_pid(pid)
      assert {:ok, {:direct, %LlamaCppEx.Model{}, _}} = ModelManager.route("d")
    end

    test "fetch_model returns the raw model for server and direct modes" do
      {:ok, _} = ModelManager.load("s", {:path, "s.gguf"}, mode: :server)
      assert {:ok, %LlamaCppEx.Model{}} = ModelManager.fetch_model("s")

      {:ok, _} = ModelManager.load("d", {:path, "d.gguf"}, mode: :direct)
      assert {:ok, %LlamaCppEx.Model{}} = ModelManager.fetch_model("d")

      assert {:error, :not_loaded} = ModelManager.fetch_model("missing")
    end

    test "embed refuses non-embedding models" do
      {:ok, _} = ModelManager.load("chat", {:path, "chat.gguf"}, mode: :server)
      assert {:error, :not_embedding_model} = ModelManager.embed("chat", "text")

      {:ok, _} = ModelManager.load("plain", {:path, "p.gguf"}, mode: :direct)
      assert {:error, :not_embedding_model} = ModelManager.embed("plain", "text")
    end

    test "stream raises for a model that is not ready" do
      assert_raise ArgumentError, fn ->
        ModelManager.stream("ghost", "hi") |> Enum.to_list()
      end
    end
  end

  describe "default routing" do
    setup do
      start_manager()
      :ok
    end

    test "load with default: true sets the default and :default routes to it" do
      {:ok, _} =
        ModelManager.load("chat", {:path, "chat.gguf"}, default: true, fake_reply: {:ok, "d"})

      assert ModelManager.default() == "chat"
      assert {:ok, "d"} = generate_via_route(:default)
    end

    test "set_default/1 updates the default" do
      {:ok, _} = ModelManager.load("a", {:path, "a.gguf"}, fake_reply: {:ok, "from-a"})
      {:ok, _} = ModelManager.load("b", {:path, "b.gguf"}, fake_reply: {:ok, "from-b"})

      assert :ok = ModelManager.set_default("b")
      assert ModelManager.default() == "b"
      assert {:ok, "from-b"} = generate_via_route(:default)
    end

    test "set_default on a missing model errors" do
      assert {:error, :not_loaded} = ModelManager.set_default("nope")
    end

    test "unloading the default clears it" do
      {:ok, _} = ModelManager.load("a", {:path, "a.gguf"}, default: true)
      assert ModelManager.default() == "a"
      :ok = ModelManager.unload("a")
      assert ModelManager.default() == nil
    end
  end

  describe "memory budget" do
    test "an integer budget is a combined pool that refuses over-budget loads" do
      start_manager(memory_budget: 5_000)

      # n_gpu_layers: 0 keeps it in RAM so the test is GPU-independent; the
      # combined pool sums RAM + VRAM either way.
      assert {:error, {:insufficient_memory, device: :total, required: 6_000, available: 5_000}} =
               ModelManager.load("big", {:path, "big.gguf"},
                 mode: :direct,
                 n_gpu_layers: 0,
                 fake_bytes: 6_000
               )

      refute ModelManager.loaded?("big")
    end

    test "accounts for already-resident models when checking the budget" do
      start_manager(memory_budget: 5_000)

      assert {:ok, _} =
               ModelManager.load("a", {:path, "a.gguf"},
                 mode: :direct,
                 n_gpu_layers: 0,
                 fake_bytes: 3_000
               )

      assert {:error, {:insufficient_memory, device: :total, required: 3_000, available: 2_000}} =
               ModelManager.load("b", {:path, "b.gguf"},
                 mode: :direct,
                 n_gpu_layers: 0,
                 fake_bytes: 3_000
               )
    end

    test "a per-device budget refuses on the over-budget GPU" do
      # Requires at least one GPU device (e.g. Metal/CUDA). Skipped on CPU-only.
      case LlamaCppEx.devices() |> Enum.filter(&(&1.type in [:gpu, :igpu])) do
        [%{gpu_index: gi} | _] ->
          start_manager(memory_budget: %{ram: :infinity, vram: %{gi => 5_000}})

          assert {:error,
                  {:insufficient_memory, device: {:gpu, ^gi}, required: 6_000, available: 5_000}} =
                   ModelManager.load("big", {:path, "big.gguf"},
                     mode: :direct,
                     n_gpu_layers: -1,
                     split_mode: :none,
                     main_gpu: gi,
                     fake_bytes: 6_000
                   )

        [] ->
          :ok
      end
    end

    test "infinity budget always fits" do
      start_manager(memory_budget: :infinity)

      assert {:ok, _} =
               ModelManager.load("huge", {:path, "huge.gguf"},
                 mode: :direct,
                 fake_bytes: 999_999_999_999
               )
    end
  end

  describe "backing server crash" do
    setup do
      start_manager()
      :ok
    end

    test "marks the model :error when its server goes down" do
      {:ok, _} = ModelManager.load("chat", {:path, "chat.gguf"})
      {:ok, {:server, pid, _}} = ModelManager.route("chat")

      # Monitor from the test too. When we receive our own :DOWN, the manager's
      # monitor message is already enqueued at the same process-death event,
      # ahead of the sync call below — so once that call returns, the entry has
      # been marked :error. Deterministic, no polling.
      ref = Process.monitor(pid)
      Process.exit(pid, :kill)
      assert_receive {:DOWN, ^ref, :process, ^pid, _}

      # A sync call drains the manager mailbox after the :DOWN it handled.
      assert {:error, :not_loaded} = ModelManager.unload(make_ref())

      assert {:ok, %{status: :error}} = ModelManager.info("chat")
      assert {:error, {:not_ready, :error}} = ModelManager.generate("chat", "hi")
    end
  end

  describe "autoload" do
    test "loads models listed in the :models option on start" do
      start_manager(
        models: [
          {"a", {:path, "a.gguf"}},
          {"b", {:path, "b.gguf"}, mode: :direct}
        ]
      )

      # Autoload runs synchronously in handle_continue, before any external call
      # is served; a sync call guarantees it has completed before we assert.
      assert {:error, :not_loaded} = ModelManager.unload(make_ref())

      assert {:ok, %{mode: :server, status: :ready}} = ModelManager.info("a")
      assert {:ok, %{mode: :direct, status: :ready}} = ModelManager.info("b")
    end
  end

  describe "async load" do
    setup do
      start_manager()
      :ok
    end

    test "a slow load does not block other manager calls" do
      parent = self()

      slow =
        Task.async(fn ->
          ModelManager.load("slow", {:path, "slow.gguf"}, fake_block: parent)
        end)

      # "slow" is now parked inside resolve_source, mid-load.
      assert_receive {:resolving, resolver_pid}, 1_000
      assert {:ok, %{status: :loading}} = ModelManager.info("slow")

      # The manager mailbox is free: another load completes and reads answer.
      assert {:ok, "fast"} = ModelManager.load("fast", {:path, "fast.gguf"})
      assert ModelManager.loaded?("fast")

      send(resolver_pid, :release)
      assert {:ok, "slow"} = Task.await(slow)
      assert ModelManager.loaded?("slow")
    end

    test "rejects a second load of the same id while the first is in flight" do
      parent = self()

      first =
        Task.async(fn ->
          ModelManager.load("dup", {:path, "dup.gguf"}, fake_block: parent)
        end)

      assert_receive {:resolving, resolver_pid}, 1_000
      assert {:error, :already_loaded} = ModelManager.load("dup", {:path, "dup.gguf"})

      send(resolver_pid, :release)
      assert {:ok, "dup"} = Task.await(first)
    end
  end
end
