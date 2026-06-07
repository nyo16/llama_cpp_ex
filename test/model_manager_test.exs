defmodule LlamaCppEx.ModelManagerTest do
  # async: false — the manager registers under a fixed name and owns named ETS
  # tables, so tests must not run concurrently.
  use ExUnit.Case, async: false

  alias LlamaCppEx.ModelManager

  # A stub standing in for LlamaCppEx.Server. It answers the two messages the
  # manager's dispatch sends it: {:generate, prompt, max_tokens} and :get_model.
  # Started unlinked so killing it (DOWN test) doesn't take down the manager.
  defmodule StubServer do
    use GenServer

    def start(reply), do: GenServer.start(__MODULE__, reply)

    @impl true
    def init(reply), do: {:ok, reply}

    @impl true
    def handle_call({:generate, _prompt, _max}, _from, reply), do: {:reply, reply, reply}
    def handle_call(:get_model, _from, reply), do: {:reply, fake_model(), reply}

    defp fake_model, do: %LlamaCppEx.Model{ref: make_ref()}
  end

  # Fake Backend: no real GGUF files, no native loads. Behaviour is driven by
  # keys threaded through the load opts (`fake_bytes`, `fake_*_error`, ...).
  defmodule FakeIO do
    @behaviour LlamaCppEx.ModelManager.Backend

    @impl true
    def resolve_source(source, opts) do
      case Keyword.get(opts, :fake_resolve_error) do
        nil ->
          path = with {:path, p} <- source, do: p
          {:ok, to_string(path), Keyword.get(opts, :fake_bytes, 1_000)}

        reason ->
          {:error, reason}
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
      assert {:ok, "hello"} = ModelManager.generate("chat", "hi")
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
      assert {:ok, "d"} = ModelManager.generate(:default, "hi")
    end

    test "set_default/1 updates the default" do
      {:ok, _} = ModelManager.load("a", {:path, "a.gguf"}, fake_reply: {:ok, "from-a"})
      {:ok, _} = ModelManager.load("b", {:path, "b.gguf"}, fake_reply: {:ok, "from-b"})

      assert :ok = ModelManager.set_default("b")
      assert ModelManager.default() == "b"
      assert {:ok, "from-b"} = ModelManager.generate(:default, "hi")
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

      Process.exit(pid, :kill)

      # The :DOWN is delivered by the runtime asynchronously, so poll until the
      # manager has marked the entry :error.
      eventually(fn -> match?({:ok, %{status: :error}}, ModelManager.info("chat")) end)

      assert {:error, {:not_ready, :error}} = ModelManager.generate("chat", "hi")
    end
  end

  defp eventually(fun, retries \\ 100) do
    cond do
      fun.() -> :ok
      retries == 0 -> flunk("condition not met within timeout")
      true -> Process.sleep(10) && eventually(fun, retries - 1)
    end
  end
end
