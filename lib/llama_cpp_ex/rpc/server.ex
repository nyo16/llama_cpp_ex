defmodule LlamaCppEx.RPC.Server do
  @moduledoc """
  The worker side of the llama.cpp RPC backend: serves this node's devices to a
  remote client.

      children = [
        {LlamaCppEx.RPC.Server,
         endpoint: "10.100.64.2:50052",
         cache_dir: Path.expand("~/.cache/llama.cpp/rpc"),
         n_threads: 10}
      ]

  ## Bind to the fabric address, not localhost

  `endpoint` is required and there is deliberately no default. Upstream's default
  is `127.0.0.1`, which is useless here twice over: a remote client cannot reach
  it, and **RDMA can never engage on it**, because the transport selects an HCA
  by matching a GID against the socket's *local* address.

  Nothing about this endpoint is authenticated or encrypted. It is a plain TCP
  port that accepts commands to allocate memory and execute compute graphs. Bind
  it to a point-to-point fabric address, never to a routable interface.

  ## The tensor cache is worth having

  `:cache_dir` enables upstream's content-addressed cache for tensors over
  10 MiB. Without it every model load re-pushes the whole remote share across the
  network; with it a warm load is close to free. There is no default — pass a
  path or accept the cost knowingly.

  ## Which transport did I get?

  Transport selection is silent auto-negotiation. There is no env var, no
  endpoint scheme and no return value that tells you. The only signal is the
  worker's own log with `GGML_RPC_DEBUG=1`: the absence of
  `RDMA activate failed, staying on TCP` means RDMA is live. To force TCP for an
  A/B, set `GGML_RDMA_DEV` to a device name that does not exist.

  ## This process cannot stop the server

  `ggml_backend_rpc_start_server` never returns — its accept loop is
  `while (true)` and the cleanup after it is unreachable — so the native server
  runs on a detached thread that outlives this GenServer. `terminate/2` therefore
  does nothing but say so. **The OS process is the unit of restart**, and that
  matters in practice: upstream's RPC worker is reported by multiple independent
  users to grow RSS during inference and never release it, so a long-lived worker
  wants restarting between runs and watching during them.

  Starting it twice on one endpoint fails cleanly at the bind, which is the one
  guard rail this design does provide — and it is also why this child is
  **`restart: :temporary`**. A restart could not succeed: the listening socket
  lives on that detached thread inside the same OS process, so it survives this
  GenServer dying, and the next `init/1` would fail `EADDRINUSE` at the
  pre-flight bind (`SO_REUSEADDR` does not let you bind over a socket that is
  actively listening). Under the default `:permanent` a single crash would retry,
  fail identically, exhaust `max_restarts` and take the whole supervision subtree
  down — reporting a bind error rather than the original cause. `:temporary` is
  what "the OS process is the unit of restart" means as a child spec.

  This module exists to own a native resource's lifecycle and to be supervised,
  which is the only reason to add a process. It carries no state a function could
  not compute.
  """

  use GenServer, restart: :temporary

  require Logger

  alias LlamaCppEx.NIF

  @type option ::
          {:endpoint, String.t()}
          | {:cache_dir, String.t() | nil}
          | {:n_threads, pos_integer()}
          | {:devices, [String.t()]}
          | {:name, GenServer.name()}

  @doc """
  Starts the RPC worker.

  ## Options

    * `:endpoint` — **required**, `"host:port"`. Bind to a fabric address.
    * `:cache_dir` — content-addressed tensor cache directory. Default `nil`
      (no cache).
    * `:n_threads` — CPU threads for the served backends. Default `4`, matching
      upstream. On a DGX Spark use `10` and pin the process to cores `5-9,15-19`;
      the performance and efficiency clusters are interleaved, so the default
      affinity spreads work across both.
    * `:devices` — device names to serve, e.g. `["CUDA0"]`. Default: every
      non-CPU device, falling back to the CPU device.

  Returns `{:error, {:rpc, reason}}` rather than starting when the endpoint
  cannot be served. Notable reasons:

    * `:rpc_unsupported` — built without `LLAMA_RPC=1`.
    * `:bind_timeout` — the native thread never reached `listen`.
    * `:no_devices` — nothing to serve.
    * a string — a bad endpoint, an unknown device name, or the bind's `errno`.
  """
  @spec start_link([option()]) :: GenServer.on_start()
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name)
    GenServer.start_link(__MODULE__, opts, if(name, do: [name: name], else: []))
  end

  @doc "The endpoint this worker is serving, and the device names it exposes."
  @spec info(GenServer.server()) :: %{endpoint: String.t(), devices: [String.t()]}
  def info(server), do: GenServer.call(server, :info)

  @impl true
  def init(opts) do
    # Without this, `terminate/2` below never runs in the case it was written
    # for. GenServer only turns a parent exit signal into a `terminate/2` call
    # when the process traps exits, and a Supervisor shuts a child down with
    # `exit(pid, :shutdown)` — so the supervised deployment in this module's own
    # child-spec example died silently and the "the native server keeps running"
    # warning never reached the operator who most needs it.
    Process.flag(:trap_exit, true)

    endpoint = Keyword.fetch!(opts, :endpoint)
    cache_dir = Keyword.get(opts, :cache_dir)
    n_threads = Keyword.get(opts, :n_threads, 4)
    devices = Keyword.get(opts, :devices, [])

    if cache_dir, do: File.mkdir_p!(cache_dir)

    NIF.backend_init()

    case NIF.rpc_start_server(endpoint, cache_dir || "", n_threads, devices) do
      {:ok, served} ->
        Logger.info(
          "RPC server listening on #{endpoint}, serving #{Enum.join(served, ", ")} " <>
            "(#{n_threads} threads, cache #{cache_dir || "disabled"})"
        )

        {:ok, %{endpoint: endpoint, devices: served}}

      {:error, reason} ->
        {:stop, {:rpc, reason}}
    end
  end

  @impl true
  def handle_call(:info, _from, state), do: {:reply, state, state}

  @impl true
  def terminate(_reason, state) do
    # Honest rather than reassuring: upstream's accept loop never returns and
    # there is no shutdown hook, so the native thread and its port outlive this
    # process. Only exiting the VM frees them.
    Logger.warning(
      "RPC server process stopping, but the native server on #{state.endpoint} keeps running. " <>
        "Restart the VM to release the port."
    )

    :ok
  end
end
