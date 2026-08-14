defmodule LlamaCppEx.RPC do
  @moduledoc """
  Remote ggml devices over the llama.cpp RPC backend.

  An RPC *worker* runs `LlamaCppEx.RPC.Server` and exposes its local devices on a
  TCP endpoint. A *client* registers that endpoint with `add_server/1`, after
  which the remote devices appear in `LlamaCppEx.devices/0` and can hold part of
  a model — so a model larger than one machine's memory can be loaded across two.

      # on the worker
      {:ok, _} = LlamaCppEx.RPC.Server.start_link(endpoint: "10.100.64.2:50052")

      # on the client, before loading
      {:ok, 1} = LlamaCppEx.RPC.add_server("10.100.64.2:50052")
      {:ok, model} = LlamaCppEx.Model.load(path, split_mode: :layer, tensor_split: [0.5, 0.5])

  `LlamaCppEx.Model.load/2` accepts `:rpc_servers` and does the registration for
  you, in the right order.

  ## Build requirement

  The RPC backend is not compiled in by default. Build with `LLAMA_RPC=1`;
  otherwise every function here returns `{:error, :rpc_unsupported}`.

      LLAMA_RPC=1 LLAMA_BACKEND=cuda mix compile

  On Linux the transport auto-negotiates RDMA when both peers have a usable HCA,
  which is a build-time choice (`LLAMA_RPC_RDMA`, default on) with no runtime
  switch. See `LlamaCppEx.RPC.Server` for how to tell which transport you got.

  > #### A peer failure kills the VM {: .error}
  >
  > This is a property of upstream llama.cpp, not of this binding, and stating it
  > is better than hiding it. Every client-side RPC command checks its result
  > with `RPC_STATUS_ASSERT`, which is `GGML_ABORT` — so a peer that crashes,
  > a network that drops, or a malformed response **terminates the OS process**,
  > taking the BEAM with it. There is no error return, no retry and no reconnect
  > to catch.
  >
  > Registration is the exception and the reason this module is shaped the way it
  > is: an unreachable endpoint is reported as `{:error, :unreachable}` rather
  > than aborting, so a two-node setup can be *checked* before a model load and
  > is only *fatal* during one. Check with `ping/1` before you load; treat the VM
  > as the unit of restart afterwards. Real fault isolation would mean running
  > the RPC client in a separate OS process, which is a different architecture.

  ## Two device orderings, and they disagree

  `LlamaCppEx.devices/0` reports the ggml **registry**, which is registration
  order: local backends first, RPC endpoints appended as they register.
  llama.cpp builds a **different** list for placement at load time and inserts
  RPC devices at the **front** of it, to minimise network transfers. So with one
  local GPU and one endpoint, `devices/0` shows `[CUDA0, CPU, RPC0]` while
  `tensor_split` indexes `[RPC0, CUDA0]` — `tensor_split: [0.25, 0.75]` puts 25%
  on the **remote** node, and `main_gpu: 0` selects it.

  A backwards split still produces correct tokens and simply runs badly, so
  nothing tells you. `:gpu_index` from `devices/0` does **not** index
  `:tensor_split` once a remote device exists. Pass `:devices` to
  `LlamaCppEx.Model.load/2` — it is used verbatim — and stop guessing:

      LlamaCppEx.Model.load(path,
        rpc_servers: ["10.100.64.2:50052"],
        devices: ["CUDA0", "RPC0"],
        split_mode: :layer,
        tensor_split: [0.6, 0.4])

  See `docs/multi-gpu.md` for the worked example.
  """

  alias LlamaCppEx.NIF

  @type endpoint :: String.t()
  @type error :: :rpc_unsupported | :unreachable

  @doc """
  Whether this build has the RPC backend compiled in.

  `false` means the NIF was built without `LLAMA_RPC=1`, and every other function
  in this module will return `{:error, :rpc_unsupported}`.

      iex> LlamaCppEx.RPC.supported?()
      false

  Worth checking rather than inferring from an error, because
  `:rpc_unsupported` and `:unreachable` are easy to confuse and mean very
  different things: the first is a build problem, the second is a network or
  version-mismatch problem. Code that treats them alike will eventually paper
  over an artifact built with the wrong flags.
  """
  @spec supported?() :: boolean()
  def supported?, do: NIF.rpc_supported()

  @doc """
  Registers a remote endpoint's devices in the global device registry.

  Returns the number of devices the endpoint contributed. Idempotent: upstream
  memoizes the registration per endpoint, so registering the same endpoint twice
  succeeds and reports `0` added the second time.

  Must be called **before** `LlamaCppEx.Model.load/2`, because tensor placement
  is computed from the devices that exist at load time.

      iex> LlamaCppEx.RPC.add_server("10.100.64.2:50052")
      {:ok, 1}

  ## Errors

    * `{:error, :rpc_unsupported}` — the NIF was built without `LLAMA_RPC=1`.
    * `{:error, :unreachable}` — nothing answered, or the peer's RPC protocol
      major/minor did not match ours. Upstream collapses both to a null
      registration, and a null registration is silently ignored by
      `ggml_backend_register`, so this check is the only thing standing between
      you and a model that quietly loads onto the wrong devices.
  """
  @spec add_server(endpoint()) :: {:ok, non_neg_integer()} | {:error, error()}
  def add_server(endpoint) when is_binary(endpoint) do
    NIF.backend_init()
    NIF.rpc_add_server(endpoint)
  end

  @doc """
  Registers several endpoints, in order.

  Stops at the first failure and reports which endpoint failed, because a
  partially registered set would place tensors somewhere nobody intended.

      iex> LlamaCppEx.RPC.add_servers(["10.100.64.2:50052", "10.100.64.3:50052"])
      {:ok, 2}
  """
  @spec add_servers([endpoint()]) ::
          {:ok, non_neg_integer()} | {:error, {endpoint(), error()}}
  def add_servers(endpoints) when is_list(endpoints) do
    Enum.reduce_while(endpoints, {:ok, 0}, fn endpoint, {:ok, total} ->
      case add_server(endpoint) do
        {:ok, n} -> {:cont, {:ok, total + n}}
        {:error, reason} -> {:halt, {:error, {endpoint, reason}}}
      end
    end)
  end

  @doc """
  The registered remote devices, in placement order.

  A filtered view of `LlamaCppEx.devices/0`. Two things to know about what comes
  back:

    * `:type` is always `:gpu`, even when the remote server exposes only a CPU
      device — upstream hardcodes it with a TODO.
    * `:description` is the endpoint string, which is the only way to tell two
      remote devices apart.

  """
  @spec devices() :: [map()]
  def devices do
    Enum.filter(LlamaCppEx.devices(), &(&1.backend == "RPC"))
  end

  @doc """
  Reports whether an endpoint is reachable and speaks a compatible protocol.

  This is `add_server/1` under a name that says what it is good for. It has the
  same side effect — a reachable endpoint stays registered — and it is **not** a
  health check you can repeat to monitor a peer: the answer is memoized after the
  first success, and once a model is loaded a dead peer aborts the VM rather than
  failing a probe.

  Use it once, before loading, to turn "the model landed on the wrong devices"
  into a clear error.
  """
  @spec ping(endpoint()) :: :ok | {:error, error()}
  def ping(endpoint) when is_binary(endpoint) do
    case add_server(endpoint) do
      {:ok, _} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end
end
