defmodule LlamaCppEx.Model do
  @moduledoc """
  Model loading and introspection.
  """

  @enforce_keys [:ref]
  defstruct [:ref, load_mtp: false]

  @type t :: %__MODULE__{ref: reference(), load_mtp: boolean()}

  @tuning_option_keys [
    :main_gpu,
    :split_mode,
    :tensor_split,
    :use_mmap,
    :use_mlock,
    :use_direct_io,
    :check_tensors,
    :rpc_servers,
    :devices
  ]

  @structural_option_keys [:n_gpu_layers, :vocab_only, :load_mtp]

  @doc """
  Options that are safe for a caller to forward from user-supplied opts.

  `LlamaCppEx.Server` selects its model options with this function rather than
  keeping its own copy of the list.
  """
  @spec tuning_option_keys() :: [atom()]
  def tuning_option_keys, do: @tuning_option_keys

  @doc """
  Options a caller must set explicitly rather than forward blindly.

  `:vocab_only` in particular must never be forwarded into a server — it would
  load a model with no weights.
  """
  @spec structural_option_keys() :: [atom()]
  def structural_option_keys, do: @structural_option_keys

  @doc """
  Loads a GGUF model from the given file path.

  ## Options

    * `:n_gpu_layers` - Number of layers to offload to GPU. Use `-1` for all layers.
      Defaults to `99` (offload all layers).
    * `:use_mmap` - Whether to memory-map the model file. Defaults to `true`.
    * `:main_gpu` - GPU device index for single-GPU mode. Defaults to `0`.
    * `:split_mode` - How to split the model across devices: `:none`, `:layer`,
      `:row` or `:tensor`. Defaults to `:none`. Only `:none` and `:layer` are
      generally usable at this llama.cpp version — see the note below.
    * `:tensor_split` - List of floats specifying the proportion of work per GPU
      (e.g. `[0.5, 0.5]` for two GPUs). Defaults to `[]`.
    * `:use_mlock` - Pin model memory in RAM to prevent swapping. Implies `:use_mmap`.
      Defaults to `false`.
    * `:use_direct_io` - Bypass page cache when loading (takes precedence over mmap).
      Defaults to `false`.
    * `:vocab_only` - Load vocabulary and metadata only, skip weights. Defaults to `false`.
    * `:check_tensors` - Validate model tensor data on load. Defaults to `false`,
      because the check walks every tensor and costs real time on a large model.
    * `:load_mtp` - Load the Multi-Token Prediction head's layers, for use with
      `LlamaCppEx.MTP`. Defaults to `false`, matching upstream, so that callers
      who are not doing speculative decoding do not pay for the extra tensors.
      Required for `LlamaCppEx.MTP.init/2`, which refuses a model loaded without
      it — the layers cannot be added after the fact.
    * `:rpc_servers` - Endpoints (`"host:port"`) to register before loading, so
      their remote devices can hold part of the model. Defaults to `[]`. Requires
      a build with `LLAMA_RPC=1`. See `LlamaCppEx.RPC`. Note that llama.cpp puts
      remote devices **first** in its automatic placement list — which is not the
      order `LlamaCppEx.devices/0` reports — so `tensor_split: [0.25, 0.75]`
      gives 25% to the first remote endpoint. Pass `:devices` to avoid guessing.
    * `:devices` - Device names, e.g. `["CUDA0", "RPC0"]`, used **verbatim** as
      the placement list: no reordering, no dedup, no CPU filtering. Defaults to
      `[]`, which lets llama.cpp build the list itself. Set this whenever more
      than one device is in play, because the automatic list is **not** the order
      `LlamaCppEx.devices/0` reports — it puts RPC devices first — so
      `:tensor_split` and `:main_gpu` would index a list you never saw. With
      `:devices` set, they index this one.

  > #### Split modes at llama.cpp b10830 (`465e49b9`) {: .warning}
  >
  > `:layer` splits contiguous layer ranges across devices, one KV cache per
  > device, and is the only mode that works across hosts.
  >
  > `:row` throws at load time on CUDA: ggml-cuda no longer exports
  > `ggml_backend_split_buffer_type`, so `llama_model_load` raises
  > `device CUDA0 does not support split buffers`. It is kept mapped to its
  > upstream value rather than removed, because the enum is upstream's, but do
  > not build on it. Only SYCL still declares a split buffer type.
  >
  > `:tensor` is real tensor parallelism via a Meta device, added in
  > llama.cpp #19378. It forces flash attention on, refuses some architectures,
  > disables backend sampling, and its CUDA all-reduce is `ncclCommInitAll` —
  > a single-process, all-local-GPUs API. **It cannot span hosts**, so it is not
  > "tp=2 across two machines". See `docs/dgx-spark.md` for the measurements.

  > #### Load mode {: .info}
  >
  > llama.cpp collapsed its three loading booleans into one `load_mode` enum, so
  > these options resolve to a single mode. `:use_direct_io` takes precedence
  > over everything and selects `dio`; otherwise `:use_mlock` and `:use_mmap`
  > combine — both true selects `mmap_mlock`, `:use_mlock` alone selects `mlock`
  > (read into anonymous memory, no mapping), `:use_mmap` alone selects `mmap`,
  > and all false selects `none`.

  > #### Untrusted models {: .warning}
  >
  > GGUF parsing happens in llama.cpp's C++ loader, and `:check_tensors` defaults
  > to `false` for every source — including files fetched by
  > `LlamaCppEx.Hub.download/3`, which verifies a download against the SHA-256
  > HuggingFace publishes but cannot vouch for what the repository owner uploaded.
  > `load/2` receives a bare path and has no notion of provenance, so it cannot
  > raise that default on its own: pass `check_tensors: true` explicitly for any
  > model whose publisher you do not trust.

  ## Examples

      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", n_gpu_layers: -1)
      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", split_mode: :layer, tensor_split: [0.5, 0.5])
      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", vocab_only: true)

  """
  @spec load(String.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def load(path, opts \\ []) do
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 99)
    use_mmap = Keyword.get(opts, :use_mmap, true)
    main_gpu = Keyword.get(opts, :main_gpu, 0)
    split_mode = Keyword.get(opts, :split_mode, :none) |> encode_split_mode()
    tensor_split = Keyword.get(opts, :tensor_split, [])
    use_mlock = Keyword.get(opts, :use_mlock, false)
    use_direct_io = Keyword.get(opts, :use_direct_io, false)
    vocab_only = Keyword.get(opts, :vocab_only, false)
    check_tensors = Keyword.get(opts, :check_tensors, false)
    load_mtp = Keyword.get(opts, :load_mtp, false)
    # Read here rather than in a private helper: test/option_forwarding_test.exs
    # checks both that `load/2` accepts the key and that it is one of
    # `tuning_option_keys/0`, and a helper satisfies only half of that.
    rpc_servers = Keyword.get(opts, :rpc_servers, [])
    devices = Keyword.get(opts, :devices, [])

    # Registration has to happen before the load, not during it: tensor
    # placement is computed from the devices that exist when
    # llama_model_load_from_file runs. An unreachable endpoint is reported here
    # rather than silently dropped, because ggml_backend_register no-ops on a
    # null registration and the model would then load onto the wrong devices.
    with :ok <- register_rpc_servers(rpc_servers),
         {:ok, ref} <-
           LlamaCppEx.NIF.model_load(
             path,
             n_gpu_layers,
             use_mmap,
             main_gpu,
             split_mode,
             tensor_split,
             use_mlock,
             use_direct_io,
             vocab_only,
             check_tensors,
             load_mtp,
             devices
           ) do
      {:ok, %__MODULE__{ref: ref, load_mtp: load_mtp}}
    end
  end

  defp register_rpc_servers([]), do: :ok

  defp register_rpc_servers(endpoints) do
    case LlamaCppEx.RPC.add_servers(endpoints) do
      {:ok, _n} -> :ok
      {:error, {endpoint, reason}} -> {:error, "RPC endpoint #{endpoint}: #{reason}"}
    end
  end

  @doc false
  # Exposed for tests: the mapping is upstream's `llama_split_mode` enum and a
  # silent drift here would place tensors somewhere nobody asked for.
  # 0 none, 1 layer, 2 row, 3 tensor (llama.h).
  def encode_split_mode(:none), do: 0
  def encode_split_mode(:layer), do: 1
  def encode_split_mode(:row), do: 2
  def encode_split_mode(:tensor), do: 3

  def encode_split_mode(other) do
    raise ArgumentError,
          "unknown split_mode #{inspect(other)}, expected :none, :layer, :row or :tensor"
  end

  @doc "Returns the training context size of the model."
  @spec n_ctx_train(t()) :: integer()
  def n_ctx_train(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_ctx_train(ref)

  @doc "Returns the embedding dimension of the model."
  @spec n_embd(t()) :: integer()
  def n_embd(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_embd(ref)

  @doc """
  Returns the output-side embedding width — the row width an MTP draft head
  consumes. Equal to `n_embd/1` for every architecture currently in tree; it is
  a distinct number because `LlamaCppEx.MTP` matches it across the target and a
  separate drafter GGUF.
  """
  @spec n_embd_out(t()) :: integer()
  def n_embd_out(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_embd_out(ref)

  @doc """
  Returns the number of MTP / "next-N" prediction layers in the checkpoint, or
  `0` when it carries no MTP head. Note this reports what the *file* contains;
  the layers are only actually loaded when the model was opened with
  `load_mtp: true`.
  """
  @spec n_layer_nextn(t()) :: non_neg_integer()
  def n_layer_nextn(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_layer_nextn(ref)

  @doc "Returns a human-readable description of the model."
  @spec desc(t()) :: String.t()
  def desc(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_desc(ref)

  @doc "Returns the model file size in bytes."
  @spec size(t()) :: integer()
  def size(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_size(ref)

  @doc "Returns the number of model parameters."
  @spec n_params(t()) :: integer()
  def n_params(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_params(ref)

  @doc """
  Returns the chat template string embedded in the model, or `nil` if none.
  """
  @spec chat_template(t()) :: String.t() | nil
  def chat_template(%__MODULE__{ref: ref}) do
    case LlamaCppEx.NIF.model_chat_template(ref) do
      "" -> nil
      template -> template
    end
  end
end
