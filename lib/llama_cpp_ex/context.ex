defmodule LlamaCppEx.Context do
  @moduledoc """
  Inference context with KV cache.
  """

  @enforce_keys [:ref, :model]
  defstruct [:ref, :model]

  @type t :: %__MODULE__{ref: reference(), model: LlamaCppEx.Model.t()}

  @doc """
  Creates a new inference context for the given model.

  ## Options

  ### Core

    * `:n_ctx` - Context size (max tokens). Defaults to `2048`.
    * `:n_batch` - Max tokens per decode batch. Defaults to `n_ctx`.
    * `:n_ubatch` - Max tokens per micro-batch. Defaults to `512`.
    * `:n_threads` - Number of threads for generation. Defaults to system CPU count.
    * `:n_threads_batch` - Number of threads for prompt processing. Defaults to `:n_threads`.
    * `:n_seq_max` - Max number of concurrent sequences. Defaults to `1`.
    * `:embeddings` - Enable embedding extraction. Defaults to `false`.
    * `:pooling_type` - Pooling type for embeddings: `:unspecified`, `:none`, `:mean`,
      `:cls`, `:last`, `:rank`. Defaults to `:unspecified`.

  ### KV Cache Quantization

    * `:type_k` - Data type for K cache. Reduces memory at the cost of precision.
      Values: `:f16` (default), `:f32`, `:q8_0`, `:q4_0`, `:q4_1`, `:q5_0`, `:q5_1`, `:bf16`.
    * `:type_v` - Data type for V cache. Same values as `:type_k`. Defaults to `:f16`.

  ### Flash Attention & GPU Offload

    * `:flash_attn` - Flash Attention mode: `:auto` (default), `:enabled`, `:disabled`.
    * `:offload_kqv` - Offload KQV ops and KV cache to GPU. Defaults to `true`.
    * `:op_offload` - Offload host tensor operations to device. Defaults to `true`.

  ### RoPE Scaling (Context Extension)

    * `:rope_scaling_type` - RoPE scaling mode: `:unspecified` (default), `:none`,
      `:linear`, `:yarn`, `:longrope`.
    * `:rope_freq_base` - RoPE base frequency. `0.0` uses model default.
    * `:rope_freq_scale` - RoPE frequency scale. `0.0` uses model default.
    * `:yarn_ext_factor` - YaRN extrapolation mix factor. `-1.0` to disable.
    * `:yarn_attn_factor` - YaRN magnitude scaling. `-1.0` to disable.
    * `:yarn_beta_fast` - YaRN low correction dimension. `-1.0` to disable.
    * `:yarn_beta_slow` - YaRN high correction dimension. `-1.0` to disable.
    * `:yarn_orig_ctx` - YaRN original context length. `0` to disable.

  ### Misc

    * `:attention_type` - Attention type: `:unspecified` (default), `:causal`, `:non_causal`.
      Use `:non_causal` for embedding models.
    * `:no_perf` - Disable performance timing. Defaults to `true`.
    * `:swa_full` - Use full-size sliding window attention cache. Defaults to `true`.

  """
  @spec create(LlamaCppEx.Model.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def create(%LlamaCppEx.Model{ref: model_ref} = model, opts \\ []) do
    # Core
    n_threads = Keyword.get(opts, :n_threads, System.schedulers_online())
    n_ctx = Keyword.get(opts, :n_ctx, 2048)
    n_batch = Keyword.get(opts, :n_batch, n_ctx)
    n_ubatch = Keyword.get(opts, :n_ubatch, 512)
    n_threads_batch = Keyword.get(opts, :n_threads_batch, n_threads)
    embeddings = Keyword.get(opts, :embeddings, false)
    pooling_type = Keyword.get(opts, :pooling_type, :unspecified) |> pooling_type_to_int()
    n_seq_max = Keyword.get(opts, :n_seq_max, 1)

    # KV cache quantization
    type_k = Keyword.get(opts, :type_k, :f16) |> ggml_type_to_int()
    type_v = Keyword.get(opts, :type_v, :f16) |> ggml_type_to_int()

    # Flash attention & GPU offload
    flash_attn = Keyword.get(opts, :flash_attn, :auto) |> flash_attn_to_int()
    offload_kqv = Keyword.get(opts, :offload_kqv, true)
    op_offload = Keyword.get(opts, :op_offload, true)

    # RoPE scaling
    rope_scaling_type =
      Keyword.get(opts, :rope_scaling_type, :unspecified) |> rope_scaling_to_int()

    rope_freq_base = Keyword.get(opts, :rope_freq_base, 0.0) / 1
    rope_freq_scale = Keyword.get(opts, :rope_freq_scale, 0.0) / 1
    yarn_ext_factor = Keyword.get(opts, :yarn_ext_factor, -1.0) / 1
    yarn_attn_factor = Keyword.get(opts, :yarn_attn_factor, -1.0) / 1
    yarn_beta_fast = Keyword.get(opts, :yarn_beta_fast, -1.0) / 1
    yarn_beta_slow = Keyword.get(opts, :yarn_beta_slow, -1.0) / 1
    yarn_orig_ctx = Keyword.get(opts, :yarn_orig_ctx, 0)

    # Misc
    attention_type = Keyword.get(opts, :attention_type, :unspecified) |> attention_type_to_int()
    no_perf = Keyword.get(opts, :no_perf, true)
    swa_full = Keyword.get(opts, :swa_full, true)

    case LlamaCppEx.NIF.context_create(
           model_ref,
           n_ctx,
           n_batch,
           n_ubatch,
           n_threads,
           n_threads_batch,
           embeddings,
           pooling_type,
           n_seq_max,
           type_k,
           type_v,
           flash_attn,
           offload_kqv,
           op_offload,
           rope_scaling_type,
           rope_freq_base,
           rope_freq_scale,
           yarn_ext_factor,
           yarn_attn_factor,
           yarn_beta_fast,
           yarn_beta_slow,
           yarn_orig_ctx,
           attention_type,
           no_perf,
           swa_full
         ) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, model: model}}
      {:error, _} = error -> error
    end
  end

  @doc "Returns the context size."
  @spec n_ctx(t()) :: integer()
  def n_ctx(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.context_n_ctx(ref)

  @doc "Returns the max number of sequences."
  @spec n_seq_max(t()) :: integer()
  def n_seq_max(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.context_n_seq_max(ref)

  @doc "Clears the KV cache."
  @spec clear(t()) :: :ok
  def clear(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.memory_clear(ref)

  @doc """
  Decodes a list of tokens through the model.
  """
  @spec decode(t(), [integer()]) :: :ok | {:error, String.t()}
  def decode(%__MODULE__{ref: ref}, tokens) when is_list(tokens) do
    LlamaCppEx.NIF.decode(ref, tokens)
  end

  @doc """
  Runs the generation loop: decodes prompt tokens and generates up to `max_tokens` new tokens.

  Returns the generated text (not including the prompt).

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.

  """
  @spec generate(t(), LlamaCppEx.Sampler.t(), [integer()], keyword()) ::
          {:ok, String.t()} | {:error, String.t()}
  def generate(
        %__MODULE__{ref: ctx_ref},
        %LlamaCppEx.Sampler{ref: sampler_ref},
        tokens,
        opts \\ []
      ) do
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    LlamaCppEx.NIF.generate(ctx_ref, sampler_ref, tokens, max_tokens)
  end

  # --- Enum mappings ---

  defp pooling_type_to_int(:unspecified), do: -1
  defp pooling_type_to_int(:none), do: 0
  defp pooling_type_to_int(:mean), do: 1
  defp pooling_type_to_int(:cls), do: 2
  defp pooling_type_to_int(:last), do: 3
  defp pooling_type_to_int(:rank), do: 4
  defp pooling_type_to_int(n) when is_integer(n), do: n

  defp ggml_type_to_int(:f32), do: 0
  defp ggml_type_to_int(:f16), do: 1
  defp ggml_type_to_int(:q4_0), do: 2
  defp ggml_type_to_int(:q4_1), do: 3
  defp ggml_type_to_int(:q5_0), do: 6
  defp ggml_type_to_int(:q5_1), do: 7
  defp ggml_type_to_int(:q8_0), do: 8
  defp ggml_type_to_int(:bf16), do: 30
  defp ggml_type_to_int(n) when is_integer(n), do: n

  defp flash_attn_to_int(:auto), do: -1
  defp flash_attn_to_int(:disabled), do: 0
  defp flash_attn_to_int(:enabled), do: 1
  defp flash_attn_to_int(n) when is_integer(n), do: n

  defp rope_scaling_to_int(:unspecified), do: -1
  defp rope_scaling_to_int(:none), do: 0
  defp rope_scaling_to_int(:linear), do: 1
  defp rope_scaling_to_int(:yarn), do: 2
  defp rope_scaling_to_int(:longrope), do: 3
  defp rope_scaling_to_int(n) when is_integer(n), do: n

  defp attention_type_to_int(:unspecified), do: -1
  defp attention_type_to_int(:causal), do: 0
  defp attention_type_to_int(:non_causal), do: 1
  defp attention_type_to_int(n) when is_integer(n), do: n
end
