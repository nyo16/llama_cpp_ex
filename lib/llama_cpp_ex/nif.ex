defmodule LlamaCppEx.NIF do
  @moduledoc false
  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:llama_cpp_ex), ~c"llama_cpp_ex_nif")
    :erlang.load_nif(path, 0)
  end

  # Converts an ErlangError raised by a NIF into an error tuple. The
  # NIF-not-loaded error is re-raised: it signals a build/packaging problem,
  # not bad input, and must not be flattened into a caller-visible string.
  @doc false
  def error_tuple(%ErlangError{original: :not_loaded} = e, _label, stacktrace),
    do: reraise(e, stacktrace)

  def error_tuple(%ErlangError{original: original}, label, _stacktrace),
    do: {:error, "#{label} failed: #{inspect(original)}"}

  # Backend
  def backend_init, do: :erlang.nif_error(:not_loaded)
  def backend_free, do: :erlang.nif_error(:not_loaded)

  # Devices
  def device_list, do: :erlang.nif_error(:not_loaded)

  # Model
  def model_load(
        _path,
        _n_gpu_layers,
        _use_mmap,
        _main_gpu,
        _split_mode,
        _tensor_split,
        _use_mlock,
        _use_direct_io,
        _vocab_only,
        _check_tensors
      ),
      do: :erlang.nif_error(:not_loaded)

  def model_n_ctx_train(_model), do: :erlang.nif_error(:not_loaded)
  def model_n_embd(_model), do: :erlang.nif_error(:not_loaded)
  def model_desc(_model), do: :erlang.nif_error(:not_loaded)
  def model_size(_model), do: :erlang.nif_error(:not_loaded)
  def model_n_params(_model), do: :erlang.nif_error(:not_loaded)
  def model_chat_template(_model), do: :erlang.nif_error(:not_loaded)

  # Vocab
  def vocab_n_tokens(_model), do: :erlang.nif_error(:not_loaded)
  def vocab_bos(_model), do: :erlang.nif_error(:not_loaded)
  def vocab_eos(_model), do: :erlang.nif_error(:not_loaded)
  def vocab_is_eog(_model, _token), do: :erlang.nif_error(:not_loaded)

  # Tokenization
  def tokenize(_model, _text, _add_special, _parse_special), do: :erlang.nif_error(:not_loaded)
  def detokenize(_model, _tokens), do: :erlang.nif_error(:not_loaded)
  def token_to_piece(_model, _token), do: :erlang.nif_error(:not_loaded)

  # Context
  def context_create(
        _model,
        _n_ctx,
        _n_batch,
        _n_ubatch,
        _n_threads,
        _n_threads_batch,
        _embeddings,
        _pooling_type,
        _n_seq_max,
        _type_k,
        _type_v,
        _flash_attn,
        _offload_kqv,
        _op_offload,
        _rope_scaling_type,
        _rope_freq_base,
        _rope_freq_scale,
        _yarn_ext_factor,
        _yarn_attn_factor,
        _yarn_beta_fast,
        _yarn_beta_slow,
        _yarn_orig_ctx,
        _attention_type,
        _no_perf,
        _swa_full,
        _ctx_type,
        _n_rs_seq
      ),
      do: :erlang.nif_error(:not_loaded)

  def context_n_ctx(_ctx), do: :erlang.nif_error(:not_loaded)
  def context_n_seq_max(_ctx), do: :erlang.nif_error(:not_loaded)
  def context_n_rs_seq(_ctx), do: :erlang.nif_error(:not_loaded)

  # Sampler
  def sampler_init(
        _model,
        _seed,
        _temp,
        _top_k,
        _top_p,
        _min_p,
        _penalty_repeat,
        _penalty_freq,
        _penalty_present,
        _grammar_str,
        _grammar_root
      ),
      do: :erlang.nif_error(:not_loaded)

  def sampler_accept(_sampler, _token), do: :erlang.nif_error(:not_loaded)
  def sampler_reset(_sampler), do: :erlang.nif_error(:not_loaded)
  def sampler_sample(_sampler, _ctx), do: :erlang.nif_error(:not_loaded)

  # Decode
  def decode(_ctx, _tokens), do: :erlang.nif_error(:not_loaded)

  # Memory
  def memory_clear(_ctx), do: :erlang.nif_error(:not_loaded)
  def memory_seq_rm(_ctx, _seq_id, _p0, _p1), do: :erlang.nif_error(:not_loaded)
  def memory_seq_cp(_ctx, _seq_id_src, _seq_id_dst, _p0, _p1), do: :erlang.nif_error(:not_loaded)
  def memory_seq_keep(_ctx, _seq_id), do: :erlang.nif_error(:not_loaded)
  def memory_seq_pos_max(_ctx, _seq_id), do: :erlang.nif_error(:not_loaded)
  def context_can_seq_rm(_ctx), do: :erlang.nif_error(:not_loaded)

  # Chat template
  def chat_apply_template(_template, _messages, _add_assistant),
    do: :erlang.nif_error(:not_loaded)

  # Jinja chat template (via common library)
  def chat_apply_template_jinja(
        _model,
        _messages,
        _add_assistant,
        _enable_thinking,
        _extra_kwargs
      ),
      do: :erlang.nif_error(:not_loaded)

  # Streaming generation (sends messages to caller_pid tagged with ref)
  def generate_tokens(_ctx, _sampler, _prompt_tokens, _max_tokens, _caller_pid, _ref),
    do: :erlang.nif_error(:not_loaded)

  # Speculative decoding (MTP)
  def speculative_init(_ctx_tgt, _ctx_dft, _n_draft), do: :erlang.nif_error(:not_loaded)
  def speculative_stats(_spec), do: :erlang.nif_error(:not_loaded)
  def speculative_print_stats(_spec), do: :erlang.nif_error(:not_loaded)

  def generate_mtp_tokens(
        _spec,
        _sampler,
        _prompt_tokens,
        _max_tokens,
        _emit_stats_every,
        _caller_pid,
        _ref
      ),
      do: :erlang.nif_error(:not_loaded)

  # High-level generation
  def generate(_ctx, _sampler, _prompt_tokens, _max_tokens), do: :erlang.nif_error(:not_loaded)

  # Embeddings
  def embed_decode(_ctx, _tokens, _seq_id), do: :erlang.nif_error(:not_loaded)
  def embed_batch_decode(_ctx, _sequences), do: :erlang.nif_error(:not_loaded)
  def get_embeddings(_ctx, _seq_id, _normalize), do: :erlang.nif_error(:not_loaded)

  # Batched inference
  def prefill(_ctx, _tokens, _seq_id), do: :erlang.nif_error(:not_loaded)
  def decode_batch(_ctx, _sampler, _entries), do: :erlang.nif_error(:not_loaded)
  def decode_token(_ctx, _token_id, _pos, _seq_id), do: :erlang.nif_error(:not_loaded)

  # Continuous batching
  def batch_eval(_ctx, _entries), do: :erlang.nif_error(:not_loaded)
  def sampler_sample_at(_sampler, _ctx, _idx), do: :erlang.nif_error(:not_loaded)

  # JSON Schema to Grammar
  def json_schema_to_grammar_nif(_json_str), do: :erlang.nif_error(:not_loaded)
end
