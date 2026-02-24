defmodule LlamaCppEx.NIF do
  @moduledoc false
  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:llama_cpp_ex), ~c"llama_cpp_ex_nif")
    :erlang.load_nif(path, 0)
  end

  # Backend
  def backend_init, do: :erlang.nif_error(:not_loaded)
  def backend_free, do: :erlang.nif_error(:not_loaded)

  # Model
  def model_load(_path, _n_gpu_layers, _use_mmap), do: :erlang.nif_error(:not_loaded)
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
        _n_seq_max
      ),
      do: :erlang.nif_error(:not_loaded)

  def context_n_ctx(_ctx), do: :erlang.nif_error(:not_loaded)
  def context_n_seq_max(_ctx), do: :erlang.nif_error(:not_loaded)

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

  # Chat template
  def chat_apply_template(_template, _messages, _add_assistant),
    do: :erlang.nif_error(:not_loaded)

  # Streaming generation (sends messages to caller_pid tagged with ref)
  def generate_tokens(_ctx, _sampler, _prompt_tokens, _max_tokens, _caller_pid, _ref),
    do: :erlang.nif_error(:not_loaded)

  # High-level generation
  def generate(_ctx, _sampler, _prompt_tokens, _max_tokens), do: :erlang.nif_error(:not_loaded)

  # Embeddings
  def embed_decode(_ctx, _tokens, _seq_id), do: :erlang.nif_error(:not_loaded)
  def get_embeddings(_ctx, _seq_id, _normalize), do: :erlang.nif_error(:not_loaded)

  # Batched inference
  def prefill(_ctx, _tokens, _seq_id), do: :erlang.nif_error(:not_loaded)
  def decode_batch(_ctx, _sampler, _entries), do: :erlang.nif_error(:not_loaded)
  def decode_token(_ctx, _token_id, _pos, _seq_id), do: :erlang.nif_error(:not_loaded)

  # Continuous batching
  def batch_eval(_ctx, _entries), do: :erlang.nif_error(:not_loaded)
  def sampler_sample_at(_sampler, _ctx, _idx), do: :erlang.nif_error(:not_loaded)
end
