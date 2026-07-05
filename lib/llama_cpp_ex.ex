defmodule LlamaCppEx do
  @moduledoc """
  Elixir bindings for llama.cpp.

  Provides a high-level API for loading GGUF models and generating text.

  ## Quick Start

      # Initialize the backend (once per application)
      :ok = LlamaCppEx.init()

      # Load a model
      {:ok, model} = LlamaCppEx.load_model("model.gguf", n_gpu_layers: -1)

      # Generate text
      {:ok, text} = LlamaCppEx.generate(model, "Once upon a time", max_tokens: 200)

  ## Lower-level API

  For fine-grained control, use the individual modules:

    * `LlamaCppEx.Model` - Model loading and introspection
    * `LlamaCppEx.Context` - Inference context with KV cache
    * `LlamaCppEx.Sampler` - Token sampling configuration
    * `LlamaCppEx.Tokenizer` - Text tokenization and detokenization
    * `LlamaCppEx.Embedding` - Embedding generation

  """

  alias LlamaCppEx.{
    Chat,
    ChatCompletion,
    ChatCompletionChunk,
    Context,
    Embedding,
    Grammar,
    Model,
    Sampler,
    Server,
    Thinking,
    Tokenizer
  }

  @context_opt_keys [
    :n_threads,
    :n_threads_batch,
    :n_batch,
    :n_ubatch,
    :type_k,
    :type_v,
    :flash_attn,
    :offload_kqv,
    :op_offload,
    :rope_scaling_type,
    :rope_freq_base,
    :rope_freq_scale,
    :yarn_ext_factor,
    :yarn_attn_factor,
    :yarn_beta_fast,
    :yarn_beta_slow,
    :yarn_orig_ctx,
    :attention_type,
    :no_perf,
    :swa_full
  ]

  # Sampling options forwarded to Sampler.create/2 by the generation entry
  # points. Keep in sync with the options documented on generate/3.
  @sampler_opt_keys [
    :seed,
    :temp,
    :top_k,
    :top_p,
    :min_p,
    :penalty_repeat,
    :penalty_freq,
    :penalty_present,
    :grammar,
    :grammar_root
  ]

  # Chat-templating options split off before the rest flows to generation.
  @chat_opt_keys [:add_assistant, :enable_thinking, :chat_template_kwargs]

  @doc """
  Initializes the llama.cpp backend. Call once at application start.
  """
  @spec init() :: :ok
  def init do
    LlamaCppEx.NIF.backend_init()
  end

  @doc """
  Lists the available ggml backend devices (GPUs, integrated GPUs, accelerators,
  and the CPU).

  Each entry is a map with `:index` (ggml device order), `:gpu_index` (0-based
  among GPU/IGPU devices, matching `:tensor_split`'s index space, or `nil` for
  non-GPU devices), `:name`, `:description`, `:type` (`:gpu`, `:igpu`, `:cpu`,
  `:accel`, or `:other`), `:backend` (e.g. `"CUDA"`, `"Metal"`), `:memory_total`,
  and `:memory_free` (bytes). On Metal, unified memory means a single device.

  ## Examples

      :ok = LlamaCppEx.init()
      LlamaCppEx.devices()
      #=> [%{gpu_index: 0, type: :gpu, name: "NVIDIA RTX 4090",
      #      memory_total: 24_000_000_000, memory_free: 23_500_000_000, ...}, ...]

  """
  @spec devices() :: [map()]
  def devices do
    LlamaCppEx.NIF.backend_init()

    LlamaCppEx.NIF.device_list()
    |> Enum.map(fn %{gpu_index: gi} = dev ->
      %{dev | gpu_index: if(gi < 0, do: nil, else: gi)}
    end)
  end

  @doc """
  Loads a GGUF model from the given file path.

  See `LlamaCppEx.Model.load/2` for options.
  """
  @spec load_model(String.t(), keyword()) :: {:ok, Model.t()} | {:error, String.t()}
  def load_model(path, opts \\ []) do
    Model.load(path, opts)
  end

  @doc """
  Downloads a GGUF model from HuggingFace Hub and loads it.

  Requires the optional `:req` dependency.

  ## Examples

      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model_from_hub(
        "Qwen/Qwen3-4B-GGUF",
        "qwen3-4b-q4_k_m.gguf",
        n_gpu_layers: -1
      )

  ## Options

  Accepts all options from `load_model/2` plus:

    * `:cache_dir` - Local cache directory for downloaded models.
    * `:token` - HuggingFace API token for private repos.
    * `:progress` - Download progress callback.
    * `:revision` - Git revision (branch, tag, commit). Defaults to `"main"`.

  """
  @spec load_model_from_hub(String.t(), String.t(), keyword()) ::
          {:ok, Model.t()} | {:error, String.t()}
  def load_model_from_hub(repo_id, filename, opts \\ []) do
    {hub_opts, model_opts} = Keyword.split(opts, [:cache_dir, :token, :progress, :revision])

    with {:ok, path} <- LlamaCppEx.Hub.download(repo_id, filename, hub_opts) do
      load_model(path, model_opts)
    end
  end

  @doc """
  Generates text from a prompt.

  Creates a temporary context and sampler, tokenizes the prompt, runs generation,
  and returns the generated text.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:n_ctx` - Context size. Defaults to `2048`.
    * `:temp` - Sampling temperature. `0.0` for greedy. Defaults to `0.8`.
    * `:top_k` - Top-K filtering. Defaults to `40`.
    * `:top_p` - Top-P (nucleus) filtering. Defaults to `0.95`.
    * `:min_p` - Min-P filtering. Defaults to `0.05`.
    * `:seed` - Random seed. Defaults to random.
    * `:penalty_repeat` - Repetition penalty. Defaults to `1.0`.
    * `:penalty_freq` - Frequency penalty (0.0–2.0). Defaults to `0.0`.
    * `:penalty_present` - Presence penalty (0.0–2.0). Defaults to `0.0`.
    * `:grammar` - GBNF grammar string for constrained generation.
    * `:grammar_root` - Root rule name for grammar. Defaults to `"root"`.
    * `:json_schema` - JSON Schema map for structured output. Automatically converted
      to a GBNF grammar. Cannot be used together with `:grammar`. Tip: set
      `"additionalProperties" => false` for tighter grammars.

  """
  @spec generate(Model.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def generate(%Model{} = model, prompt, opts \\ []) when is_binary(prompt) do
    cfg = opts |> resolve_grammar_opts() |> gen_config()
    {:ok, tokens} = Tokenizer.encode(model, prompt)

    with {:ok, ctx, sampler} <- create_gen_resources(model, tokens, cfg) do
      Context.generate(ctx, sampler, tokens, max_tokens: cfg.max_tokens)
    end
  end

  @doc """
  Returns a lazy stream of generated text chunks (tokens).

  Each element is a string (the text piece for one token). The stream ends
  when an end-of-generation token is produced or `max_tokens` is reached.

  Accepts the same options as `generate/3`.

  ## Examples

      model
      |> LlamaCppEx.stream("Tell me a story", max_tokens: 500)
      |> Enum.each(&IO.write/1)

  """
  @spec stream(Model.t(), String.t(), keyword()) :: Enumerable.t()
  def stream(%Model{} = model, prompt, opts \\ []) when is_binary(prompt) do
    cfg = opts |> resolve_grammar_opts() |> gen_config()

    Stream.resource(
      fn ->
        # Start: tokenize, create context+sampler, spawn generator
        {:ok, tokens} = Tokenizer.encode(model, prompt)
        {:ok, ctx, sampler} = create_gen_resources(model, tokens, cfg)
        {ref, gen_pid} = spawn_generator(ctx, sampler, tokens, cfg.max_tokens)
        {ref, gen_pid, cfg.timeout}
      end,
      fn {ref, _gen_pid, timeout} = state ->
        receive do
          {^ref, {:token, _id, text}} -> {[text], state}
          {^ref, outcome} when outcome in [:eog, :done] -> {:halt, state}
          {^ref, {:error, _reason}} -> {:halt, state}
        after
          timeout -> {:halt, state}
        end
      end,
      fn {ref, gen_pid, _timeout} -> stop_generator(ref, gen_pid) end
    )
  end

  # --- Shared generation plumbing ---

  # Option handling shared by the generation entry points.
  defp gen_config(opts) do
    %{
      max_tokens: Keyword.get(opts, :max_tokens, 256),
      n_ctx: Keyword.get(opts, :n_ctx, 2048),
      timeout: Keyword.get(opts, :timeout, 60_000),
      sampler_opts: Keyword.take(opts, @sampler_opt_keys),
      ctx_opts: Keyword.take(opts, @context_opt_keys)
    }
  end

  defp split_chat_opts(opts), do: Keyword.split(opts, @chat_opt_keys)

  # Creates a context sized to fit prompt + generation, plus its sampler.
  defp create_gen_resources(model, tokens, cfg) do
    ctx_size = max(cfg.n_ctx, length(tokens) + cfg.max_tokens)

    with {:ok, ctx} <- Context.create(model, Keyword.put(cfg.ctx_opts, :n_ctx, ctx_size)),
         {:ok, sampler} <- Sampler.create(model, cfg.sampler_opts) do
      {:ok, ctx, sampler}
    end
  end

  # Runs the NIF token generator in a linked process that sends
  # {ref, {:token, id, text} | :eog | :done | {:error, reason}} to the caller.
  defp spawn_generator(ctx, sampler, tokens, max_tokens) do
    ref = make_ref()
    parent = self()

    gen_pid =
      spawn_link(fn ->
        LlamaCppEx.NIF.generate_tokens(ctx.ref, sampler.ref, tokens, max_tokens, parent, ref)
      end)

    {ref, gen_pid}
  end

  # Kills a generator (if still running) and drains its remaining messages.
  defp stop_generator(ref, gen_pid) do
    Process.unlink(gen_pid)
    Process.exit(gen_pid, :kill)
    flush_stream_messages(ref)
  end

  defp flush_stream_messages(ref) do
    receive do
      {^ref, _} -> flush_stream_messages(ref)
    after
      0 -> :ok
    end
  end

  # OpenAI-style finish_reason. :eog/:done come from the stateless generator
  # protocol; :max_tokens from the Server's completion metadata.
  defp finish_reason(:eog), do: "stop"
  defp finish_reason(:done), do: "length"
  defp finish_reason(:max_tokens), do: "length"

  @doc """
  Applies the chat template and generates a response.

  ## Options

  Accepts all options from `generate/3` plus:

    * `:template` - Custom chat template string. Defaults to the model's embedded template.

  ## Examples

      {:ok, reply} = LlamaCppEx.chat(model, [
        %{role: "system", content: "You are helpful."},
        %{role: "user", content: "What is Elixir?"}
      ], max_tokens: 200)

  """
  @spec chat(Model.t(), [Chat.message()], keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def chat(%Model{} = model, messages, opts \\ []) when is_list(messages) do
    {chat_opts, gen_opts} = opts |> resolve_grammar_opts() |> split_chat_opts()
    {:ok, prompt} = Chat.apply_template(model, messages, chat_opts)
    generate(model, prompt, gen_opts)
  end

  @doc """
  Returns a lazy stream of chat response chunks.

  Applies the chat template and streams the generated response token by token.
  Accepts same options as `chat/3`.
  """
  @spec stream_chat(Model.t(), [Chat.message()], keyword()) :: Enumerable.t()
  def stream_chat(%Model{} = model, messages, opts \\ []) when is_list(messages) do
    {chat_opts, gen_opts} = opts |> resolve_grammar_opts() |> split_chat_opts()
    {:ok, prompt} = Chat.apply_template(model, messages, chat_opts)
    stream(model, prompt, gen_opts)
  end

  @doc """
  Generates an OpenAI-compatible chat completion response.

  Applies the chat template, runs generation, and returns a `%ChatCompletion{}`
  struct with choices, usage counts, and finish reason.

  ## Options

  Accepts all options from `generate/3` plus:

    * `:template` - Custom chat template string. Defaults to the model's embedded template.

  ## Examples

      {:ok, completion} = LlamaCppEx.chat_completion(model, [
        %{role: "user", content: "What is Elixir?"}
      ], max_tokens: 200)

      completion.choices |> hd() |> Map.get(:message) |> Map.get(:content)

  ## Server routing

  When given a running `LlamaCppEx.Server` (pid or name) instead of a
  `%Model{}`, the request is served by the batching server: chat templating
  and tokenization happen in the caller, and the multi-turn prompt benefits
  from the server's prefix cache. Server-only request options like
  `:session` and `:cache_prompt` are forwarded.
  """
  @spec chat_completion(Model.t() | GenServer.server(), [Chat.message()], keyword()) ::
          {:ok, ChatCompletion.t()} | {:error, term()}
  def chat_completion(model_or_server, messages, opts \\ [])

  def chat_completion(%Model{} = model, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} = opts |> resolve_grammar_opts() |> split_chat_opts()
    cfg = gen_config(gen_opts)

    with {:ok, prompt} <- Chat.apply_template(model, messages, chat_opts),
         {:ok, prompt_tokens} <- Tokenizer.encode(model, prompt) do
      {:ok, ctx, sampler} = create_gen_resources(model, prompt_tokens, cfg)
      {ref, gen_pid} = spawn_generator(ctx, sampler, prompt_tokens, cfg.max_tokens)

      {texts, finish_reason, completion_tokens} = collect_completion_tokens(ref, cfg.timeout)

      # On timeout the generator may still be running — kill it and drain any
      # stragglers so they don't pollute the caller's mailbox.
      stop_generator(ref, gen_pid)

      completion =
        build_chat_completion(
          Model.desc(model),
          length(prompt_tokens),
          Enum.join(texts),
          finish_reason,
          completion_tokens,
          chat_opts
        )

      {:ok, completion}
    end
  end

  def chat_completion(server, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} = opts |> resolve_grammar_opts() |> split_chat_opts()
    model = Server.get_model(server)

    with {:ok, prompt} <- Chat.apply_template(model, messages, chat_opts),
         {:ok, prompt_tokens} <- Tokenizer.encode(model, prompt),
         {:ok, result} <- Server.complete_tokens(server, prompt_tokens, gen_opts) do
      completion =
        build_chat_completion(
          Model.desc(model),
          length(prompt_tokens),
          result.text,
          finish_reason(result.finish_reason),
          result.completion_tokens,
          chat_opts
        )

      {:ok, completion}
    end
  end

  defp build_chat_completion(
         model_desc,
         prompt_tokens,
         raw_text,
         finish_reason,
         completion_tokens,
         chat_opts
       ) do
    enable_thinking = Keyword.get(chat_opts, :enable_thinking, false)

    {reasoning_content, content} =
      if enable_thinking do
        {rc, c} = Thinking.parse(raw_text)
        {if(rc == "", do: nil, else: rc), c}
      else
        {nil, raw_text}
      end

    %ChatCompletion{
      id: "chatcmpl-" <> random_hex(12),
      object: "chat.completion",
      created: System.os_time(:second),
      model: model_desc,
      choices: [
        %{
          index: 0,
          message: %{
            role: "assistant",
            content: content,
            reasoning_content: reasoning_content
          },
          finish_reason: finish_reason
        }
      ],
      usage: %{
        prompt_tokens: prompt_tokens,
        completion_tokens: completion_tokens,
        total_tokens: prompt_tokens + completion_tokens
      }
    }
  end

  @doc """
  Returns a lazy stream of OpenAI-compatible chat completion chunks.

  Each element is a `%ChatCompletionChunk{}` struct. The first chunk contains
  `delta: %{role: "assistant", content: ""}`. Subsequent chunks contain
  `delta: %{content: "token_text"}`. The final chunk contains the `finish_reason`.

  All chunks share the same `id` and `created` timestamp.

  ## Options

  Accepts same options as `chat_completion/3`.

  ## Examples

      model
      |> LlamaCppEx.stream_chat_completion(messages, max_tokens: 200)
      |> Enum.each(fn chunk ->
        chunk.choices |> hd() |> get_in([:delta, :content]) |> IO.write()
      end)

  """
  @spec stream_chat_completion(Model.t() | GenServer.server(), [Chat.message()], keyword()) ::
          Enumerable.t()
  def stream_chat_completion(model_or_server, messages, opts \\ [])

  def stream_chat_completion(%Model{} = model, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} = opts |> resolve_grammar_opts() |> split_chat_opts()
    cfg = gen_config(gen_opts)

    Stream.resource(
      fn ->
        {:ok, prompt} = Chat.apply_template(model, messages, chat_opts)
        {:ok, tokens} = Tokenizer.encode(model, prompt)
        {:ok, ctx, sampler} = create_gen_resources(model, tokens, cfg)
        {ref, gen_pid} = spawn_generator(ctx, sampler, tokens, cfg.max_tokens)

        chat_chunk_state(ref, cfg.timeout, Model.desc(model), chat_opts)
        |> Map.put(:gen_pid, gen_pid)
      end,
      fn
        %{phase: :first} = state ->
          {[chunk(state, %{role: "assistant", content: ""}, nil)], %{state | phase: :streaming}}

        %{phase: :streaming, ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token, _id, text}} ->
              token_chunks(state, text)

            {^ref, outcome} when outcome in [:eog, :done] ->
              {[chunk(state, %{}, finish_reason(outcome))], %{state | phase: :done}}

            {^ref, {:error, _reason}} ->
              {:halt, state}
          after
            timeout -> {:halt, state}
          end

        %{phase: :done} = state ->
          {:halt, state}
      end,
      fn %{ref: ref, gen_pid: gen_pid} -> stop_generator(ref, gen_pid) end
    )
  end

  # Server-routed streaming: templating/tokenization in the caller, tokens
  # served by the batching server with prefix caching.
  def stream_chat_completion(server, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} = opts |> resolve_grammar_opts() |> split_chat_opts()
    timeout = Keyword.get(gen_opts, :timeout, 30_000)

    Stream.resource(
      fn ->
        model = Server.get_model(server)
        {:ok, prompt} = Chat.apply_template(model, messages, chat_opts)
        {:ok, tokens} = Tokenizer.encode(model, prompt)
        ref = make_ref()
        :ok = Server.subscribe_stream_tokens(server, tokens, ref, gen_opts)

        chat_chunk_state(ref, timeout, Model.desc(model), chat_opts)
      end,
      fn
        %{phase: :first} = state ->
          {[chunk(state, %{role: "assistant", content: ""}, nil)], %{state | phase: :streaming}}

        %{phase: :streaming, ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token, text}} ->
              token_chunks(state, text)

            {^ref, {:done, reason}} ->
              {[chunk(state, %{}, finish_reason(reason))], %{state | phase: :done}}

            {^ref, {:error, _reason}} ->
              {:halt, state}
          after
            timeout -> {:halt, state}
          end

        %{phase: :done} = state ->
          {:halt, state}
      end,
      fn %{ref: ref} -> flush_stream_messages(ref) end
    )
  end

  # Shared per-stream state for the chat-completion chunk builders.
  defp chat_chunk_state(ref, timeout, model_desc, chat_opts) do
    enable_thinking = Keyword.get(chat_opts, :enable_thinking, false)

    %{
      ref: ref,
      timeout: timeout,
      id: "chatcmpl-" <> random_hex(12),
      created: System.os_time(:second),
      model: model_desc,
      phase: :first,
      enable_thinking: enable_thinking,
      thinking_parser: if(enable_thinking, do: Thinking.stream_parser(thinking: true), else: nil)
    }
  end

  # Emits the chunk(s) for one generated token, routing through the thinking
  # parser when enabled (one token can yield reasoning and content chunks).
  defp token_chunks(%{enable_thinking: true} = state, text) do
    {events, new_parser} = Thinking.feed(state.thinking_parser, text)
    state = %{state | thinking_parser: new_parser}

    chunks =
      Enum.map(events, fn
        {:thinking, t} -> chunk(state, %{reasoning_content: t}, nil)
        {:content, t} -> chunk(state, %{content: t}, nil)
      end)

    {chunks, state}
  end

  defp token_chunks(state, text) do
    {[chunk(state, %{content: text}, nil)], state}
  end

  defp chunk(state, delta, finish_reason) do
    %ChatCompletionChunk{
      id: state.id,
      object: "chat.completion.chunk",
      created: state.created,
      model: state.model,
      choices: [%{index: 0, delta: delta, finish_reason: finish_reason}]
    }
  end

  defp collect_completion_tokens(ref, timeout) do
    collect_completion_tokens(ref, timeout, [], 0)
  end

  defp collect_completion_tokens(ref, timeout, texts, count) do
    receive do
      {^ref, {:token, _id, text}} ->
        collect_completion_tokens(ref, timeout, [text | texts], count + 1)

      {^ref, outcome} when outcome in [:eog, :done] ->
        {Enum.reverse(texts), finish_reason(outcome), count}

      {^ref, {:error, _reason}} ->
        {Enum.reverse(texts), "stop", count}
    after
      timeout -> {Enum.reverse(texts), "length", count}
    end
  end

  defp random_hex(n) do
    :crypto.strong_rand_bytes(n) |> Base.encode16(case: :lower)
  end

  defp resolve_grammar_opts(opts) do
    grammar = Keyword.get(opts, :grammar)
    json_schema = Keyword.get(opts, :json_schema)

    cond do
      grammar && json_schema ->
        raise ArgumentError, "cannot use both :grammar and :json_schema options"

      json_schema ->
        gbnf = Grammar.from_json_schema!(json_schema)
        opts |> Keyword.delete(:json_schema) |> Keyword.put(:grammar, gbnf)

      true ->
        opts
    end
  end

  @doc """
  Computes an embedding for a single text.

  See `LlamaCppEx.Embedding.embed/3` for options.
  """
  @spec embed(Model.t(), String.t(), keyword()) :: {:ok, Embedding.t()} | {:error, String.t()}
  def embed(%Model{} = model, text, opts \\ []) do
    Embedding.embed(model, text, opts)
  end

  @doc """
  Computes embeddings for multiple texts.

  See `LlamaCppEx.Embedding.embed_batch/3` for options.
  """
  @spec embed_batch(Model.t(), [String.t()], keyword()) ::
          {:ok, [Embedding.t()]} | {:error, String.t()}
  def embed_batch(%Model{} = model, texts, opts \\ []) do
    Embedding.embed_batch(model, texts, opts)
  end
end
