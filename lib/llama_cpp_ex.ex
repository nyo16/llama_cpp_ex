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
    Model,
    Context,
    Sampler,
    Tokenizer,
    Chat,
    Embedding,
    ChatCompletion,
    ChatCompletionChunk
  }

  @doc """
  Initializes the llama.cpp backend. Call once at application start.
  """
  @spec init() :: :ok
  def init do
    LlamaCppEx.NIF.backend_init()
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

  """
  @spec generate(Model.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def generate(%Model{} = model, prompt, opts \\ []) when is_binary(prompt) do
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    n_ctx = Keyword.get(opts, :n_ctx, 2048)

    sampler_opts =
      Keyword.take(opts, [
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
      ])

    # Tokenize prompt
    {:ok, tokens} = Tokenizer.encode(model, prompt)

    # Ensure context is large enough for prompt + generation
    ctx_size = max(n_ctx, length(tokens) + max_tokens)

    ctx_opts =
      opts
      |> Keyword.take([:n_threads, :n_threads_batch, :n_batch, :n_ubatch])
      |> Keyword.put(:n_ctx, ctx_size)

    with {:ok, ctx} <- Context.create(model, ctx_opts),
         {:ok, sampler} <- Sampler.create(model, sampler_opts) do
      Context.generate(ctx, sampler, tokens, max_tokens: max_tokens)
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
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    n_ctx = Keyword.get(opts, :n_ctx, 2048)
    timeout = Keyword.get(opts, :timeout, 60_000)

    sampler_opts =
      Keyword.take(opts, [
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
      ])

    ctx_opts =
      Keyword.take(opts, [:n_threads, :n_threads_batch, :n_batch, :n_ubatch])

    Stream.resource(
      fn ->
        # Start: tokenize, create context+sampler, spawn generator
        {:ok, tokens} = Tokenizer.encode(model, prompt)
        ctx_size = max(n_ctx, length(tokens) + max_tokens)
        {:ok, ctx} = Context.create(model, Keyword.put(ctx_opts, :n_ctx, ctx_size))
        {:ok, sampler} = Sampler.create(model, sampler_opts)

        ref = make_ref()
        parent = self()

        gen_pid =
          spawn_link(fn ->
            LlamaCppEx.NIF.generate_tokens(
              ctx.ref,
              sampler.ref,
              tokens,
              max_tokens,
              parent,
              ref
            )
          end)

        {ref, gen_pid, timeout}
      end,
      fn {ref, _gen_pid, timeout} = state ->
        receive do
          {^ref, {:token, _id, text}} -> {[text], state}
          {^ref, :eog} -> {:halt, state}
          {^ref, :done} -> {:halt, state}
          {^ref, {:error, _reason}} -> {:halt, state}
        after
          timeout -> {:halt, state}
        end
      end,
      fn {ref, gen_pid, _timeout} ->
        # Kill generator if still running, flush remaining messages
        Process.unlink(gen_pid)
        Process.exit(gen_pid, :kill)
        flush_stream_messages(ref)
      end
    )
  end

  defp flush_stream_messages(ref) do
    receive do
      {^ref, _} -> flush_stream_messages(ref)
    after
      0 -> :ok
    end
  end

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
    {chat_opts, gen_opts} = Keyword.split(opts, [:template, :add_assistant])
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
    {chat_opts, gen_opts} = Keyword.split(opts, [:template, :add_assistant])
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

  """
  @spec chat_completion(Model.t(), [Chat.message()], keyword()) ::
          {:ok, ChatCompletion.t()} | {:error, term()}
  def chat_completion(%Model{} = model, messages, opts \\ []) when is_list(messages) do
    {chat_opts, gen_opts} = Keyword.split(opts, [:template, :add_assistant])
    max_tokens = Keyword.get(gen_opts, :max_tokens, 256)
    n_ctx = Keyword.get(gen_opts, :n_ctx, 2048)
    timeout = Keyword.get(gen_opts, :timeout, 60_000)

    sampler_opts =
      Keyword.take(gen_opts, [
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
      ])

    ctx_opts =
      Keyword.take(gen_opts, [:n_threads, :n_threads_batch, :n_batch, :n_ubatch])

    with {:ok, prompt} <- Chat.apply_template(model, messages, chat_opts),
         {:ok, prompt_tokens} <- Tokenizer.encode(model, prompt) do
      ctx_size = max(n_ctx, length(prompt_tokens) + max_tokens)
      {:ok, ctx} = Context.create(model, Keyword.put(ctx_opts, :n_ctx, ctx_size))
      {:ok, sampler} = Sampler.create(model, sampler_opts)

      ref = make_ref()
      parent = self()

      spawn_link(fn ->
        LlamaCppEx.NIF.generate_tokens(
          ctx.ref,
          sampler.ref,
          prompt_tokens,
          max_tokens,
          parent,
          ref
        )
      end)

      {texts, finish_reason, completion_tokens} = collect_completion_tokens(ref, timeout)

      completion = %ChatCompletion{
        id: "chatcmpl-" <> random_hex(12),
        object: "chat.completion",
        created: System.os_time(:second),
        model: Model.desc(model),
        choices: [
          %{
            index: 0,
            message: %{role: "assistant", content: Enum.join(texts)},
            finish_reason: finish_reason
          }
        ],
        usage: %{
          prompt_tokens: length(prompt_tokens),
          completion_tokens: completion_tokens,
          total_tokens: length(prompt_tokens) + completion_tokens
        }
      }

      {:ok, completion}
    end
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
  @spec stream_chat_completion(Model.t(), [Chat.message()], keyword()) :: Enumerable.t()
  def stream_chat_completion(%Model{} = model, messages, opts \\ []) when is_list(messages) do
    {chat_opts, gen_opts} = Keyword.split(opts, [:template, :add_assistant])
    max_tokens = Keyword.get(gen_opts, :max_tokens, 256)
    n_ctx = Keyword.get(gen_opts, :n_ctx, 2048)
    timeout = Keyword.get(gen_opts, :timeout, 60_000)

    sampler_opts =
      Keyword.take(gen_opts, [
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
      ])

    ctx_opts =
      Keyword.take(gen_opts, [:n_threads, :n_threads_batch, :n_batch, :n_ubatch])

    Stream.resource(
      fn ->
        {:ok, prompt} = Chat.apply_template(model, messages, chat_opts)
        {:ok, tokens} = Tokenizer.encode(model, prompt)
        ctx_size = max(n_ctx, length(tokens) + max_tokens)
        {:ok, ctx} = Context.create(model, Keyword.put(ctx_opts, :n_ctx, ctx_size))
        {:ok, sampler} = Sampler.create(model, sampler_opts)

        id = "chatcmpl-" <> random_hex(12)
        created = System.os_time(:second)
        model_name = Model.desc(model)

        ref = make_ref()
        parent = self()

        gen_pid =
          spawn_link(fn ->
            LlamaCppEx.NIF.generate_tokens(
              ctx.ref,
              sampler.ref,
              tokens,
              max_tokens,
              parent,
              ref
            )
          end)

        %{
          ref: ref,
          gen_pid: gen_pid,
          timeout: timeout,
          id: id,
          created: created,
          model: model_name,
          phase: :first
        }
      end,
      fn
        %{phase: :first} = state ->
          chunk = %ChatCompletionChunk{
            id: state.id,
            object: "chat.completion.chunk",
            created: state.created,
            model: state.model,
            choices: [%{index: 0, delta: %{role: "assistant", content: ""}, finish_reason: nil}]
          }

          {[chunk], %{state | phase: :streaming}}

        %{phase: :streaming, ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token, _id, text}} ->
              chunk = %ChatCompletionChunk{
                id: state.id,
                object: "chat.completion.chunk",
                created: state.created,
                model: state.model,
                choices: [%{index: 0, delta: %{content: text}, finish_reason: nil}]
              }

              {[chunk], state}

            {^ref, :eog} ->
              final_chunk = %ChatCompletionChunk{
                id: state.id,
                object: "chat.completion.chunk",
                created: state.created,
                model: state.model,
                choices: [%{index: 0, delta: %{}, finish_reason: "stop"}]
              }

              {[final_chunk], %{state | phase: :done}}

            {^ref, :done} ->
              final_chunk = %ChatCompletionChunk{
                id: state.id,
                object: "chat.completion.chunk",
                created: state.created,
                model: state.model,
                choices: [%{index: 0, delta: %{}, finish_reason: "length"}]
              }

              {[final_chunk], %{state | phase: :done}}

            {^ref, {:error, _reason}} ->
              {:halt, state}
          after
            timeout -> {:halt, state}
          end

        %{phase: :done} = state ->
          {:halt, state}
      end,
      fn %{ref: ref, gen_pid: gen_pid} ->
        Process.unlink(gen_pid)
        Process.exit(gen_pid, :kill)
        flush_stream_messages(ref)
      end
    )
  end

  defp collect_completion_tokens(ref, timeout) do
    collect_completion_tokens(ref, timeout, [], 0)
  end

  defp collect_completion_tokens(ref, timeout, texts, count) do
    receive do
      {^ref, {:token, _id, text}} ->
        collect_completion_tokens(ref, timeout, [text | texts], count + 1)

      {^ref, :eog} ->
        {Enum.reverse(texts), "stop", count}

      {^ref, :done} ->
        {Enum.reverse(texts), "length", count}

      {^ref, {:error, _reason}} ->
        {Enum.reverse(texts), "stop", count}
    after
      timeout -> {Enum.reverse(texts), "length", count}
    end
  end

  defp random_hex(n) do
    :crypto.strong_rand_bytes(n) |> Base.encode16(case: :lower)
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
