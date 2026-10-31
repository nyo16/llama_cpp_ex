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
    * `LlamaCppEx.Decision` - Typed decisions with a decision model

  """

  alias LlamaCppEx.{
    Chat,
    ChatCompletion,
    ChatCompletionChunk,
    Context,
    Decision,
    Embedding,
    Generator,
    Grammar,
    Model,
    Options,
    Sampler,
    Server,
    Thinking,
    Tokenizer,
    UTF8Stream
  }

  # Context and sampling options are owned by the modules that consume them —
  # see Context.tuning_option_keys/0 and Sampler.option_keys/0. Do not copy the
  # lists here; three copies had already drifted apart.

  # Chat-templating options split off before the rest flows to generation. Owned
  # by LlamaCppEx.Chat.
  #
  # `:template` used to be here and documented on chat/3 and chat_completion/3 as
  # "Custom chat template string", but nothing ever read it: the chat functions
  # take the model and use its embedded template. An accepted-and-ignored option
  # is worse than no option, so it is gone and a caller passing it now gets an
  # unknown-option error.
  @chat_opt_keys Chat.template_option_keys()

  # Tool calling, accepted by chat_completion/3 and stream_chat_completion/3
  # only: chat/3 returns text, which has nowhere to put a tool call.
  @tool_opt_keys Chat.tool_option_keys()

  # Options this module reads itself, as opposed to forwarding.
  @own_opt_keys [:max_tokens, :n_ctx, :timeout, :grammar, :json_schema]

  # The complete accepted key set for the generation entry points. Assembled from
  # the owning modules at compile time so it cannot drift; a compile-time
  # dependency means this module recompiles when either list changes.
  @gen_opt_keys Enum.uniq(
                  @own_opt_keys ++
                    @chat_opt_keys ++
                    Sampler.option_keys() ++
                    Context.tuning_option_keys()
                )

  @completion_opt_keys @gen_opt_keys ++ @tool_opt_keys

  # Server-routed calls accept what the *server* accepts, plus this module's own
  # chat-templating and schema options. Assembling from @gen_opt_keys instead
  # dragged in `:n_ctx` and all 20 `Context.tuning_option_keys()`: 21 keys that
  # passed this gate and were then rejected by `Server`'s inner
  # `Options.validate!/3`, raising an ArgumentError that named
  # `LlamaCppEx.Server.complete_tokens/3` — a function the user never called.
  # A request cannot resize a running server's KV cache, so the honest answer is
  # to reject those keys here, naming this function.
  #
  # `:json_schema` stays: resolve_grammar_opts/1 turns it into `:grammar` before
  # anything is forwarded, so the server never sees the key.
  @server_opt_keys Enum.uniq(
                     [:max_tokens, :timeout, :json_schema] ++
                       @chat_opt_keys ++ @tool_opt_keys ++ Server.request_option_keys()
                   )

  # Sampler options a tool-call grammar sets; a caller's own would conflict.
  @grammar_opt_keys [:grammar, :json_schema, :grammar_lazy, :grammar_triggers, :grammar_prefill]

  @doc """
  Initializes the llama.cpp backend. Call once at application start.
  """
  @spec init() :: :ok
  def init do
    LlamaCppEx.NIF.backend_init()
  end

  # ggml_log_level values (ggml.h): DEBUG 1, INFO 2, WARN 3, ERROR 4; CONT (5),
  # above every level, prints nothing. And common_log's verbosity threshold
  # (common/log.h): DEBUG 5, INFO 3, WARN 2, ERROR 1; 0 keeps only tool output,
  # which the library never writes, so -1 silences it.
  @log_levels %{
    debug: {1, 5},
    info: {2, 3},
    warning: {3, 2},
    error: {4, 1},
    none: {5, -1}
  }

  @doc """
  Sets the lowest level of llama.cpp's own log output that is printed to stderr.

  `level` is `:debug`, `:info` (the default), `:warning`, `:error` or `:none`.
  It covers llama.cpp, ggml (backends, Metal/CUDA setup) and the chat-template
  code, for the whole VM. `:info` prints what llama.cpp prints by default
  except its DEBUG lines, which include one per generated token while a lazy
  tool-call grammar waits for its trigger.

  To set it at startup, configure it and the application applies it when it
  starts:

      config :llama_cpp_ex, log_level: :warning

  """
  @spec set_log_level(:debug | :info | :warning | :error | :none) :: :ok
  def set_log_level(level) do
    case Map.fetch(@log_levels, level) do
      {:ok, {min_level, common_threshold}} ->
        LlamaCppEx.NIF.log_set_level(min_level, common_threshold)

      :error ->
        raise ArgumentError,
              "invalid log level #{inspect(level)}, expected one of " <>
                inspect([:debug, :info, :warning, :error, :none])
    end
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

  ## Errors

  Returns `{:error, {:invalid_json_schema, message}}` when llama.cpp cannot
  convert `:json_schema` to a grammar (an unresolvable `$ref`, say), with the
  converter's message. Passing both `:grammar` and `:json_schema` raises
  `ArgumentError`.

  """
  @spec generate(Model.t(), String.t(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def generate(%Model{} = model, prompt, opts \\ []) when is_binary(prompt) do
    opts = Options.validate!(opts, @gen_opt_keys, "LlamaCppEx.generate/3")

    with {:ok, opts} <- resolve_grammar_opts(opts),
         {:ok, tokens} <- Tokenizer.encode(model, prompt) do
      generate_from_tokens(model, tokens, gen_config(opts, :blocking))
    end
  end

  defp generate_from_tokens(model, tokens, cfg) do
    with {:ok, ctx, sampler} <- create_gen_resources(model, tokens, cfg) do
      Context.generate(ctx, sampler, tokens, max_tokens: cfg.max_tokens)
    end
  end

  @doc """
  Returns a lazy stream of generated text chunks (tokens).

  Each element is a string (the text piece for one token). The stream ends
  when an end-of-generation token is produced or `max_tokens` is reached.

  Accepts the same options as `generate/3`, except that `:timeout` bounds the
  wait for the *next* chunk rather than the whole generation and therefore
  defaults to `#{LlamaCppEx.Options.stream_timeout()}` ms.

  ## Errors

  A generation failure is emitted as a final `{:error, reason}` element rather
  than truncating the stream silently, matching `LlamaCppEx.Server.stream/3`.
  Elements are otherwise always strings, so `Enum.join/1` on a stream that may
  fail is unsafe — match on the element shape if you care.

  A `:json_schema` that cannot be converted is reported the same way, before
  anything is generated: the stream's only element is
  `{:error, {:invalid_json_schema, message}}`.

  ## Examples

      model
      |> LlamaCppEx.stream("Tell me a story", max_tokens: 500)
      |> Enum.each(&IO.write/1)

  """
  @spec stream(Model.t(), String.t(), keyword()) :: Enumerable.t()
  def stream(%Model{} = model, prompt, opts \\ []) when is_binary(prompt) do
    opts = Options.validate!(opts, @gen_opt_keys, "LlamaCppEx.stream/3")

    case resolve_grammar_opts(opts) do
      {:ok, opts} ->
        token_stream(model, fn -> Tokenizer.encode(model, prompt) end, gen_config(opts, :stream))

      {:error, reason} ->
        [{:error, reason}]
    end
  end

  # The text stream behind stream/3 and stream_chat/3. `fetch_tokens` runs in
  # the start function, so tokenizing a long prompt waits for the first pull.
  defp token_stream(model, fetch_tokens, cfg) do
    Stream.resource(
      fn ->
        # Start: tokenize, create context+sampler, spawn generator. Failures
        # here surface as an {:error, reason} element rather than a raise from
        # inside the first Enum.take — Stream.resource start functions run
        # lazily, so a hard match would blow up in the consumer's face at a
        # point unrelated to the call that configured it.
        with {:ok, tokens} <- fetch_tokens.(),
             {:ok, ctx, sampler} <- create_gen_resources(model, tokens, cfg) do
          {ref, gen_pid, cancel} = spawn_generator(ctx, sampler, tokens, cfg.max_tokens)
          {ref, gen_pid, cancel, cfg.timeout, _utf8_pending = ""}
        else
          {:error, reason} -> {:setup_failed, reason}
        end
      end,
      fn
        {:setup_failed, reason} ->
          {[{:error, reason}], {:flushed, nil, nil, nil}}

        {:flushed, _ref, _gen_pid, _cancel} = state ->
          {:halt, state}

        {ref, gen_pid, cancel, timeout, pending} ->
          receive do
            {^ref, {:token, _id, text}} ->
              # Emit only whole codepoints; hold back a split multibyte tail.
              {out, pending} = UTF8Stream.push(pending, text)
              emitted = if out == "", do: [], else: [out]
              {emitted, {ref, gen_pid, cancel, timeout, pending}}

            {^ref, outcome} when outcome in [:eog, :done] ->
              flush_then_halt(pending, {:flushed, ref, gen_pid, cancel})

            {^ref, {:error, reason}} ->
              {flush_pending(pending) ++ [{:error, reason}], {:flushed, ref, gen_pid, cancel}}
          after
            timeout ->
              {flush_pending(pending) ++ [{:error, :timeout}], {:flushed, ref, gen_pid, cancel}}
          end
      end,
      fn
        {:setup_failed, _reason} -> :ok
        {:flushed, nil, nil, nil} -> :ok
        {:flushed, ref, gen_pid, cancel} -> stop_generator(ref, gen_pid, cancel)
        {ref, gen_pid, cancel, _timeout, _pending} -> stop_generator(ref, gen_pid, cancel)
      end
    )
  end

  # Emits held-back trailing bytes (if any) as a final element, then halts on
  # the next pull.
  defp flush_then_halt("", state), do: {:halt, state}
  defp flush_then_halt(pending, state), do: {[pending], state}

  defp flush_pending(""), do: []
  defp flush_pending(pending), do: [pending]

  # --- Shared generation plumbing ---

  # Option handling shared by the generation entry points. `mode` picks the
  # :timeout default; both defaults are owned by LlamaCppEx.Options.
  defp gen_config(opts, mode) do
    %{
      max_tokens: Keyword.get(opts, :max_tokens, 256),
      n_ctx: Keyword.get(opts, :n_ctx, 2048),
      timeout: Options.timeout(opts, mode),
      sampler_opts: Keyword.take(opts, Sampler.option_keys()),
      ctx_opts: Keyword.take(opts, Context.tuning_option_keys())
    }
  end

  defp split_chat_opts(opts), do: Keyword.split(opts, @chat_opt_keys ++ @tool_opt_keys)

  # A tool-call grammar comes from the chat template; a caller's own grammar
  # would replace it, so the two together are a programmer error, as in
  # llama-server ("Cannot use custom grammar constraints with tools").
  defp check_tool_grammar!(chat_opts, gen_opts) do
    if Chat.tool_grammar?(chat_opts) and
         Enum.any?(@grammar_opt_keys, &Keyword.has_key?(gen_opts, &1)) do
      raise ArgumentError,
            "cannot use :grammar or :json_schema together with :tools; the chat template " <>
              "supplies the tool-call grammar (pass tool_choice: :none to use your own)"
    end
  end

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
  # The lifecycle protocol (cancel flag, unlink, kill, drain) lives in
  # LlamaCppEx.Generator — it used to be spelled out at three call sites.
  defp spawn_generator(ctx, sampler, tokens, max_tokens) do
    {:ok, gen} =
      Generator.start(fn {parent, ref, cancel} ->
        LlamaCppEx.NIF.generate_tokens(
          ctx.ref,
          sampler.ref,
          tokens,
          max_tokens,
          parent,
          ref,
          cancel
        )
      end)

    {gen.ref, gen.pid, gen.cancel}
  end

  defp stop_generator(ref, gen_pid, cancel) do
    Generator.stop(%{ref: ref, pid: gen_pid, cancel: cancel})
  end

  defp flush_stream_messages(ref), do: Generator.drain(ref)

  # OpenAI-style finish_reason. :eog/:done come from the stateless generator
  # protocol; :max_tokens from the Server's completion metadata.
  defp finish_reason(:eog), do: "stop"
  defp finish_reason(:done), do: "length"
  defp finish_reason(:max_tokens), do: "length"

  @doc """
  Applies the chat template and generates a response.

  ## Options

  Accepts all options from `generate/3` plus the chat-templating options from
  `LlamaCppEx.Chat.apply_template/3` (`:add_assistant`, `:enable_thinking`,
  `:chat_template_kwargs`).

  Special-token text in message content (`<|im_start|>`, `[INST]`, ...) is
  tokenized as text, so a message cannot forge a turn of the conversation; see
  `LlamaCppEx.Chat`.

  ## Examples

      {:ok, reply} = LlamaCppEx.chat(model, [
        %{role: "system", content: "You are helpful."},
        %{role: "user", content: "What is Elixir?"}
      ], max_tokens: 200)

  """
  @spec chat(Model.t(), [Chat.message()], keyword()) :: {:ok, String.t()} | {:error, term()}
  def chat(%Model{} = model, messages, opts \\ []) when is_list(messages) do
    {chat_opts, gen_opts} =
      opts
      |> Options.validate!(@gen_opt_keys, "LlamaCppEx.chat/3")
      |> split_chat_opts()

    with {:ok, gen_opts} <- resolve_grammar_opts(gen_opts),
         {:ok, %{tokens: tokens}} <- Chat.prepare(model, messages, chat_opts) do
      generate_from_tokens(model, tokens, gen_config(gen_opts, :blocking))
    end
  end

  @doc """
  Returns a lazy stream of chat response chunks.

  Applies the chat template and streams the generated response token by token.
  Accepts same options as `chat/3`.

  Like `stream/3`, a failure is emitted as a final `{:error, reason}` element.
  """
  @spec stream_chat(Model.t(), [Chat.message()], keyword()) :: Enumerable.t()
  def stream_chat(%Model{} = model, messages, opts \\ []) when is_list(messages) do
    {chat_opts, gen_opts} =
      opts
      |> Options.validate!(@gen_opt_keys, "LlamaCppEx.stream_chat/3")
      |> split_chat_opts()

    with {:ok, gen_opts} <- resolve_grammar_opts(gen_opts),
         {:ok, %{tokens: tokens}} <- Chat.prepare(model, messages, chat_opts) do
      token_stream(model, fn -> {:ok, tokens} end, gen_config(gen_opts, :stream))
    else
      {:error, reason} -> [{:error, reason}]
    end
  end

  @doc """
  Generates an OpenAI-compatible chat completion response.

  Applies the chat template, runs generation, and returns a `%ChatCompletion{}`
  struct with choices, usage counts, and finish reason.

  ## Options

  Accepts all options from `generate/3` plus the chat-templating options from
  `LlamaCppEx.Chat.apply_template/3` (`:add_assistant`, `:enable_thinking`,
  `:chat_template_kwargs`), and:

    * `:tools` - Tools the model may call, in the OpenAI shape:
      `[%{type: "function", function: %{name: "get_weather", description: "...",
      parameters: json_schema}}]`. The model's chat template renders them and
      supplies the grammar and parser for its tool-call format; a template
      without tool support returns `{:error, reason}` saying so.
    * `:tool_choice` - `:auto` (default: the model decides), `:none` (the
      tools are left out of the prompt, so the reply is text), `:required` (the
      grammar does not let the reply end without a call; the model may write
      text before it), or `%{type: "function", function: %{name: name}}` to
      offer only that tool and require a call to it.
    * `:parallel_tool_calls` - Allow several calls in one reply. Defaults to
      `false`.

  With tools, a reply that calls one has `finish_reason: "tool_calls"` and its
  message carries `tool_calls: [%{id: "call_…", type: "function", function:
  %{name: name, arguments: json_text}}]`, with `content: nil` when the model
  wrote nothing else. Answer each call with a message `%{role: "tool",
  tool_call_id: id, content: result}`, after the assistant message that made
  it, and call again. `reasoning_content` is split out by the template's own
  parser, whatever `:enable_thinking` says. `:grammar` and `:json_schema`
  cannot be combined with tools (`ArgumentError`) unless `tool_choice: :none`.

  ## Examples

      {:ok, completion} = LlamaCppEx.chat_completion(model, [
        %{role: "user", content: "What is Elixir?"}
      ], max_tokens: 200)

      completion.choices |> hd() |> Map.get(:message) |> Map.get(:content)

  ## Errors

  Returns `{:error, reason}` when templating, tokenization, or generation fails,
  `{:error, :timeout}` when generation does not complete within `:timeout`,
  `{:error, {:invalid_json_schema, message}}` when `:json_schema` cannot be
  converted to a grammar, and `{:error, {:chat_parse_failed, message}}` when a
  reply does not parse in the template's tool-call format.

  > #### Breaking change in 0.8.40 {: .warning}
  >
  > Both cases previously returned `{:ok, %ChatCompletion{}}` — a NIF error was
  > reported as `finish_reason: "stop"` and a timeout as `"length"`, so a caller
  > matching only `{:ok, _}` silently used truncated or empty output. The error
  > branch of this function's `@spec` was unreachable.

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
    {chat_opts, gen_opts} =
      opts
      |> Options.validate!(@completion_opt_keys, "LlamaCppEx.chat_completion/3")
      |> split_chat_opts()

    check_tool_grammar!(chat_opts, gen_opts)

    with {:ok, gen_opts} <- resolve_grammar_opts(gen_opts),
         {:ok, prep} <- Chat.prepare(model, messages, chat_opts),
         cfg = gen_config(gen_opts ++ prep.sampler_opts, :blocking),
         {:ok, ctx, sampler} <- create_gen_resources(model, prep.tokens, cfg) do
      {ref, gen_pid, cancel} = spawn_generator(ctx, sampler, prep.tokens, cfg.max_tokens)
      outcome = collect_completion_tokens(ref, cfg.timeout, prep.session)

      # Always tear the generator down: on timeout it is still running, and its
      # stragglers would otherwise pollute the caller's mailbox. It is also how a
      # reply that reached a stop string ends early.
      stop_generator(ref, gen_pid, cancel)

      with {:ok, reply} <- outcome,
           {:ok, message, finish} <- completion_message(reply, prep.session, chat_opts) do
        {:ok,
         build_chat_completion(
           Model.desc(model),
           length(prep.tokens),
           message,
           finish,
           reply.count
         )}
      end
    end
  end

  def chat_completion(server, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} =
      opts
      |> Options.validate!(@server_opt_keys, "LlamaCppEx.chat_completion/3")
      |> split_chat_opts()

    check_tool_grammar!(chat_opts, gen_opts)

    with {:ok, gen_opts} <- resolve_grammar_opts(gen_opts),
         {:ok, model} <- Server.fetch_model(server),
         {:ok, prep} <- Chat.prepare(model, messages, chat_opts),
         {:ok, result} <-
           Server.complete_tokens(server, prep.tokens, gen_opts ++ prep.sampler_opts),
         reply = %{
           texts: [result.text],
           finish: finish_reason(result.finish_reason),
           count: result.completion_tokens
         },
         reply = feed_session(prep.session, result.token_ids, reply),
         {:ok, message, finish} <- completion_message(reply, prep.session, chat_opts) do
      {:ok,
       build_chat_completion(Model.desc(model), length(prep.tokens), message, finish, reply.count)}
    end
  end

  # The server generates without knowing the template's stop strings; the
  # session finds them in the finished reply and cuts it there.
  defp feed_session(nil, _token_ids, reply), do: reply

  defp feed_session(session, token_ids, reply) do
    case Chat.append(session, token_ids) do
      :ok -> reply
      :stop -> %{reply | finish: "stop"}
    end
  end

  # The assistant message of a finished reply. Without tools it is the raw text,
  # with `<think>` split out when `:enable_thinking` is set; with tools, the
  # template's parser splits content, reasoning and tool calls.
  defp completion_message(reply, nil, chat_opts) do
    raw_text = Enum.join(reply.texts)

    {reasoning_content, content} =
      if Keyword.get(chat_opts, :enable_thinking, false) do
        {rc, c} = Thinking.parse(raw_text)
        {if(rc == "", do: nil, else: rc), c}
      else
        {nil, raw_text}
      end

    {:ok, %{role: "assistant", content: content, reasoning_content: reasoning_content},
     reply.finish}
  end

  defp completion_message(reply, session, _chat_opts) do
    with {:ok, _diffs, parsed} <- Chat.parse(session, true) do
      {:ok, tool_message(parsed), tool_finish_reason(reply.finish, parsed)}
    end
  end

  defp tool_message(%{tool_calls: []} = parsed) do
    %{
      role: "assistant",
      content: parsed.content,
      reasoning_content: nil_if_empty(parsed.reasoning_content)
    }
  end

  defp tool_message(parsed) do
    %{
      role: "assistant",
      content: nil_if_empty(parsed.content),
      reasoning_content: nil_if_empty(parsed.reasoning_content),
      tool_calls: parsed.tool_calls
    }
  end

  # OpenAI: a reply that ended by calling tools finishes with "tool_calls".
  defp tool_finish_reason("stop", %{tool_calls: [_ | _]}), do: "tool_calls"
  defp tool_finish_reason(finish, _parsed), do: finish

  defp nil_if_empty(""), do: nil
  defp nil_if_empty(text), do: text

  defp build_chat_completion(model_desc, prompt_tokens, message, finish_reason, completion_tokens) do
    %ChatCompletion{
      id: "chatcmpl-" <> random_hex(12),
      object: "chat.completion",
      created: System.os_time(:second),
      model: model_desc,
      choices: [%{index: 0, message: message, finish_reason: finish_reason}],
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

  With `:tools`, deltas come from the template's parser instead: `%{content: _}`,
  `%{reasoning_content: _}`, and `%{tool_calls: [%{index: i, id: id, type:
  "function", function: %{name: name, arguments: ""}}]}` when a call starts,
  then `%{tool_calls: [%{index: i, function: %{arguments: fragment}}]}` as its
  arguments arrive, as OpenAI streams them. A reply that called a tool ends with
  `finish_reason: "tool_calls"`. Concatenating a call's argument fragments gives
  the `arguments` `chat_completion/3` returns.

  ## Options

  Accepts same options as `chat_completion/3`. `:timeout` bounds the wait for the
  next chunk rather than the whole generation, so it defaults to
  `#{LlamaCppEx.Options.stream_timeout()}` ms for both the model- and
  server-routed forms. It previously flipped between 60s and 30s depending on the
  type of the first argument.

  ## Errors

  A failure is emitted as a final `{:error, reason}` element, and a chunk timeout
  as `{:error, :timeout}`, rather than truncating the stream silently. Every other
  element is a `%ChatCompletionChunk{}`. A `:json_schema` that cannot be
  converted makes `{:error, {:invalid_json_schema, message}}` the only element.

  ## Examples

      model
      |> LlamaCppEx.stream_chat_completion(messages, max_tokens: 200)
      |> Enum.each(fn
        %LlamaCppEx.ChatCompletionChunk{} = chunk ->
          chunk.choices |> hd() |> get_in([:delta, :content]) |> IO.write()

        {:error, reason} ->
          IO.warn("generation failed: \#{inspect(reason)}")
      end)

  """
  @spec stream_chat_completion(Model.t() | GenServer.server(), [Chat.message()], keyword()) ::
          Enumerable.t()
  def stream_chat_completion(model_or_server, messages, opts \\ [])

  def stream_chat_completion(%Model{} = model, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} =
      opts
      |> Options.validate!(@completion_opt_keys, "LlamaCppEx.stream_chat_completion/3")
      |> split_chat_opts()

    check_tool_grammar!(chat_opts, gen_opts)

    case resolve_grammar_opts(gen_opts) do
      {:ok, gen_opts} -> model_chat_chunk_stream(model, messages, chat_opts, gen_opts)
      {:error, reason} -> [{:error, reason}]
    end
  end

  # Server-routed streaming: templating/tokenization in the caller, tokens
  # served by the batching server with prefix caching.
  def stream_chat_completion(server, messages, opts) when is_list(messages) do
    {chat_opts, gen_opts} =
      opts
      |> Options.validate!(@server_opt_keys, "LlamaCppEx.stream_chat_completion/3")
      |> split_chat_opts()

    check_tool_grammar!(chat_opts, gen_opts)

    case resolve_grammar_opts(gen_opts) do
      {:ok, gen_opts} -> server_chat_chunk_stream(server, messages, chat_opts, gen_opts)
      {:error, reason} -> [{:error, reason}]
    end
  end

  defp model_chat_chunk_stream(model, messages, chat_opts, gen_opts) do
    Stream.resource(
      fn ->
        with {:ok, prep} <- Chat.prepare(model, messages, chat_opts),
             cfg = gen_config(gen_opts ++ prep.sampler_opts, :stream),
             {:ok, ctx, sampler} <- create_gen_resources(model, prep.tokens, cfg) do
          {ref, gen_pid, cancel} = spawn_generator(ctx, sampler, prep.tokens, cfg.max_tokens)

          chat_chunk_state(ref, cfg.timeout, Model.desc(model), chat_opts, prep.session)
          |> Map.put(:gen_pid, gen_pid)
          |> Map.put(:cancel, cancel)
        else
          {:error, reason} -> %{phase: :setup_failed, reason: reason}
        end
      end,
      fn
        %{phase: :setup_failed, reason: reason} = state ->
          {[{:error, reason}], %{state | phase: :done}}

        %{phase: :first} = state ->
          {[chunk(state, %{role: "assistant", content: ""}, nil)], %{state | phase: :streaming}}

        %{phase: :streaming, ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token, id, _text}} when state.session != nil ->
              session_token_chunks(state, id)

            {^ref, {:token, _id, text}} ->
              # Whole codepoints only — the thinking parser and consumers
              # must never see a split multibyte sequence.
              {out, pending} = UTF8Stream.push(state.utf8_pending, text)
              state = %{state | utf8_pending: pending}
              if out == "", do: {[], state}, else: token_chunks(state, out)

            {^ref, outcome} when outcome in [:eog, :done] ->
              finish_chunks(state, finish_reason(outcome))

            {^ref, {:error, reason}} ->
              halt_with_error(state, reason)
          after
            timeout -> halt_with_error(state, :timeout)
          end

        %{phase: :done} = state ->
          {:halt, state}
      end,
      fn
        %{phase: :done, gen_pid: gen_pid, ref: ref, cancel: cancel} ->
          stop_generator(ref, gen_pid, cancel)

        %{ref: ref, gen_pid: gen_pid, cancel: cancel} ->
          stop_generator(ref, gen_pid, cancel)

        %{} ->
          # Setup never got as far as spawning a generator.
          :ok
      end
    )
  end

  defp server_chat_chunk_stream(server, messages, chat_opts, gen_opts) do
    timeout = Options.timeout(gen_opts, :stream)

    Stream.resource(
      fn -> start_server_chat_stream(server, messages, chat_opts, gen_opts, timeout) end,
      fn
        %{phase: :setup_failed, reason: reason} = state ->
          {[{:error, reason}], %{state | phase: :done}}

        %{phase: :first} = state ->
          {[chunk(state, %{role: "assistant", content: ""}, nil)], %{state | phase: :streaming}}

        %{phase: :streaming, ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token_id, id}} ->
              session_token_chunks(state, id)

            {^ref, {:token, text}} ->
              token_chunks(state, text)

            {^ref, {:done, reason}} ->
              finish_chunks(state, finish_reason(reason))

            {^ref, {:error, reason}} ->
              halt_with_error(state, reason)
          after
            timeout -> halt_with_error(state, :timeout)
          end

        %{phase: :done} = state ->
          {:halt, state}
      end,
      fn
        %{ref: ref, phase: phase, aborted?: aborted?} ->
          # `phase: :done` is not the same as "finished cleanly": `halt_with_error/2`
          # sets it too, both on a mid-stream error and on a per-chunk timeout.
          # Reading only that flag meant a *timed-out* stream never cancelled — the
          # slot stayed occupied and the model kept decoding to `max_tokens` for a
          # consumer that had gone. That is precisely the leak `stop_generator/3`
          # prevents on the model-routed path, reintroduced on the server path.
          if aborted? or phase != :done, do: Server.cancel(server, ref)
          flush_stream_messages(ref)

        %{} ->
          :ok
      end
    )
  end

  defp start_server_chat_stream(server, messages, chat_opts, gen_opts, timeout) do
    with {:ok, model} <- Server.fetch_model(server),
         {:ok, prep} <- Chat.prepare(model, messages, chat_opts),
         ref = make_ref(),
         # A tool-call reply is rendered and parsed here, from token ids.
         ids = if(prep.session, do: [token_ids: true], else: []),
         :ok <-
           Server.subscribe_stream_tokens(
             server,
             prep.tokens,
             ref,
             gen_opts ++ prep.sampler_opts ++ ids
           ) do
      chat_chunk_state(ref, timeout, Model.desc(model), chat_opts, prep.session)
    else
      {:error, reason} -> %{phase: :setup_failed, reason: reason}
    end
  end

  # One generated token of a tool-calling reply: feed it to the session, emit
  # what the partial parse adds. Reaching a stop string ends the reply; the
  # generation behind it is still running, hence `aborted?`.
  defp session_token_chunks(state, id) do
    case Chat.append(state.session, [id]) do
      :ok ->
        case Chat.parse(state.session, false) do
          {:ok, diffs, nil} -> {Enum.map(diffs, &diff_chunk(state, &1)), state}
          {:error, reason} -> halt_with_error(state, reason)
        end

      :stop ->
        finish_chunks(%{state | aborted?: true}, "stop")
    end
  end

  # The end of a reply: the last of the text, then the finish chunk.
  defp finish_chunks(%{session: nil} = state, finish) do
    {tail, state} = flush_pending_chunks(state)
    {tail ++ [chunk(state, %{}, finish)], %{state | phase: :done}}
  end

  defp finish_chunks(state, finish) do
    case Chat.parse(state.session, true) do
      {:ok, diffs, parsed} ->
        chunks = Enum.map(diffs, &diff_chunk(state, &1))

        {chunks ++ [chunk(state, %{}, tool_finish_reason(finish, parsed))],
         %{state | phase: :done}}

      {:error, reason} ->
        halt_with_error(state, reason)
    end
  end

  defp diff_chunk(state, {:content, text}), do: chunk(state, %{content: text}, nil)

  defp diff_chunk(state, {:reasoning_content, text}),
    do: chunk(state, %{reasoning_content: text}, nil)

  # OpenAI sends a call's id, type and name once, in its first delta.
  defp diff_chunk(state, {:tool_call, tc}) do
    function = if tc.name == "", do: %{}, else: %{name: tc.name}
    call = %{index: tc.index, function: Map.put(function, :arguments, tc.arguments)}
    call = if tc.id == "", do: call, else: Map.merge(call, %{id: tc.id, type: "function"})
    chunk(state, %{tool_calls: [call]}, nil)
  end

  # Emits any held-back text, then the error, then halts. Shared by both
  # streaming clauses so the two cannot drift apart again.
  #
  # `aborted?` is what the server-routed after-function reads: it has to
  # distinguish "the generation ended" from "we stopped listening", and `:phase`
  # cannot carry both because this function has to set `:done` to halt.
  defp halt_with_error(state, reason) do
    {tail, state} = flush_pending_chunks(state)
    {tail ++ [{:error, reason}], %{state | phase: :done, aborted?: true}}
  end

  defp flush_pending_chunks(%{utf8_pending: ""} = state), do: {[], state}

  defp flush_pending_chunks(%{utf8_pending: pending} = state) do
    token_chunks(%{state | utf8_pending: ""}, pending)
  end

  # Shared per-stream state for the chat-completion chunk builders.
  defp chat_chunk_state(ref, timeout, model_desc, chat_opts, session) do
    enable_thinking = session == nil and Keyword.get(chat_opts, :enable_thinking, false)

    %{
      ref: ref,
      timeout: timeout,
      id: "chatcmpl-" <> random_hex(12),
      created: System.os_time(:second),
      model: model_desc,
      phase: :first,
      aborted?: false,
      utf8_pending: "",
      session: session,
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

  # A failed generation and a timeout are errors, not successes. Reporting them
  # as finish_reason "stop"/"length" inside {:ok, %ChatCompletion{}} meant a
  # caller matching only {:ok, _} silently consumed truncated or empty output,
  # and made this function's error branch unreachable.
  #
  # With a chat session, each token also goes to it, and the reply ends early
  # at one of the template's stop strings.
  @spec collect_completion_tokens(reference(), timeout(), reference() | nil) ::
          {:ok, %{texts: [String.t()], finish: String.t(), count: non_neg_integer()}}
          | {:error, term()}
  defp collect_completion_tokens(ref, timeout, session) do
    collect_completion_tokens(ref, timeout, session, [], 0)
  end

  defp collect_completion_tokens(ref, timeout, session, texts, count) do
    receive do
      {^ref, {:token, id, text}} ->
        texts = [text | texts]

        if session != nil and Chat.append(session, [id]) == :stop do
          {:ok, %{texts: Enum.reverse(texts), finish: "stop", count: count + 1}}
        else
          collect_completion_tokens(ref, timeout, session, texts, count + 1)
        end

      {^ref, outcome} when outcome in [:eog, :done] ->
        {:ok, %{texts: Enum.reverse(texts), finish: finish_reason(outcome), count: count}}

      {^ref, {:error, reason}} ->
        {:error, reason}
    after
      timeout -> {:error, :timeout}
    end
  end

  defp random_hex(n) do
    :crypto.strong_rand_bytes(n) |> Base.encode16(case: :lower)
  end

  # Turns `:json_schema` into `:grammar`. A schema is data, often from user
  # config, so one llama.cpp cannot convert is an error tuple rather than a
  # raise; passing both options is a programmer error and still raises.
  defp resolve_grammar_opts(opts) do
    grammar = Keyword.get(opts, :grammar)
    json_schema = Keyword.get(opts, :json_schema)

    cond do
      grammar && json_schema ->
        raise ArgumentError, "cannot use both :grammar and :json_schema options"

      json_schema ->
        case Grammar.from_json_schema(json_schema) do
          {:ok, gbnf} ->
            {:ok, opts |> Keyword.delete(:json_schema) |> Keyword.put(:grammar, gbnf)}

          {:error, message} ->
            {:error, {:invalid_json_schema, message}}
        end

      true ->
        {:ok, opts}
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

  @doc """
  Answers typed questions about `state` with a decision model.

  Creates a `LlamaCppEx.Decision` (and its context) for this one call. To send
  several requests, create it once with `LlamaCppEx.Decision.new/2` and call
  `LlamaCppEx.Decision.decide/3` on it. See `LlamaCppEx.Decision` for the shape
  of `state`, `questions` and the answers, and `LlamaCppEx.Decision.new/2` for
  options.
  """
  @spec decide(Model.t(), term(), Decision.questions(), keyword()) ::
          {:ok, Decision.result()} | {:error, String.t()}
  def decide(%Model{} = model, state, questions, opts \\ []) do
    with {:ok, decision} <- Decision.new(model, opts) do
      Decision.decide(decision, state, questions)
    end
  end
end
