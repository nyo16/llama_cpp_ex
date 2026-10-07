defmodule LlamaCppEx.Decision do
  @moduledoc """
  Typed decisions with a decision model: llama.cpp's `/v1/systemone` API,
  in-process.

  A decision model answers typed questions about a `state` in one forward pass
  per prompt; no token is generated. Each answer is a probability distribution,
  so it comes with a confidence instead of free text. This is llama-server's
  [TypeSafe-compatible System One API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api),
  ported into the NIF: same request shape, same prompts, same answers.

  ## Models

  The model file says what kind of decision model it is
  (`"<arch>.decision.type"`); `model_type/1` reads it. Supported types:

    * `:openjev`, `:lev`, `:nimble`, `:pplx_decider`, `:lfm2_d1` - read the
      logits of one label token per option (`:nimble` lists every question of
      the request in each prompt, `:lev` asks each choice twice with the
      options reversed, `:pplx_decider` uses one- or two-letter label codes
      like `:lev`, `:lfm2_d1` picks label codes per question and accepts a
      `nil` state).
    * `:kev` - scores each option by a dot product of hidden states.
    * `:laya` - a ModernBERT encoder; one score per `[MASK]` marker.
    * `:clef` - reads every question in one prompt and decides them jointly.

  Decision GGUFs are published in ggml-org's "Decision models" collection on
  Hugging Face, for example `ggml-org/Laya-GGUF`, `ggml-org/Clef-Flash-GGUF` and
  `ggml-org/OpenJev-GGUF`.

  ## Questions

  `questions` maps a question id to a question. Each question has:

    * `:type` - `:choice`, `:score` or `:noul` (strings work too).
    * `:instructions` - the question. A string, or any JSON-encodable term,
      which the model is given as JSON text.
    * `:criteria` - depends on `:type`:
      * `:choice` - the options, mapping each option key to its description
        (`nil` for none). The number of options is capped by the model: 52
        for openjev, 255 for laya, clef, pplx-decider and lfm2-d1.
      * `:score` - a list of 2 to 10 level descriptions, lowest first.
      * `:noul` - optional; `%{"true" => ..., "false" => ...}` descriptions.

  The questions of a request are answered independently, except with clef,
  which decides them jointly.

  Order matters to the model: the options of a choice are shown in the order
  given, and with clef and nimble so are the questions. A map is taken in
  Elixir's map order (sorted keys for up to 32 entries); pass a keyword list or
  a list of `{key, value}` tuples to fix the order yourself. Ids and option keys
  come back exactly as given, atoms included.

  ## State

  A string, or any JSON-encodable term (a map, a list), which the model is given
  as JSON text. A list of chat messages, or a map with a `"messages"` list,
  works the same way.

  ## Answers

    * `:choice` - `%{type: :choice, choice: key, probabilities: %{key => p}, confidence: c}`
    * `:score` - `%{type: :score, score: s, legend: %{0 => desc, ...},
      probabilities: %{0 => p, ...}, confidence: c}`, where `score` is the
      probability-weighted level index and can fall between two levels.
    * `:noul` - `%{type: :noul, noul: p}`, the probability that the answer is
      true.

  `confidence` runs from 0 (every option equally likely) to 1. Probabilities
  are scaled with the temperatures stored in the model file; they are not
  guaranteed to be calibrated for your data.

  ## Example

      {:ok, model} = LlamaCppEx.load_model("Laya-Q8_0.gguf", n_gpu_layers: -1)
      {:ok, decision} = LlamaCppEx.Decision.new(model)

      {:ok, %{answers: answers}} =
        LlamaCppEx.Decision.decide(
          decision,
          "Customer message: I was charged twice for my order last week.",
          route: [
            type: :choice,
            instructions: "Which team should handle this?",
            criteria: [billing: "payments and refunds", shipping: nil, technical: nil]
          ],
          angry: [type: :noul, instructions: "Is the customer angry?"],
          urgency: [
            type: :score,
            instructions: "How urgent is this?",
            criteria: ["can wait", "this week", "today", "right now"]
          ]
        )

      answers.route.choice
      #=> :billing (0.992 with Laya-Q8_0)

  ## Limits

    * Text only. Upstream also takes images for openjev and clef through
      libmtmd, which this build does not link; a request with images is
      rejected.
    * laya and clef evaluate the whole prompt in one batch, so it has to fit in
      `:n_batch` (2048 tokens by default, see `new/2`).
    * A `%Decision{}` drives one context and is not safe to use from two
      processes at once, like `LlamaCppEx.Context`. Give each process its own,
      or serialize calls through one process.
  """

  alias LlamaCppEx.{Context, Model, Options}

  @enforce_keys [:ref, :context, :type]
  defstruct [:ref, :context, :type]

  @type decision_type ::
          :openjev | :lev | :kev | :nimble | :laya | :clef | :pplx_decider | :lfm2_d1

  @type t :: %__MODULE__{
          ref: reference(),
          context: Context.t(),
          type: decision_type()
        }

  @type id :: atom() | String.t()

  @type question ::
          %{optional(atom() | String.t()) => term()} | [{atom() | String.t(), term()}]

  @type questions :: %{optional(id()) => question()} | [{id(), question()}]

  @type answer ::
          %{
            type: :choice,
            choice: id(),
            probabilities: %{optional(id()) => float()},
            confidence: float()
          }
          | %{
              type: :score,
              score: float(),
              legend: %{optional(non_neg_integer()) => term()},
              probabilities: %{optional(non_neg_integer()) => float()},
              confidence: float()
            }
          | %{type: :noul, noul: float()}

  @type result :: %{
          answers: %{optional(id()) => answer()},
          usage: %{input_tokens: non_neg_integer(), output_tokens: 0}
        }

  @types %{
    "openjev" => :openjev,
    "lev" => :lev,
    "kev" => :kev,
    "nimble" => :nimble,
    "laya" => :laya,
    "clef" => :clef,
    "pplx-decider" => :pplx_decider,
    "lfm2-d1" => :lfm2_d1
  }

  # Mirror upstream's common_init_result and can_share_prompt; decision.cpp
  # checks the first of these against the context it is handed.
  @reads_embeddings [:laya, :kev, :clef]
  @shares_prompt [:openjev, :lev, :kev, :nimble, :pplx_decider, :lfm2_d1]

  @default_n_ctx 4096
  @default_n_batch_embeddings 2048

  # :n_batch is a tuning key already; :n_ctx is structural, read here.
  @new_opt_keys Enum.uniq([:n_ctx | Context.tuning_option_keys()])

  @doc """
  The decision type of a model.

  Returns `nil` for a model that is not a decision model and `:unknown` for a
  decision type this build does not support.
  """
  @spec model_type(Model.t()) :: decision_type() | :unknown | nil
  def model_type(%Model{ref: ref}) do
    case LlamaCppEx.NIF.decision_model_type(ref) do
      "" -> nil
      name -> Map.get(@types, name, :unknown)
    end
  end

  @doc """
  Creates a decision engine on a new context for `model`.

  The context is shaped for the model's decision type: laya, kev and clef read
  the embeddings output, so their context has embeddings on and no pooling; the
  types that share a prompt prefix across questions (openjev, lev, kev, nimble,
  pplx-decider, lfm2-d1) get a second sequence to evaluate that prefix once per
  request.

  ## Options

    * `:n_ctx` - Context size. Defaults to `#{@default_n_ctx}`. A prompt holds
      the state, one question and its options (all questions for clef and
      nimble), so a long state needs a larger context.
    * `:n_batch` - Max tokens per batch. For laya, kev and clef it is also the
      micro-batch size and caps the prompt that can be evaluated at once;
      defaults to `min(n_ctx, #{@default_n_batch_embeddings})` for them and to
      `LlamaCppEx.Context`'s default otherwise.

  Any `LlamaCppEx.Context.tuning_option_keys/0` option (`:n_threads`,
  `:flash_attn`, ...) is passed through to `LlamaCppEx.Context.create/2`.
  """
  @spec new(Model.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(%Model{} = model, opts \\ []) do
    Options.validate!(opts, @new_opt_keys, "LlamaCppEx.Decision.new/2")

    case LlamaCppEx.NIF.decision_model_type(model.ref) do
      "" ->
        {:error, "the model is not a decision model"}

      name ->
        case Map.fetch(@types, name) do
          {:ok, type} -> create(model, type, opts)
          :error -> {:error, "unsupported decision model type: #{name}"}
        end
    end
  end

  defp create(model, type, opts) do
    with {:ok, ctx} <- Context.create(model, context_opts(type, opts)),
         {:ok, ref} <- LlamaCppEx.NIF.decision_init(ctx.ref) do
      {:ok, %__MODULE__{ref: ref, context: ctx, type: type}}
    end
  end

  defp context_opts(type, opts) do
    n_ctx = Keyword.get(opts, :n_ctx, @default_n_ctx)

    embeddings =
      if type in @reads_embeddings do
        n_batch = Keyword.get(opts, :n_batch, min(n_ctx, @default_n_batch_embeddings))
        [embeddings: true, pooling_type: :none, n_batch: n_batch, n_ubatch: n_batch]
      else
        []
      end

    sharing = if type in @shares_prompt, do: [n_seq_max: 2, kv_unified: true], else: []

    # Keyword.get/3 takes the first match, so these structural values win over
    # anything the caller forwards.
    [n_ctx: n_ctx] ++ embeddings ++ sharing ++ Keyword.take(opts, Context.tuning_option_keys())
  end

  @doc """
  Answers `questions` about `state`.

  See the module doc for the shape of `state`, `questions` and the answers.
  Returns `{:error, reason}` for a request the model rejects (a missing
  `:instructions`, an unknown `:type`, too many options, a prompt that does not
  fit the context) with the same messages as llama-server.

  Raises `ArgumentError` if two question ids, or two option keys of one
  question, turn into the same string.
  """
  @spec decide(t(), term(), questions()) :: {:ok, result()} | {:error, String.t()}
  def decide(%__MODULE__{ref: ref}, state, questions) do
    {request, ids} = encode_request(state, questions)

    with {:ok, response} <- LlamaCppEx.NIF.decision_decide(ref, request) do
      {:ok, decode_response(response, ids)}
    end
  end

  # --- Request ---
  #
  # Built as JSON text by hand rather than through JSON.encode!/1 on one map,
  # because the order of the questions and of the options of a choice reaches
  # the model, and a map does not keep insertion order. Leaf values (state,
  # instructions, descriptions) go through JSON.encode_to_iodata!/1 as usual.
  #
  # `ids` maps each question id, as a JSON string, to the id as given and to
  # the option keys of that question as given, so the answers come back keyed
  # the way the caller wrote them.

  defp encode_request(state, questions) do
    pairs = pairs!(questions, "questions")
    check_unique!(pairs, "question id")

    {encoded, ids} =
      Enum.map_reduce(pairs, %{}, fn {id, question}, ids ->
        {fields, keys} = encode_question(question)
        {{key_string(id), object(fields)}, Map.put(ids, key_string(id), {id, keys})}
      end)

    request = object([{"state", JSON.encode_to_iodata!(state)}, {"questions", object(encoded)}])
    {IO.iodata_to_binary(request), ids}
  end

  defp encode_question(question) do
    Enum.map_reduce(pairs!(question, "a question"), %{}, fn {field, value}, keys ->
      case key_string(field) do
        "criteria" ->
          {criteria, keys} = encode_criteria(value)
          {{"criteria", criteria}, keys}

        name ->
          {{name, JSON.encode_to_iodata!(value)}, keys}
      end
    end)
  end

  # A list of {key, description} pairs is a choice's options in that order; any
  # other list is a score's levels. decision.cpp validates the shape per type.
  defp encode_criteria(criteria) when is_map(criteria) do
    encode_options(Map.to_list(criteria))
  end

  defp encode_criteria([_ | _] = criteria) do
    if Enum.all?(criteria, &key_pair?/1) do
      encode_options(criteria)
    else
      {JSON.encode_to_iodata!(criteria), %{}}
    end
  end

  defp encode_criteria(criteria), do: {JSON.encode_to_iodata!(criteria), %{}}

  defp encode_options(options) do
    check_unique!(options, "option key")

    fields =
      Enum.map(options, fn {key, description} ->
        {key_string(key), JSON.encode_to_iodata!(description)}
      end)

    {object(fields), Map.new(options, fn {key, _} -> {key_string(key), key} end)}
  end

  defp object(fields) do
    [
      "{",
      Enum.map_intersperse(fields, ",", fn {key, value} ->
        [JSON.encode_to_iodata!(key), ":", value]
      end),
      "}"
    ]
  end

  defp pairs!(term, _what) when is_map(term), do: Map.to_list(term)

  defp pairs!(term, what) when is_list(term) do
    if Enum.all?(term, &key_pair?/1) do
      term
    else
      raise ArgumentError,
            "#{what} must be a map or a list of {key, value} tuples, got: #{inspect(term)}"
    end
  end

  defp pairs!(term, what) do
    raise ArgumentError,
          "#{what} must be a map or a list of {key, value} tuples, got: #{inspect(term)}"
  end

  defp key_pair?({key, _}) when is_atom(key) or is_binary(key), do: true
  defp key_pair?(_), do: false

  defp key_string(key) when is_binary(key), do: key
  defp key_string(key) when is_atom(key), do: Atom.to_string(key)

  defp check_unique!(pairs, what) do
    pairs
    |> Enum.map(fn {key, _} -> key_string(key) end)
    |> Enum.frequencies()
    |> Enum.find(fn {_, n} -> n > 1 end)
    |> case do
      nil -> :ok
      {key, _} -> raise ArgumentError, "duplicate #{what}: #{inspect(key)}"
    end
  end

  # --- Response ---

  defp decode_response(response, ids) do
    %{"answers" => answers, "usage" => usage} = JSON.decode!(response)

    %{
      answers:
        Map.new(answers, fn {id, answer} ->
          {original, keys} = Map.fetch!(ids, id)
          {original, decode_answer(answer, keys)}
        end),
      usage: %{input_tokens: usage["input_tokens"], output_tokens: usage["output_tokens"]}
    }
  end

  defp decode_answer(%{"type" => "choice"} = answer, keys) do
    option = &Map.get(keys, &1, &1)

    %{
      type: :choice,
      choice: option.(answer["choice"]),
      probabilities: Map.new(answer["probabilities"], fn {key, p} -> {option.(key), p} end),
      confidence: answer["confidence"]
    }
  end

  defp decode_answer(%{"type" => "score"} = answer, _keys) do
    %{
      type: :score,
      score: answer["score"],
      legend: Map.new(answer["legend"], fn {level, desc} -> {String.to_integer(level), desc} end),
      probabilities:
        Map.new(answer["probabilities"], fn {level, p} -> {String.to_integer(level), p} end),
      confidence: answer["confidence"]
    }
  end

  defp decode_answer(%{"type" => "noul"} = answer, _keys) do
    %{type: :noul, noul: answer["noul"]}
  end
end
