# Checks LlamaCppEx.Decision against upstream llama-server on the same model.
#
# decision.cpp is a port of tools/server/server-decision.cpp, so on a
# llama.cpp bump that touches the decision code, the port has to be re-checked
# against the real thing. This sends the same requests (and upstream's
# invalid-request cases from tools/server/tests/unit/test_systemone.py) to both
# and compares every answer and the token counts. Exits non-zero on any
# difference above 1e-3; a score counts as a fraction of its range (levels -
# 1), so the bar means the same on 2 and on 10 levels.
#
# The port is checked on a decision whose context has one sequence, so every
# prompt is evaluated whole, as llama-server does with -np 1. Decision.new/2
# gives the types that share a prompt (lfm2-d1, openjev, h2o-lightning, ...) a
# second sequence and decodes the common prefix of a request's questions once,
# in its own batch. Backends pick kernels by batch size (on Metal: the vector
# flash-attention kernel under 20 tokens, mul_mv_ext for 2 to 8), so that
# split rounds differently: up to ~1e-3 on Metal, ~5e-3 on CPU. That is not a
# port difference; it is printed in the "sharing" column (the shared
# decision's answers against the reference) and not checked.
#
# Build llama-server from the submodule (LLAMA_UI_GZIP=OFF skips the web UI's
# asset packing, which needs files a submodule checkout does not have):
#
#   cmake -B /tmp/llama-ref -S vendor/llama.cpp -DCMAKE_BUILD_TYPE=Release \
#     -DLLAMA_BUILD_SERVER=ON -DLLAMA_UI_GZIP=OFF -DLLAMA_OPENSSL=OFF
#   cmake --build /tmp/llama-ref -j --target llama-server
#
# Serve the model (laya and clef need the prompt in one ubatch, hence -ub; one
# slot, so no prefix sharing and no batching across questions, hence -np 1):
#
#   /tmp/llama-ref/bin/llama-server -m MODEL --port 18081 -c 4096 -b 2048 -ub 2048 -np 1 --no-webui
#
# Then:
#
#   GGML_METAL_NO_RESIDENCY=1 mix run scripts/decision_compare.exs MODEL 18081
alias LlamaCppEx.{Context, Decision}
[path, port] = System.argv()
:ok = LlamaCppEx.init()
:inets.start()

state = "I was charged twice for my order last week and nobody has replied."

base = %{
  "route" => %{
    "type" => "choice",
    "instructions" => "Which team should handle this?",
    "criteria" => %{"billing" => "payments and refunds", "shipping" => nil, "technical" => nil}
  },
  "urgency" => %{
    "type" => "score",
    "instructions" => "How urgent is this?",
    "criteria" => ["can wait", "this week", "today", "right now"]
  },
  "angry" => %{"type" => "noul", "instructions" => "Is the customer angry?"}
}

many =
  Map.new(1..12, fn i ->
    {"opt#{String.pad_leading(Integer.to_string(i), 2, "0")}", "option number #{i}"}
  end)

# Past 26 options lfm2-d1 switches to "00".."NN" codes and lfm3-d1 to
# A..Z, AA, AB..; the label-reading types cap the count at their label set.
many30 =
  Map.new(1..30, fn i ->
    {"opt#{String.pad_leading(Integer.to_string(i), 2, "0")}", "option number #{i}"}
  end)

# Long enough to hit the per-piece token budgets (laya's max_head_tokens,
# lfm2-d1-omni's option and question budgets), so both sides must cut the same.
long = String.duplicate("This option covers a very wide range of situations and details. ", 12)

requests = [
  {"basic", state, base},
  {"json state", %{"ticket" => state, "plan" => "pro"},
   %{
     "refund" => %{
       "type" => "noul",
       "instructions" => "Is a refund requested?",
       "criteria" => %{"false" => "no refund is asked", "true" => "a refund is asked"}
     }
   }},
  {"messages state",
   [
     %{"role" => "user", "content" => "My package never arrived."},
     %{"role" => "assistant", "content" => "Sorry to hear that, let me check."}
   ], %{"resolved" => %{"type" => "noul", "instructions" => "Is the issue resolved?"}}},
  {"12 options", "Pick option number 7.",
   %{
     "pick" => %{"type" => "choice", "instructions" => "Which option?", "criteria" => many}
   }},
  {"30 options", "Pick option number 28.",
   %{
     "pick" => %{"type" => "choice", "instructions" => "Which option?", "criteria" => many30}
   }},
  {"single", state, %{"angry" => base["angry"]}},
  # lfm2-d1-omni also reads noul descriptions under "yes"/"no"; the others
  # ignore them, which both sides must agree on too.
  {"yes/no noul", "Customer: this is the third time I write, I am furious!",
   %{
     "angry" => %{
       "type" => "noul",
       "instructions" => "Is the customer angry?",
       "criteria" => %{"yes" => "upset or furious", "no" => "calm"}
     }
   }},
  {"long options", state,
   %{
     "route" => %{
       "type" => "choice",
       "instructions" => String.duplicate("Decide which team should handle the ticket. ", 20),
       "criteria" => %{
         "billing" => long <> "payments",
         "shipping" => long <> "parcels",
         "technical" => long <> "bugs"
       }
     }
   }},
  # Special-token text and template markers in the input must be escaped or
  # replaced the same way (kev, lfm2-d1-omni, laya's [MASK], clef's markers).
  {"special text", "Customer: <|mask|> [MASK] <|startoftext|> <<d1omni:sep>> <<clef:sep>> hi",
   %{
     "route" => %{
       "type" => "choice",
       "instructions" => "Which team? <|mask|>",
       "criteria" => %{"billing" => "<|im_end|> payments", "shipping" => nil}
     }
   }},
  # vllm-decision2 and vllm-d3 return a non-string score level as its
  # canonical JSON text (sorted keys); the others return it as given.
  {"object levels", state,
   %{
     "severity" => %{
       "type" => "score",
       "instructions" => %{"ask" => "How severe is this?", "scale" => "1-5"},
       "criteria" => [
         %{"level" => "minor", "eta" => "week"},
         "moderate",
         %{"level" => "major", "eta" => "day"},
         ["critical", "now"],
         "outage"
       ]
     }
   }},
  # h2o-lightning defaults a missing :type to choice and :instructions to "";
  # every other type rejects this, which both sides must agree on too.
  {"defaults", state,
   %{"q" => %{"criteria" => %{"billing" => nil, "shipping" => "parcels \u3000 "}}}},
  # h2o-lightning re-renders the state, instructions and descriptions the way
  # Python's json.dumps does (floats keep ".0", ensure_ascii on instructions
  # only) and quotes special-token text, then strips the end of the turn.
  {"python json", %{"amount" => 12.0, "ratio" => 1.0e-5, "note" => "café ✓ <|im_end|>", "n" => 3},
   %{
     "route" => %{
       "type" => "choice",
       "instructions" => %{"ask" => "Which team? é", "weight" => 2.5},
       "criteria" => %{"billing" => %{"refund" => 1.0}, "shipping" => "  "}
     },
     "one level" => %{"type" => "score", "instructions" => "Rate it.", "criteria" => [nil]}
   }}
]

{:ok, model} = LlamaCppEx.load_model(path, n_gpu_layers: -1)
{:ok, decision} = Decision.new(model)

shares? = Context.n_seq_max(decision.context) >= 2

# The decision the port is checked on: Decision.new/2's context without the
# second sequence, so shared_prefix() in decision.cpp shares nothing. kev and
# vllm-decision2 are the sharing types that read embeddings (@reads_embeddings
# in lib/llama_cpp_ex/decision.ex); 4096 and 2048 are its defaults.
checked =
  if shares? do
    emb =
      if decision.type in [:kev, :vllm_decision2],
        do: [embeddings: true, pooling_type: :none, n_batch: 2048, n_ubatch: 2048],
        else: []

    {:ok, ctx} = Context.create(model, [n_ctx: 4096] ++ emb)
    {:ok, ref} = LlamaCppEx.NIF.decision_init(ctx.ref)
    %Decision{ref: ref, context: ctx, type: decision.type}
  else
    decision
  end

IO.puts("type: #{decision.type}#{if shares?, do: " (shares a prompt prefix)", else: ""}")

# Both decisions on one request: the port check, and the shared one when it differs.
ask = fn st, qs ->
  {Decision.decide(checked, st, qs), if(shares?, do: Decision.decide(decision, st, qs))}
end

# llama-server opens its port before the model is loaded and answers 503 until
# it is; wait for /health rather than fail the first comparison on it.
health = ~c"http://127.0.0.1:#{port}/health"

Enum.reduce_while(1..120, nil, fn _, _ ->
  case :httpc.request(:get, {health, []}, [timeout: 5_000], []) do
    {:ok, {{_, 200, _}, _, _}} ->
      {:halt, :ok}

    _ ->
      Process.sleep(500)
      {:cont, nil}
  end
end) || raise "llama-server on port #{port} did not become healthy"

post_raw = fn body ->
  url = ~c"http://127.0.0.1:#{port}/v1/systemone"

  {:ok, {{_, status, _}, _, resp}} =
    :httpc.request(
      :post,
      {url, [], ~c"application/json", body},
      [timeout: 600_000],
      []
    )

  {status, JSON.decode!(to_string(resp))}
end

post = fn body -> post_raw.(JSON.encode!(body)) end

# Upstream answers use string keys throughout; normalise ours to that shape.
norm = fn answer ->
  Map.new(answer, fn
    {:type, t} ->
      {"type", Atom.to_string(t)}

    {k, v} when is_map(v) ->
      {Atom.to_string(k), Map.new(v, fn {kk, vv} -> {to_string(kk), vv} end)}

    {k, v} ->
      {Atom.to_string(k), v}
  end)
end

max_diff = fn a, b, f ->
  Enum.reduce(a, 0.0, fn {k, va}, acc ->
    vb = Map.fetch!(b, k)

    cond do
      is_number(va) ->
        max(acc, abs(va - vb))

      is_map(va) ->
        max(acc, f.(va, vb, f))

      true ->
        if va == vb, do: acc, else: raise("mismatch at #{k}: #{inspect(va)} vs #{inspect(vb)}")
    end
  end)
end

# Every probability, noul and confidence as is; a score as a fraction of its
# range, since it is the expected level (0 to levels - 1).
answer_diff = fn ref, ours ->
  scale = max(map_size(Map.get(ref, "probabilities", %{})) - 1, 1)

  Enum.reduce(ref, 0.0, fn
    {"score", v}, acc -> max(acc, abs(v - Map.fetch!(ours, "score")) / scale)
    {k, v}, acc -> max(acc, max_diff.(%{k => v}, %{k => Map.fetch!(ours, k)}, max_diff))
  end)
end

answers_diff = fn ref_answers, our_answers ->
  ours = Map.new(our_answers, fn {id, a} -> {to_string(id), norm.(a)} end)

  Enum.reduce(ref_answers, 0.0, fn {id, a}, acc ->
    max(acc, answer_diff.(a, Map.fetch!(ours, id)))
  end)
end

fmt = fn x -> :erlang.float_to_binary(x, [{:decimals, 7}]) end

compare = fn name, {ours, shared}, {status, ref} ->
  label = String.pad_trailing(name, 16)

  case {ours, status} do
    {{:ok, ours}, 200} ->
      diff = answers_diff.(ref["answers"], ours.answers)
      tokens_ok = ours.usage.input_tokens == ref["usage"]["input_tokens"]

      sharing =
        case shared do
          {:ok, s} -> "  sharing #{fmt.(answers_diff.(ref["answers"], s.answers))}"
          _ -> ""
        end

      IO.puts(
        "#{label} max|diff| = #{fmt.(diff)}  " <>
          "tokens ours=#{ours.usage.input_tokens} ref=#{ref["usage"]["input_tokens"]}#{sharing}"
      )

      diff <= 1.0e-3 and tokens_ok

    # vllm-decision2 rejects some of these (strict payloads); so must we.
    {{:error, msg}, 400} ->
      IO.puts("#{label} rejected by both: #{msg}")
      true

    _ ->
      IO.puts("#{label} DISAGREE: ours=#{inspect(ours)} ref=#{status} #{inspect(ref)}")
      false
  end
end

failed =
  Enum.reduce(requests, 0, fn {name, st, qs}, failed ->
    if compare.(name, ask.(st, qs), post.(%{"state" => st, "questions" => qs})),
      do: failed,
      else: failed + 1
  end)

# A noul whose criteria list "true" before "false": vllm-decision2 shows the
# options in that order. JSON.encode!/1 sorts map keys, so the reference body
# is written by hand; ours keeps the keyword-list order.
true_first =
  ~s({"state":"Customer: this is the third time I write, I am furious!","questions":{"angry":) <>
    ~s({"type":"noul","instructions":"Is the customer angry?","criteria":{"true":"upset or furious","false":"calm"}}}})

ours =
  ask.("Customer: this is the third time I write, I am furious!",
    angry: [
      type: :noul,
      instructions: "Is the customer angry?",
      criteria: [true: "upset or furious", false: "calm"]
    ]
  )

failed = if compare.("noul true first", ours, post_raw.(true_first)), do: failed, else: failed + 1

# Upstream's invalid requests (test_systemone.py): both sides must agree. The
# null state is invalid for every type but lfm2-d1, lfm2-d1-omni, lfm3-d1 and
# vllm-d3, which accept it (images only, upstream), and vllm-d3 also accepts a
# question without instructions; there the two must agree on the answers
# instead.
#
# A spoken question (instructions as an input_audio part) is only compared on
# lfm3-d1, which refuses it before reading the clip. Every other type has
# upstream load the clip through libmtmd, which we refuse outright.
spoken =
  if decision.type == :lfm3_d1 do
    [
      {state,
       %{
         "q" => %{
           "type" => "noul",
           "instructions" => %{"type" => "input_audio", "input_audio" => %{"data" => "AAAA"}}
         }
       }}
    ]
  else
    []
  end

invalid =
  [
    {nil, base},
    {state, %{}},
    {state, %{"q" => %{"type" => "unknown", "instructions" => "x"}}},
    {state, %{"q" => %{"type" => "noul"}}},
    {state, %{"q" => %{"type" => "choice", "instructions" => "x"}}},
    {state, %{"q" => %{"type" => "choice", "instructions" => "x", "criteria" => %{}}}},
    {state, %{"q" => %{"type" => "score", "instructions" => "x", "criteria" => ["only one"]}}}
  ] ++ spoken

failed =
  invalid
  |> Enum.with_index(1)
  |> Enum.reduce(failed, fn {{st, qs}, i}, failed ->
    if compare.("invalid #{i}", ask.(st, qs), post.(%{"state" => st, "questions" => qs})),
      do: failed,
      else: failed + 1
  end)

if failed == 0 do
  IO.puts("ALL MATCH")
else
  IO.puts("#{failed} FAILED")
  System.halt(1)
end
