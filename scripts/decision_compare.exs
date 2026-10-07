# Checks LlamaCppEx.Decision against upstream llama-server on the same model.
#
# decision.cpp is a port of tools/server/server-decision.cpp, so on a
# llama.cpp bump that touches the decision code, the port has to be re-checked
# against the real thing. This sends the same requests (and upstream's
# invalid-request cases from tools/server/tests/unit/test_systemone.py) to both
# and compares every probability and the token counts. Exits non-zero on any
# difference above 1e-3.
#
# Build llama-server from the submodule (LLAMA_UI_GZIP=OFF skips the web UI's
# asset packing, which needs files a submodule checkout does not have):
#
#   cmake -B /tmp/llama-ref -S vendor/llama.cpp -DCMAKE_BUILD_TYPE=Release \
#     -DLLAMA_BUILD_SERVER=ON -DLLAMA_UI_GZIP=OFF -DLLAMA_OPENSSL=OFF
#   cmake --build /tmp/llama-ref -j --target llama-server
#
# Serve the model (laya and clef need the prompt in one ubatch, hence -ub):
#
#   /tmp/llama-ref/bin/llama-server -m MODEL --port 18081 -c 4096 -b 2048 -ub 2048 --no-webui
#
# Then:
#
#   GGML_METAL_NO_RESIDENCY=1 mix run scripts/decision_compare.exs MODEL 18081
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
  {"single", state, %{"angry" => base["angry"]}}
]

{:ok, model} = LlamaCppEx.load_model(path, n_gpu_layers: -1)
{:ok, decision} = LlamaCppEx.Decision.new(model)
IO.puts("type: #{decision.type}")

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

post = fn body ->
  url = ~c"http://127.0.0.1:#{port}/v1/systemone"

  {:ok, {{_, status, _}, _, resp}} =
    :httpc.request(
      :post,
      {url, [], ~c"application/json", JSON.encode!(body)},
      [timeout: 600_000],
      []
    )

  {status, JSON.decode!(to_string(resp))}
end

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

failed =
  Enum.reduce(requests, 0, fn {name, st, qs}, failed ->
    {:ok, ours} = LlamaCppEx.Decision.decide(decision, st, qs)
    {200, ref} = post.(%{"state" => st, "questions" => qs})

    ours_answers = Map.new(ours.answers, fn {id, a} -> {id, norm.(a)} end)
    diff = max_diff.(ref["answers"], ours_answers, max_diff)
    tokens_ok = ours.usage.input_tokens == ref["usage"]["input_tokens"]

    IO.puts(
      "#{String.pad_trailing(name, 16)} max|diff| = #{:erlang.float_to_binary(diff, [{:decimals, 7}])}  " <>
        "tokens ours=#{ours.usage.input_tokens} ref=#{ref["usage"]["input_tokens"]}"
    )

    if diff > 1.0e-3 or not tokens_ok, do: failed + 1, else: failed
  end)

# Upstream's invalid requests (test_systemone.py): both sides must agree. The
# null state is invalid for every type but lfm2-d1, which accepts it (images
# only, upstream); there the two must agree on the answers instead.
invalid = [
  {nil, base},
  {state, %{}},
  {state, %{"q" => %{"type" => "unknown", "instructions" => "x"}}},
  {state, %{"q" => %{"type" => "noul"}}},
  {state, %{"q" => %{"type" => "choice", "instructions" => "x"}}},
  {state, %{"q" => %{"type" => "choice", "instructions" => "x", "criteria" => %{}}}},
  {state, %{"q" => %{"type" => "score", "instructions" => "x", "criteria" => ["only one"]}}}
]

failed =
  Enum.reduce(invalid, failed, fn {st, qs}, failed ->
    ours = LlamaCppEx.Decision.decide(decision, st, qs)
    {status, ref} = post.(%{"state" => st, "questions" => qs})

    case {ours, status} do
      {{:error, _}, 400} ->
        IO.puts(
          "invalid: ours=#{inspect(ours)} ref=400 #{inspect(get_in(ref, ["error", "message"]))}"
        )

        failed

      {{:ok, %{answers: answers, usage: usage}}, 200} ->
        diff =
          max_diff.(ref["answers"], Map.new(answers, fn {id, a} -> {id, norm.(a)} end), max_diff)

        tokens_ok = usage.input_tokens == ref["usage"]["input_tokens"]

        IO.puts(
          "accepted by both: max|diff| = #{:erlang.float_to_binary(diff, [{:decimals, 7}])} tokens ours=#{usage.input_tokens} ref=#{ref["usage"]["input_tokens"]}"
        )

        if diff > 1.0e-3 or not tokens_ok, do: failed + 1, else: failed

      _ ->
        IO.puts("DISAGREE: ours=#{inspect(ours)} ref=#{status} #{inspect(ref)}")
        failed + 1
    end
  end)

if failed == 0 do
  IO.puts("ALL MATCH")
else
  IO.puts("#{failed} FAILED")
  System.halt(1)
end
