# Plain decode against every speculative drafter, through LlamaCppEx.MTP.
#
#   scripts/spark/remote.sh --env MIX_ENV=bench --big-cores spark-1 \
#     mix run bench/spark_spec.exs <target.gguf> [config ...]
#
# A config is `plain`, `mtp:<n>` (the head inside the target GGUF),
# `dflash:<n>:<drafter.gguf>` or `dspark:<n>:<drafter.gguf>`. With none given:
# plain, mtp:1..4. Each config runs the same four chat-templated prompts twice,
# greedy, 256 tokens, thinking off, and reports the median decode rate (first
# token to last, so prefill is excluded) and the draft acceptance.
#
# Greedy speculative decoding is lossless, so every config's text is also
# compared with the plain run's: a drafter that changes the output is a bug,
# whatever it does to the rate. Same prompts as the llama-server runs in
# docs/dgx-spark.md, for a like-for-like comparison.

alias LlamaCppEx.{Chat, MTP}

[target | configs] = System.argv()
configs = if configs == [], do: ["plain", "mtp:1", "mtp:2", "mtp:3", "mtp:4"], else: configs
n_gen = 256

prompts = [
  "Write a Python function that parses an ISO-8601 date string and returns a datetime, with docstring and error handling.",
  "Explain step by step how to compute the derivative of x^3 * sin(x).",
  "Write a short essay about the history of the printing press.",
  "List ten tips for writing maintainable Elixir code, one sentence each."
]

LlamaCppEx.set_log_level(:error)
:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(target, n_gpu_layers: -1, load_mtp: true)

templated =
  Enum.map(prompts, fn p ->
    {:ok, text} =
      Chat.apply_template(model, [%{role: "user", content: p}], enable_thinking: false)

    text
  end)

defmodule Spec do
  def median(xs) do
    s = Enum.sort(xs)
    n = length(s)
    if rem(n, 2) == 1, do: Enum.at(s, div(n, 2)), else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
  end

  # {text, tokens_after_first / seconds}
  def run(enum) do
    {pieces, first, last, n} =
      Enum.reduce(enum, {[], nil, nil, 0}, fn
        {:error, reason}, _ -> raise "generation failed: #{inspect(reason)}"
        piece, {acc, f, _, n} when is_binary(piece) ->
          now = System.monotonic_time(:microsecond)
          {[piece | acc], f || now, now, n + 1}
      end)

    {pieces |> Enum.reverse() |> IO.iodata_to_binary(), (n - 1) * 1_000_000 / max(last - first, 1)}
  end
end

opts = [max_tokens: n_gen, temp: 0.0]

IO.puts("target: #{Path.basename(target)}\n")
IO.puts(
  "| config | median t/s | range | vs plain | acceptance | output == plain | ms/iter draft / verify / ckpt / sample / other |"
)

IO.puts("|---|---|---|---|---|---|---|")

{reference, _} =
  Enum.reduce(configs, {nil, nil}, fn config, {ref_texts, ref_rate} ->
    {runner, cleanup, label} =
      case String.split(config, ":", parts: 3) do
        ["plain"] ->
          {fn prompt -> Spec.run(LlamaCppEx.stream(model, prompt, Keyword.put(opts, :n_ctx, 4096))) end,
           fn -> nil end, "plain"}

        [type, n | rest] ->
          draft =
            case rest do
              [path] -> elem(LlamaCppEx.load_model(path, n_gpu_layers: -1), 1)
              [] -> nil
            end

          {:ok, mtp} =
            MTP.init(model,
              type: String.to_existing_atom(type),
              draft_model: draft,
              n_draft: String.to_integer(n),
              n_ctx: 4096
            )

          {fn prompt -> Spec.run(MTP.stream(mtp, prompt, opts)) end, fn -> MTP.stats(mtp) end,
           "#{type} n=#{mtp.n_draft}"}
      end

    _warm = runner.(hd(templated))
    results = for _ <- 1..2, prompt <- templated, do: runner.(prompt)
    texts = results |> Enum.take(length(templated)) |> Enum.map(&elem(&1, 0))
    rates = Enum.map(results, &elem(&1, 1))
    med = Spec.median(rates)

    {acc, split} =
      case cleanup.() do
        %{acceptance_rate: r, iters: iters, timing_us: t} ->
          per = fn key -> Float.round(Map.fetch!(t, key) / max(iters, 1) / 1000, 2) end

          {"#{Float.round(r * 100, 1)}%",
           Enum.map_join([:draft, :verify, :ckpt, :sample, :other], " / ", &per.(&1))}

        _ ->
          {"-", "-"}
      end

    same = if ref_texts, do: to_string(texts == ref_texts), else: "-"
    vs = if ref_rate, do: "#{Float.round(med / ref_rate, 2)}x", else: "-"

    IO.puts(
      "| #{label} | #{Float.round(med, 2)} | #{Float.round(Enum.min(rates), 1)}-#{Float.round(Enum.max(rates), 1)} | #{vs} | #{acc} | #{same} | #{split} |"
    )

    :erlang.garbage_collect()
    {ref_texts || texts, ref_rate || med}
  end)

_ = reference
