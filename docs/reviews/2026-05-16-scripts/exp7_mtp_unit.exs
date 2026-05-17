# Exp 7 — MTP unit-ish tests
#
# Three targeted invariants:
#   (a) Determinism: same prompt + temp:0.0 + seed:42 → identical output across
#       two MTP.generate calls.
#   (b) Stats consistency: iters * (n_draft + 1) >= tokens_emitted; acceptance
#       rate in [0, 1].
#   (c) Reuse safety: 3 sequential MTP.generate calls on the SAME %MTP{} with
#       different prompts → each output is non-empty and coherent (no
#       contamination from the previous call's KV state).
#
# Run:
#   MIX_ENV=dev mix run --no-start /tmp/llama_review/exp7_mtp_unit.exs

mtp_path =
  System.get_env("LLAMA_MTP_MODEL_PATH") ||
    Path.expand("~/Downloads/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf")

if !File.exists?(mtp_path) do
  IO.puts(:stderr, "MTP model not found: #{mtp_path}")
  System.halt(1)
end

:ok = LlamaCppEx.init()
IO.puts("Loading MTP model: #{Path.basename(mtp_path)}")
{:ok, model} = LlamaCppEx.Model.load(mtp_path, n_gpu_layers: 999)
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 4096, flash_attn: :enabled)

IO.puts("\n=== (a) Determinism (greedy, temp:0.0, seed:42) ===")

prompt = "The first three prime numbers are"

{:ok, t1} = LlamaCppEx.MTP.generate(mtp, prompt, max_tokens: 50, temp: 0.0, top_k: 1, seed: 42)
{:ok, t2} = LlamaCppEx.MTP.generate(mtp, prompt, max_tokens: 50, temp: 0.0, top_k: 1, seed: 42)

IO.puts("Run 1: #{inspect(t1)}")
IO.puts("Run 2: #{inspect(t2)}")
IO.puts("Identical? #{t1 == t2}")

if t1 != t2 do
  IO.puts("FAIL: MTP greedy generation is non-deterministic")
else
  IO.puts("PASS: MTP greedy generation is deterministic")
end

IO.puts("\n=== (b) Stats consistency ===")

# Reset stats by fresh init for this test
{:ok, mtp_fresh} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 4096, flash_attn: :enabled)

{:ok, _} =
  LlamaCppEx.MTP.generate(mtp_fresh, "Write a short paragraph about cats.",
    max_tokens: 80,
    temp: 0.7,
    seed: 7
  )

s = LlamaCppEx.MTP.stats(mtp_fresh)
IO.inspect(s, label: "MTP stats")

# Invariants:
#   iters * (n_draft + 1) >= tokens_emitted  (each iter emits at most n_draft+1)
#   0.0 <= acceptance_rate <= 1.0
#   drafts_accepted <= drafts_generated
#   tokens_per_sec > 0 (if any tokens emitted)
#
# Each iter consumes n_draft drafts and emits 1..n_draft+1 tokens. So
# tokens_emitted >= iters (lower bound) and tokens_emitted <= iters * (n_draft+1) (upper bound).
upper_bound = s.iters * (s.n_draft + 1)
lower_bound = s.iters

invariants = [
  {"acceptance_rate in [0,1]", s.acceptance_rate >= 0.0 and s.acceptance_rate <= 1.0},
  {"drafts_accepted <= drafts_generated", s.drafts_accepted <= s.drafts_generated},
  {"tokens_emitted <= iters * (n_draft+1)", s.tokens_emitted <= upper_bound},
  {"tokens_emitted >= iters", s.tokens_emitted >= lower_bound},
  {"tokens_per_sec > 0", s.tokens_per_sec > 0.0}
]

for {name, ok} <- invariants do
  IO.puts("  #{if ok, do: "PASS", else: "FAIL"}: #{name}")
end

IO.puts("\n=== (c) Reuse safety: 3 sequential calls with different prompts ===")

prompts = [
  "The capital of Japan is",
  "Two plus two equals",
  "In the Renaissance, the artist Leonardo da Vinci painted"
]

results =
  Enum.map(prompts, fn p ->
    {:ok, text} =
      LlamaCppEx.MTP.generate(mtp, p, max_tokens: 40, temp: 0.0, top_k: 1, seed: 42)

    {p, text}
  end)

all_nonempty = Enum.all?(results, fn {_, t} -> byte_size(t) > 0 end)

IO.puts("All non-empty? #{all_nonempty}")

for {p, t} <- results do
  IO.puts("  #{inspect(p)}")
  IO.puts("    -> #{inspect(t)}")
end

# Coarse contamination check: would the previous prompt's terms leak into the
# next? With greedy + seed pinned, output should reflect only the current
# prompt. We check: does the 2nd call's output contain words from the 1st
# prompt that don't make sense in the 2nd context?
{_, t1} = Enum.at(results, 0)
{_, t2} = Enum.at(results, 1)
{_, t3} = Enum.at(results, 2)

# Trivial sanity: outputs should be distinct.
distinct = MapSet.size(MapSet.new([t1, t2, t3])) == 3
IO.puts("All outputs distinct? #{distinct}")

# Sanity-check: call MTP.generate a second time with the SAME first prompt;
# should match the very first result.
{:ok, t1_again} =
  LlamaCppEx.MTP.generate(mtp, Enum.at(prompts, 0),
    max_tokens: 40,
    temp: 0.0,
    top_k: 1,
    seed: 42
  )

IO.puts("Replay of first prompt matches initial? #{t1 == t1_again}")

if t1 != t1_again do
  IO.puts(
    "FAIL: same prompt + seed produced different output across calls — KV reuse contaminates output"
  )
end

IO.puts("\nDone.")
