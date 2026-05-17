# Exp 5 — API edge cases
#
# For each input below: does it succeed cleanly, return {:error, _}, or crash?
# A library should never crash on user-supplied params — always {:error, _}.
#
# Run:
#   MIX_ENV=dev mix run --no-start /tmp/llama_review/exp5_edge_cases.exs

model_path =
  System.get_env("MODEL_PATH") || Path.expand("~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

if !File.exists?(model_path) do
  IO.puts(:stderr, "model not found: #{model_path}")
  System.halt(1)
end

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.Model.load(model_path, n_gpu_layers: 999)

run_case = fn name, fun ->
  result =
    try do
      fun.()
    rescue
      e -> {:rescue, e.__struct__, Exception.message(e)}
    catch
      kind, reason -> {:catch, kind, reason}
    end

  IO.puts("  #{name}:")

  case result do
    {:ok, _} -> IO.puts("    OK")
    :ok -> IO.puts("    OK (:ok)")
    {:error, e} -> IO.puts("    {:error, #{inspect(e, limit: 80)}}")
    {:rescue, type, msg} -> IO.puts("    RESCUE #{inspect(type)}: #{msg}")
    {:catch, kind, reason} -> IO.puts("    CATCH #{kind}: #{inspect(reason, limit: 80)}")
    other -> IO.puts("    OTHER: #{inspect(other, limit: 100)}")
  end

  result
end

IO.puts("\n--- LlamaCppEx.generate edge cases ---")

run_case.("empty prompt", fn ->
  LlamaCppEx.generate(model, "", max_tokens: 10)
end)

run_case.("max_tokens: 0", fn ->
  LlamaCppEx.generate(model, "Hello", max_tokens: 0)
end)

run_case.("max_tokens: 1", fn ->
  LlamaCppEx.generate(model, "Hello", max_tokens: 1)
end)

run_case.("max_tokens: -1", fn ->
  LlamaCppEx.generate(model, "Hello", max_tokens: -1)
end)

IO.puts("\n--- Context.create edge cases ---")

run_case.("n_ctx: 1", fn ->
  LlamaCppEx.Context.create(model, n_ctx: 1)
end)

run_case.("n_ctx: 0", fn ->
  LlamaCppEx.Context.create(model, n_ctx: 0)
end)

run_case.("n_ctx: -1", fn ->
  LlamaCppEx.Context.create(model, n_ctx: -1)
end)

run_case.("n_ctx larger than train", fn ->
  LlamaCppEx.Context.create(model, n_ctx: 1_000_000)
end)

IO.puts("\n--- Sampler edge cases ---")

run_case.("top_k: 0 (= disabled)", fn ->
  LlamaCppEx.Sampler.create(model, top_k: 0, temp: 1.0)
end)

run_case.("top_k: 1, temp: 0.0 (strict greedy)", fn ->
  LlamaCppEx.Sampler.create(model, top_k: 1, temp: 0.0, seed: 42)
end)

run_case.("temp: -1.0", fn ->
  LlamaCppEx.Sampler.create(model, temp: -1.0)
end)

run_case.("temp: 100.0 (extreme high)", fn ->
  LlamaCppEx.Sampler.create(model, temp: 100.0)
end)

run_case.("top_p: 0.0", fn ->
  LlamaCppEx.Sampler.create(model, top_p: 0.0, temp: 1.0)
end)

IO.puts("\n--- Determinism check (top_k: 1, temp: 0.0, seed: 42) ---")

{:ok, ctx1} = LlamaCppEx.Context.create(model, n_ctx: 1024)
{:ok, sampler1} = LlamaCppEx.Sampler.create(model, top_k: 1, temp: 0.0, seed: 42)
prompt = "The capital of France is"
{:ok, t1} = LlamaCppEx.generate(model, prompt, max_tokens: 20, top_k: 1, temp: 0.0, seed: 42)
IO.puts("  Run 1: #{inspect(t1)}")

{:ok, t2} = LlamaCppEx.generate(model, prompt, max_tokens: 20, top_k: 1, temp: 0.0, seed: 42)
IO.puts("  Run 2: #{inspect(t2)}")

IO.puts("  identical? #{t1 == t2}")

IO.puts("\n--- MTP edge cases (n_draft validation) ---")

mtp_model =
  case System.get_env("LLAMA_MTP_MODEL_PATH") do
    nil -> Path.expand("~/Downloads/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf")
    path -> path
  end

if File.exists?(mtp_model) do
  {:ok, mm} = LlamaCppEx.Model.load(mtp_model, n_gpu_layers: 999)

  run_case.("MTP n_draft: 0", fn ->
    LlamaCppEx.MTP.init(mm, n_draft: 0, n_ctx: 1024)
  end)

  run_case.("MTP n_draft: -3", fn ->
    LlamaCppEx.MTP.init(mm, n_draft: -3, n_ctx: 1024)
  end)

  run_case.("MTP n_draft: nil", fn ->
    LlamaCppEx.MTP.init(mm, n_draft: nil, n_ctx: 1024)
  end)
else
  IO.puts("  (skipping MTP cases — model not at #{mtp_model})")
end

IO.puts("\n--- Tokenizer edge cases ---")

run_case.("tokenize empty string", fn ->
  LlamaCppEx.Tokenizer.encode(model, "")
end)

run_case.("tokenize 100 KB string", fn ->
  big = String.duplicate("a ", 50_000)
  LlamaCppEx.Tokenizer.encode(model, big)
end)

run_case.("decode empty token list", fn ->
  LlamaCppEx.Tokenizer.decode(model, [])
end)

run_case.("decode invalid token (-1)", fn ->
  LlamaCppEx.Tokenizer.decode(model, [-1])
end)

run_case.("decode out-of-vocab token (10_000_000)", fn ->
  LlamaCppEx.Tokenizer.decode(model, [10_000_000])
end)

IO.puts("\n--- context_can_seq_rm side-effect demonstration ---")
{:ok, ctx_demo} = LlamaCppEx.Context.create(model, n_ctx: 1024)
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello world")
:ok = LlamaCppEx.NIF.decode(ctx_demo.ref, tokens)
pos_before = LlamaCppEx.NIF.memory_seq_pos_max(ctx_demo.ref, 0)
IO.puts("  pos_max before context_can_seq_rm: #{pos_before}")
_ = LlamaCppEx.NIF.context_can_seq_rm(ctx_demo.ref)
pos_after = LlamaCppEx.NIF.memory_seq_pos_max(ctx_demo.ref, 0)
IO.puts("  pos_max after  context_can_seq_rm: #{pos_after}")

if pos_before > 0 and pos_after < pos_before do
  IO.puts("  CONFIRMED: context_can_seq_rm silently wiped KV memory")
end
