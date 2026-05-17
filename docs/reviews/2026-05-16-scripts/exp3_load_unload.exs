# Exp 3 — model load/unload memory leak
#
# Hypothesis: Repeated Model.load → Context.create → drop refs → GC does not
# return native memory.
#
# Method:
#   - Capture RSS before.
#   - In a loop of N: load model, create context, decode some tokens, drop
#     references, force GC.
#   - Capture RSS after each iteration.
#   - Plateau = healthy (file mmap'd). Linear growth = leak.
#
# Run:
#   MIX_ENV=dev mix run --no-start /tmp/llama_review/exp3_load_unload.exs

model_path =
  System.get_env("MODEL_PATH") || Path.expand("~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

if !File.exists?(model_path) do
  IO.puts(:stderr, "model not found: #{model_path}")
  System.halt(1)
end

:ok = LlamaCppEx.init()

pid = System.pid()

rss = fn ->
  {out, 0} = System.cmd("ps", ["-o", "rss=", "-p", pid])
  out |> String.trim() |> String.to_integer()
end

iters = 10

IO.puts("Initial RSS: #{rss.()} KB")

samples =
  for i <- 1..iters do
    {:ok, model} = LlamaCppEx.Model.load(model_path, n_gpu_layers: 999)
    {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048)
    {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello world this is a test")
    :ok = LlamaCppEx.NIF.decode(ctx.ref, tokens)

    # Drop everything
    _ = model
    _ = ctx
    _ = tokens

    # Force GC several times; resources are released on GC of the BEAM term.
    for _ <- 1..3, do: :erlang.garbage_collect()
    Process.sleep(200)

    r = rss.()
    IO.puts("  iter #{String.pad_leading(to_string(i), 2)}: RSS = #{r} KB")
    r
  end

# Compute slope: best-fit RSS growth per iter from iter 3 onward
# (skip warmup variance).
later = Enum.drop(samples, 2)

slope_per_iter =
  if length(later) >= 2 do
    first = hd(later)
    last = List.last(later)
    (last - first) / (length(later) - 1)
  else
    0
  end

IO.puts("\n=== Verdict ===")
IO.puts("Avg RSS growth per iter (iters 3..N): #{Float.round(slope_per_iter / 1, 1)} KB")

cond do
  slope_per_iter > 100_000 ->
    IO.puts("LEAK: > 100 MB/iter growth — full model size being leaked each load")

  slope_per_iter > 10_000 ->
    IO.puts("LEAK SUSPECT: > 10 MB/iter growth")

  slope_per_iter > 1_000 ->
    IO.puts("MINOR LEAK: > 1 MB/iter growth")

  true ->
    IO.puts("No leak: RSS plateaus")
end
