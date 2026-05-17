# Exp 6 — Server concurrency stress
#
# Hypothesis: Server.tick loop and queue handling stay coherent under N
# concurrent callers, with mixed generate/stream. Prefix cache should not
# return another caller's response.
#
# Method:
#   - n_parallel=2, 4 concurrent Tasks, each Server.generate with a unique
#     trailing suffix. Verify each got their own response.
#   - n_parallel=4, 16 concurrent Tasks, mixed generate + stream.
#   - Throughout: check get_stats returns sensible numbers, no error logs.
#
# Run:
#   MIX_ENV=dev mix run --no-start /tmp/llama_review/exp6_concurrency.exs

model_path =
  System.get_env("MODEL_PATH") || Path.expand("~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

if !File.exists?(model_path) do
  IO.puts(:stderr, "model not found: #{model_path}")
  System.halt(1)
end

:ok = LlamaCppEx.init()

# Test 1: 4 concurrent Tasks, n_parallel=2, with shared prefix.
IO.puts("=== Test 1: 4 concurrent generate, n_parallel=2, cache_prompt=true ===")

{:ok, srv1} =
  LlamaCppEx.Server.start_link(
    model_path: model_path,
    n_gpu_layers: 999,
    n_parallel: 2,
    n_ctx: 4096,
    cache_prompt: true,
    flash_attn: :enabled
  )

# Warmup
{:ok, _} = LlamaCppEx.Server.generate(srv1, "warmup", max_tokens: 4)

prefix =
  "You are a careful assistant. Answer the user's question with a single number. " <>
    "Be precise. The user's question is:\n\n"

suffixes = [
  "How many planets in the Solar System?",
  "How many days in a non-leap year?",
  "How many cards in a standard deck?",
  "How many sides on a hexagon?"
]

t0 = System.monotonic_time(:millisecond)

tasks_1 =
  Enum.map(suffixes, fn s ->
    Task.async(fn ->
      {:ok, text} =
        LlamaCppEx.Server.generate(srv1, prefix <> s,
          max_tokens: 32,
          timeout: 60_000
        )

      {s, text}
    end)
  end)

results_1 = Task.await_many(tasks_1, 60_000)
t_elapsed_1 = System.monotonic_time(:millisecond) - t0

IO.puts("Elapsed: #{t_elapsed_1}ms")

for {s, r} <- results_1 do
  IO.puts("  Q: #{s}")
  IO.puts("    A: #{inspect(String.slice(r, 0, 80))}")
end

# Sanity: each output should be non-empty.
all_nonempty_1 = Enum.all?(results_1, fn {_, r} -> byte_size(r) > 0 end)
IO.puts("All non-empty? #{all_nonempty_1}")

stats_after_1 = LlamaCppEx.Server.get_stats(srv1)
IO.puts("Stats after: #{inspect(stats_after_1)}")

GenServer.stop(srv1, :normal)

# Test 2: 16 concurrent mixed generate/stream, n_parallel=4.
IO.puts("\n=== Test 2: 16 concurrent mixed generate/stream, n_parallel=4 ===")

{:ok, srv2} =
  LlamaCppEx.Server.start_link(
    model_path: model_path,
    n_gpu_layers: 999,
    n_parallel: 4,
    n_ctx: 4096,
    cache_prompt: false,
    flash_attn: :enabled
  )

{:ok, _} = LlamaCppEx.Server.generate(srv2, "warmup", max_tokens: 4)

prompts_2 =
  for i <- 1..16 do
    "Write one sentence about the color #{i} of the rainbow."
  end

t1 = System.monotonic_time(:millisecond)

# Even indices: generate; odd indices: stream.
tasks_2 =
  prompts_2
  |> Enum.with_index()
  |> Enum.map(fn {p, i} ->
    Task.async(fn ->
      try do
        if rem(i, 2) == 0 do
          case LlamaCppEx.Server.generate(srv2, p, max_tokens: 16, timeout: 90_000) do
            {:ok, t} -> {:gen, i, byte_size(t)}
            {:error, e} -> {:gen_err, i, e}
          end
        else
          chunks =
            srv2
            |> LlamaCppEx.Server.stream(p, max_tokens: 16, timeout: 60_000)
            |> Enum.to_list()

          {:stream, i, length(chunks), IO.iodata_to_binary(chunks) |> byte_size()}
        end
      catch
        kind, reason -> {:caught, i, kind, reason}
      end
    end)
  end)

results_2 = Task.await_many(tasks_2, 120_000)
t_elapsed_2 = System.monotonic_time(:millisecond) - t1

IO.puts("Elapsed: #{t_elapsed_2}ms")

success_count =
  Enum.count(results_2, fn
    {:gen, _, n} when n > 0 -> true
    {:stream, _, _, n} when n > 0 -> true
    _ -> false
  end)

errors = Enum.filter(results_2, fn
  {:gen_err, _, _} -> true
  {:caught, _, _, _} -> true
  _ -> false
end)

IO.puts("Successes: #{success_count}/16")
IO.puts("Errors: #{inspect(errors, limit: 100)}")

stats_after_2 = LlamaCppEx.Server.get_stats(srv2)
IO.puts("Stats after: #{inspect(stats_after_2)}")

GenServer.stop(srv2, :normal)

IO.puts("\nDone.")
