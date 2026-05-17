# Exp 2 — what happens to a Server slot when the stream consumer dies?
#
# Hypothesis: Server.stream/2 does not Process.monitor the caller pid.
# Killing the consumer should leak the slot until max_tokens completes.
#
# Method:
#   1. Start a Server with n_parallel=1.
#   2. Spawn a Task that calls Server.stream/2 with max_tokens=512, pulls 3
#      tokens, then exits.
#   3. Sample Server.get_stats every 200ms for ~10s; record active_slots over
#      time.
#   4. Also: send a second :generate request right after the consumer dies.
#      If the slot is leaked, this should sit in the queue until the zombie
#      slot finishes max_tokens generation.
#
# Run:
#   MIX_ENV=dev mix run --no-start /tmp/llama_review/exp2_caller_death.exs

model_path =
  System.get_env("MODEL_PATH") || Path.expand("~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

if !File.exists?(model_path) do
  IO.puts(:stderr, "model not found: #{model_path}")
  System.halt(1)
end

:ok = LlamaCppEx.init()

{:ok, server} =
  LlamaCppEx.Server.start_link(
    model_path: model_path,
    n_gpu_layers: 999,
    n_parallel: 1,
    n_ctx: 4096,
    cache_prompt: false
  )

# Warm up
{:ok, _} = LlamaCppEx.Server.generate(server, "warmup", max_tokens: 4)
Process.sleep(200)

stats_before = LlamaCppEx.Server.get_stats(server)
IO.puts("Initial stats: #{inspect(stats_before)}")

# Spawn a consumer that pulls 3 tokens then exits.
t_consumer_start = System.monotonic_time(:millisecond)

consumer =
  spawn(fn ->
    stream =
      LlamaCppEx.Server.stream(
        server,
        "Write a long detailed essay about the history of the Roman Empire, " <>
          "covering at least five centuries with specific dates and emperors.",
        max_tokens: 512
      )

    # Pull just 3 tokens then exit.
    chunks = Enum.take(stream, 3)
    IO.puts("Consumer pulled: #{inspect(chunks)} — exiting now")
  end)

# Sample stats every 200ms for ~30s.
ref = Process.monitor(consumer)
sampler_pid = self()

monitor_task =
  Task.async(fn ->
    sample_loop = fn sample_loop, samples ->
      receive do
        :stop ->
          Enum.reverse(samples)
      after
        200 ->
          stats = LlamaCppEx.Server.get_stats(server)
          t = System.monotonic_time(:millisecond) - t_consumer_start
          sample_loop.(sample_loop, [{t, stats} | samples])
      end
    end

    sample_loop.(sample_loop, [])
  end)

# Wait for consumer to die.
t_consumer_died =
  receive do
    {:DOWN, ^ref, :process, _, _} ->
      System.monotonic_time(:millisecond) - t_consumer_start
  after
    10_000 -> nil
  end

IO.puts("Consumer died at t=#{t_consumer_died}ms")

# Try a second generate request — does it block while zombie slot runs?
t_second_call_start = System.monotonic_time(:millisecond)

second_result =
  try do
    LlamaCppEx.Server.generate(server, "Hello", max_tokens: 8, timeout: 60_000)
  catch
    :exit, reason -> {:exit, reason}
  end

t_second_call_done = System.monotonic_time(:millisecond) - t_second_call_start
IO.puts("Second generate returned in #{t_second_call_done}ms: #{inspect(second_result)}")

# Let the monitor run for a few more seconds.
Process.sleep(3000)
send(monitor_task.pid, :stop)
samples = Task.await(monitor_task, 5_000)

IO.puts("\n=== Timeline (t_ms, active, idle, queued) ===")

for {t, s} <- samples do
  IO.puts("  t=#{String.pad_leading(to_string(t), 6)}ms  active=#{s.active_slots}  idle=#{s.idle_slots}  queued=#{s.queue_depth}")
end

IO.puts("\n=== Verdict ===")

# If the slot freed within ~1 tick after consumer death, no leak.
# If it stayed active until the second generate finished, slot was leaked.
late_active =
  Enum.filter(samples, fn {t, _} -> t > (t_consumer_died || 0) + 500 end)
  |> Enum.any?(fn {_t, s} -> s.active_slots > 0 end)

if late_active do
  IO.puts("LEAK CONFIRMED: slot stayed active >500ms after consumer death")
else
  IO.puts("No leak: slot was reclaimed promptly")
end

GenServer.stop(server, :normal)
