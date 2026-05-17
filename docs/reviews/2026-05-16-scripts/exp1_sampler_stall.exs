# Exp 1 — does sampler_sample (NOT marked dirty) stall the BEAM scheduler?
#
# Method:
#   - Heartbeat process: every 1ms, sends itself a message via Process.send_after
#     and measures actual-fire vs scheduled-fire delta.
#   - Workload: N parallel processes hammering LlamaCppEx.NIF.sampler_sample/2
#     in a tight loop, each on a warm context.
#   - If the NIF blocks an OS scheduler thread > 1 ms, heartbeat latency spikes.
#
# Run:
#   MIX_ENV=dev mix run --no-start /tmp/llama_review/exp1_sampler_stall.exs

model_path =
  System.get_env("MODEL_PATH") || Path.expand("~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

if !File.exists?(model_path) do
  IO.puts(:stderr, "model not found: #{model_path}")
  System.halt(1)
end

:ok = LlamaCppEx.init()
IO.puts("schedulers_online = #{System.schedulers_online()}")

{:ok, model} = LlamaCppEx.Model.load(model_path, n_gpu_layers: 999)

# Build N warm contexts so we can sample from each in parallel without contention.
n_workers = 4

ctxs_and_samplers =
  for _ <- 1..n_workers do
    {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048)
    {:ok, sampler} = LlamaCppEx.Sampler.create(model, temp: 0.7, top_k: 40, top_p: 0.95)
    {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Tell me a story about the ocean.")
    :ok = LlamaCppEx.NIF.decode(ctx.ref, tokens)
    {ctx, sampler}
  end

# Heartbeat: schedule a self-message every 1ms; measure actual fire latency.
# If the BEAM scheduler is stalled, fire latency >> 1ms.
heartbeat_pid =
  spawn(fn ->
    Process.flag(:priority, :high)
    parent = Process.get(:parent_pid)
    history = []

    loop = fn loop, history, t0 ->
      receive do
        :stop ->
          send(Process.get(:parent_pid), {:heartbeat_done, Enum.reverse(history)})

        :tick ->
          now = System.monotonic_time(:microsecond)
          # We asked send_after for 1ms = 1000 µs. Anything more is scheduler lag.
          lag = (now - t0) - 1000
          history = [lag | history]
          t1 = System.monotonic_time(:microsecond)
          Process.send_after(self(), :tick, 1)
          loop.(loop, history, t1)
      end
    end

    Process.put(:parent_pid, parent)
    t0 = System.monotonic_time(:microsecond)
    Process.send_after(self(), :tick, 1)
    loop.(loop, [], t0)
  end)

Process.put(:parent_pid, self())

# Tell the heartbeat who its parent is so it can send results back.
send(heartbeat_pid, :init)
:erlang.suspend_process(heartbeat_pid)
# That's hacky — use a different approach: re-spawn properly with parent baked in.
:erlang.resume_process(heartbeat_pid)

# Simpler: a re-spawned heartbeat with parent in closure.
Process.exit(heartbeat_pid, :kill)
flush = fn flush -> receive do _ -> flush.(flush) after 0 -> :ok end end
flush.(flush)

parent = self()

heartbeat_pid =
  spawn(fn ->
    Process.flag(:priority, :high)

    loop = fn loop, history, t_prev ->
      receive do
        :stop ->
          send(parent, {:heartbeat_done, Enum.reverse(history)})

        :tick ->
          now = System.monotonic_time(:microsecond)
          lag = (now - t_prev) - 1000
          Process.send_after(self(), :tick, 1)
          loop.(loop, [lag | history], now)
      end
    end

    Process.send_after(self(), :tick, 1)
    loop.(loop, [], System.monotonic_time(:microsecond))
  end)

# Phase 1: idle baseline — no sampler work for 1s.
Process.sleep(1000)
send(heartbeat_pid, :stop)

idle_lags =
  receive do
    {:heartbeat_done, lags} -> lags
  after
    2000 -> []
  end

# Phase 2: hammer phase — N workers loop sampler_sample for 2s.
parent = self()

heartbeat_pid =
  spawn(fn ->
    Process.flag(:priority, :high)

    loop = fn loop, history, t_prev ->
      receive do
        :stop ->
          send(parent, {:heartbeat_done, Enum.reverse(history)})

        :tick ->
          now = System.monotonic_time(:microsecond)
          lag = (now - t_prev) - 1000
          Process.send_after(self(), :tick, 1)
          loop.(loop, [lag | history], now)
      end
    end

    Process.send_after(self(), :tick, 1)
    loop.(loop, [], System.monotonic_time(:microsecond))
  end)

t_start = System.monotonic_time(:millisecond)
deadline = t_start + 2000

worker_pids =
  for {ctx, sampler} <- ctxs_and_samplers do
    spawn_link(fn ->
      sample_loop = fn sample_loop, history ->
        if System.monotonic_time(:millisecond) >= deadline do
          send(parent, {:worker_done, self(), Enum.reverse(history)})
        else
          t0 = System.monotonic_time(:microsecond)
          _ = LlamaCppEx.NIF.sampler_sample(sampler.ref, ctx.ref)
          t1 = System.monotonic_time(:microsecond)
          sample_loop.(sample_loop, [t1 - t0 | history])
        end
      end

      sample_loop.(sample_loop, [])
    end)
  end

# Wait for all workers to finish
worker_results =
  for _ <- worker_pids do
    receive do
      {:worker_done, _pid, history} -> history
    after
      30_000 -> []
    end
  end

send(heartbeat_pid, :stop)

busy_lags =
  receive do
    {:heartbeat_done, lags} -> lags
  after
    2000 -> []
  end

percentile = fn list, p ->
  if list == [] do
    0
  else
    sorted = Enum.sort(list)
    idx = max(0, min(length(sorted) - 1, trunc(p * length(sorted))))
    Enum.at(sorted, idx)
  end
end

IO.puts("\n=== Idle phase (no workers) ===")
IO.puts("heartbeat samples: #{length(idle_lags)}")

IO.puts(
  "heartbeat lag (µs): p50=#{percentile.(idle_lags, 0.5)} p99=#{percentile.(idle_lags, 0.99)} max=#{Enum.max(idle_lags, fn -> 0 end)}"
)

IO.puts("\n=== Busy phase (#{n_workers} workers hammering sampler_sample) ===")
all_samples = Enum.flat_map(worker_results, & &1)
IO.puts("sampler_sample calls completed: #{length(all_samples)}")

IO.puts(
  "sampler_sample µs: p50=#{percentile.(all_samples, 0.5)} p95=#{percentile.(all_samples, 0.95)} p99=#{percentile.(all_samples, 0.99)} max=#{Enum.max(all_samples, fn -> 0 end)}"
)

IO.puts("heartbeat samples: #{length(busy_lags)}")

IO.puts(
  "heartbeat lag (µs): p50=#{percentile.(busy_lags, 0.5)} p99=#{percentile.(busy_lags, 0.99)} max=#{Enum.max(busy_lags, fn -> 0 end)}"
)

# Verdict
p99_lag = percentile.(busy_lags, 0.99)
p99_sample = percentile.(all_samples, 0.99)

IO.puts("\n=== Verdict ===")
IO.puts("sampler_sample p99 = #{p99_sample} µs")

if p99_sample > 1_000 do
  IO.puts("  > 1 ms — confirms sampler_sample is a scheduler hazard if not dirty.")
end

IO.puts("heartbeat lag p99 under load = #{p99_lag} µs (vs idle p99 = #{percentile.(idle_lags, 0.99)})")

if p99_lag > 5_000 do
  IO.puts("  > 5 ms — scheduler stall CONFIRMED")
else
  IO.puts("  < 5 ms — schedulers not visibly stalled (sample work fits in normal scheduling)")
end
