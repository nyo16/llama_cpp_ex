Code.require_file("helpers.exs", __DIR__)

# Decode latency under one process-placement condition. Driven by
# scripts/spark/cpuidle-matrix.sh, which runs it once per condition; run it
# directly only to sanity-check a single setting.
#
# Deliberately narrow: TTFT and steady-state decode on a short prompt, which is
# where per-token wakeup latency shows up. A long prompt buries it under compute.
#
# The hypothesis under test is that cpuidle exit latency costs inter-token time.
# On these boxes LPI-3 exit is 433 us and holding cores out of deep idle takes
# ICMP RTT from 1.2 ms to 0.028 ms — a 43x effect on the network path. Whether it
# also costs decode is a different question, and this measures it rather than
# assuming either way.

label = System.get_env("SPARK_COND") || "unlabelled"
samples = String.to_integer(System.get_env("SPARK_SAMPLES") || "7")
decode_steps = 64

server = Bench.Helpers.start_server(n_parallel: 1, n_ctx: 4096, cache_prompt: false)
model = Bench.Helpers.await_model(server)
prompt = Bench.Helpers.prompt_of_tokens(model, 101)

time_ms = fn fun ->
  t0 = System.monotonic_time(:microsecond)
  fun.()
  (System.monotonic_time(:microsecond) - t0) / 1000
end

median = fn values ->
  sorted = Enum.sort(values)
  len = length(sorted)

  case rem(len, 2) do
    1 -> Enum.at(sorted, div(len, 2))
    0 -> (Enum.at(sorted, div(len, 2) - 1) + Enum.at(sorted, div(len, 2))) / 2
  end
end

gen = fn n -> {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: n) end
gen.(4)

ones = for _ <- 1..samples, do: time_ms.(fn -> gen.(1) end)
manys = for _ <- 1..samples, do: time_ms.(fn -> gen.(1 + decode_steps) end)

t_one = median.(ones)
t_many = median.(manys)
per_decode = (t_many - t_one) / decode_steps

# p99-ish tail on TTFT: the cpuidle story is a latency-tail story, so a median
# alone would hide exactly the effect being looked for.
worst_ttft = Enum.max(ones)

IO.puts(
  "RESULT\t#{label}\t#{Float.round(t_one, 2)}\t#{Float.round(worst_ttft, 2)}\t" <>
    "#{Float.round(1000 / per_decode, 2)}\t#{Float.round(per_decode, 3)}"
)

GenServer.stop(server)
