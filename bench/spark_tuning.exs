Code.require_file("helpers.exs", __DIR__)

# One-variable-at-a-time runtime tuning matrix for a DGX Spark.
#
#   scripts/spark/remote.sh --env MIX_ENV=bench --big-cores spark-1 \
#     mix run bench/spark_tuning.exs
#
# Every row changes exactly one thing against the same baseline, so a number can
# be attributed. The deliverable is a short "use these settings" table, not a
# data dump — anything that moves less than a few percent gets reported as "no
# effect" and dropped from the recommendation.
#
# Unified memory makes several of these genuinely different from a discrete-GPU
# box: mlock and "offload to GPU" are the same physical DRAM here, so the usual
# advice about keeping weights off the host does not apply.

alias LlamaCppEx.Server

decode_steps = 64
samples = 3
prompt_tokens = 1024

defmodule Tuning do
  def time_ms(fun) do
    t0 = System.monotonic_time(:microsecond)
    fun.()
    (System.monotonic_time(:microsecond) - t0) / 1000
  end

  def median(values) do
    sorted = Enum.sort(values)
    len = length(sorted)

    case rem(len, 2) do
      1 -> Enum.at(sorted, div(len, 2))
      0 -> (Enum.at(sorted, div(len, 2) - 1) + Enum.at(sorted, div(len, 2))) / 2
    end
  end

  def run(label, opts, prompt_tokens, decode_steps, samples) do
    IO.write("  #{String.pad_trailing(label, 26)}")

    try do
      # Server.start_link/1 returns before the model is loaded — init/1 does
      # only what is cheap and the load happens in handle_continue/2. So the
      # first GenServer.call is what actually waits for it, and the timed region
      # has to include one or the load time reads as zero.
      load_ms =
        time_ms(fn ->
          {:ok, server} = Server.start_link(opts)
          Process.put(:srv, server)
          Process.put(:model, Bench.Helpers.await_model(server))
        end)

      server = Process.get(:srv)
      model = Process.get(:model)
      prompt = Bench.Helpers.prompt_of_tokens(model, prompt_tokens)
      {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)
      n_prompt = length(tokens)

      gen = fn n -> {:ok, _} = Server.generate(server, prompt, max_tokens: n) end
      gen.(4)

      t_one = median(for _ <- 1..samples, do: time_ms(fn -> gen.(1) end))
      t_many = median(for _ <- 1..samples, do: time_ms(fn -> gen.(1 + decode_steps) end))
      per_decode = (t_many - t_one) / decode_steps

      GenServer.stop(server)

      row = %{
        label: label,
        load_s: load_ms / 1000,
        prefill_tps: n_prompt * 1000 / (t_one - per_decode),
        decode_tps: 1000 / per_decode
      }

      IO.puts(
        "load #{Float.round(row.load_s, 1)}s  " <>
          "prefill #{Float.round(row.prefill_tps, 0)} t/s  " <>
          "decode #{Float.round(row.decode_tps, 2)} t/s"
      )

      row
    rescue
      e ->
        IO.puts("FAILED: #{Exception.message(e)}")
        %{label: label, error: Exception.message(e)}
    end
  end

  def table(title, rows, baseline_label) do
    baseline = Enum.find(rows, &(&1.label == baseline_label))

    IO.puts("\n### #{title}\n")
    IO.puts("| setting | load s | prefill t/s | decode t/s | decode vs baseline |")
    IO.puts("|---|---|---|---|---|")

    for r <- rows do
      if Map.has_key?(r, :error) do
        IO.puts("| #{r.label} | FAILED: #{r.error} | | | |")
      else
        delta =
          if baseline && baseline[:decode_tps] do
            pct = (r.decode_tps / baseline.decode_tps - 1) * 100
            "#{if pct >= 0, do: "+", else: ""}#{Float.round(pct, 1)}%"
          else
            "-"
          end

        IO.puts(
          "| #{r.label} | #{Float.round(r.load_s, 1)} | #{Float.round(r.prefill_tps, 0)} | " <>
            "#{Float.round(r.decode_tps, 2)} | #{delta} |"
        )
      end
    end
  end
end

model_path = System.get_env("LLAMA_MODEL_PATH") || raise "LLAMA_MODEL_PATH is required"
base = [model_path: model_path, n_parallel: 1, n_ctx: 4096, temp: 0.0, cache_prompt: false]

go = fn label, extra ->
  Tuning.run(label, Keyword.merge(base, extra), prompt_tokens, decode_steps, samples)
end

IO.puts(
  "\n#{Path.basename(model_path)}, #{prompt_tokens}-token prompt, #{decode_steps} decode steps\n"
)

# --- Batch sizes -------------------------------------------------------------
# The server defaults n_batch to min(n_ctx, 2048). Prefill is the only thing
# that should care.
IO.puts("batch:")

batch =
  [
    go.("n_batch default", []),
    go.("n_batch 512", n_batch: 512),
    go.("n_batch 4096", n_batch: 4096),
    go.("n_ubatch 256", n_ubatch: 256),
    go.("n_ubatch 1024", n_ubatch: 1024)
  ]

# --- Flash attention ---------------------------------------------------------
IO.puts("\nflash attention:")

flash = [
  go.("flash_attn auto", flash_attn: :auto),
  go.("flash_attn enabled", flash_attn: :enabled),
  go.("flash_attn disabled", flash_attn: :disabled)
]

# --- KV cache type -----------------------------------------------------------
# On unified memory the KV cache competes with the weights for the same DRAM
# bandwidth that decode is already limited by, so quantizing it is not only a
# capacity trade here.
IO.puts("\nKV cache type:")

kv = [
  go.("KV f16 (default)", []),
  go.("KV q8_0", type_k: :q8_0, type_v: :q8_0),
  go.("KV q4_0", type_k: :q4_0, type_v: :q4_0)
]

# --- Load mode ---------------------------------------------------------------
# mmap, mlock and direct I/O collapse into llama.cpp's single load_mode. On a
# unified-memory part "offloading" to the GPU and holding pages in RAM are the
# same physical memory, which is why mlock is worth measuring rather than
# dismissing.
IO.puts("\nload mode:")

load_mode = [
  go.("mmap (default)", use_mmap: true),
  go.("mlock + mmap", use_mmap: true, use_mlock: true),
  go.("direct I/O", use_direct_io: true),
  go.("no mmap", use_mmap: false)
]

# --- Offload -----------------------------------------------------------------
# There should be no reason to ever partially offload on this hardware: the GPU
# and the CPU address the same 121 GiB. Confirm, then say so in the docs.
IO.puts("\noffload:")

offload = [
  go.("n_gpu_layers 99", n_gpu_layers: 99),
  go.("n_gpu_layers 0 (CPU)", n_gpu_layers: 0)
]

# --- Concurrency -------------------------------------------------------------
# Throughput versus latency. Decode t/s here is per-request, so a drop with more
# slots is expected; the question is how gentle it is.
IO.puts("\nconcurrency:")

parallel =
  for n <- [1, 4, 8] do
    go.("n_parallel #{n}", n_parallel: n, n_ctx: 4096 * n)
  end

Tuning.table("Batch sizes", batch, "n_batch default")
Tuning.table("Flash attention", flash, "flash_attn auto")
Tuning.table("KV cache type", kv, "KV f16 (default)")
Tuning.table("Load mode", load_mode, "mmap (default)")
Tuning.table("Offload", offload, "n_gpu_layers 99")
Tuning.table("Concurrency (decode t/s is per-request)", parallel, "n_parallel 1")
