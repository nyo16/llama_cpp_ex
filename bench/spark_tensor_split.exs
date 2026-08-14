Code.require_file("helpers.exs", __DIR__)

# The tp=2 spike: does `split_mode: :tensor` do anything useful here?
#
#   # 6.1 — single node, Meta device over one local GPU
#   scripts/spark/remote.sh --env MIX_ENV=bench spark-1 \
#     mix run bench/spark_tensor_split.exs local
#
#   # 6.2 — two nodes, Meta device over [CUDA0, RPC0]
#   scripts/spark/rpc-worker.sh start spark-2
#   scripts/spark/remote.sh --env MIX_ENV=bench --env LLAMA_RPC=1 spark-1 \
#     mix run bench/spark_tensor_split.exs remote
#
# "tp=2" is vLLM vocabulary. In llama.cpp at b10362 it means
# LLAMA_SPLIT_MODE_TENSOR, added in #19378, which builds a Meta device wrapping
# N real devices. Everything interesting about it is in how the all-reduce is
# implemented:
#
#   - ggml_backend_cuda_comm_init returns nullptr the moment ANY member backend
#     is not CUDA (ggml-cuda.cu:1209-1213). With an RPC device in the set, the
#     CUDA all-reduce — NCCL or the internal pipeline — never engages at all.
#   - What runs instead is the meta backend's generic butterfly, which moves
#     data with ggml_backend_tensor_{set,get}_2d. The RPC backend leaves both
#     2-D hooks NULL, and ggml-backend.cpp:360,382 then fall back to a LOOP of
#     n_copies separate 1-D transfers. That is a performance cliff, not a
#     failure — which is why this is worth running rather than reasoning about.
#
# GGML_CUDA_ALLREDUCE selects the CUDA path where one applies: nccl | internal |
# none. It is read in ggml_backend_cuda_comm_init and is the only lever.
#
# Time-box for the remote leg: 90 minutes. Capture the exact failure if it
# fails — the failure IS the deliverable.

alias LlamaCppEx.Server

mode =
  case System.argv() do
    [m | _] when m in ["local", "remote"] -> m
    _ -> "local"
  end

endpoint = System.get_env("SPARK_RPC_ENDPOINT") || "10.100.64.2:50052"
model_path = System.get_env("LLAMA_MODEL_PATH") || raise "LLAMA_MODEL_PATH is required"
decode_steps = 32
prompt_tokens = 512

defmodule Spike do
  def time_ms(fun) do
    t0 = System.monotonic_time(:microsecond)
    fun.()
    (System.monotonic_time(:microsecond) - t0) / 1000
  end

  def median(v) do
    s = Enum.sort(v)
    n = length(s)

    if rem(n, 2) == 1,
      do: Enum.at(s, div(n, 2)),
      else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
  end

  # Everything is wrapped, because the point of the exercise is to survive and
  # report the failure rather than to succeed. A GGML_ABORT would take the VM
  # with it and no rescue can help — that outcome is recorded from the outside,
  # by the exit status and the tail of the log.
  def attempt(label, opts, prompt_tokens, decode_steps) do
    IO.puts("\n=== #{label}")

    try do
      # start_link returns before the model is loaded (handle_continue/2 does
      # the work), so the first GenServer.call is what waits. Timing has to
      # include one, and here that also means an unsupported configuration
      # surfaces as an exit from the call rather than a silent success.
      load_ms =
        time_ms(fn ->
          {:ok, server} = Server.start_link(opts)
          Process.put(:srv, {:ok, server})
          Process.put(:model, Bench.Helpers.await_model(server))
        end)

      case Process.get(:srv) do
        {:ok, server} ->
          model = Process.get(:model)
          prompt = Bench.Helpers.prompt_of_tokens(model, prompt_tokens)
          {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)
          n_prompt = length(tokens)

          gen = fn n -> Server.generate(server, prompt, max_tokens: n) end
          {:ok, _} = gen.(4)

          t_one = median(for _ <- 1..3, do: time_ms(fn -> {:ok, _} = gen.(1) end))
          t_many = median(for _ <- 1..3, do: time_ms(fn -> {:ok, _} = gen.(1 + decode_steps) end))
          per_decode = (t_many - t_one) / decode_steps

          {:ok, text} = gen.(16)
          GenServer.stop(server)

          row = %{
            label: label,
            load_s: load_ms / 1000,
            prefill_tps: n_prompt * 1000 / (t_one - per_decode),
            decode_tps: 1000 / per_decode,
            sample: String.slice(text, 0, 40)
          }

          IO.puts("  load     #{Float.round(row.load_s, 1)} s")
          IO.puts("  prefill  #{Float.round(row.prefill_tps, 1)} t/s")
          IO.puts("  decode   #{Float.round(row.decode_tps, 2)} t/s")
          IO.puts("  output   #{inspect(row.sample)}")
          row

        {:error, reason} ->
          IO.puts("  REFUSED at load: #{inspect(reason)}")
          %{label: label, error: inspect(reason)}
      end
    rescue
      e ->
        IO.puts("  RAISED: #{Exception.message(e)}")
        %{label: label, error: Exception.message(e)}
    catch
      kind, value ->
        IO.puts("  #{kind}: #{inspect(value)}")
        %{label: label, error: "#{kind} #{inspect(value)}"}
    end
  end

  def table(rows, baseline_label) do
    baseline = Enum.find(rows, &(&1.label == baseline_label && !Map.has_key?(&1, :error)))

    IO.puts("\n| configuration | load s | prefill t/s | decode t/s | vs baseline |")
    IO.puts("|---|---|---|---|---|")

    for r <- rows do
      if Map.has_key?(r, :error) do
        IO.puts("| #{r.label} | — | — | — | **#{r.error}** |")
      else
        delta =
          if baseline do
            pct = (r.decode_tps / baseline.decode_tps - 1) * 100
            "#{if pct >= 0, do: "+", else: ""}#{Float.round(pct, 1)}%"
          else
            "-"
          end

        IO.puts(
          "| #{r.label} | #{Float.round(r.load_s, 1)} | #{Float.round(r.prefill_tps, 1)} | " <>
            "#{Float.round(r.decode_tps, 2)} | #{delta} |"
        )
      end
    end
  end
end

base = [model_path: model_path, n_parallel: 1, n_ctx: 4096, temp: 0.0, cache_prompt: false]

rows =
  case mode do
    "local" ->
      # 6.1: does the Meta device cost anything at all with one local GPU, and
      # is our architecture even accepted? llm_arch_supports_sm_tensor is a
      # blocklist (llama-arch.cpp:1009-1042), so most architectures pass —
      # qwen3 and gpt-oss both do. flash attention is force-enabled by the mode
      # (llama-context.cpp), so :none is compared with it on to keep the
      # comparison about the Meta device rather than about flash attention.
      [
        Spike.attempt(
          "split_mode :none (flash on)",
          base ++ [split_mode: :none, flash_attn: :enabled],
          prompt_tokens,
          decode_steps
        ),
        Spike.attempt(
          "split_mode :tensor, 1 local GPU",
          base ++ [split_mode: :tensor],
          prompt_tokens,
          decode_steps
        )
      ]

    "remote" ->
      case LlamaCppEx.RPC.add_server(endpoint) do
        {:ok, n} ->
          IO.puts("registered #{endpoint}: #{n} device(s)")

        {:error, reason} ->
          IO.puts(:stderr, "cannot reach #{endpoint}: #{inspect(reason)}")
          System.halt(1)
      end

      devices = LlamaCppEx.devices()
      local = Enum.find(devices, &(&1.type in [:gpu, :igpu] and &1.backend != "RPC"))
      remote = Enum.find(devices, &(&1.backend == "RPC"))
      pair = [local.name, remote.name]

      IO.puts("Meta device over #{inspect(pair)}")

      IO.puts(
        "GGML_CUDA_ALLREDUCE=#{System.get_env("GGML_CUDA_ALLREDUCE") || "(unset — Linux default is nccl)"}"
      )

      [
        Spike.attempt(
          "layer split, 2 nodes (reference)",
          base ++ [devices: pair, split_mode: :layer, tensor_split: [0.5, 0.5]],
          prompt_tokens,
          decode_steps
        ),
        Spike.attempt(
          "tensor split, 2 nodes",
          base ++ [devices: pair, split_mode: :tensor],
          prompt_tokens,
          decode_steps
        )
      ]
  end

Spike.table(
  rows,
  if(mode == "local", do: "split_mode :none (flash on)", else: "layer split, 2 nodes (reference)")
)
