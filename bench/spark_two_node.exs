Code.require_file("helpers.exs", __DIR__)

# Two-node measurement suite for a pair of DGX Sparks.
#
#   scripts/spark/rpc-worker.sh start spark-2
#   scripts/spark/remote.sh --env MIX_ENV=bench --env LLAMA_RPC=1 spark-1 \
#     mix run bench/spark_two_node.exs <bench>
#
# Benches, each answering one question:
#
#   b1   RPC overhead, controlled. A model that FITS one node, run single-node
#        and then layer-split across two. Isolates the cost of crossing the
#        network from the benefit of more memory. Expect decode roughly
#        unchanged and cold load materially worse.
#   b2   The actual point. A model that does NOT fit one node: two-node RPC
#        versus single-node paging weights off NVMe. This is the number that
#        justifies the second Spark, and it is allowed to come out negative.
#   b3   RDMA versus TCP, in tokens. There is no runtime switch, so TCP is
#        forced with GGML_RDMA_DEV set to a device that does not exist.
#   b4   Batching amortisation. Per-token RTT is fixed per *graph*, not per
#        token, so n_parallel should amortise it. Measure, do not assume.
#
# Every run reports worker RSS before and after: upstream's worker is reported
# to grow and never release, and RSS is a column here rather than an afterthought.

alias LlamaCppEx.Server

endpoint = System.get_env("SPARK_RPC_ENDPOINT") || "10.100.64.2:50052"
# The worker's address on the fabric — not the control node's ssh alias, which
# does not resolve from here.
worker_node = System.get_env("SPARK_WORKER_HOST") || "10.100.64.2"

which =
  case System.argv() do
    [w | _] -> w
    [] -> "b1"
  end

defmodule TwoNode do
  # Overridable because B2 is a different kind of run: a 142 GB model paged off
  # NVMe on one node decodes slowly enough that the default 3x65 tokens would
  # take the better part of an hour to say something a tenth of that already
  # says.
  @decode_steps String.to_integer(System.get_env("SPARK_DECODE_STEPS") || "64")
  @samples String.to_integer(System.get_env("SPARK_SAMPLES") || "3")

  def time_ms(fun) do
    t0 = System.monotonic_time(:microsecond)
    fun.()
    (System.monotonic_time(:microsecond) - t0) / 1000
  end

  def median([_ | _] = values) do
    sorted = Enum.sort(values)
    len = length(sorted)

    case rem(len, 2) do
      1 -> Enum.at(sorted, div(len, 2))
      0 -> (Enum.at(sorted, div(len, 2) - 1) + Enum.at(sorted, div(len, 2))) / 2
    end
  end

  # Same derivation as bench/spark_baseline.exs: two generations from one prompt
  # with prompt caching off differ by exactly K decode steps.
  def split(server, prompt, n_prompt, samples \\ @samples) do
    gen = fn n -> {:ok, _} = Server.generate(server, prompt, max_tokens: n) end
    gen.(4)

    t_one = median(for _ <- 1..samples, do: time_ms(fn -> gen.(1) end))
    t_many = median(for _ <- 1..samples, do: time_ms(fn -> gen.(1 + @decode_steps) end))

    per_decode = (t_many - t_one) / @decode_steps

    %{
      n_prompt: n_prompt,
      ttft_ms: t_one,
      prefill_tps: n_prompt * 1000 / (t_one - per_decode),
      decode_tps: 1000 / per_decode
    }
  end

  # RSS of the remote worker, in MiB. The leak is on the other machine, so this
  # has to cross the fabric: the client cannot see it, and the control node's
  # ssh aliases do not resolve from here. Needs `remote.sh --forward-agent`,
  # since the nodes hold no keys for each other.
  #
  # Returns nil rather than 0 when it cannot read — a missing measurement and a
  # 0 MiB worker are very different claims, and one of them is a lie.
  def worker_rss(host) do
    args = [
      "-o",
      "BatchMode=yes",
      "-o",
      "StrictHostKeyChecking=no",
      "-o",
      "UserKnownHostsFile=/dev/null",
      "-o",
      "ConnectTimeout=5",
      host,
      "pid=$(systemctl --user show llama-rpc-worker --property=MainPID --value); " <>
        "awk '/^VmRSS:/{print int($2/1024)}' /proc/$pid/status 2>/dev/null"
    ]

    # stderr stays separate: with UserKnownHostsFile=/dev/null ssh prints
    # "Warning: Permanently added ..." on every connection, and merging it in
    # made every reading unparseable — which then showed up as an honest-looking
    # "n/a" instead of a number.
    case System.cmd("ssh", args) do
      {out, 0} ->
        out
        |> String.split("\n", trim: true)
        |> List.last()
        |> to_string()
        |> Integer.parse()
        |> case do
          {mib, _} -> mib
          :error -> nil
        end

      _ ->
        nil
    end
  rescue
    _ -> nil
  end

  def rss_str(nil), do: "n/a"
  def rss_str(mib), do: "#{mib}"

  def measure(label, model_path, server_opts, prompt_tokens, worker_node) do
    IO.puts("\n--- #{label}")
    rss_before = worker_rss(worker_node)

    # Server.start_link/1 returns before the model is loaded: init/1 stays cheap
    # and handle_continue/2 does the work, so the first GenServer.call is what
    # waits for it. Load time is a headline number here — a two-node cold load
    # pushes every remote tensor across the network — so the timed region has to
    # include one call or it reads as zero.
    load_ms =
      time_ms(fn ->
        {:ok, server} = Server.start_link(server_opts ++ [model_path: model_path])
        Process.put(:server, {:ok, server})
        Process.put(:model, Bench.Helpers.await_model(server))
      end)

    case Process.get(:server) do
      {:ok, server} ->
        model = Process.get(:model)
        prompt = Bench.Helpers.prompt_of_tokens(model, prompt_tokens)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)

        result = split(server, prompt, length(tokens))
        rss_after = worker_rss(worker_node)

        GenServer.stop(server)

        row =
          Map.merge(result, %{
            label: label,
            load_ms: load_ms,
            rss_before: rss_before,
            rss_after: rss_after
          })

        report(row)
        row

      {:error, reason} ->
        IO.puts("  FAILED: #{inspect(reason)}")
        %{label: label, error: reason}
    end
  end

  def report(r) do
    IO.puts("  load          #{Float.round(r.load_ms / 1000, 1)} s")
    IO.puts("  prompt        #{r.n_prompt} tokens")
    IO.puts("  TTFT          #{Float.round(r.ttft_ms, 1)} ms")
    IO.puts("  prefill       #{Float.round(r.prefill_tps, 1)} t/s")
    IO.puts("  decode        #{Float.round(r.decode_tps, 2)} t/s")
    IO.puts("  worker RSS    #{rss_str(r.rss_before)} -> #{rss_str(r.rss_after)} MiB")
  end

  def table(rows) do
    IO.puts("\n| run | load s | prompt | TTFT ms | prefill t/s | decode t/s | worker RSS MiB |")
    IO.puts("|---|---|---|---|---|---|---|")

    for r <- rows do
      if Map.has_key?(r, :error) do
        IO.puts("| #{r.label} | FAILED: #{inspect(r.error)} | | | | | |")
      else
        IO.puts(
          "| #{r.label} | #{Float.round(r.load_ms / 1000, 1)} | #{r.n_prompt} | " <>
            "#{Float.round(r.ttft_ms, 1)} | #{Float.round(r.prefill_tps, 1)} | " <>
            "#{Float.round(r.decode_tps, 2)} | #{rss_str(r.rss_before)} -> #{rss_str(r.rss_after)} |"
        )
      end
    end
  end
end

models = %{
  "120b" =>
    Path.join(
      System.get_env("HOME"),
      "models/ggml-org/gpt-oss-120b-GGUF/main/gpt-oss-120b-MXFP4.gguf"
    ),
  "235b" =>
    Path.join(
      System.get_env("HOME"),
      "models/unsloth/Qwen3-235B-A22B-GGUF/main/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf"
    )
}

# Register up front so the remote device exists before any load. Failing here is
# the whole reason add_server reports instead of aborting: after a load, a peer
# problem takes the VM with it.
register = fn ->
  case LlamaCppEx.RPC.add_server(endpoint) do
    {:ok, n} when n >= 1 ->
      IO.puts("registered #{endpoint}: #{n} remote device(s)")
      :ok

    {:ok, 0} ->
      :ok

    {:error, reason} ->
      IO.puts(:stderr, "cannot reach #{endpoint}: #{inspect(reason)}")
      IO.puts(:stderr, "start it with: scripts/spark/rpc-worker.sh start #{worker_node}")
      System.halt(1)
  end
end

# :devices is named explicitly in every two-node run. The automatic placement
# list puts RPC devices first, which is not the order LlamaCppEx.devices/0
# reports, and a backwards split still produces correct tokens while
# benchmarking badly. Naming them makes the split mean what it says.
device_names = fn ->
  devices = LlamaCppEx.devices()
  local = Enum.find(devices, &(&1.type in [:gpu, :igpu] and &1.backend != "RPC"))
  remote = Enum.find(devices, &(&1.backend == "RPC"))
  {local.name, remote.name}
end

n_ctx = 4096
prompt_tokens = String.to_integer(System.get_env("SPARK_PROMPT_TOKENS") || "1024")

case which do
  "b1" ->
    # Controlled: the model fits one node, so anything the second node costs is
    # pure RPC overhead rather than a memory benefit.
    path = models["120b"]
    File.exists?(path) || raise "missing #{path}"

    single =
      TwoNode.measure(
        "120b single-node",
        path,
        [n_parallel: 1, n_ctx: n_ctx, cache_prompt: false],
        prompt_tokens,
        worker_node
      )

    register.()
    {local, remote} = device_names.()

    two =
      TwoNode.measure(
        "120b two-node 50/50",
        path,
        [
          n_parallel: 1,
          n_ctx: n_ctx,
          cache_prompt: false,
          devices: [local, remote],
          split_mode: :layer,
          tensor_split: [0.5, 0.5]
        ],
        prompt_tokens,
        worker_node
      )

    TwoNode.table([single, two])

  "b2" ->
    # The point. 142.1 GB of weights against 121 GiB (130.0 GB) of unified
    # memory.
    #
    # The single-node leg is OFF by default, because it does not produce a
    # number: it produces a global OOM. Measured on spark-1 —
    #
    #   oom-kill: constraint=CONSTRAINT_NONE, global_oom
    #   Out of memory: Killed process 1599 (avahi-daemon)
    #   NVRM: Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal
    #
    # The box survived, but the OOM killer took out an unrelated system service
    # (mDNS stopped resolving afterwards, from a different machine's point of
    # view) and the run died. Unified memory is why: there is no separate VRAM
    # to spill into, so "offload it all" and "keep it in RAM" compete for the
    # same 130 GB and mmap cannot save you. Set SPARK_INCLUDE_SINGLE=1 to
    # reproduce it deliberately.
    path = models["235b"]
    File.exists?(path) || raise "missing #{path} — run scripts/spark/fetch_models.exs 235b"

    single =
      if System.get_env("SPARK_INCLUDE_SINGLE") == "1" do
        [
          TwoNode.measure(
            "235b single-node (mmap overflow)",
            path,
            [n_parallel: 1, n_ctx: n_ctx, cache_prompt: false, use_mmap: true],
            prompt_tokens,
            worker_node
          )
        ]
      else
        IO.puts("\n--- 235b single-node: SKIPPED (OOMs the node; SPARK_INCLUDE_SINGLE=1 to try)")
        []
      end

    register.()
    {local, remote} = device_names.()

    two =
      TwoNode.measure(
        "235b two-node 50/50",
        path,
        [
          n_parallel: 1,
          n_ctx: n_ctx,
          cache_prompt: false,
          devices: [local, remote],
          split_mode: :layer,
          tensor_split: [0.5, 0.5]
        ],
        prompt_tokens,
        worker_node
      )

    TwoNode.table(single ++ [two])

  "b3" ->
    # Transport A/B. Run this twice: once normally, once with the worker AND this
    # process started under GGML_RDMA_DEV=nonexistent. There is no runtime
    # switch — that variable and a -DGGML_RPC_RDMA=OFF build are the only levers.
    path = models["120b"]
    register.()
    {local, remote} = device_names.()

    transport = if System.get_env("GGML_RDMA_DEV") in [nil, ""], do: "RDMA", else: "TCP (forced)"

    row =
      TwoNode.measure(
        "120b two-node over #{transport}",
        path,
        [
          n_parallel: 1,
          n_ctx: n_ctx,
          cache_prompt: false,
          devices: [local, remote],
          split_mode: :layer,
          tensor_split: [0.5, 0.5]
        ],
        prompt_tokens,
        worker_node
      )

    TwoNode.table([row])

  "b4" ->
    # Per-token RTT is fixed per graph, not per token, so a bigger batch should
    # amortise it.
    path = models["120b"]
    register.()
    {local, remote} = device_names.()

    rows =
      for n_parallel <- [1, 4, 8] do
        TwoNode.measure(
          "120b two-node n_parallel=#{n_parallel}",
          path,
          [
            n_parallel: n_parallel,
            n_ctx: n_ctx * n_parallel,
            cache_prompt: false,
            devices: [local, remote],
            split_mode: :layer,
            tensor_split: [0.5, 0.5]
          ],
          prompt_tokens,
          worker_node
        )
      end

    TwoNode.table(rows)

  other ->
    IO.puts(:stderr, "unknown bench #{inspect(other)}; expected b1, b2, b3 or b4")
    System.halt(2)
end
