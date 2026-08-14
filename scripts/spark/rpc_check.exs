# Two-node bring-up verification. Proves each link in the chain before any
# benchmark is allowed to mean anything.
#
#   scripts/spark/rpc-worker.sh start spark-2 --debug
#   scripts/spark/remote.sh --env LLAMA_RPC=1 --env MIX_ENV=test spark-1 \
#     mix run scripts/spark/rpc_check.exs 10.100.64.2:50052
#   scripts/spark/rpc-worker.sh logs spark-2
#
# Set GGML_RPC_DEBUG=1 for the client side of the transport negotiation. The
# only way to know whether you got RDMA or TCP is that log: selection is silent
# auto-negotiation with no env var, no endpoint scheme and no return value.

endpoint =
  case System.argv() do
    [e | _] -> e
    [] -> raise "usage: mix run scripts/spark/rpc_check.exs <host:port>"
  end

model_path = System.get_env("LLAMA_MODEL_PATH") || raise "LLAMA_MODEL_PATH is required"

defmodule Check do
  def start, do: Agent.start_link(fn -> [] end, name: __MODULE__)

  def step(label, fun) do
    IO.write("  " <> String.pad_trailing(label, 46))

    # Statement form, not `IO.puts(...) || :ok`. `IO.puts/1` returns `:ok`, which
    # is truthy, so `||` short-circuited and BOTH branches evaluated to `:ok` --
    # `outcome` was always `:ok`, `results` never held an `:error`, and the
    # `finish/0` gate below was dead code. This script printed
    # "all N checks passed" and exited 0 even when the model failed to load
    # across two nodes, which is the one thing it exists to prevent.
    outcome =
      case fun.() do
        {:ok, detail} ->
          IO.puts("PASS  #{detail}")
          :ok

        {:error, detail} ->
          IO.puts("FAIL  #{detail}")
          :error
      end

    Agent.update(__MODULE__, &[outcome | &1])
    outcome
  end

  def finish do
    results = Agent.get(__MODULE__, & &1)

    IO.puts("")

    if :error in results do
      IO.puts("FAILED — do not benchmark until this is clean.")
      System.halt(1)
    else
      IO.puts("all #{length(results)} checks passed")
    end
  end
end

Check.start()

# --- 1. Registration ---------------------------------------------------------
#
# An unreachable endpoint and a HELLO version mismatch both collapse to a null
# registration upstream, and a null registration is silently ignored by
# ggml_backend_register. Checking the device count is the only thing standing
# between us and a model that quietly loads entirely onto the local GPU while we
# benchmark "two nodes".

before = length(LlamaCppEx.devices())

Check.step("register #{endpoint}", fn ->
  case LlamaCppEx.RPC.add_server(endpoint) do
    {:ok, n} when n >= 1 -> {:ok, "#{n} device(s) added"}
    {:ok, 0} -> {:error, "already registered in this VM"}
    {:error, reason} -> {:error, inspect(reason)}
  end
end)

devices = LlamaCppEx.devices()
remote = Enum.filter(devices, &(&1.backend == "RPC"))
local = Enum.find(devices, &(&1.type in [:gpu, :igpu] and &1.backend != "RPC"))

Check.step("device registry grew", fn ->
  if length(devices) > before,
    do: {:ok, "#{before} -> #{length(devices)} devices"},
    else: {:error, "still #{before} — ggml_backend_register no-opped"}
end)

gib = fn bytes -> Float.round(bytes / (1024 * 1024 * 1024), 1) end

Check.step("remote device reports real memory", fn ->
  case remote do
    [d | _] when d.memory_total > 0 ->
      {:ok, "#{d.name} #{gib.(d.memory_total)} GiB total, #{gib.(d.memory_free)} GiB free"}

    [d | _] -> {:error, "#{d.name} reports #{d.memory_total} bytes"}
    [] -> {:error, ~s(no device with backend "RPC")}
  end
end)

Check.step("a local accelerator is still visible", fn ->
  if local, do: {:ok, "#{local.name} (#{local.type})"}, else: {:error, "none"}
end)

IO.puts("\ndevice registry order — what LlamaCppEx.devices/0 reports:")

Enum.each(devices, fn d ->
  IO.puts(
    "  [#{d.index}] #{String.pad_trailing(d.name, 8)}#{String.pad_trailing(d.backend, 7)}" <>
      "#{String.pad_trailing(to_string(d.type), 7)}gpu_index=#{inspect(d.gpu_index)}  #{d.description}"
  )
end)

IO.puts("""

  Placement order is NOT this order. llama.cpp rebuilds the list at load time
  with RPC devices FIRST (src/llama.cpp:263-273), so tensor_split indexes a
  different list. Naming :devices below removes the ambiguity entirely.
""")

if remote == [] or local == nil do
  Check.finish()
  System.halt(0)
end

[remote_dev | _] = remote

# --- 2. A model loads and generates across the pair --------------------------

IO.puts("loading #{Path.basename(model_path)} across [#{local.name}, #{remote_dev.name}] 50/50\n")

load_started = System.monotonic_time(:millisecond)

load =
  LlamaCppEx.Model.load(model_path,
    n_gpu_layers: 99,
    devices: [local.name, remote_dev.name],
    split_mode: :layer,
    tensor_split: [0.5, 0.5]
  )

load_ms = System.monotonic_time(:millisecond) - load_started

case load do
  {:ok, model} ->
    Check.step("model loads across two nodes", fn -> {:ok, "#{load_ms} ms"} end)

    Check.step("generates across the pair", fn ->
      case LlamaCppEx.generate(model, "The capital of France is", max_tokens: 16, temp: 0.0) do
        {:ok, text} when byte_size(text) > 0 -> {:ok, inspect(String.slice(text, 0, 48))}
        {:ok, ""} -> {:error, "empty output"}
        other -> {:error, inspect(other)}
      end
    end)

    # --- 3. The decode fast path ---------------------------------------------
    #
    # Decode across an RPC device is only affordable because a repeated graph
    # collapses to a 4-byte GRAPH_RECOMPUTE. A cache miss re-serialises every
    # tensor descriptor on every token, which is the difference between free and
    # ruinous. The cache keys on the graph uid being unchanged since the last
    # graph on that device, so anything that varies the batch shape per token
    # reverts to the slow path.
    #
    # Two runs of the same shape: if the second is not materially faster per
    # token than the first, the cache is not being hit.
    warm = fn n ->
      t0 = System.monotonic_time(:millisecond)
      {:ok, _} = LlamaCppEx.generate(model, "Count from one:", max_tokens: n, temp: 0.0)
      System.monotonic_time(:millisecond) - t0
    end

    first = warm.(32)
    second = warm.(32)

    Check.step("steady-state decode", fn ->
      tps = Float.round(32_000 / second, 1)
      {:ok, "#{tps} tok/s (first pass #{first} ms, second #{second} ms)"}
    end)

    IO.puts("""

      Now check the worker journal for the transport and the graph cache:

        scripts/spark/rpc-worker.sh logs spark-2 200 | grep -E 'RDMA|GRAPH'

      Expect 'RDMA activated', and GRAPH_RECOMPUTE dominating GRAPH_COMPUTE in
      the steady state. 'RDMA activate failed, staying on TCP' means you are
      measuring the ~19 us TCP path, not the 1.39 us RDMA one.
    """)

  {:error, reason} ->
    Check.step("model loads across two nodes", fn -> {:error, inspect(reason)} end)
end

Check.finish()
