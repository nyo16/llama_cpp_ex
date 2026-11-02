# End-to-end check of the CUDA build on a Spark, single node or across two.
#
#   # one node: two chat models and an embedding model resident at once
#   scripts/spark/remote.sh --env MIX_ENV=test --env LLAMA_RPC=1 spark-1 \
#     mix run scripts/spark/e2e.exs single
#
#   # two nodes: a model layer-split over the fabric next to a local one
#   scripts/spark/rpc-worker.sh start spark-2
#   scripts/spark/remote.sh --env MIX_ENV=test --env LLAMA_RPC=1 spark-1 \
#     mix run scripts/spark/e2e.exs two-node 10.100.64.2:50052
#
# Every check asserts on what the model said, not only on {:ok, _}: a model
# that loads onto the wrong device, or a split that corrupts activations,
# still returns {:ok, text}. Exits non-zero on any failure.
#
# Models default to the files on spark-1 and can be overridden:
#   SPARK_E2E_BIG    Qwen3.8-27B NVFP4 (dense, hybrid GDN, MTP head)
#   SPARK_E2E_MOE    Qwen3-30B-A3B Q4_K_M
#   SPARK_E2E_EMBED  Qwen3-Embedding-0.6B
#   SPARK_E2E_TOOLS  a model whose template supports tools

home = System.user_home!()

big =
  System.get_env("SPARK_E2E_BIG") ||
    Path.join(home, "models/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/main/Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf")

moe =
  System.get_env("SPARK_E2E_MOE") ||
    Path.join(home, "models/Qwen/Qwen3-30B-A3B-GGUF/main/Qwen3-30B-A3B-Q4_K_M.gguf")

embed =
  System.get_env("SPARK_E2E_EMBED") ||
    Path.join(home, "models/test/Qwen3-Embedding-0.6B-f16.gguf")

defmodule E2E do
  alias LlamaCppEx.ModelManager

  def start, do: Agent.start_link(fn -> [] end, name: __MODULE__)

  def step(label, fun) do
    IO.write("  " <> String.pad_trailing(label, 52))
    t0 = System.monotonic_time(:millisecond)

    outcome =
      try do
        fun.()
      rescue
        e -> {:error, Exception.message(e)}
      catch
        kind, reason -> {:error, "#{kind}: #{inspect(reason)}"}
      end

    ms = System.monotonic_time(:millisecond) - t0

    case outcome do
      {:ok, detail} ->
        IO.puts("PASS  #{detail} (#{ms} ms)")
        Agent.update(__MODULE__, &[:ok | &1])

      {:error, detail} ->
        IO.puts("FAIL  #{detail}")
        Agent.update(__MODULE__, &[:error | &1])
    end
  end

  def finish do
    results = Agent.get(__MODULE__, & &1)
    failed = Enum.count(results, &(&1 == :error))

    if failed == 0 do
      IO.puts("\nall #{length(results)} checks passed")
    else
      IO.puts("\n#{failed} of #{length(results)} checks FAILED")
      System.halt(1)
    end
  end

  def content({:ok, %{choices: [%{message: message} | _]}}), do: message.content || ""
  def content(other), do: raise("unexpected reply: #{inspect(other, limit: 5)}")

  # Factual prompts with one checkable answer each. Thinking is off so the
  # answer is in `content`, and the token budget leaves room for a sentence.
  def ask(target, question, expect, opts \\ []) do
    reply =
      target
      |> chat(question, opts)
      |> content()

    if reply =~ expect,
      do: {:ok, inspect(String.slice(String.trim(reply), 0, 40))},
      else: {:error, "expected #{inspect(expect)} in #{inspect(reply)}"}
  end

  defp chat({:manager, id}, question, opts),
    do: ModelManager.chat_completion(id, [%{role: "user", content: question}], chat_opts(opts))

  defp chat(server, question, opts),
    do: LlamaCppEx.chat_completion(server, [%{role: "user", content: question}], chat_opts(opts))

  defp chat_opts(opts),
    do: Keyword.merge([max_tokens: 64, temp: 0.0, enable_thinking: false], opts)

  def questions do
    [
      {"What is the capital of France? One word.", ~r/Paris/i},
      {"What is 12 times 12? Reply with the number only.", ~r/144/},
      {"Which planet is known as the Red Planet? One word.", ~r/Mars/i},
      {"What is the chemical symbol for gold? Reply with the symbol only.", ~r/\bAu\b/}
    ]
  end

  def concurrent(target, n) do
    qs = questions() |> Stream.cycle() |> Enum.take(n)

    results =
      qs
      |> Task.async_stream(fn {q, e} -> ask(target, q, e) end,
        max_concurrency: n,
        timeout: 300_000
      )
      |> Enum.map(fn {:ok, r} -> r end)

    case Enum.reject(results, &match?({:ok, _}, &1)) do
      [] -> {:ok, "#{n}/#{n} correct"}
      bad -> {:error, inspect(bad)}
    end
  end

  def stream_text(server, question) do
    chunks =
      server
      |> LlamaCppEx.stream_chat_completion([%{role: "user", content: question}],
        max_tokens: 64,
        temp: 0.0,
        enable_thinking: false
      )
      |> Enum.to_list()

    text =
      chunks
      |> Enum.flat_map(fn
        %{choices: [%{delta: %{content: c}} | _]} when is_binary(c) -> [c]
        _ -> []
      end)
      |> Enum.join()

    {length(chunks), text}
  end

  def cosine(a, b) do
    dot = Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum()
    norm = fn v -> :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2))) end
    dot / (norm.(a) * norm.(b))
  end

  def mem_gb do
    {out, 0} = System.cmd("free", ["-b"])
    [_, total, used | _] = out |> String.split("\n") |> Enum.at(1) |> String.split()
    "#{Float.round(String.to_integer(used) / 1.0e9, 1)} / #{Float.round(String.to_integer(total) / 1.0e9, 1)} GB used"
  end
end

{:ok, _} = E2E.start()
LlamaCppEx.set_log_level(:error)
:ok = LlamaCppEx.init()

case System.argv() do
  ["single" | _] ->
    IO.puts("\n== single node: two chat models + one embedding model resident\n")
    {:ok, _} = LlamaCppEx.ModelSupervisor.start_link(memory_budget: :infinity)
    alias LlamaCppEx.ModelManager

    E2E.step("CUDA0 is the GPU and reports unified memory", fn ->
      case Enum.find(LlamaCppEx.devices(), &(&1.backend == "CUDA")) do
        %{name: "CUDA0", type: :igpu} = d -> {:ok, "#{d.description}, #{div(d.memory_total, 1_000_000_000)} GB"}
        other -> {:error, inspect(other)}
      end
    end)

    E2E.step("load NVFP4 27B (server, n_parallel 4)", fn ->
      {:ok, _} = ModelManager.load("big", {:path, big}, n_gpu_layers: -1, n_parallel: 4, n_ctx: 16_384)
      {:ok, Path.basename(big)}
    end)

    E2E.step("load Qwen3-30B-A3B MoE (server, n_parallel 4)", fn ->
      {:ok, _} = ModelManager.load("moe", {:path, moe}, n_gpu_layers: -1, n_parallel: 4, n_ctx: 16_384)
      {:ok, Path.basename(moe)}
    end)

    E2E.step("load embedding model (direct)", fn ->
      {:ok, _} = ModelManager.load("embed", {:path, embed}, capabilities: [:embed], n_gpu_layers: -1)
      {:ok, Path.basename(embed)}
    end)

    E2E.step("three models resident", fn ->
      ids = ModelManager.list() |> Enum.map(& &1.id) |> Enum.sort()
      if ids == ["big", "embed", "moe"], do: {:ok, E2E.mem_gb()}, else: {:error, inspect(ids)}
    end)

    for {q, e} <- E2E.questions() |> Enum.take(2), id <- ["big", "moe"] do
      E2E.step("#{id}: #{String.slice(q, 0, 30)}", fn -> E2E.ask({:manager, id}, q, e) end)
    end

    E2E.step("both chat models at once, 8 requests each", fn ->
      [a, b] =
        Task.await_many(
          [Task.async(fn -> E2E.concurrent({:manager, "big"}, 8) end),
           Task.async(fn -> E2E.concurrent({:manager, "moe"}, 8) end)],
          600_000
        )

      case {a, b} do
        {{:ok, x}, {:ok, y}} -> {:ok, "big #{x}, moe #{y}"}
        _ -> {:error, inspect({a, b})}
      end
    end)

    E2E.step("embeddings rank a paraphrase above an unrelated text", fn ->
      {:ok, a} = ModelManager.embed("embed", "The cat sat on the mat.")
      {:ok, b} = ModelManager.embed("embed", "A cat was sitting on a rug.")
      {:ok, c} = ModelManager.embed("embed", "Quarterly revenue grew by eight percent.")
      {sim, far} = {E2E.cosine(a, b), E2E.cosine(a, c)}
      if sim > far, do: {:ok, "#{Float.round(sim, 3)} > #{Float.round(far, 3)}"}, else: {:error, "#{sim} <= #{far}"}
    end)

    E2E.step("unload one model, the other keeps serving", fn ->
      :ok = ModelManager.unload("moe")
      E2E.ask({:manager, "big"}, "What is the capital of France? One word.", ~r/Paris/i)
    end)

  ["two-node", endpoint | _] ->
    IO.puts("\n== two nodes over #{endpoint}: split model + local model\n")

    E2E.step("worker answers the HELLO", fn ->
      case LlamaCppEx.RPC.ping(endpoint) do
        :ok -> {:ok, inspect(Enum.map(LlamaCppEx.RPC.devices(), & &1.name))}
        other -> {:error, inspect(other)}
      end
    end)

    {:ok, split} =
      LlamaCppEx.Server.start_link(
        model_path: big,
        rpc_servers: [endpoint],
        devices: ["CUDA0", "RPC0"],
        split_mode: :layer,
        tensor_split: [0.5, 0.5],
        n_gpu_layers: -1,
        n_parallel: 4,
        n_ctx: 16_384
      )

    E2E.step("NVFP4 27B loads split across CUDA0 + RPC0", fn ->
      {:ok, _} = LlamaCppEx.Server.generate(split, "Hello", max_tokens: 1)
      {:ok, E2E.mem_gb()}
    end)

    {:ok, local} =
      LlamaCppEx.Server.start_link(
        model_path: moe,
        devices: ["CUDA0"],
        n_gpu_layers: -1,
        n_parallel: 4,
        n_ctx: 16_384
      )

    E2E.step("Qwen3-30B-A3B loads locally beside it", fn ->
      {:ok, _} = LlamaCppEx.Server.generate(local, "Hello", max_tokens: 1)
      {:ok, E2E.mem_gb()}
    end)

    for {q, e} <- E2E.questions() do
      E2E.step("split: #{String.slice(q, 0, 36)}", fn -> E2E.ask(split, q, e) end)
    end

    E2E.step("split: streaming chat completion", fn ->
      {n, text} = E2E.stream_text(split, "What is the capital of France? One word.")
      if text =~ ~r/Paris/i, do: {:ok, "#{n} chunks, #{inspect(String.trim(text))}"}, else: {:error, inspect(text)}
    end)

    E2E.step("split + local at once, 8 requests each", fn ->
      [a, b] =
        Task.await_many(
          [Task.async(fn -> E2E.concurrent(split, 8) end), Task.async(fn -> E2E.concurrent(local, 8) end)],
          600_000
        )

      case {a, b} do
        {{:ok, x}, {:ok, y}} -> {:ok, "split #{x}, local #{y}"}
        _ -> {:error, inspect({a, b})}
      end
    end)

    E2E.step("split: tool call", fn ->
      tools = [
        %{
          type: "function",
          function: %{
            name: "get_weather",
            description: "Get the current weather for a city",
            parameters: %{
              type: "object",
              properties: %{city: %{type: "string"}},
              required: ["city"]
            }
          }
        }
      ]

      {:ok, %{choices: [%{message: msg, finish_reason: reason} | _]}} =
        LlamaCppEx.chat_completion(split, [%{role: "user", content: "What's the weather in Paris right now?"}],
          tools: tools,
          max_tokens: 512,
          temp: 0.0
        )

      case msg do
        %{tool_calls: [%{function: %{name: "get_weather", arguments: args}} | _]} ->
          if args =~ ~r/Paris/i, do: {:ok, "#{reason}: #{args}"}, else: {:error, args}

        other ->
          {:error, inspect(other, limit: 8)}
      end
    end)

    GenServer.stop(local)
    GenServer.stop(split)

  _ ->
    IO.puts("usage: mix run scripts/spark/e2e.exs single | two-node <host:port>")
    System.halt(2)
end

E2E.finish()
