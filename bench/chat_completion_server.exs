Code.require_file("helpers.exs", __DIR__)

# Chat-completion routing + abandoned-stream benchmark (plan 5.3).
#
# Part 1 — multi-turn chat_completion: stateless %Model{} path (fresh context
# + full prefill per turn) vs Server-routed path (continuous batching +
# prefix cache). Reports wall time for a K-turn conversation.
#
# Part 2 — abandoned streams: aggregate throughput of complete requests while
# a fraction of streams is abandoned after a few chunks. With cancellation,
# abandoned slots free immediately instead of decoding to max_tokens.
#
# Run: LLAMA_MODEL_PATH=... mix run bench/chat_completion_server.exs

defmodule Bench.ChatCompletionServer do
  @moduledoc false

  @turns 4
  @max_tokens 24

  def run do
    model_path = System.get_env("LLAMA_MODEL_PATH") || raise "Set LLAMA_MODEL_PATH"

    IO.puts("Chat completion: stateless vs Server-routed (#{@turns} turns)")

    {:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

    stateless_ms =
      time_conversation(fn msgs, opts -> LlamaCppEx.chat_completion(model, msgs, opts) end)

    IO.puts("  stateless    : #{stateless_ms} ms")

    server = Bench.Helpers.start_server(n_parallel: 2, n_ctx: 8192)

    server_ms =
      time_conversation(fn msgs, opts -> LlamaCppEx.chat_completion(server, msgs, opts) end)

    IO.puts("  server-routed: #{server_ms} ms")
    GenServer.stop(server)

    IO.puts("")
    IO.puts("Abandoned streams: 8 requests, half abandoned after 3 chunks")

    for label <- ["with cancellation"] do
      ms = abandoned_stream_wall()
      IO.puts("  #{label}: 4 full + 4 abandoned in #{ms} ms")
    end
  end

  defp time_conversation(complete_fun) do
    base = [
      %{role: "system", content: "You are terse. Answer with at most one sentence."}
    ]

    t0 = System.monotonic_time(:millisecond)

    Enum.reduce(1..@turns, base, fn turn, msgs ->
      msgs = msgs ++ [%{role: "user", content: "Question #{turn}: name #{turn} colors."}]

      {:ok, completion} =
        complete_fun.(msgs, max_tokens: @max_tokens, temp: 0.0, timeout: 120_000)

      [%{message: %{content: content}}] = completion.choices
      msgs ++ [%{role: "assistant", content: content}]
    end)

    System.monotonic_time(:millisecond) - t0
  end

  defp abandoned_stream_wall do
    model_path = System.get_env("LLAMA_MODEL_PATH")
    server = Bench.Helpers.start_server(n_parallel: 2, n_ctx: 8192)
    _warm = LlamaCppEx.Server.generate(server, "hi", max_tokens: 4)

    t0 = System.monotonic_time(:millisecond)

    tasks =
      for i <- 1..8 do
        Task.async(fn ->
          if rem(i, 2) == 0 do
            # Abandon after 3 chunks — cancellation should free the slot.
            LlamaCppEx.Server.stream(server, "Endless story #{i}:", max_tokens: 200)
            |> Enum.take(3)
          else
            {:ok, _} =
              LlamaCppEx.Server.generate(server, "Short answer #{i}: 2+2=",
                max_tokens: 16,
                timeout: 120_000
              )
          end
        end)
      end

    Task.await_many(tasks, 300_000)
    wall = System.monotonic_time(:millisecond) - t0
    GenServer.stop(server)
    _ = model_path
    wall
  end
end

Bench.ChatCompletionServer.run()
