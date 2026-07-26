defmodule LlamaCppEx.ServerSmokeTest do
  # Integration tests for Server behaviors that need a real model. Run with:
  #
  #   LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf mix test --include smoke
  #
  # async: false — each test starts its own server against the GPU.
  use ExUnit.Case, async: false

  @moduletag :smoke
  @moduletag timeout: 300_000

  @gen_model System.get_env("LLAMA_SMOKE_GEN_MODEL")

  if @gen_model && File.exists?(@gen_model) do
    alias LlamaCppEx.Server

    # No explicit teardown: the server is linked and traps exits, so it stops
    # itself (running terminate/2) when the test process exits.
    defp start_server(opts) do
      defaults = [
        model_path: @gen_model,
        n_gpu_layers: -1,
        n_parallel: 2,
        n_ctx: 2048,
        temp: 0.0
      ]

      {:ok, server} = Server.start_link(Keyword.merge(defaults, opts))
      server
    end

    defp attach_collector(event) do
      parent = self()
      handler_id = {__MODULE__, self(), event}

      :telemetry.attach(
        handler_id,
        event,
        fn _event, measurements, metadata, _ ->
          send(parent, {:telemetry, measurements, metadata})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach(handler_id) end)
    end

    defp next_telemetry(timeout \\ 60_000) do
      receive do
        {:telemetry, m, meta} -> {m, meta}
      after
        timeout -> flunk("expected telemetry event")
      end
    end

    test "empty token list errors immediately on both call types" do
      server = start_server([])

      assert {:error, :empty_prompt} = Server.generate_tokens(server, [])
      assert [{:error, :empty_prompt}] = Server.stream_tokens(server, []) |> Enum.to_list()
    end

    test "per-request cache_prompt overrides: hit, no-reuse, no-retention" do
      server = start_server(n_parallel: 1)
      attach_collector([:llama_cpp_ex, :server, :request, :start])

      prompt = "User: Name three colors.\nAssistant:"
      {:ok, reply} = Server.generate(server, prompt, max_tokens: 12)
      {%{prefix_cache_tokens: 0}, _} = next_telemetry()

      # Exact-prefix continuation hits the cache (works on :full models too).
      turn2 = prompt <> reply <> "\nUser: One more.\nAssistant:"
      {:ok, reply2} = Server.generate(server, turn2, max_tokens: 12)
      {%{prefix_cache_tokens: hit}, _} = next_telemetry()
      assert hit > 0

      # cache_prompt: false → no reuse now, no retention after.
      turn3 = turn2 <> reply2 <> "\nUser: Again.\nAssistant:"
      {:ok, _} = Server.generate(server, turn3, max_tokens: 8, cache_prompt: false)
      {%{prefix_cache_tokens: 0}, _} = next_telemetry()

      {:ok, _} = Server.generate(server, turn3, max_tokens: 4)
      {%{prefix_cache_tokens: 0}, _} = next_telemetry()
    end

    test "session affinity keeps interleaved conversations on their slots" do
      server = start_server([])
      attach_collector([:llama_cpp_ex, :server, :request, :start])

      # The two prompts must share no leading tokens. pick_cached_slot/2 reuses a
      # slot when longest-common-prefix / prompt_len > 0.1, so a shared opener
      # (both prompts used to start with "Chat ") is enough to route b1 onto a1's
      # slot on some tokenizers — this test then failed on Llama-3.2 while
      # passing on Qwen3.5. Keep the first tokens distinct.
      a1 = "Weather log. User: Name three colors.\nAssistant:"
      b1 = "Zoology notes, unrelated. User: Name three animals.\nAssistant:"

      {:ok, ra} = Server.generate(server, a1, max_tokens: 12, session: :a)
      {_, %{seq_id: slot_a}} = next_telemetry()
      {:ok, rb} = Server.generate(server, b1, max_tokens: 12, session: :b)
      {_, %{seq_id: slot_b}} = next_telemetry()
      assert slot_a != slot_b

      {:ok, _} =
        Server.generate(server, a1 <> ra <> "\nUser: More.\nAssistant:",
          max_tokens: 8,
          session: :a
        )

      {%{prefix_cache_tokens: hit_a}, %{seq_id: slot_a2}} = next_telemetry()

      {:ok, _} =
        Server.generate(server, b1 <> rb <> "\nUser: More.\nAssistant:",
          max_tokens: 8,
          session: :b
        )

      {%{prefix_cache_tokens: hit_b}, %{seq_id: slot_b2}} = next_telemetry()

      assert slot_a2 == slot_a
      assert slot_b2 == slot_b
      assert hit_a > 0
      assert hit_b > 0
    end

    test "max_queue rejects overflow immediately, queued work still completes" do
      server = start_server(n_parallel: 1, max_queue: 1)

      long =
        Task.async(fn ->
          Server.generate(server, "Write a long story:", max_tokens: 200, timeout: 240_000)
        end)

      Process.sleep(300)

      queued =
        Task.async(fn -> Server.generate(server, "2+2=", max_tokens: 4, timeout: 240_000) end)

      Process.sleep(200)

      assert {:error, :queue_full} = Server.generate(server, "3+3=", max_tokens: 4)

      assert [{:error, :queue_full}] =
               Server.stream(server, "4+4=", max_tokens: 4) |> Enum.to_list()

      assert {:ok, _} = Task.await(long, 240_000)
      assert {:ok, _} = Task.await(queued, 240_000)
    end

    test "halting a stream early cancels generation and frees the slot" do
      server = start_server(n_parallel: 1)
      attach_collector([:llama_cpp_ex, :server, :request, :done])

      _ = Server.stream(server, "Write an endless story:", max_tokens: 400) |> Enum.take(3)

      {%{generated_tokens: n}, %{stop_reason: :cancelled}} = next_telemetry()
      assert n < 50

      # Slot is immediately usable.
      assert {:ok, _} = Server.generate(server, "2+2=", max_tokens: 4)
    end

    test "a request exceeding its context budget fails alone" do
      server = start_server(n_parallel: 2, n_ctx: 256)

      long =
        Task.async(fn ->
          Server.generate(server, "Write a story:", max_tokens: 400, timeout: 240_000)
        end)

      short =
        Task.async(fn ->
          Process.sleep(200)
          Server.generate(server, "2+2=", max_tokens: 4, timeout: 240_000)
        end)

      assert {:error, :context_full} = Task.await(long, 240_000)
      assert {:ok, _} = Task.await(short, 240_000)

      # Server is still healthy afterwards.
      assert {:ok, _} = Server.generate(server, "The sky is", max_tokens: 4)
    end
  else
    @tag :skip
    test "server smoke tests skipped — set LLAMA_SMOKE_GEN_MODEL" do
      :ok
    end
  end
end
