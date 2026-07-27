defmodule LlamaCppEx.ServerSmokeTest do
  # Integration tests for Server behaviors that need a real model. Run with:
  #
  #   GGML_METAL_NO_RESIDENCY=1 \
  #   LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf mix test --include smoke
  #
  # On Metal, GGML_METAL_NO_RESIDENCY=1 keeps a passing run from aborting with
  # exit 134 while the VM tears down; test/test_helper.exs explains why.
  #
  # async: false — each test starts its own server against the GPU.
  use ExUnit.Case, async: false

  alias LlamaCppEx.Server

  @moduletag :smoke
  @moduletag timeout: 300_000

  # No explicit teardown: the server is linked and traps exits, so it stops
  # itself (running terminate/2) when the test process exits.
  defp start_server(opts) do
    defaults = [
      model_path: LlamaCppEx.TestModels.path!(:gen),
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

  # Blocks until the server's own stats satisfy `predicate`. A GenServer.call is
  # serialized behind the server's mailbox, so a satisfied predicate is an
  # observed fact rather than a guess about how long something takes.
  defp await_stats(server, predicate, attempts \\ 600)

  defp await_stats(server, _predicate, 0) do
    flunk("server never reached the expected state: #{inspect(Server.get_stats(server))}")
  end

  defp await_stats(server, predicate, attempts) do
    stats = Server.get_stats(server)

    if predicate.(stats) do
      stats
    else
      Process.sleep(5)
      await_stats(server, predicate, attempts - 1)
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

  # The two Process.sleep/1 calls this used to open with were a race, and a fast
  # small model lost it: the "long" generation finished before the third request
  # arrived, so the queue was empty and nothing was rejected. Both waits are now
  # on observed state (`get_stats/1`), which is what they were approximating.
  test "max_queue rejects overflow immediately, queued work still completes" do
    server = start_server(n_parallel: 1, max_queue: 1)

    long =
      Task.async(fn ->
        Server.generate(server, "Write a long story:", max_tokens: 400, timeout: 240_000)
      end)

    await_stats(server, &(&1.active_slots == 1))

    queued =
      Task.async(fn -> Server.generate(server, "2+2=", max_tokens: 4, timeout: 240_000) end)

    await_stats(server, &(&1.queue_depth == 1))

    # The only slot is busy and the one queue place is taken, so the next two
    # requests must be refused rather than queued until their call timeout.
    assert {:error, :queue_full} = Server.generate(server, "3+3=", max_tokens: 4)

    assert [{:error, :queue_full}] =
             Server.stream(server, "4+4=", max_tokens: 4) |> Enum.to_list()

    assert {:ok, _} = Task.await(long, 240_000)
    assert {:ok, _} = Task.await(queued, 240_000)
  end

  test "the default queue is deep enough to absorb a burst" do
    # @default_max_queue was 0, which made the reject branch dead code and the
    # documented :queue_full error unreachable — the queue was unbounded and each
    # entry holds a whole token list.
    server = start_server(n_parallel: 1)
    assert Server.get_stats(server).queue_depth == 0

    tasks =
      for i <- 1..8 do
        Task.async(fn ->
          Server.generate(server, "#{i}+#{i}=", max_tokens: 4, timeout: 240_000)
        end)
      end

    for task <- tasks do
      assert {:ok, _} = Task.await(task, 240_000)
    end
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

  # `handle_info/2` used to clause-match only :tick, :DOWN and :EXIT, so one
  # stray message was a FunctionClauseError that killed the model, dropped the
  # %Model{}/%Context{} refs and failed every in-flight request — and because
  # ModelManager's children are restart: :temporary, it never came back.
  test "an unexpected message is ignored, not fatal" do
    server = start_server(n_parallel: 1)
    assert {:ok, _} = Server.generate(server, "2+2=", max_tokens: 4)

    ref = Process.monitor(server)

    for msg <- [:unexpected_message, {:tick, :extra}, {:some, :tuple}, "a string", 42] do
      send(server, msg)
    end

    refute_receive {:DOWN, ^ref, :process, ^server, _}, 200
    assert Process.alive?(server)

    # Still serving, and the model handle survived.
    assert {:ok, _} = Server.generate(server, "The sky is", max_tokens: 4)
    assert {:ok, %LlamaCppEx.Model{}} = Server.fetch_model(server)
  end

  test "an unexpected message during generation does not disturb it" do
    server = start_server(n_parallel: 1)

    task =
      Task.async(fn ->
        Server.generate(server, "Write a short story:", max_tokens: 64, timeout: 240_000)
      end)

    Process.sleep(50)
    send(server, :unexpected_message)

    assert {:ok, text} = Task.await(task, 240_000)
    assert byte_size(text) > 0
  end

  # C-1: a generation failure routed through a server used to come back as
  # {:ok, %ChatCompletion{finish_reason: "stop"}}.
  test "chat_completion/3 propagates a server-side generation error" do
    server = start_server(n_parallel: 1, n_ctx: 256)
    messages = [%{role: "user", content: "Write a very long story about a cat."}]

    result = LlamaCppEx.chat_completion(server, messages, max_tokens: 400)

    assert {:error, :context_full} = result
    refute match?({:ok, %LlamaCppEx.ChatCompletion{}}, result)

    # And the server is fine afterwards.
    assert {:ok, %LlamaCppEx.ChatCompletion{}} =
             LlamaCppEx.chat_completion(server, messages, max_tokens: 4)
  end

  # Security M6. `:cache_scope` is inert unless every reuse path honours it: the
  # slot's own retained KV, another slot's live KV, and the RAM prompt cache all
  # hand one request's KV to another when their prompts share a prefix.
  test "a cached prefix is never reused across cache scopes" do
    server = start_server(n_parallel: 1, prompt_cache_ram_mb: 64)
    attach_collector([:llama_cpp_ex, :server, :request, :start])

    prompt =
      "Internal memo, confidential. " <>
        String.duplicate("The quarterly figures are enclosed below. ", 20) <>
        "\nUser: Summarize.\nAssistant:"

    {:ok, _} = Server.generate(server, prompt, max_tokens: 8, cache_scope: "tenant-a")
    assert {%{prefix_cache_tokens: 0}, _} = next_telemetry()

    # Same tenant, same prompt: the cache is exactly what it is for.
    {:ok, _} = Server.generate(server, prompt, max_tokens: 8, cache_scope: "tenant-a")
    {%{prefix_cache_tokens: same_scope}, _} = next_telemetry()
    assert same_scope > 0

    # Another tenant, byte-identical prompt: no reuse. This is the leak.
    {:ok, _} = Server.generate(server, prompt, max_tokens: 8, cache_scope: "tenant-b")
    {%{prefix_cache_tokens: cross_scope}, _} = next_telemetry()
    assert cross_scope == 0

    # The default scope is a pool of its own, not a wildcard that matches any.
    {:ok, _} = Server.generate(server, prompt, max_tokens: 8)
    {%{prefix_cache_tokens: default_scope}, _} = next_telemetry()
    assert default_scope == 0

    # ...and it caches for itself like any other scope.
    {:ok, _} = Server.generate(server, prompt, max_tokens: 8)
    {%{prefix_cache_tokens: default_again}, _} = next_telemetry()
    assert default_again > 0
  end

  # W-6. `stream/3` and `stream_tokens/3` truncated silently on a per-token
  # timeout, which is indistinguishable from a completed generation and
  # contradicted their own `@doc` and the three facade pipelines. `:timeout` is 1
  # ms, so the wait expires before the first token on any model.
  #
  # W-5. The request is still generating when the consumer gives up, so the
  # cleanup has to cancel it. On the facade's server path the after-function read
  # `phase != :done` to decide, and `halt_with_error/2` sets `phase: :done` —
  # so a timed-out stream held its slot and decoded to `max_tokens` for a consumer
  # that had gone. The slot assertion is the half that catches that.
  describe "a stream whose per-token timeout expires" do
    @long_prompt "User: Write a very long story about the sea.\nAssistant:"

    test "Server.stream/3 emits {:error, :timeout} and releases the slot" do
      server = start_server(n_parallel: 1)

      elements =
        server
        |> Server.stream(@long_prompt, max_tokens: 400, timeout: 1)
        |> Enum.to_list()

      assert List.last(elements) == {:error, :timeout}
      assert_slots_released(server)
    end

    test "Server.stream_tokens/3 emits {:error, :timeout} and releases the slot" do
      server = start_server(n_parallel: 1)
      {:ok, tokens} = LlamaCppEx.Tokenizer.encode(Server.get_model(server), @long_prompt)

      elements =
        server
        |> Server.stream_tokens(tokens, max_tokens: 400, timeout: 1)
        |> Enum.to_list()

      assert List.last(elements) == {:error, :timeout}
      assert_slots_released(server)
    end

    test "the facade's server-routed stream cancels the request it abandoned" do
      server = start_server(n_parallel: 1)
      messages = [%{role: "user", content: "Write a very long story about the sea."}]

      chunks =
        server
        |> LlamaCppEx.stream_chat_completion(messages, max_tokens: 400, timeout: 1)
        |> Enum.to_list()

      assert List.last(chunks) == {:error, :timeout}
      assert_slots_released(server)
    end

    test "the server keeps serving afterwards" do
      server = start_server(n_parallel: 1)

      _ = server |> Server.stream(@long_prompt, max_tokens: 400, timeout: 1) |> Enum.to_list()
      assert_slots_released(server)

      assert {:ok, text} = Server.generate(server, "2 + 2 =", max_tokens: 4)
      assert is_binary(text)
    end
  end

  # A cancelled request frees its slot and empties the queue. Polled through the
  # server's own stats, so a pass is an observed fact rather than a sleep.
  defp assert_slots_released(server) do
    stats =
      await_stats(server, fn s ->
        s.active_slots == 0 and s.prefilling_slots == 0 and s.queue_depth == 0
      end)

    assert stats.active_slots == 0
    assert stats.queue_depth == 0
  end
end
