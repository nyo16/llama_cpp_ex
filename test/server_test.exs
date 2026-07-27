defmodule LlamaCppEx.ServerTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Server
  alias LlamaCppEx.Server.PromptCache
  alias LlamaCppEx.Server.Slots
  alias LlamaCppEx.TestSlots

  import LlamaCppEx.TestSlots, only: [idle_slot: 1, idle_slot: 2]

  # --- Pure slot/cache logic (no model files needed) ---

  describe "pick_cached_slot/2 (similarity threshold + LRU fallback)" do
    test "picks the slot with the best match when it clears the 10% threshold" do
      slots = [
        idle_slot(0, cached_tokens: [1, 2, 3, 4, 5, 6, 7, 8], t_last_used: 100),
        idle_slot(1, cached_tokens: [1, 2, 9, 9], t_last_used: 200)
      ]

      # 8/10 of the prompt matches slot 0. The returned LCP is the whole point of
      # the {seq_id, lcp} shape: the caller used to recompute it against the
      # chosen slot, ~69 µs per request at a 32k prompt.
      assert Slots.pick_cached_slot(slots, [1, 2, 3, 4, 5, 6, 7, 8, 20, 21]) == {0, 8}
    end

    test "falls back to LRU when the best match is below the threshold" do
      # Slot 1 holds a long valuable cache; the new prompt barely matches it.
      slots = [
        idle_slot(0, t_last_used: 100),
        idle_slot(1, cached_tokens: Enum.to_list(1..200), t_last_used: 200)
      ]

      prompt = [1] ++ Enum.to_list(900..950)
      # 1/52 match < 0.1 → LRU (slot 0, oldest) — protects slot 1's cache. The
      # reported LCP is 0, not the rejected 1: nothing of this prompt is cached
      # on the slot we picked.
      assert Slots.pick_cached_slot(slots, prompt) == {0, 0}
    end

    test "a match of exactly 10% does not clear the threshold" do
      slots = [
        idle_slot(0, cached_tokens: [1], t_last_used: 200),
        idle_slot(1, t_last_used: 100)
      ]

      prompt = Enum.to_list(1..10)

      # 1/10 == 0.1, and the rule is strictly greater — so LRU (slot 1) wins.
      assert Slots.pick_cached_slot(slots, prompt) == {1, 0}

      # One more matching token (2/10) clears it and slot 0 is reused.
      assert Slots.pick_cached_slot(
               [idle_slot(0, cached_tokens: [1, 2], t_last_used: 200), idle_slot(1)],
               prompt
             ) == {0, 2}
    end

    test "pick_lru_slot returns the least recently used" do
      slots = [
        idle_slot(0, t_last_used: 300),
        idle_slot(1, t_last_used: 100),
        idle_slot(2, t_last_used: 200)
      ]

      assert Slots.pick_lru_slot(slots) == 1
    end
  end

  describe "pick_cached_slot/2 and pick_lru_slot/1 boundaries" do
    test "no idle slots yields nil rather than Enum.EmptyError" do
      assert Slots.pick_cached_slot([], [1, 2, 3]) == nil
      assert Slots.pick_lru_slot([]) == nil
    end

    test "an empty prompt takes the LRU slot rather than dividing by zero" do
      slots = [
        idle_slot(0, cached_tokens: [1, 2, 3], t_last_used: 300),
        idle_slot(1, cached_tokens: [4, 5, 6], t_last_used: 100)
      ]

      # `lcp / prompt_len` was an ArithmeticError here. Nothing of an empty
      # prompt can be cached, so the match length is 0 by construction.
      assert Slots.pick_cached_slot(slots, []) == {1, 0}
    end

    test "an empty prompt with no idle slots is still nil" do
      assert Slots.pick_cached_slot([], []) == nil
    end

    test "a single idle slot is picked whether or not it matches" do
      assert Slots.pick_cached_slot([idle_slot(3, cached_tokens: [1, 2, 3])], [1, 2, 3]) == {3, 3}
      assert Slots.pick_cached_slot([idle_slot(3, cached_tokens: [9])], [1, 2, 3]) == {3, 0}
    end
  end

  describe "session_slot_if_idle/3" do
    test "returns the session's slot when idle" do
      state = %{sessions: %{"conv-a" => 1}}
      idle = [idle_slot(0), idle_slot(1)]
      assert Slots.session_slot_if_idle(state.sessions, "conv-a", idle) == 1
    end

    test "returns nil when the session's slot is busy" do
      state = %{sessions: %{"conv-a" => 1}}
      idle = [idle_slot(0)]
      assert Slots.session_slot_if_idle(state.sessions, "conv-a", idle) == nil
    end

    test "returns nil for unknown or nil sessions" do
      state = %{sessions: %{}}
      idle = [idle_slot(0)]
      assert Slots.session_slot_if_idle(state.sessions, "ghost", idle) == nil
      assert Slots.session_slot_if_idle(state.sessions, nil, idle) == nil
    end
  end

  # W-19. `:session` was a *global* keyspace: affinity routed on the session id
  # while only prefix *reuse* checked `:cache_scope`, so a guessed session id let
  # one scope claim the slot another scope was using and evict its prefix cache.
  # A DoS rather than a KV leak — the scope check still cleared the KV on mismatch
  # — but worth closing while the feature is new.
  #
  # `Slots.session_slot_if_idle/3` is agnostic about the key; the fix is that
  # `LlamaCppEx.Server` builds `{cache_scope, session}`. These pin the property
  # from the map's side, which is what the server's callers observe.
  describe "session affinity is keyed by {cache_scope, session}" do
    test "the same session id under two scopes does not share a slot" do
      sessions = %{{"tenant-a", "conv-1"} => 1}
      idle = [idle_slot(0), idle_slot(1)]

      assert Slots.session_slot_if_idle(sessions, {"tenant-a", "conv-1"}, idle) == 1

      # Tenant B guesses the id. Under a bare-session key this returned 1 and
      # evicted tenant A's cache.
      assert Slots.session_slot_if_idle(sessions, {"tenant-b", "conv-1"}, idle) == nil

      # And the default (nil) scope is a keyspace of its own, not a wildcard.
      assert Slots.session_slot_if_idle(sessions, {nil, "conv-1"}, idle) == nil
    end

    test "an unscoped session keeps its own affinity" do
      sessions = %{{nil, "conv-1"} => 1}
      idle = [idle_slot(0), idle_slot(1)]

      assert Slots.session_slot_if_idle(sessions, {nil, "conv-1"}, idle) == 1
      assert Slots.session_slot_if_idle(sessions, {"tenant-a", "conv-1"}, idle) == nil
    end
  end

  # W-15. `TestSlots` is a hand-maintained copy of `idle_slot_fields/3`, and its
  # entire value is *completeness*: a batching strategy or slot helper that reads a
  # field the fixture omits must raise `KeyError` in the test rather than quietly
  # matching a stub. All 26 fields agree today, and `TestSlots.base_fields/0` was
  # exported specifically so a drift test could be written — it had zero callers.
  # That is the `F5`/`F18` hole recurring in a new shape: a seam built for a test
  # that was never written.
  describe "TestSlots does not drift from a real slot" do
    # The three fields that deliberately live outside the per-request reset: slot
    # metadata that must survive `reset_slot/2`.
    @outside_the_reset [:sampler, :session, :t_last_used]

    test "base_fields/0 is idle_slot_fields/3 plus exactly those three" do
      real = idle_slot_field_names()
      fixture = TestSlots.base_fields() |> Map.keys() |> Enum.sort()
      expected = Enum.sort(real ++ @outside_the_reset)

      assert fixture == expected,
             """
             LlamaCppEx.TestSlots.base_fields/0 has drifted from
             LlamaCppEx.Server.idle_slot_fields/3.

             In the real slot, missing from the fixture: #{inspect(expected -- fixture)}
             In the fixture, not in a real slot:          #{inspect(fixture -- expected)}

             Add the field to @base in test/support/test_slots.exs (or remove it),
             so every strategy test keeps running against a complete slot.
             """
    end

    test "the three excluded fields really are outside idle_slot_fields/3" do
      # The complement: if one of them were folded into the reset map, the test
      # above would still pass while the fixture silently disagreed about whether
      # a reset clears it.
      for field <- @outside_the_reset do
        refute field in idle_slot_field_names(),
               "#{inspect(field)} is now part of idle_slot_fields/3 — it is reset per " <>
                 "request, so drop it from @outside_the_reset here"
      end
    end

    # `idle_slot_fields/3` is private, so its key set is read from the source. The
    # map is a literal, so the scan is exact — the same technique
    # option_forwarding_test.exs uses, for the same reason.
    defp idle_slot_field_names do
      [_, body | _] =
        "lib/llama_cpp_ex/server.ex"
        |> File.read!()
        |> String.split("defp idle_slot_fields(")

      body
      |> String.split(~r/\n  end\n/, parts: 2)
      |> hd()
      |> then(&Regex.scan(~r/^\s+([a-z_0-9]+):/m, &1))
      |> Enum.map(fn [_, key] -> String.to_existing_atom(key) end)
      |> Enum.sort()
    end
  end

  describe "donor_prefix_match/2 (only fed tokens count)" do
    test "idle donor matches against its cached tokens" do
      slot = TestSlots.slot(idle_slot(0, cached_tokens: [1, 2, 3, 4]))
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 9]) == 3
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4, 5, 6]) == 4
      assert Slots.donor_prefix_match(slot, [9, 9]) == 0
      assert Slots.donor_prefix_match(slot, []) == 0
    end

    test "an idle donor with an empty cache matches nothing" do
      slot = TestSlots.slot(idle_slot(0))
      assert Slots.donor_prefix_match(slot, [1, 2, 3]) == 0
    end

    test "prefilling donor is capped at prefill_pos" do
      slot = TestSlots.slot(TestSlots.prefilling_slot(0, [1, 2, 3, 4, 5, 6], 2))
      # Prompt matches 6 tokens, but only 2 are in the KV so far.
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4, 5, 6]) == 2
    end

    test "prefilling donor short of its cap reports the real match" do
      slot = TestSlots.slot(TestSlots.prefilling_slot(0, [1, 2, 3, 4, 5, 6], 5))
      assert Slots.donor_prefix_match(slot, [1, 2, 9]) == 2
    end

    test "a prefilling donor that has fed nothing matches nothing" do
      slot = TestSlots.slot(TestSlots.prefilling_slot(0, [1, 2, 3], 0))
      assert Slots.donor_prefix_match(slot, [1, 2, 3]) == 0
    end

    test "generating donor is capped at fed position" do
      slot = TestSlots.slot(TestSlots.generating_slot(0, [1, 2, 3], 9, generated: [4, 5]))

      # Fed history is [1, 2, 3, 4, 5]; pos caps the match at 5.
      assert slot.generated_token_ids == [5, 4]
      assert slot.pos == 5
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4, 5, 6]) == 5
    end

    test "a generating donor's pos caps a longer textual match" do
      # A donor whose bookkeeping lags its token lists must not over-report:
      # only `pos` tokens are actually in the KV cache.
      slot =
        TestSlots.slot(TestSlots.generating_slot(0, [1, 2, 3], 9, generated: [4, 5], pos: 4))

      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4, 5, 6]) == 4
    end
  end

  # `donor_prefix_match/2` for a :generating slot was rewritten to walk
  # `prompt_tokens` and then the reversed `generated_token_ids` without
  # materialising `prompt_tokens ++ Enum.reverse(generated_token_ids)` (85.4 µs
  # and ~460 KB of garbage per candidate slot per request at a 32k prompt). The
  # old expression is the specification, so it is spelled out here and the
  # rewrite is held to it — especially at the seam between the two lists, which
  # is the only place a hand-rolled two-list walk can plausibly diverge.
  describe "donor_prefix_match/2 :generating matches the pre-rewrite semantics" do
    defp reference_match(slot, tokens) do
      fed = slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids)
      min(Slots.common_prefix_length(tokens, fed), slot.pos)
    end

    defp gen_slot(prompt, generated, overrides \\ []) do
      TestSlots.slot(
        TestSlots.generating_slot(0, prompt, 99, [generated: generated] ++ overrides)
      )
    end

    test "the seam: a match ending exactly at the end of prompt_tokens" do
      slot = gen_slot([1, 2, 3], [4, 5])
      # Diverges on the first generated token — the walk must stop at 3, not
      # restart the tail comparison from the head of `tokens`.
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 99, 100]) == 3

      assert Slots.donor_prefix_match(slot, [1, 2, 3, 99, 100]) ==
               reference_match(slot, [1, 2, 3, 99, 100])
    end

    test "the seam: a match running through prompt_tokens into the generated tail" do
      slot = gen_slot([1, 2, 3], [4, 5])
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4]) == 4
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4, 5]) == 5
      assert Slots.donor_prefix_match(slot, [1, 2, 3, 4, 9]) == 4
    end

    test "tokens ending exactly at the prompt/generated seam" do
      slot = gen_slot([1, 2, 3], [4, 5])
      # `tokens` runs out precisely when `prompt_tokens` does: the walk reports
      # 3 and must not step into the tail with an empty remainder.
      assert Slots.donor_prefix_match(slot, [1, 2, 3]) == 3
      assert Slots.donor_prefix_match(slot, [1, 2, 3]) == reference_match(slot, [1, 2, 3])
    end

    test "agrees with the reference over the full case matrix" do
      cases = [
        # {prompt_tokens, generated (chronological), tokens}
        {[], [], []},
        {[], [], [1, 2]},
        {[], [4, 5], [4, 5, 6]},
        {[1, 2, 3], [], []},
        {[1, 2, 3], [], [1, 2, 3]},
        {[1, 2, 3], [], [1, 2, 3, 4]},
        {[1, 2, 3], [4], [1, 2, 3, 4]},
        {[1, 2, 3], [4, 5], [0]},
        {[1, 2, 3], [4, 5], [1]},
        {[1, 2, 3], [4, 5], [1, 2]},
        {[1, 2, 3], [4, 5], [1, 2, 3]},
        {[1, 2, 3], [4, 5], [1, 2, 3, 4]},
        {[1, 2, 3], [4, 5], [1, 2, 3, 4, 5]},
        {[1, 2, 3], [4, 5], [1, 2, 3, 4, 5, 6]},
        {[1, 2, 3], [4, 5], [1, 2, 9, 4, 5]},
        {[1, 2, 3], [4, 5], [1, 2, 3, 9, 5]},
        {[1, 1, 1], [1, 1], [1, 1, 1, 1, 1, 1]},
        {[1, 2], [2, 1], [1, 2, 2, 1, 2]}
      ]

      for {prompt, generated, tokens} <- cases,
          pos_cap <- [nil, 0, 1, length(prompt) + length(generated)] do
        overrides = if pos_cap, do: [pos: pos_cap], else: []
        slot = gen_slot(prompt, generated, overrides)

        assert Slots.donor_prefix_match(slot, tokens) == reference_match(slot, tokens),
               """
               donor_prefix_match/2 diverged from the pre-rewrite expression
                 prompt_tokens: #{inspect(prompt)}
                 generated:     #{inspect(generated)}
                 tokens:        #{inspect(tokens)}
                 pos:           #{slot.pos}
                 got:           #{Slots.donor_prefix_match(slot, tokens)}
                 expected:      #{reference_match(slot, tokens)}
               """
      end
    end
  end

  # Moving the model load into handle_continue/2 is what makes these reachable:
  # init/1 now does only what is cheap and can fail fast, so a misconfiguration
  # is reported synchronously by start_link/1 and needs no GGUF file at all. It
  # used to be an opaque MatchError raised after a multi-hundred-MB load.
  describe "start_link/1 fails fast on misconfiguration" do
    setup do
      # `{:stop, reason}` from init/1 exits the linked child with that reason, so
      # an untrapped caller would be taken down with it.
      Process.flag(:trap_exit, true)
      :ok
    end

    test "a model path that does not exist" do
      assert Server.start_link(model_path: "/nonexistent/model.gguf") ==
               {:error, {:model_not_found, "/nonexistent/model.gguf"}}
    end

    @tag :tmp_dir
    test "a model path that is a directory, not a file", %{tmp_dir: tmp_dir} do
      assert Server.start_link(model_path: tmp_dir) == {:error, {:model_not_found, tmp_dir}}
    end

    test "no :model_path at all" do
      assert Server.start_link([]) == {:error, {:missing_option, :model_path}}
    end

    test "a :model_path that is not a string" do
      assert Server.start_link(model_path: :nope) == {:error, {:invalid_model_path, :nope}}
    end

    @tag :tmp_dir
    test "a typo'd :batch_strategy module", %{tmp_dir: tmp_dir} do
      # A typo used to surface as an UndefinedFunctionError inside
      # handle_info(:tick) — after the model was already resident.
      assert Server.start_link(
               model_path: placeholder_gguf(tmp_dir),
               batch_strategy: LlamaCppEx.Server.Strategy.DecodeMaximall
             ) ==
               {:error,
                {:invalid_batch_strategy,
                 {:module_not_available, LlamaCppEx.Server.Strategy.DecodeMaximall}}}
    end

    @tag :tmp_dir
    test "a :batch_strategy that is not a module at all", %{tmp_dir: tmp_dir} do
      assert Server.start_link(
               model_path: placeholder_gguf(tmp_dir),
               batch_strategy: "DecodeMaximal"
             ) == {:error, {:invalid_batch_strategy, "DecodeMaximal"}}
    end

    @tag :tmp_dir
    test "a real module that does not implement the behaviour", %{tmp_dir: tmp_dir} do
      assert Server.start_link(model_path: placeholder_gguf(tmp_dir), batch_strategy: Enum) ==
               {:error, {:invalid_batch_strategy, {:build_batch_4_not_exported, Enum}}}
    end

    @tag :tmp_dir
    @tag capture_log: true
    test "the three shipped strategies all pass validation", %{tmp_dir: tmp_dir} do
      # The complement: validation rejects typos, not the real strategies. Each
      # gets past init/1 and fails later, on the placeholder file's contents.
      for strategy <- [
            LlamaCppEx.Server.Strategy.DecodeMaximal,
            LlamaCppEx.Server.Strategy.PrefillPriority,
            LlamaCppEx.Server.Strategy.Balanced
          ] do
        result =
          Server.start_link(model_path: placeholder_gguf(tmp_dir), batch_strategy: strategy)

        refute match?({:error, {:invalid_batch_strategy, _}}, result)

        with {:ok, pid} <- result do
          # init/1 accepted it; the load in handle_continue/2 is what fails.
          ref = Process.monitor(pid)
          assert_receive {:DOWN, ^ref, :process, ^pid, {:load_failed, _}}, 30_000
        end
      end
    end

    # File.regular?/1 is all init/1 checks, so an empty file gets past the path
    # check and on to the option checks that follow it.
    defp placeholder_gguf(tmp_dir) do
      path = Path.join(tmp_dir, "placeholder.gguf")
      File.write!(path, "")
      path
    end
  end

  # `get_model/1`'s @spec claimed a total function while the implementation
  # exited with `{:noproc, ...}`, so callers had no documented way to handle a
  # dead server. Both functions read the Registry, so neither needs one running.
  describe "fetch_model/1 and get_model/1 without a live server" do
    test "fetch_model/1 reports :noproc for anything that does not resolve" do
      dead = spawn(fn -> :ok end)
      ref = Process.monitor(dead)
      assert_receive {:DOWN, ^ref, :process, ^dead, _}

      for server <- [:no_such_server, nil, dead, {:global, :nope}] do
        assert Server.fetch_model(server) == {:error, :noproc},
               "#{inspect(server)} should report :noproc"
      end
    end

    test "get_model/1 raises ArgumentError instead of exiting" do
      error = assert_raise ArgumentError, fn -> Server.get_model(:no_such_server) end

      message = Exception.message(error)
      assert message =~ "no model available for :no_such_server"
      assert message =~ ":noproc"
      assert message =~ "fetch_model/1"
    end

    test "get_model/1 does not exit the caller" do
      # The old implementation exited with {:noproc, {GenServer, :call, ...}},
      # which a caller could only survive by catching an exit.
      task = Task.async(fn -> catch_error(Server.get_model(:no_such_server)) end)
      assert %ArgumentError{} = Task.await(task)
    end

    # The exit shapes that escaped. `fetch_model/1`'s @spec promised
    # `{:ok, _} | {:error, :noproc | :not_ready}` while its `catch` handled only
    # `:noproc`, `:normal` and `:timeout` — so `handle_continue/2`'s
    # `{:stop, {:load_failed, reason}, state}` exited the *caller*, from inside
    # both `Stream.resource` start-functions where nothing can catch it. Same
    # defect the function was written to fix, one layer down.
    #
    # A stub GenServer reproduces it exactly: `fetch_model/1` resolves a live pid,
    # finds no Registry entry, and falls through to the `:get_model` call.
    defmodule Stopping do
      @moduledoc false
      use GenServer

      def start_link(reason), do: GenServer.start_link(__MODULE__, reason)

      @impl true
      def init(reason), do: {:ok, reason}

      @impl true
      def handle_call(:get_model, _from, reason), do: {:stop, reason, reason}
    end

    @exit_reasons [
      {{:load_failed, "no such file"}, {:load_failed, "no such file"}},
      {{:shutdown, :supervisor_said_so}, :noproc},
      {:shutdown, :noproc},
      {:killed, :noproc},
      {:normal, :noproc},
      {:a_reason_nobody_catalogued, :a_reason_nobody_catalogued}
    ]

    for {reason, expected} <- @exit_reasons do
      # The stub stops abnormally on purpose; its SASL report is not a finding.
      @tag capture_log: true
      test "fetch_model/1 turns a #{inspect(reason)} exit into #{inspect(expected)}" do
        Process.flag(:trap_exit, true)
        {:ok, pid} = Stopping.start_link(unquote(Macro.escape(reason)))

        assert Server.fetch_model(pid) == {:error, unquote(Macro.escape(expected))}

        assert_receive {:EXIT, ^pid, _}
      end
    end

    @tag capture_log: true
    test "no exit reason escapes fetch_model/1 as an exit" do
      # The catch-all is the point: an uncatalogued reason must still become a
      # value, because every caller is inside a stream start-function or a `with`.
      Process.flag(:trap_exit, true)
      {:ok, pid} = Stopping.start_link({:totally, :novel, [:shape]})

      task = Task.async(fn -> Server.fetch_model(pid) end)
      assert {:error, {:totally, :novel, [:shape]}} = Task.await(task)

      assert_receive {:EXIT, ^pid, _}
    end
  end

  describe "PromptCache bookkeeping" do
    defp entry(tokens, bytes, scope \\ nil),
      do: %{tokens: tokens, len: length(tokens), bytes: bytes, bin: "", scope: scope}

    defp cache(entries, budget_bytes) do
      %PromptCache{
        entries: entries,
        bytes: Enum.sum(Enum.map(entries, & &1.bytes)),
        budget_bytes: budget_bytes
      }
    end

    test "new/1 turns megabytes into a byte budget, and 0 disables the cache" do
      assert PromptCache.new(4).budget_bytes == 4 * 1024 * 1024
      assert PromptCache.enabled?(PromptCache.new(1))

      disabled = PromptCache.new(0)
      assert disabled.budget_bytes == 0
      refute PromptCache.enabled?(disabled)
    end

    test "size/1 counts resident entries" do
      assert PromptCache.size(PromptCache.new(1)) == 0
      assert PromptCache.size(cache([entry([1], 10), entry([2], 10)], 1_000)) == 2
    end

    test "covers?/4 detects a covering entry" do
      entries = [entry([1, 2, 3, 4, 5], 100)]
      assert PromptCache.covers?(entries, [1, 2, 3], 3)
      refute PromptCache.covers?(entries, [1, 9, 3], 3)
      refute PromptCache.covers?(entries, [1, 2, 3, 4, 5, 6, 7], 7)
    end

    test "covers?/4 takes a %PromptCache{} as well as a bare entry list" do
      entries = [entry([1, 2, 3, 4, 5], 100)]
      assert PromptCache.covers?(cache(entries, 1_000), [1, 2, 3], 3)
      refute PromptCache.covers?(PromptCache.new(1), [1, 2, 3], 3)
    end

    test "covers?/4 refuses to cross a cache scope" do
      entries = [entry([1, 2, 3, 4, 5], 100, "tenant-a")]

      assert PromptCache.covers?(entries, [1, 2, 3], 3, "tenant-a")
      refute PromptCache.covers?(entries, [1, 2, 3], 3, "tenant-b")
      # The default scope is a pool of its own, not a wildcard.
      refute PromptCache.covers?(entries, [1, 2, 3], 3)
      refute PromptCache.covers?([entry([1, 2, 3], 100)], [1, 2, 3], 3, "tenant-a")
    end

    test "evict_to_budget/1 drops oldest entries first and reports them" do
      {cache, evicted} =
        PromptCache.evict_to_budget(
          cache([entry([1], 400), entry([2], 300), entry([3], 200)], 500)
        )

      assert Enum.map(evicted, & &1.tokens) == [[1]]
      assert Enum.map(cache.entries, & &1.tokens) == [[2], [3]]
      assert cache.bytes == 500
    end

    test "evict_to_budget/1 keeps dropping until it is under budget" do
      {cache, evicted} =
        PromptCache.evict_to_budget(
          cache([entry([1], 400), entry([2], 300), entry([3], 200)], 250)
        )

      assert Enum.map(evicted, & &1.tokens) == [[1], [2]]
      assert Enum.map(cache.entries, & &1.tokens) == [[3]]
      assert cache.bytes == 200
    end

    test "evict_to_budget/1 is a no-op within budget" do
      within = cache([entry([1], 100)], 100)
      assert PromptCache.evict_to_budget(within) == {within, []}
    end

    test "evict_to_budget/1 survives an inconsistent empty cache" do
      inconsistent = %PromptCache{entries: [], bytes: 999, budget_bytes: 10}
      assert {%PromptCache{entries: [], bytes: 0}, []} = PromptCache.evict_to_budget(inconsistent)
    end

    test "save/4 short-circuits before touching the NIF" do
      # Each of these returns without a context reference, which is the point:
      # `state_seq_get_size` alone is ~60 µs at n_ctx=131072 and `get_data`
      # copies the whole blob, so the free checks must rule the save out first.
      # A real reference would be needed if any of them fell through.
      slot = %{cached_pos: 64, cached_tokens: Enum.to_list(1..64), cache_scope: nil}

      assert PromptCache.save(PromptCache.new(0), :not_a_ref, 0, slot) ==
               {PromptCache.new(0), nil, []}

      tiny = %{slot | cached_pos: 31, cached_tokens: Enum.to_list(1..31)}
      empty = PromptCache.new(1)
      assert PromptCache.save(empty, :not_a_ref, 0, tiny) == {empty, nil, []}

      covered = cache([entry(Enum.to_list(1..64), 100)], 1024 * 1024)
      assert PromptCache.save(covered, :not_a_ref, 0, slot) == {covered, nil, []}
    end
  end

  # `best_candidate/4` was the private `best_ram_candidate/3` and had no test at
  # all; it decides whether a KV-sized memcpy is worth doing and whether one
  # request may read another's cached KV.
  describe "PromptCache.best_candidate/4" do
    defp entries_cache(entries), do: cache(entries, 1024 * 1024)

    test "an empty cache has no candidate" do
      assert PromptCache.best_candidate(PromptCache.new(1), [1, 2, 3], nil, :part) == nil
    end

    test "returns the entry and its usable prefix length" do
      e = entry(Enum.to_list(1..100), 500)
      tokens = Enum.to_list(1..50) ++ [999]

      assert {^e, 50} = PromptCache.best_candidate(entries_cache([e]), tokens, nil, :part)
    end

    test "caps the match one token short, so there is always a token to decode" do
      e = entry([1, 2, 3, 4], 500)

      # A full match would leave nothing to run through the model, and therefore
      # no logits for the first sampled token.
      assert {^e, 3} = PromptCache.best_candidate(entries_cache([e]), [1, 2, 3, 4], nil, :part)
    end

    test "applies the f_keep bar: a sliver of a large entry is not worth restoring" do
      e = entry(Enum.to_list(1..100), 500)

      # 24/100 < 0.25
      assert PromptCache.best_candidate(
               entries_cache([e]),
               Enum.to_list(1..24) ++ [999],
               nil,
               :part
             ) == nil

      # 25/100 == 0.25 clears it.
      assert {^e, 25} =
               PromptCache.best_candidate(
                 entries_cache([e]),
                 Enum.to_list(1..25) ++ [999],
                 nil,
                 :part
               )
    end

    test "a zero-length match is never a candidate" do
      e = entry([1, 2, 3, 4], 500)
      assert PromptCache.best_candidate(entries_cache([e]), [9, 9, 9], nil, :part) == nil
    end

    test "a :full seq_rm model can only use an exact-length entry" do
      # The unusable tail cannot be trimmed on a hybrid GDN model, so a partial
      # match would leave stale KV past the reused prefix.
      partial = entry(Enum.to_list(1..100), 500)
      tokens = Enum.to_list(1..60) ++ [999]

      assert {^partial, 60} =
               PromptCache.best_candidate(entries_cache([partial]), tokens, nil, :part)

      assert PromptCache.best_candidate(entries_cache([partial]), tokens, nil, :full) == nil

      exact = entry(Enum.to_list(1..60), 500)
      assert {^exact, 60} = PromptCache.best_candidate(entries_cache([exact]), tokens, nil, :full)
    end

    test "only entries in the request's scope are considered" do
      mine = entry(Enum.to_list(1..100), 500, "tenant-a")
      theirs = entry(Enum.to_list(1..100), 500, "tenant-b")
      tokens = Enum.to_list(1..60) ++ [999]

      assert {^mine, 60} =
               PromptCache.best_candidate(
                 entries_cache([theirs, mine]),
                 tokens,
                 "tenant-a",
                 :part
               )

      assert PromptCache.best_candidate(entries_cache([theirs]), tokens, "tenant-a", :part) == nil
      assert PromptCache.best_candidate(entries_cache([theirs]), tokens, nil, :part) == nil
    end

    test "picks the longest match among several candidates" do
      short = entry(Enum.to_list(1..40), 500)
      long = entry(Enum.to_list(1..80), 500)
      tokens = Enum.to_list(1..70) ++ [999]

      assert {^long, 70} =
               PromptCache.best_candidate(entries_cache([short, long]), tokens, nil, :part)
    end
  end
end
