defmodule LlamaCppEx.OptionsTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Options

  # `Options` exists because there were six hand-rolled
  # `Keyword.get(opts, :timeout, ...)` calls across three modules, split 60s/30s,
  # and `stream_chat_completion/3` picked its default from the *type of its first
  # argument*. Giving the defaults an owner is only half the fix; nothing asserted
  # the contract, and the `:stream` clause was likely never executed in the default
  # suite. `option_forwarding_test.exs` covers `validate!/3`; this covers the
  # timeouts.

  describe "the default timeouts" do
    test "a blocking call gets the whole-generation budget" do
      assert Options.blocking_timeout() == 60_000
    end

    test "a streaming call gets the tighter per-chunk budget" do
      assert Options.stream_timeout() == 30_000
    end

    # The distinction is the whole point: a stream's timeout bounds the wait for
    # the *next* chunk, so it must be strictly tighter than a budget that has to
    # cover a whole generation. Two equal values would mean one of the two call
    # styles is mis-documented.
    test "the streaming budget is tighter than the blocking one" do
      assert Options.stream_timeout() < Options.blocking_timeout()
    end

    test "both are positive integers, so they can be passed to receive/after" do
      for value <- [Options.blocking_timeout(), Options.stream_timeout()] do
        assert is_integer(value) and value > 0
      end
    end
  end

  describe "timeout/2" do
    test "falls back to the mode's default when :timeout is absent" do
      assert Options.timeout([], :blocking) == Options.blocking_timeout()
      assert Options.timeout([], :stream) == Options.stream_timeout()
      assert Options.timeout([temp: 0.0], :blocking) == Options.blocking_timeout()
      assert Options.timeout([temp: 0.0], :stream) == Options.stream_timeout()
    end

    test "an explicit :timeout wins in both modes" do
      assert Options.timeout([timeout: 1], :blocking) == 1
      assert Options.timeout([timeout: 1], :stream) == 1
    end

    test "passes :infinity through rather than treating it as absent" do
      # `Keyword.get/3` would return the default for a *missing* key, not for a
      # present one whose value happens to be an atom.
      assert Options.timeout([timeout: :infinity], :blocking) == :infinity
      assert Options.timeout([timeout: :infinity], :stream) == :infinity
    end

    test "an explicit nil is a value, not an absence" do
      # Documenting the actual behaviour: `Keyword.get/3` only defaults on a
      # missing key, so `timeout: nil` produces nil and a `receive ... after nil`
      # would raise. Callers must not spell "use the default" as nil.
      assert Options.timeout([timeout: nil], :blocking) == nil
    end

    test "the first :timeout wins, matching Keyword.get/3" do
      assert Options.timeout([timeout: 5, timeout: 9], :stream) == 5
    end
  end
end
