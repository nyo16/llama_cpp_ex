defmodule LlamaCppEx.Server.RequestTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Server.Request

  # `Request` replaced a positional 7-tuple that was constructed at two sites,
  # destructured at five, and pattern-matched positionally inside two
  # `:queue.filter/2` callbacks — and then shipped with no direct coverage at all.
  # Its whole reason to exist is that `:from` and `:stream_pid` are easy to
  # transpose, so the invariant "exactly one of the two shapes is populated" is
  # what these tests pin.

  describe "sync/4" do
    test "populates :from and leaves the stream fields nil" do
      from = {self(), make_ref()}
      request = Request.sync([1, 2, 3], 16, from, temp: 0.0)

      assert request.tokens == [1, 2, 3]
      assert request.max_tokens == 16
      assert request.from == from
      assert request.opts == [temp: 0.0]
      assert request.stream_pid == nil
      assert request.stream_ref == nil
    end
  end

  describe "stream/5" do
    test "populates the stream fields and leaves :from nil" do
      ref = make_ref()
      request = Request.stream([4, 5], 8, self(), ref, session: "s")

      assert request.tokens == [4, 5]
      assert request.max_tokens == 8
      assert request.stream_pid == self()
      assert request.stream_ref == ref
      assert request.opts == [session: "s"]
      assert request.from == nil
    end
  end

  describe "type/1" do
    test "reports the populated shape, not the constructor that was called" do
      assert Request.type(Request.sync([1], 1, {self(), make_ref()}, [])) == :sync
      assert Request.type(Request.stream([1], 1, self(), make_ref(), [])) == :stream
    end

    # The discriminant is `stream_pid`, so a hand-built struct with neither shape
    # populated reports `:sync`. Pinned because `Server.reject_request/3` and
    # `fail_slot/3` both branch on the two fields directly and must agree with it.
    test "a struct with neither shape populated is :sync" do
      assert Request.type(%Request{tokens: [1], max_tokens: 1, opts: []}) == :sync
    end
  end

  describe "consumer_pid/1" do
    test "is the stream consumer for a stream request" do
      assert Request.consumer_pid(Request.stream([1], 1, self(), make_ref(), [])) == self()
    end

    test "is the caller's pid for a sync request" do
      ref = make_ref()
      assert Request.consumer_pid(Request.sync([1], 1, {self(), ref}, [])) == self()
    end

    # `init_slot/4` monitors this pid so a dead consumer frees its slot instead of
    # generating to max_tokens. `nil` means "nothing to monitor", and
    # `Process.monitor(nil)` would raise — the guard is what keeps that off the
    # request path.
    test "is nil when there is nobody to monitor" do
      assert Request.consumer_pid(%Request{tokens: [1], max_tokens: 1, opts: []}) == nil
    end

    test "prefers the stream consumer when both are somehow set" do
      other = spawn(fn -> :ok end)

      request = %Request{
        tokens: [1],
        max_tokens: 1,
        opts: [],
        from: {other, :tag},
        stream_pid: self()
      }

      assert Request.consumer_pid(request) == self()
    end
  end

  describe "the struct's enforced keys" do
    test "a request without tokens, max_tokens or opts cannot be built" do
      for missing <- [:tokens, :max_tokens, :opts] do
        fields = %{tokens: [1], max_tokens: 1, opts: []} |> Map.delete(missing)

        assert_raise ArgumentError, ~r/#{missing}/, fn -> struct!(Request, fields) end
      end
    end
  end
end
