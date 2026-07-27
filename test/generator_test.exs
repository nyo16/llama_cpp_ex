defmodule LlamaCppEx.GeneratorTest do
  # async: false — the stray-exit property is about *this* process's mailbox, and
  # the runner processes are spawn_linked to it.
  use ExUnit.Case, async: false

  alias LlamaCppEx.Generator

  # `Generator` is the consolidation of three hand-written copies of the
  # streaming-NIF lifecycle dance and had zero direct coverage, which is not a
  # coincidence: the stray-`{:EXIT, _, :normal}` bug lived here and none of the
  # three copies' tests could have caught it, because they all drove the
  # generator through a `Stream.resource` whose consumer did not trap exits.
  #
  # Nothing here needs a model. `cancel_flag_new/0` is a NIF but takes no
  # arguments and allocates a bare resource, and a runner function that returns
  # immediately exercises the exact race that matters: the runner exits `:normal`
  # before `stop/1` gets to it.

  defp noop_runner, do: fn {_parent, _ref, _cancel} -> :ok end

  describe "start/1" do
    test "returns a ref, a live-or-finished runner pid, and a cancel flag" do
      {:ok, gen} = Generator.start(noop_runner())

      assert is_reference(gen.ref)
      assert is_pid(gen.pid)
      assert is_reference(gen.cancel)

      Generator.stop(gen)
    end

    test "the runner receives the parent pid, ref and cancel flag" do
      parent = self()

      {:ok, gen} =
        Generator.start(fn {p, ref, cancel} -> send(parent, {:handed, p, ref, cancel}) end)

      assert_receive {:handed, ^parent, ref, cancel}
      assert ref == gen.ref
      assert cancel == gen.cancel

      Generator.stop(gen)
    end
  end

  describe "stop/1" do
    test "is idempotent" do
      {:ok, gen} = Generator.start(noop_runner())

      assert Generator.stop(gen) == :ok
      assert Generator.stop(gen) == :ok
      assert Generator.stop(gen) == :ok
    end

    test "drains messages already queued for the ref" do
      parent = self()

      {:ok, gen} =
        Generator.start(fn {p, ref, _cancel} ->
          send(p, {ref, {:token, 1, "a"}})
          send(p, {ref, {:token, 2, "b"}})
          send(p, {ref, :done})
          send(p, {:sentinel, self()})
        end)

      # The sentinel is not tagged with the ref, so it proves the drain stopped at
      # the generator's own messages instead of emptying the mailbox.
      assert_receive {:sentinel, _}

      assert Generator.stop(gen) == :ok
      refute_received {_ref, _anything}
    end

    test "leaves messages for other refs alone" do
      {:ok, a} = Generator.start(noop_runner())
      {:ok, b} = Generator.start(noop_runner())

      send(self(), {b.ref, {:token, 1, "keep me"}})

      Generator.stop(a)

      assert_received {ref, {:token, 1, "keep me"}}
      assert ref == b.ref

      Generator.stop(b)
    end

    # The bug: `Process.unlink/1` does not remove an exit signal that is already in
    # the mailbox, and the runner exits `:normal` the instant the NIF returns —
    # which for a stream is normally just before the after-function runs. A
    # trapping consumer got a stray `{:EXIT, pid, :normal}`; one using the
    # `{:stop, reason, state}` shape (`LlamaCppEx.Server` does exactly that in its
    # own `{:EXIT, _, reason}` clause) shut down *silently*. Reproduced 200/200
    # before the fix.
    test "leaves no stray exit signal for a trapping consumer" do
      strays =
        Enum.count(1..200, fn _ ->
          parent = self()

          spawn(fn ->
            Process.flag(:trap_exit, true)
            {:ok, gen} = Generator.start(fn {_p, _r, _c} -> :ok end)

            # Let the runner finish, so its :normal exit signal is queued before
            # stop/1 unlinks. This is the common case, not a rare race.
            Process.sleep(5)
            Generator.stop(gen)

            stray =
              receive do
                {:EXIT, _pid, _reason} = msg -> msg
              after
                0 -> nil
              end

            send(parent, {:stray, stray})
          end)

          assert_receive {:stray, stray}, 5_000
          stray != nil
        end)

      assert strays == 0, "#{strays}/200 consumers were left a stray exit signal"
    end

    test "still cancels a runner that is genuinely still going" do
      parent = self()

      {:ok, gen} =
        Generator.start(fn {p, ref, _cancel} ->
          send(p, {ref, :started})
          Process.sleep(:infinity)
        end)

      assert_receive {ref, :started}
      assert ref == gen.ref
      runner = gen.pid

      down = Process.monitor(runner)
      assert Generator.stop(gen) == :ok
      assert_receive {:DOWN, ^down, :process, ^runner, :killed}

      # And killing it did not take this process with it.
      assert Process.alive?(parent)
    end
  end

  describe "drain/1" do
    test "discards only the given ref's messages and always returns :ok" do
      ref = make_ref()
      other = make_ref()

      send(self(), {ref, :a})
      send(self(), {other, :b})
      send(self(), {ref, :c})

      assert Generator.drain(ref) == :ok

      assert_received {^other, :b}
      refute_received {^ref, _}
    end

    test "is a no-op on an empty mailbox" do
      assert Generator.drain(make_ref()) == :ok
    end
  end
end
