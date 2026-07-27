defmodule LlamaCppEx.Generator do
  @moduledoc false
  # Lifecycle protocol for the streaming NIF generators.
  #
  # The streaming NIFs (`generate_tokens/7`, `generate_mtp_tokens/9`) run a whole
  # generation loop inside one dirty-scheduler call, sending messages tagged with
  # a caller-supplied ref. Driving one correctly is a six-step dance:
  #
  #   1. allocate a cancel flag resource
  #   2. spawn_link a process that enters the NIF
  #   3. on teardown, set the cancel flag
  #   4. unlink, so killing the runner does not take the consumer with it
  #   5. kill the runner
  #   6. drain the mailbox of any messages already in flight
  #
  # Step 3 is the one that is easy to get wrong and impossible to notice: killing
  # the process alone does NOT interrupt a running NIF, so without the flag the
  # dirty scheduler keeps decoding to `max_tokens` for a consumer that has gone
  # away. Step 4 is the other one: without the unlink, the kill in step 5
  # propagates to the caller.
  #
  # This was implemented three times (twice in `LlamaCppEx`, once in
  # `LlamaCppEx.MTP`) — exactly the shape where a cancel or leak bug hides in the
  # copy nobody looked at.

  @doc """
  Starts a generator and returns its handle.

  `fun` receives `{caller_pid, ref, cancel_ref}` and must enter the streaming NIF.
  It runs in a linked process.
  """
  @spec start(({pid(), reference(), reference()} -> any())) ::
          {:ok, %{ref: reference(), pid: pid(), cancel: reference()}}
  def start(fun) when is_function(fun, 1) do
    ref = make_ref()
    parent = self()
    cancel = LlamaCppEx.NIF.cancel_flag_new()
    pid = spawn_link(fn -> fun.({parent, ref, cancel}) end)
    {:ok, %{ref: ref, pid: pid, cancel: cancel}}
  end

  @doc """
  Cancels a generator and drains its remaining messages.

  Idempotent, and safe to call on an already-finished generator. Always call it
  from a `Stream.resource/3` after-function — a consumer that stops early is the
  case this exists for.
  """
  @spec stop(%{ref: reference(), pid: pid(), cancel: reference()}) :: :ok
  def stop(%{ref: ref, pid: pid, cancel: cancel}) do
    LlamaCppEx.NIF.request_cancel(cancel)

    # `Process.unlink/1` stops *future* exit signals but does not remove one
    # already in the mailbox, and the runner exits `:normal` the instant the NIF
    # returns — which is the common case, since a stream is usually torn down
    # right after its last token. So a trapping consumer was left holding a stray
    # `{:EXIT, pid, :normal}`: noise for a `handle_info` catch-all, and a *silent
    # shutdown* for a consumer using the `{:stop, reason, state}` shape, which is
    # what `LlamaCppEx.Server`'s own `{:EXIT, _, reason}` clause does.
    #
    # Draining after the unlink is exact rather than best-effort: once
    # `Process.unlink/1` has returned, no further exit signal from `pid` can be
    # delivered, so anything matching is already queued and `after 0` finds it.
    # The kill therefore has to come *after* the unlink, which it does.
    Process.unlink(pid)
    Process.exit(pid, :kill)
    drain(ref, pid)
  end

  @doc "Discards any messages still queued for `ref`."
  @spec drain(reference()) :: :ok
  def drain(ref), do: drain(ref, nil)

  @doc """
  Discards messages queued for `ref`, and the runner's exit signal if `pid` is
  given and the caller traps exits.
  """
  @spec drain(reference(), pid() | nil) :: :ok
  def drain(ref, pid) do
    receive do
      {^ref, _} -> drain(ref, pid)
      {:EXIT, ^pid, _reason} when is_pid(pid) -> drain(ref, pid)
    after
      0 -> :ok
    end
  end
end
