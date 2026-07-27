defmodule LlamaCppEx.Server.Request do
  @moduledoc false
  # One generation request, from arrival through queueing into a slot.
  #
  # This replaced a positional 7-tuple `{type, tokens, max_tokens, from,
  # stream_pid, stream_ref, opts}` that was constructed at two sites, destructured
  # at five, and pattern-matched positionally inside two `:queue.filter/2`
  # callbacks — where a field's meaning was carried entirely by its index and
  # `{_type, _tokens, _max, _from, _pid, req_ref, _opts}` was the only
  # documentation. Reordering or adding a field meant finding every one of those
  # by hand.
  #
  # `:from` is set for synchronous requests and `nil` for streams; `:stream_pid`
  # and `:stream_ref` are the reverse. Exactly one of the two shapes is populated,
  # which `type/1` reports.

  @enforce_keys [:tokens, :max_tokens, :opts]
  defstruct [
    :tokens,
    :max_tokens,
    :opts,
    from: nil,
    stream_pid: nil,
    stream_ref: nil
  ]

  @type t :: %__MODULE__{
          tokens: [integer()],
          max_tokens: non_neg_integer(),
          opts: keyword(),
          from: GenServer.from() | nil,
          stream_pid: pid() | nil,
          stream_ref: reference() | nil
        }

  @doc "A synchronous request: the caller waits for a reply."
  @spec sync([integer()], non_neg_integer(), GenServer.from(), keyword()) :: t()
  def sync(tokens, max_tokens, from, opts) do
    %__MODULE__{tokens: tokens, max_tokens: max_tokens, from: from, opts: opts}
  end

  @doc "A streaming request: tokens are messaged to `pid` tagged with `ref`."
  @spec stream([integer()], non_neg_integer(), pid(), reference(), keyword()) :: t()
  def stream(tokens, max_tokens, pid, ref, opts) do
    %__MODULE__{
      tokens: tokens,
      max_tokens: max_tokens,
      stream_pid: pid,
      stream_ref: ref,
      opts: opts
    }
  end

  @doc "`:stream` or `:sync`."
  @spec type(t()) :: :stream | :sync
  def type(%__MODULE__{stream_pid: pid}) when is_pid(pid), do: :stream
  def type(%__MODULE__{}), do: :sync

  @doc "The pid whose death should cancel this request, or `nil`."
  @spec consumer_pid(t()) :: pid() | nil
  def consumer_pid(%__MODULE__{stream_pid: pid}) when is_pid(pid), do: pid
  def consumer_pid(%__MODULE__{from: {pid, _tag}}) when is_pid(pid), do: pid
  def consumer_pid(%__MODULE__{}), do: nil
end
