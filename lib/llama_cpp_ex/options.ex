defmodule LlamaCppEx.Options do
  @moduledoc """
  Single owner for option policy shared across the public entry points.

  Two things live here, both of which used to be duplicated per call site:

  ## Scalar defaults

  `Context.tuning_option_keys/0` and `LlamaCppEx.Sampler.option_keys/0` gave the
  option *lists* an owner, but the scalar defaults kept their own copies. There
  were six hand-rolled `Keyword.get(opts, :timeout, ...)` calls across three
  modules, split 60s/30s, and `LlamaCppEx.stream_chat_completion/3` picked
  between them purely on the type of its first argument. The split itself is
  intentional — see `blocking_timeout/0` and `stream_timeout/0` — but it needs
  to be stated once.

  ## Unknown-key rejection

  `Keyword.take/2` *is* the routing mechanism in this library: each consumer
  takes the keys it owns and ignores the rest. That makes an unknown key
  structurally indistinguishable from another module's key, so a typo is silently
  dropped — `generate(model, prompt, temperature: 0.1)` runs at the default
  temperature and `n_paralell: 8` runs at 4. Routing therefore cannot validate;
  only the public entry point, which knows the complete key set, can.

  This module deliberately has no dependencies on `LlamaCppEx`, `LlamaCppEx.Server`
  or `LlamaCppEx.ModelManager`, so any of them can use it without adding a cycle.
  """

  # A blocking call must complete the *whole* generation inside this budget:
  # `LlamaCppEx.generate/3`, `chat_completion/3`, `Server.generate/3`,
  # `Server.generate_tokens/3`, `Server.complete_tokens/3`, `MTP.stream/3`.
  @blocking_timeout 60_000

  # A streaming call only waits this long for the *next* chunk, so the budget is
  # per-token rather than per-request and is deliberately tighter:
  # `LlamaCppEx.stream/3`, `stream_chat_completion/3`, `Server.stream/3`,
  # `Server.stream_tokens/3`.
  @stream_timeout 30_000

  @doc "Default `:timeout` for a call that blocks until generation completes."
  @spec blocking_timeout() :: pos_integer()
  def blocking_timeout, do: @blocking_timeout

  @doc "Default `:timeout` for a streaming call, bounding the wait per chunk."
  @spec stream_timeout() :: pos_integer()
  def stream_timeout, do: @stream_timeout

  @doc """
  Reads `:timeout` from `opts`, defaulting by call shape.

  `mode` is `:blocking` or `:stream`. `:infinity` is accepted.
  """
  @spec timeout(keyword(), :blocking | :stream) :: timeout()
  def timeout(opts, :blocking), do: Keyword.get(opts, :timeout, @blocking_timeout)
  def timeout(opts, :stream), do: Keyword.get(opts, :timeout, @stream_timeout)

  @doc """
  Raises `ArgumentError` unless every key in `opts` appears in `known`.

  `label` names the entry point in the error message. Returns `opts` unchanged so
  it can sit in a pipeline. A near-miss key gets a "did you mean" hint, since the
  motivating failures are typos (`temperature` for `temp`, `n_paralell` for
  `n_parallel`) rather than wholly invented options.
  """
  @spec validate!(keyword(), [atom()], String.t()) :: keyword()
  def validate!(opts, known, label) when is_list(known) and is_binary(label) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "#{label}: options must be a keyword list, got: #{inspect(opts)}"
    end

    case Enum.uniq(Keyword.keys(opts)) -- known do
      [] ->
        opts

      unknown ->
        raise ArgumentError, unknown_option_message(unknown, known, label)
    end
  end

  defp unknown_option_message(unknown, known, label) do
    details =
      Enum.map_join(unknown, "\n", fn key ->
        case suggest(key, known) do
          nil -> "  * #{inspect(key)}"
          hint -> "  * #{inspect(key)} (did you mean #{inspect(hint)}?)"
        end
      end)

    """
    #{label}: unknown option#{if length(unknown) == 1, do: "", else: "s"}:

    #{details}

    Known options: #{known |> Enum.sort() |> Enum.map_join(", ", &inspect/1)}
    """
  end

  # Tuned against the 58-key union of every declared option set in the library.
  # 0.78 rather than 0.8 because `String.jaro_distance("temperature", "temp")` is
  # 0.7879 — the single most likely typo in this API would otherwise get no hint.
  # The nearest wrong candidate for "temperature" is `:template` at 0.7657, so it
  # still resolves to `:temp`. Dropping from 0.8 to 0.78 adds exactly one
  # mutually-confusable pair among known keys (13 -> 14), and words that are not
  # options at all stay unmatched ("elephant" tops out at 0.7083).
  @suggest_threshold 0.78

  defp suggest(key, known) do
    key_str = Atom.to_string(key)

    known
    |> Enum.map(&{&1, String.jaro_distance(key_str, Atom.to_string(&1))})
    |> Enum.filter(fn {_, score} -> score >= @suggest_threshold end)
    |> Enum.max_by(fn {_, score} -> score end, fn -> nil end)
    |> case do
      nil -> nil
      {candidate, _} -> candidate
    end
  end
end
