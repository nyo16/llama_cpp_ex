defmodule LlamaCppEx.UTF8Stream do
  @moduledoc false
  # Pending-bytes buffering for streamed token pieces. A tokenizer piece can
  # end mid-codepoint (e.g. one token per byte of a multibyte emoji), so
  # emitting pieces verbatim hands consumers invalid UTF-8. `push/2` returns
  # the longest emittable prefix and holds back a trailing incomplete
  # codepoint until the next piece completes it. Invalid bytes that cannot
  # become valid by appending more data are passed through unchanged — the
  # model produced them, hiding them would corrupt offsets.

  @doc """
  Appends `chunk` to the held-back bytes and splits off what can be emitted.

  Returns `{emit, pending}` — `emit` never ends in a partial codepoint.
  """
  @spec push(binary(), binary()) :: {binary(), binary()}
  def push(pending, chunk) when is_binary(pending) and is_binary(chunk) do
    data = pending <> chunk
    size = byte_size(data)
    holdback = incomplete_suffix_len(data, size)
    {binary_part(data, 0, size - holdback), binary_part(data, size - holdback, holdback)}
  end

  # Number of trailing bytes forming an incomplete-but-completable UTF-8
  # codepoint (0..3). Scans back at most 3 bytes: a lead byte further back
  # than that heads a sequence that is already complete or already invalid.
  defp incomplete_suffix_len(data, size) do
    Enum.find_value(1..min(size, 3)//1, 0, fn back ->
      byte = :binary.at(data, size - back)

      cond do
        byte < 0x80 -> 0
        byte < 0xC0 -> nil
        byte >= 0xF8 -> 0
        back < expected_len(byte) -> back
        true -> 0
      end
    end)
  end

  defp expected_len(byte) when byte >= 0xF0, do: 4
  defp expected_len(byte) when byte >= 0xE0, do: 3
  defp expected_len(_byte), do: 2
end
