defmodule LlamaCppEx.Thinking do
  @moduledoc """
  Parser for `<think>...</think>` blocks in thinking model output.

  Thinking models (e.g. Qwen 3.5 with `enable_thinking: true`) wrap their
  chain-of-thought reasoning in `<think>...</think>` tags. This module provides
  both a one-shot parser for complete text and a streaming parser that handles
  token boundary splits.
  """

  @think_open "<think>"
  @think_close "</think>"
  @think_open_len byte_size(@think_open)
  @think_close_len byte_size(@think_close)

  @doc """
  Splits completed text into `{reasoning_content, content}`.

  Handles both explicit `<think>...</think>` wrapping and the common case where
  the chat template already opened the `<think>` block (so generated text starts
  directly with reasoning followed by `</think>`).

  ## Examples

      iex> LlamaCppEx.Thinking.parse("<think>I need to think</think>The answer is 42")
      {"I need to think", "The answer is 42"}

      iex> LlamaCppEx.Thinking.parse("reasoning here\\n</think>\\nThe answer is 42")
      {"reasoning here", "The answer is 42"}

      iex> LlamaCppEx.Thinking.parse("Just a response")
      {"", "Just a response"}

      iex> LlamaCppEx.Thinking.parse("<think>reasoning only</think>")
      {"reasoning only", ""}

  """
  @spec parse(String.t()) :: {String.t(), String.t()}
  def parse(text) when is_binary(text) do
    case text do
      <<@think_open, rest::binary>> ->
        split_at_close(rest)

      _ ->
        # Template may have already opened <think>, so generated text starts
        # with reasoning directly followed by </think>
        case :binary.match(text, @think_close) do
          {pos, @think_close_len} ->
            reasoning = binary_part(text, 0, pos)

            content =
              binary_part(
                text,
                pos + @think_close_len,
                byte_size(text) - pos - @think_close_len
              )

            {String.trim_trailing(reasoning), String.trim_leading(content)}

          :nomatch ->
            {"", text}
        end
    end
  end

  defp split_at_close(rest) do
    case :binary.match(rest, @think_close) do
      {pos, @think_close_len} ->
        reasoning = binary_part(rest, 0, pos)

        content =
          binary_part(rest, pos + @think_close_len, byte_size(rest) - pos - @think_close_len)

        {String.trim_trailing(reasoning), String.trim_leading(content)}

      :nomatch ->
        # Unclosed think tag — treat entire content as reasoning
        {rest, ""}
    end
  end

  @doc """
  Creates a new streaming parser state.

  Use with `feed/2` to incrementally parse streamed tokens.

  ## Options

    * `:thinking` - When `true`, assumes the template already opened a
      `<think>` block, so generated text starts in thinking mode. Defaults
      to `false`.

  """
  @spec stream_parser(keyword()) :: map()
  def stream_parser(opts \\ []) do
    if Keyword.get(opts, :thinking, false) do
      %{state: :thinking, buffer: ""}
    else
      %{state: :init, buffer: ""}
    end
  end

  @doc """
  Feeds a text chunk to the streaming parser.

  Returns `{events, new_parser}` where events are `{:thinking, text}` or
  `{:content, text}` tuples.

  The parser buffers partial `<think>` and `</think>` tags to correctly handle
  token boundary splits.

  ## Examples

      parser = LlamaCppEx.Thinking.stream_parser()
      {events, parser} = LlamaCppEx.Thinking.feed(parser, "<think>")
      # events = []  (tag consumed)
      {events, parser} = LlamaCppEx.Thinking.feed(parser, "reasoning")
      # events = [{:thinking, "reasoning"}]
      {events, _parser} = LlamaCppEx.Thinking.feed(parser, "</think>answer")
      # events = [{:content, "answer"}]

  """
  @spec feed(map(), String.t()) :: {[{:thinking | :content, String.t()}], map()}
  def feed(%{state: state, buffer: buffer} = parser, text) when is_binary(text) do
    input = buffer <> text
    do_feed(state, input, parser)
  end

  # --- :init state ---
  # Looking for <think> at the start. Buffer partial prefixes.

  defp do_feed(:init, input, parser) do
    cond do
      # Full <think> tag found at start
      String.starts_with?(input, @think_open) ->
        rest = binary_part(input, @think_open_len, byte_size(input) - @think_open_len)
        # Transition to :thinking, process rest
        do_feed(:thinking, rest, %{parser | state: :thinking, buffer: ""})

      # Input could be a prefix of <think> — keep buffering
      String.starts_with?(@think_open, input) and byte_size(input) < @think_open_len ->
        {[], %{parser | buffer: input}}

      # Not a think tag — everything is content
      true ->
        events = if input != "", do: [{:content, input}], else: []
        {events, %{parser | state: :content, buffer: ""}}
    end
  end

  # --- :thinking state ---
  # Emit {:thinking, ...} events. Watch for </think>.

  defp do_feed(:thinking, input, parser) do
    case :binary.match(input, @think_close) do
      {pos, @think_close_len} ->
        # Found closing tag
        thinking_text = binary_part(input, 0, pos)

        rest =
          binary_part(input, pos + @think_close_len, byte_size(input) - pos - @think_close_len)

        events = if thinking_text != "", do: [{:thinking, thinking_text}], else: []
        rest_events = if rest != "", do: [{:content, rest}], else: []

        {events ++ rest_events, %{parser | state: :content, buffer: ""}}

      :nomatch ->
        # Check if input ends with a partial </think> prefix
        {safe, maybe_tag} = split_trailing_tag_prefix(input, @think_close)

        events = if safe != "", do: [{:thinking, safe}], else: []
        {events, %{parser | buffer: maybe_tag}}
    end
  end

  # --- :content state ---
  # Everything is content, no more tag parsing needed.

  defp do_feed(:content, input, parser) do
    events = if input != "", do: [{:content, input}], else: []
    {events, %{parser | state: :content, buffer: ""}}
  end

  # Splits input into {safe_to_emit, possible_tag_prefix} where the suffix
  # could be the beginning of the given tag.
  defp split_trailing_tag_prefix(input, tag) do
    tag_len = byte_size(tag)
    input_len = byte_size(input)

    # Check suffixes of length 1..min(tag_len-1, input_len)
    max_check = min(tag_len - 1, input_len)

    result =
      Enum.find(max_check..1//-1, fn len ->
        suffix = binary_part(input, input_len - len, len)
        prefix = binary_part(tag, 0, len)
        suffix == prefix
      end)

    case result do
      nil ->
        {input, ""}

      len ->
        safe = binary_part(input, 0, input_len - len)
        tail = binary_part(input, input_len - len, len)
        {safe, tail}
    end
  end
end
