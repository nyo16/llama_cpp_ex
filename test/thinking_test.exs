defmodule LlamaCppEx.ThinkingTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Thinking

  describe "parse/1" do
    test "splits thinking and content" do
      assert {"I need to think", "The answer is 42"} =
               Thinking.parse("<think>I need to think</think>The answer is 42")
    end

    test "no thinking block returns empty reasoning" do
      assert {"", "Just a response"} = Thinking.parse("Just a response")
    end

    test "thinking only with no content" do
      assert {"reasoning only", ""} = Thinking.parse("<think>reasoning only</think>")
    end

    test "unclosed thinking tag treats all as reasoning" do
      assert {"still thinking...", ""} = Thinking.parse("<think>still thinking...")
    end

    test "empty thinking block" do
      assert {"", "content after"} = Thinking.parse("<think></think>content after")
    end

    test "trims leading whitespace from content" do
      assert {"thought", "answer"} = Thinking.parse("<think>thought</think>\nanswer")
    end

    test "empty string" do
      assert {"", ""} = Thinking.parse("")
    end

    test "multiline thinking" do
      input = "<think>line 1\nline 2\nline 3</think>final answer"
      assert {"line 1\nline 2\nline 3", "final answer"} = Thinking.parse(input)
    end
  end

  describe "stream_parser/0 + feed/2" do
    test "basic streaming with think tags" do
      parser = Thinking.stream_parser()

      {events, parser} = Thinking.feed(parser, "<think>")
      assert events == []

      {events, parser} = Thinking.feed(parser, "reasoning here")
      assert events == [{:thinking, "reasoning here"}]

      {events, _parser} = Thinking.feed(parser, "</think>the answer")
      assert events == [{:content, "the answer"}]
    end

    test "split <think> tag across tokens" do
      parser = Thinking.stream_parser()

      {events, parser} = Thinking.feed(parser, "<thi")
      assert events == []

      {events, parser} = Thinking.feed(parser, "nk>")
      assert events == []

      {events, _parser} = Thinking.feed(parser, "thought")
      assert events == [{:thinking, "thought"}]
    end

    test "split </think> tag across tokens" do
      parser = Thinking.stream_parser()

      {[], parser} = Thinking.feed(parser, "<think>")
      {events, parser} = Thinking.feed(parser, "reasoning</thi")
      assert events == [{:thinking, "reasoning"}]

      {events, _parser} = Thinking.feed(parser, "nk>content")
      assert events == [{:content, "content"}]
    end

    test "no thinking block — immediate content" do
      parser = Thinking.stream_parser()

      {events, parser} = Thinking.feed(parser, "Hello ")
      assert events == [{:content, "Hello "}]

      {events, _parser} = Thinking.feed(parser, "world")
      assert events == [{:content, "world"}]
    end

    test "content continues after think block" do
      parser = Thinking.stream_parser()

      {[], parser} = Thinking.feed(parser, "<think>")
      {_, parser} = Thinking.feed(parser, "r")
      {_, parser} = Thinking.feed(parser, "</think>")

      {events, _parser} = Thinking.feed(parser, "more content")
      assert events == [{:content, "more content"}]
    end

    test "full flow token by token" do
      tokens = ["<", "think", ">", "step 1", "</", "think", ">", "answer"]
      parser = Thinking.stream_parser()

      {all_events, _parser} =
        Enum.reduce(tokens, {[], parser}, fn token, {acc, p} ->
          {events, p} = Thinking.feed(p, token)
          {acc ++ events, p}
        end)

      thinking_text =
        all_events
        |> Enum.filter(&match?({:thinking, _}, &1))
        |> Enum.map_join(fn {:thinking, t} -> t end)

      content_text =
        all_events
        |> Enum.filter(&match?({:content, _}, &1))
        |> Enum.map_join(fn {:content, t} -> t end)

      assert thinking_text == "step 1"
      assert content_text == "answer"
    end

    test "empty input does not emit events" do
      parser = Thinking.stream_parser()
      {events, _parser} = Thinking.feed(parser, "")
      assert events == []
    end

    test "think tag with content in same chunk" do
      parser = Thinking.stream_parser()

      {events, parser} = Thinking.feed(parser, "<think>reasoning</think>answer")
      assert events == [{:thinking, "reasoning"}, {:content, "answer"}]

      assert parser.state == :content
    end
  end
end
