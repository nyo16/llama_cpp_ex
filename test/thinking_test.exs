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

    test "implicit thinking (template already opened <think>)" do
      assert {"step by step reasoning", "The answer is 42"} =
               Thinking.parse("step by step reasoning\n</think>\nThe answer is 42")
    end

    test "implicit thinking with trailing newline in reasoning" do
      assert {"reasoning", "content"} =
               Thinking.parse("reasoning\n</think>\ncontent")
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

  describe "stream_parser(thinking: true)" do
    test "starts in thinking mode — no <think> tag needed" do
      parser = Thinking.stream_parser(thinking: true)
      assert parser.state == :thinking

      {events, parser} = Thinking.feed(parser, "reasoning here")
      assert events == [{:thinking, "reasoning here"}]

      {events, _parser} = Thinking.feed(parser, "</think>the answer")
      assert events == [{:content, "the answer"}]
    end

    test "handles </think> split across tokens" do
      parser = Thinking.stream_parser(thinking: true)

      {events, parser} = Thinking.feed(parser, "step 1</thi")
      assert events == [{:thinking, "step 1"}]

      {events, _parser} = Thinking.feed(parser, "nk>answer")
      assert events == [{:content, "answer"}]
    end

    test "full implicit thinking flow" do
      tokens = ["Let me ", "think", "...\n", "</", "think", ">\n", "42"]
      parser = Thinking.stream_parser(thinking: true)

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

      assert thinking_text == "Let me think...\n"
      assert content_text == "\n42"
    end
  end

  describe "parse/1 edge cases" do
    test "unicode content in thinking blocks" do
      input = "<think>考えています 🤔</think>答えは42です"
      assert {"考えています 🤔", "答えは42です"} = Thinking.parse(input)
    end

    test "unicode with implicit thinking (template opened <think>)" do
      input = "Réfléchissons en français\n</think>\nLa réponse est 42"
      assert {"Réfléchissons en français", "La réponse est 42"} = Thinking.parse(input)
    end

    test "nested <think> tags — only the first closing tag is matched" do
      input = "<think>outer <think>inner</think> still thinking</think>content"
      {reasoning, content} = Thinking.parse(input)
      assert reasoning == "outer <think>inner"
      assert content == "still thinking</think>content"
    end

    test "malformed closing tag is treated as plain text" do
      input = "<think>reasoning</thin>still thinking</think>answer"
      {reasoning, content} = Thinking.parse(input)
      assert reasoning == "reasoning</thin>still thinking"
      assert content == "answer"
    end

    test "very long thinking content" do
      long_reasoning = String.duplicate("step ", 10_000)
      input = "<think>#{long_reasoning}</think>done"
      {reasoning, content} = Thinking.parse(input)
      assert reasoning == String.trim_trailing(long_reasoning)
      assert content == "done"
    end
  end

  describe "feed/2 edge cases" do
    test "unicode content in streaming" do
      parser = Thinking.stream_parser()

      {[], parser} = Thinking.feed(parser, "<think>")
      {events, parser} = Thinking.feed(parser, "日本語で考える")
      assert events == [{:thinking, "日本語で考える"}]

      {events, _parser} = Thinking.feed(parser, "</think>答え")
      assert events == [{:content, "答え"}]
    end

    test "very long streaming content" do
      parser = Thinking.stream_parser()
      {[], parser} = Thinking.feed(parser, "<think>")

      long_text = String.duplicate("reasoning ", 5_000)
      {events, parser} = Thinking.feed(parser, long_text)
      assert [{:thinking, ^long_text}] = events

      {events, _parser} = Thinking.feed(parser, "</think>done")
      assert events == [{:content, "done"}]
    end
  end
end
