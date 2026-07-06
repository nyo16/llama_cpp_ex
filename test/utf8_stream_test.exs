defmodule LlamaCppEx.UTF8StreamTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.UTF8Stream

  test "plain ASCII passes through" do
    assert {"hello", ""} = UTF8Stream.push("", "hello")
  end

  test "complete multibyte codepoint passes through" do
    assert {"héllo", ""} = UTF8Stream.push("", "héllo")
    assert {"🎉", ""} = UTF8Stream.push("", "🎉")
  end

  test "holds back a split 2-byte codepoint" do
    <<a, b>> = "é"
    assert {"", <<^a>>} = UTF8Stream.push("", <<a>>)
    assert {"é", ""} = UTF8Stream.push(<<a>>, <<b>>)
  end

  test "holds back a 4-byte emoji split across four pieces" do
    <<a, b, c, d>> = "🎉"

    {out1, p1} = UTF8Stream.push("", <<a>>)
    {out2, p2} = UTF8Stream.push(p1, <<b>>)
    {out3, p3} = UTF8Stream.push(p2, <<c>>)
    {out4, p4} = UTF8Stream.push(p3, <<d>>)

    assert out1 == "" and out2 == "" and out3 == ""
    assert out4 == "🎉"
    assert p4 == ""
  end

  test "emits text before a trailing partial codepoint" do
    <<a, b, c, d>> = "🎉"
    assert {"yay ", <<^a, ^b>>} = UTF8Stream.push("", "yay " <> <<a, b>>)
    assert {"🎉!", ""} = UTF8Stream.push(<<a, b>>, <<c, d>> <> "!")
  end

  test "genuinely invalid bytes pass through rather than stall" do
    # 0xFF can never start a valid codepoint; 0x80 is a bare continuation.
    assert {<<0xFF>>, ""} = UTF8Stream.push("", <<0xFF>>)
    assert {<<0x80, 0x80, 0x80, 0x80>>, ""} = UTF8Stream.push("", <<0x80, 0x80, 0x80, 0x80>>)
  end

  test "long valid text with trailing lead byte" do
    text = String.duplicate("日本語", 10)
    <<lead, _rest::binary>> = "本"
    assert {^text, <<^lead>>} = UTF8Stream.push("", text <> <<lead>>)
  end
end
