defmodule LlamaCppEx.LogLevelTest do
  # The level is VM-global, so nothing else may run while it changes.
  use ExUnit.Case, async: false

  setup do
    on_exit(fn -> LlamaCppEx.set_log_level(:info) end)
  end

  test "accepts the documented levels" do
    for level <- [:debug, :info, :warning, :error, :none] do
      assert LlamaCppEx.set_log_level(level) == :ok
    end
  end

  # Logger's spelling is :warning; :warn and strings are a likely slip.
  test "rejects anything else, naming the accepted levels" do
    for level <- [:warn, "info", nil] do
      assert_raise ArgumentError, ~r/invalid log level .* expected one of \[:debug/, fn ->
        LlamaCppEx.set_log_level(level)
      end
    end
  end
end
