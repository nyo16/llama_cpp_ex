defmodule LlamaCppEx.ModelManager.BudgetTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.ModelManager.Budget

  describe "resolve/1" do
    test "passes :infinity and nil through as :infinity" do
      assert Budget.resolve(:infinity) == :infinity
      assert Budget.resolve(nil) == :infinity
    end

    test "takes a positive integer as a byte limit" do
      assert Budget.resolve(8_000_000_000) == 8_000_000_000
    end

    test ":auto resolves to a positive integer or :infinity" do
      case Budget.resolve(:auto) do
        :infinity -> :ok
        bytes when is_integer(bytes) and bytes > 0 -> :ok
        other -> flunk("unexpected :auto budget: #{inspect(other)}")
      end
    end
  end

  describe "estimate/2" do
    test ":direct mode is just the file size" do
      assert Budget.estimate(1_000, mode: :direct) == 1_000
    end

    test ":server mode adds a KV-cache estimate that scales with n_ctx and n_parallel" do
      base = Budget.estimate(1_000, mode: :server, n_ctx: 1024, n_parallel: 1)
      bigger_ctx = Budget.estimate(1_000, mode: :server, n_ctx: 2048, n_parallel: 1)
      more_parallel = Budget.estimate(1_000, mode: :server, n_ctx: 1024, n_parallel: 2)

      assert base > 1_000
      assert bigger_ctx > base
      assert more_parallel > base
    end

    test "defaults to :server mode" do
      assert Budget.estimate(1_000) > 1_000
    end
  end

  describe "check/3" do
    test ":infinity budget always fits" do
      assert Budget.check(:infinity, 999_999_999, 999_999_999) == :ok
    end

    test "fits when required is within available headroom" do
      assert Budget.check(1_000, 400, 500) == :ok
      assert Budget.check(1_000, 500, 500) == :ok
    end

    test "refuses with required/available when over budget" do
      assert {:error, {:insufficient_memory, required: 600, available: 500}} =
               Budget.check(1_000, 600, 500)
    end

    test "available never goes negative" do
      assert {:error, {:insufficient_memory, required: 100, available: 0}} =
               Budget.check(1_000, 100, 5_000)
    end
  end
end
