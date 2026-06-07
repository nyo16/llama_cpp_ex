defmodule LlamaCppEx.ModelManager.BudgetTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.ModelManager.Budget

  describe "resolve/2" do
    test ":infinity and nil are unlimited" do
      assert Budget.resolve(:infinity, []) == %{mode: :unlimited}
      assert Budget.resolve(nil, []) == %{mode: :unlimited}
    end

    test "an integer is a combined pool (backward-compatible)" do
      assert Budget.resolve(8_000, []) == %{mode: :combined, limit: 8_000}
    end

    test "a map resolves ram and explicit vram per device" do
      budget = Budget.resolve(%{ram: 64_000, vram: [24_000, 24_000]}, [])
      assert budget.mode == :per_device
      assert budget.ram == 64_000
      assert budget.vram == %{0 => 24_000, 1 => 24_000}
    end

    test "vram :auto reads free memory from GPU devices" do
      devices = [
        %{gpu_index: 0, memory_free: 20_000},
        %{gpu_index: 1, memory_free: 21_000}
      ]

      budget = Budget.resolve(%{ram: :infinity, vram: :auto}, devices)
      assert budget.vram == %{0 => 20_000, 1 => 21_000}
      assert budget.ram == :infinity
    end

    test "a map vram with :infinity entries normalizes" do
      budget = Budget.resolve(%{ram: 10, vram: %{0 => :infinity, 1 => 5_000}}, [])
      assert budget.vram == %{0 => :infinity, 1 => 5_000}
    end
  end

  describe "distribute/3 — placement" do
    test ":direct on a single GPU (split_mode :none) puts the full footprint on main_gpu" do
      p = Budget.distribute(6_000, [mode: :direct, n_gpu_layers: -1], 1)
      assert p == %{ram: 0, vram: %{0 => 6_000}}
    end

    test "n_gpu_layers: 0 keeps everything in RAM" do
      p = Budget.distribute(6_000, [mode: :direct, n_gpu_layers: 0], 4)
      assert p == %{ram: 6_000, vram: %{}}
    end

    test "no GPUs detected falls back to RAM" do
      p = Budget.distribute(6_000, [mode: :direct, n_gpu_layers: -1], 0)
      assert p == %{ram: 6_000, vram: %{}}
    end

    test "main_gpu selects the device under split_mode :none" do
      p = Budget.distribute(6_000, [mode: :direct, n_gpu_layers: -1, main_gpu: 5], 8)
      assert p == %{ram: 0, vram: %{5 => 6_000}}
    end

    test "tensor_split spreads across GPUs by weight" do
      p =
        Budget.distribute(
          8_000,
          [mode: :direct, n_gpu_layers: -1, split_mode: :layer, tensor_split: [1, 1, 1, 1]],
          4
        )

      assert p.ram == 0
      assert p.vram == %{0 => 2_000, 1 => 2_000, 2 => 2_000, 3 => 2_000}
    end

    test "tensor_split zeros exclude devices" do
      p =
        Budget.distribute(
          4_000,
          [mode: :direct, n_gpu_layers: -1, split_mode: :layer, tensor_split: [0, 0, 1, 1]],
          4
        )

      assert p.vram == %{2 => 2_000, 3 => 2_000}
      refute Map.has_key?(p.vram, 0)
    end

    test "empty tensor_split with :layer splits equally across all GPUs" do
      p = Budget.distribute(8_000, [mode: :direct, n_gpu_layers: -1, split_mode: :layer], 8)
      assert map_size(p.vram) == 8
      assert Enum.all?(Map.values(p.vram), &(&1 == 1_000))
    end

    test ":server mode adds a KV-cache estimate (on GPU when offload_kqv)" do
      direct = Budget.distribute(6_000, [mode: :direct, n_gpu_layers: -1], 1)

      server =
        Budget.distribute(6_000, [mode: :server, n_gpu_layers: -1, n_ctx: 1024, n_parallel: 1], 1)

      assert server.vram[0] > direct.vram[0]
    end

    test "offload_kqv: false puts the KV cache in RAM" do
      p =
        Budget.distribute(
          6_000,
          [mode: :server, n_gpu_layers: -1, offload_kqv: false, n_ctx: 1024, n_parallel: 1],
          1
        )

      assert p.ram > 0
      assert p.vram[0] == 6_000
    end
  end

  describe "check/3" do
    test ":unlimited always fits" do
      assert Budget.check(
               %{mode: :unlimited},
               %{ram: 9_9, vram: %{0 => 9_9}},
               Budget.empty_usage()
             ) ==
               :ok
    end

    test "combined pool sums RAM and all VRAM" do
      budget = %{mode: :combined, limit: 5_000}
      placement = %{ram: 1_000, vram: %{0 => 3_000}}
      assert Budget.check(budget, placement, Budget.empty_usage()) == :ok

      over = %{ram: 3_000, vram: %{0 => 3_000}}

      assert {:error, {:insufficient_memory, device: :total, required: 6_000, available: 5_000}} =
               Budget.check(budget, over, Budget.empty_usage())
    end

    test "per-device refuses on the over-budget GPU and names it" do
      budget = %{mode: :per_device, ram: :infinity, vram: %{0 => 10_000, 1 => 2_000}}
      placement = %{ram: 0, vram: %{0 => 5_000, 1 => 5_000}}

      assert {:error,
              {:insufficient_memory, device: {:gpu, 1}, required: 5_000, available: 2_000}} =
               Budget.check(budget, placement, Budget.empty_usage())
    end

    test "per-device refuses on RAM" do
      budget = %{mode: :per_device, ram: 1_000, vram: :infinity}
      placement = %{ram: 2_000, vram: %{}}

      assert {:error, {:insufficient_memory, device: :ram, required: 2_000, available: 1_000}} =
               Budget.check(budget, placement, Budget.empty_usage())
    end

    test "accounts for prior usage per device" do
      budget = %{mode: :per_device, ram: :infinity, vram: %{0 => 10_000}}
      used = Budget.add_usage(Budget.empty_usage(), %{ram: 0, vram: %{0 => 7_000}})
      placement = %{ram: 0, vram: %{0 => 4_000}}

      assert {:error,
              {:insufficient_memory, device: {:gpu, 0}, required: 4_000, available: 3_000}} =
               Budget.check(budget, placement, used)
    end

    test "unknown devices are unbounded under a per-device budget" do
      budget = %{mode: :per_device, ram: :infinity, vram: %{0 => 10_000}}
      placement = %{ram: 0, vram: %{7 => 999_999_999}}
      assert Budget.check(budget, placement, Budget.empty_usage()) == :ok
    end
  end

  describe "add_usage/2" do
    test "merges RAM and per-device VRAM" do
      a = %{ram: 100, vram: %{0 => 10, 1 => 20}}
      b = %{ram: 50, vram: %{0 => 5, 2 => 7}}
      assert Budget.add_usage(a, b) == %{ram: 150, vram: %{0 => 15, 1 => 20, 2 => 7}}
    end
  end
end
