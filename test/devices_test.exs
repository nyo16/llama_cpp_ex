defmodule LlamaCppEx.DevicesTest do
  use ExUnit.Case, async: true

  test "devices/0 lists backend devices with the expected shape" do
    devices = LlamaCppEx.devices()

    assert is_list(devices)
    # There is always at least a CPU device.
    assert devices != []

    for dev <- devices do
      assert is_integer(dev.index)
      assert dev.type in [:gpu, :igpu, :cpu, :accel, :other]
      assert is_binary(dev.name)
      assert is_binary(dev.backend)
      assert is_integer(dev.memory_total) and dev.memory_total >= 0
      assert is_integer(dev.memory_free) and dev.memory_free >= 0

      case dev.type do
        t when t in [:gpu, :igpu] -> assert is_integer(dev.gpu_index)
        _ -> assert dev.gpu_index == nil
      end
    end
  end

  test "gpu_index values are contiguous from 0 across GPU devices" do
    gpu_indices =
      LlamaCppEx.devices()
      |> Enum.filter(&(&1.type in [:gpu, :igpu]))
      |> Enum.map(& &1.gpu_index)

    assert gpu_indices == Enum.to_list(0..(length(gpu_indices) - 1)//1)
  end
end
