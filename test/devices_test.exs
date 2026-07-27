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

  # The previous version compared `gpu_indices` against a list it derived from
  # `gpu_indices` itself, so on a CPU-only host it reduced to `[] == []` and
  # asserted nothing. Splitting the devices makes both halves of the invariant
  # carry weight: whichever side is empty, the other still has to hold.
  test "gpu_index numbers the GPU devices contiguously and no others" do
    {gpus, non_gpus} = Enum.split_with(LlamaCppEx.devices(), &(&1.type in [:gpu, :igpu]))

    # :tensor_split indexes GPUs 0..n-1 in device order, with no gaps.
    assert Enum.map(gpus, & &1.gpu_index) == Enum.to_list(0..(length(gpus) - 1)//1)

    # There is always a CPU device, and a non-GPU device never claims a slot in
    # :tensor_split's index space. True on every host, GPU or not.
    assert non_gpus != []
    assert Enum.any?(non_gpus, &(&1.type == :cpu))
    assert Enum.all?(non_gpus, &is_nil(&1.gpu_index))
  end
end
