defmodule LlamaCppEx.PrecompilerTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Precompiler

  # The precompiler decides which release artifact a user downloads. Getting it
  # wrong is not a build failure, it is a NIF that either silently runs on the
  # CPU or refuses to dlopen, so the selection rules are pinned here.

  defp env(map), do: fn key -> Map.get(map, key) end
  defp libs(list), do: fn soname -> soname in list end

  @driver "libcuda.so.1"
  @cuda12 "libcudart.so.12"
  @cuda13 "libcudart.so.13"

  describe "all_supported_targets/1" do
    test "declares a CPU and a CUDA artifact per supported Linux CUDA major" do
      targets = Precompiler.all_supported_targets(:fetch)

      assert "x86_64-linux-gnu" in targets
      assert "x86_64-linux-gnu-cu12" in targets
      assert "x86_64-linux-gnu-cu13" in targets
      assert "aarch64-apple-darwin" in targets
    end

    test "every declared target is unique" do
      targets = Precompiler.all_supported_targets(:fetch)
      assert targets == Enum.uniq(targets)
    end

    test "compile mode offers only the target this machine actually is" do
      case Precompiler.current_target() do
        {:ok, target} -> assert Precompiler.all_supported_targets(:compile) == [target]
        {:error, _} -> assert Precompiler.all_supported_targets(:compile) == []
      end
    end
  end

  describe "cuda_suffix/2 detection" do
    test "no CUDA runtime and no driver selects the CPU artifact" do
      assert Precompiler.cuda_suffix(env(%{}), libs([])) == ""
    end

    test "driver plus CUDA 13 runtime selects cu13" do
      assert Precompiler.cuda_suffix(env(%{}), libs([@driver, @cuda13])) == "-cu13"
    end

    test "driver plus CUDA 12 runtime selects cu12" do
      assert Precompiler.cuda_suffix(env(%{}), libs([@driver, @cuda12])) == "-cu12"
    end

    test "both runtimes installed prefers the newer major" do
      assert Precompiler.cuda_suffix(env(%{}), libs([@driver, @cuda12, @cuda13])) == "-cu13"
    end

    # The regression this ordering exists to prevent: a CUDA artifact links
    # -lcuda, so with no driver present it cannot be dlopen'd at all. Handing a
    # toolkit-only machine a CUDA build turns a working CPU install into a NIF
    # that fails to load, which is strictly worse than running on the CPU.
    test "toolkit without a driver stays on the CPU artifact" do
      assert Precompiler.cuda_suffix(env(%{}), libs([@cuda12, @cuda13])) == ""
    end

    test "driver without any CUDA runtime stays on the CPU artifact" do
      assert Precompiler.cuda_suffix(env(%{}), libs([@driver])) == ""
    end

    test "an unpublished CUDA major is not selected" do
      assert Precompiler.cuda_suffix(env(%{}), libs([@driver, "libcudart.so.11"])) == ""
    end
  end

  describe "cuda_suffix/2 explicit variant" do
    # Release runners have the toolkit but no driver, so detection would name
    # them CPU. Each CUDA leg states its own variant instead.
    test "the variant override wins over detection" do
      assert Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => "cu12"}), libs([])) == "-cu12"
      assert Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => "cu13"}), libs([])) == "-cu13"
    end

    test "none forces the CPU artifact even on a working CUDA host" do
      full = libs([@driver, @cuda13])
      assert Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => "none"}), full) == ""
      assert Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => ""}), full) == ""
    end

    test "an unknown variant fails loudly rather than publishing a wrong name" do
      assert_raise ArgumentError, ~r/not a known CUDA variant/, fn ->
        Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => "cu11"}), libs([]))
      end

      assert_raise ArgumentError, ~r/not a known CUDA variant/, fn ->
        Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => "yes"}), libs([]))
      end
    end

    test "every variant the override accepts is a target that gets published" do
      targets = Precompiler.all_supported_targets(:fetch)

      for variant <- ["cu12", "cu13"] do
        suffix = Precompiler.cuda_suffix(env(%{"LLAMA_CUDA_VARIANT" => variant}), libs([]))
        assert ("x86_64-linux-gnu" <> suffix) in targets
      end
    end
  end

  describe "current_target/0" do
    test "names a target this build could actually download" do
      case Precompiler.current_target() do
        {:ok, target} ->
          # macOS and x86_64 Linux are published; anything else must report an
          # error so elixir_make falls back to a source build.
          assert target in Precompiler.all_supported_targets(:fetch)

        {:error, message} ->
          assert message =~ "unsupported target"
      end
    end
  end
end
