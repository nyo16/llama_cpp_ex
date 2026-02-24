defmodule LlamaCppEx.Precompiler do
  @moduledoc false
  @behaviour ElixirMake.Precompiler

  @all_targets ["aarch64-apple-darwin", "x86_64-linux-gnu"]

  @impl true
  def all_supported_targets(:fetch), do: @all_targets

  def all_supported_targets(:compile) do
    case current_target() do
      {:ok, target} -> [target]
      _ -> []
    end
  end

  @impl true
  def current_target do
    system_arch = to_string(:erlang.system_info(:system_architecture))

    cond do
      system_arch =~ ~r/aarch64.*apple.*darwin/ -> {:ok, "aarch64-apple-darwin"}
      system_arch =~ ~r/x86_64.*linux.*gnu/ -> {:ok, "x86_64-linux-gnu"}
      true -> {:error, "unsupported target: #{system_arch}"}
    end
  end

  @impl true
  def build_native(args), do: ElixirMake.Precompiler.mix_compile(args)

  @impl true
  def precompile(args, _target) do
    case ElixirMake.Precompiler.mix_compile(args) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  @impl true
  def unavailable_target(_target), do: :compile
end
