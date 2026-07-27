defmodule LlamaCppEx.TestModels do
  @moduledoc false

  # Smoke tests need real GGUF files on disk. Each kind maps to exactly one env
  # var; `test/test_helper.exs` documents how to invoke the suite with them set.
  #
  # Every lookup is a runtime call — never a module attribute — so the tests
  # always *compile*, and are skipped by tag rather than by a compile-time `if`.

  @vars %{
    gen: {"LLAMA_SMOKE_GEN_MODEL", "a chat/instruct"},
    emb: {"LLAMA_SMOKE_EMB_MODEL", "an embedding"},
    mtp: {"LLAMA_SMOKE_MTP_MODEL", "an MTP-enabled"}
  }

  @kinds Map.keys(@vars)

  @type kind :: :gen | :emb | :mtp

  @doc "Name of the environment variable holding the model path for `kind`."
  @spec var(kind()) :: String.t()
  def var(kind) when kind in @kinds, do: @vars |> Map.fetch!(kind) |> elem(0)

  @doc """
  Path to the `kind` model, or `nil` when the env var is unset, empty, or names
  a file that does not exist.
  """
  @spec path(kind()) :: String.t() | nil
  def path(kind) when kind in @kinds do
    case System.get_env(var(kind)) do
      value when value in [nil, ""] -> nil
      value -> if File.exists?(value), do: value, else: nil
    end
  end

  @doc "Like `path/1`, but raises with an actionable message instead of returning `nil`."
  @spec path!(kind()) :: String.t()
  def path!(kind) when kind in @kinds do
    path(kind) || raise(unavailable(kind))
  end

  defp unavailable(kind) do
    {var, description} = Map.fetch!(@vars, kind)

    case System.get_env(var) do
      unset when unset in [nil, ""] ->
        "#{var} is not set: point it at #{description} .gguf model file to run these tests."

      missing ->
        "#{var} is set to #{inspect(missing)}, which does not exist: " <>
          "point it at #{description} .gguf model file to run these tests."
    end
  end
end
