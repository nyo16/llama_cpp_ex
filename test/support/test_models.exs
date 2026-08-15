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
    mtp: {"LLAMA_SMOKE_MTP_MODEL", "an MTP-enabled"},
    # The head-only sidecar half of a target/draft pair, e.g. Qwen 3.8's
    # mtp-Qwen3.8-27B-Q4_0.gguf. Its own env var rather than a second use of
    # :mtp because the two files are provisioned independently and the sidecar is
    # useless without the target it was built for.
    mtp_draft: {"LLAMA_SMOKE_MTP_DRAFT_MODEL", "an MTP sidecar (head-only)"}
  }

  @kinds Map.keys(@vars)

  @type kind :: :gen | :emb | :mtp | :mtp_draft

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

  @doc """
  `seq_rm` support reported by the `kind` model: `:part` (any position range),
  `:full` (whole sequence only) or `:rs` (bounded partial rollback).

  Prefix-cache reuse is not a property of the library alone — it is a property of
  the model's memory module. Hybrid GDN architectures (Qwen 3.5 / 3.6, Mamba,
  RWKV) keep recurrent state that cannot be rolled back to an arbitrary position,
  so `llama_memory_seq_rm` refuses a partial range and the Server deliberately
  disables prefix reuse for anything needing a trim (`server.ex:812`). Tests that
  assert reuse *happened* are therefore only meaningful on `:part` models, and
  must assert the documented fallback instead on `:full` ones.

  Probed once per suite run and memoised: it costs a model load plus a two-token
  decode.
  """
  @spec seq_rm_kind(kind()) :: :part | :full | :rs | :no
  def seq_rm_kind(kind) when kind in @kinds do
    case :persistent_term.get({__MODULE__, :seq_rm_kind, kind}, :miss) do
      :miss ->
        probed = probe_seq_rm_kind(kind)
        :persistent_term.put({__MODULE__, :seq_rm_kind, kind}, probed)
        probed

      cached ->
        cached
    end
  end

  # n_gpu_layers: 0 — the probe only needs the memory module's answer, and
  # keeping it off the GPU avoids competing with the test that is about to run.
  defp probe_seq_rm_kind(kind) do
    :ok = LlamaCppEx.init()
    {:ok, model} = LlamaCppEx.load_model(path!(kind), n_gpu_layers: 0)
    {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 64)
    LlamaCppEx.NIF.context_can_seq_rm(ctx.ref)
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
