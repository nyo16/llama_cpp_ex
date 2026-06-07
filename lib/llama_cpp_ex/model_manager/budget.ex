defmodule LlamaCppEx.ModelManager.Budget do
  @moduledoc """
  Advisory memory budgeting for `LlamaCppEx.ModelManager`.

  These are pure decision functions: given a budget, the bytes already in use
  by resident models, and the estimated footprint of a new model, decide whether
  the new model fits. The manager refuses to load when a model would exceed the
  budget.

  The budget is advisory — it is a guardrail derived from GGUF file sizes plus a
  coarse KV-cache estimate, not a precise allocator. Quantization, GPU offload
  splits, and runtime overhead are approximated, not measured.
  """

  @type budget :: non_neg_integer() | :infinity

  @doc """
  Resolves a `:memory_budget` option into bytes or `:infinity`.

    * `:infinity` or `nil` → `:infinity` (no limit; fit-checks always pass).
    * `:auto` → roughly 80% of total system memory, or `:infinity` if it can't
      be determined.
    * a positive integer → taken as a byte limit.
  """
  @spec resolve(term()) :: budget()
  def resolve(:infinity), do: :infinity
  def resolve(nil), do: :infinity
  def resolve(bytes) when is_integer(bytes) and bytes > 0, do: bytes

  def resolve(:auto) do
    case system_memory_bytes() do
      bytes when is_integer(bytes) and bytes > 0 -> trunc(bytes * 0.8)
      _ -> :infinity
    end
  end

  @doc """
  Estimates the resident memory footprint of a model in bytes.

  Uses the GGUF file size as the weight baseline, plus a coarse KV-cache
  estimate for `:server` mode derived from `:n_ctx` and `:n_parallel`. Weights
  offloaded to the GPU still count toward the single coarse budget in v1.

  ## Options

    * `:mode` - `:server` (default) or `:direct`. `:direct` adds no standing
      KV-cache estimate.
    * `:n_ctx` - Context size (default `8192`). Only used for `:server` mode.
    * `:n_parallel` - Concurrent slots (default `4`). Only used for `:server` mode.
  """
  @spec estimate(non_neg_integer(), keyword()) :: non_neg_integer()
  def estimate(file_bytes, opts \\ []) when is_integer(file_bytes) and file_bytes >= 0 do
    case Keyword.get(opts, :mode, :server) do
      :server -> file_bytes + kv_cache_estimate(opts)
      _ -> file_bytes
    end
  end

  @doc """
  Decides whether a model needing `required` bytes fits given the `budget` and
  the `used` bytes already accounted for by resident models.

  Returns `:ok`, or `{:error, {:insufficient_memory, required: r, available: a}}`
  where `a` is the budget minus what is already in use.
  """
  @spec check(budget(), non_neg_integer(), non_neg_integer()) ::
          :ok
          | {:error,
             {:insufficient_memory,
              [{:required, non_neg_integer()} | {:available, non_neg_integer()}]}}
  def check(:infinity, _required, _used), do: :ok

  def check(budget, required, used) when is_integer(budget) do
    available = max(budget - used, 0)

    if required <= available do
      :ok
    else
      {:error, {:insufficient_memory, required: required, available: available}}
    end
  end

  # Coarse KV-cache size: 2 (K+V) * n_ctx * n_parallel * bytes-per-context-token.
  # The per-token constant is model-dependent; we use a conservative figure since
  # this is advisory only.
  defp kv_cache_estimate(opts) do
    n_ctx = Keyword.get(opts, :n_ctx, 8192)
    n_parallel = max(Keyword.get(opts, :n_parallel, 4), 1)
    # ~2 KiB per context token (K+V combined, coarse upper bound for small models).
    per_token = 2 * 1024
    n_ctx * per_token * n_parallel
  end

  # Best-effort total system memory. Returns bytes or nil. Shelled out once at
  # budget resolution time (manager start); failures fall back to nil.
  defp system_memory_bytes do
    case :os.type() do
      {:unix, :darwin} -> darwin_memory()
      {:unix, :linux} -> linux_memory()
      _ -> nil
    end
  rescue
    _ -> nil
  catch
    _, _ -> nil
  end

  defp darwin_memory do
    case System.cmd("sysctl", ["-n", "hw.memsize"], stderr_to_stdout: true) do
      {out, 0} -> out |> String.trim() |> String.to_integer()
      _ -> nil
    end
  end

  defp linux_memory do
    case File.read("/proc/meminfo") do
      {:ok, contents} ->
        case Regex.run(~r/MemTotal:\s+(\d+)\s+kB/, contents) do
          [_, kb] -> String.to_integer(kb) * 1024
          _ -> nil
        end

      _ ->
        nil
    end
  end
end
