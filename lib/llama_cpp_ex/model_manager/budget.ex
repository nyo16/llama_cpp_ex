defmodule LlamaCppEx.ModelManager.Budget do
  @moduledoc """
  Advisory, placement-aware memory budgeting for `LlamaCppEx.ModelManager`.

  Given a budget, the footprint a new model would occupy across devices, and what
  resident models already use, decide whether the new model fits. The manager
  refuses over-budget loads.

  ## Budget shapes

    * `:infinity` / `nil` — no limit.
    * an **integer** — a single **combined** pool: RAM plus all VRAM counts
      against one number (backward-compatible with the original single-pool
      budget).
    * `:auto` — RAM ≈ 80% of system memory, and **per-GPU** VRAM from each
      device's free memory (via `LlamaCppEx.devices/0`).
    * a map `%{ram: ram, vram: vram}` — explicit **per-device** budget. `ram`
      and each VRAM entry may be a byte limit, `:infinity`, or (for `ram`/`vram`)
      `:auto`. `vram` may be a list `[b0, b1, ...]` (indexed by GPU) or a map
      `%{gpu_index => bytes}`.

  Placement across GPUs is derived from a model's `:n_gpu_layers`, `:split_mode`,
  `:tensor_split`, and `:main_gpu`. It is advisory — quantization, compute
  buffers, fragmentation, and exact KV growth are approximated, and partial
  offload (`0 < n_gpu_layers < n_layers`) is treated coarsely as fully offloaded.
  """

  @type limit :: non_neg_integer() | :infinity
  @type placement :: %{ram: non_neg_integer(), vram: %{non_neg_integer() => non_neg_integer()}}
  @type budget ::
          %{mode: :unlimited}
          | %{mode: :combined, limit: non_neg_integer()}
          | %{mode: :per_device, ram: limit(), vram: :infinity | %{non_neg_integer() => limit()}}

  # --- Budget resolution ---

  @doc """
  Resolves a `:memory_budget` option into an internal budget.

  `gpu_devices` is the GPU subset of `LlamaCppEx.devices/0` (maps with
  `:gpu_index`, `:memory_free`); it is only consulted for `:auto` VRAM.
  """
  @spec resolve(term(), [map()]) :: budget()
  def resolve(opt, gpu_devices \\ [])

  def resolve(:infinity, _devices), do: %{mode: :unlimited}
  def resolve(nil, _devices), do: %{mode: :unlimited}

  def resolve(bytes, _devices) when is_integer(bytes) and bytes > 0,
    do: %{mode: :combined, limit: bytes}

  def resolve(:auto, devices) do
    %{
      mode: :per_device,
      ram: resolve_ram(:auto),
      vram: resolve_vram(:auto, devices)
    }
  end

  def resolve(%{} = map, devices) do
    %{
      mode: :per_device,
      ram: resolve_ram(Map.get(map, :ram, :infinity)),
      vram: resolve_vram(Map.get(map, :vram, :infinity), devices)
    }
  end

  defp resolve_ram(:auto) do
    case system_memory_bytes() do
      bytes when is_integer(bytes) and bytes > 0 -> trunc(bytes * 0.8)
      _ -> :infinity
    end
  end

  defp resolve_ram(nil), do: :infinity
  defp resolve_ram(:infinity), do: :infinity
  defp resolve_ram(bytes) when is_integer(bytes) and bytes >= 0, do: bytes

  defp resolve_vram(nil, _devices), do: :infinity
  defp resolve_vram(:infinity, _devices), do: :infinity

  defp resolve_vram(:auto, devices) do
    for %{gpu_index: gi, memory_free: free} when is_integer(gi) <- devices, into: %{} do
      {gi, free}
    end
  end

  defp resolve_vram(list, _devices) when is_list(list) do
    list |> Enum.with_index() |> Map.new(fn {bytes, i} -> {i, normalize_limit(bytes)} end)
  end

  defp resolve_vram(%{} = map, _devices) do
    Map.new(map, fn {i, bytes} -> {i, normalize_limit(bytes)} end)
  end

  defp normalize_limit(:infinity), do: :infinity
  defp normalize_limit(nil), do: :infinity
  defp normalize_limit(bytes) when is_integer(bytes) and bytes >= 0, do: bytes

  # --- Placement estimation ---

  @doc """
  Estimates how a model's footprint is distributed across RAM and GPUs.

  Returns `%{ram: bytes, vram: %{gpu_index => bytes}}`.

  ## Options (from the load opts)

    * `:mode` - `:server` adds a coarse KV-cache estimate; `:direct` adds none.
    * `:n_gpu_layers` - `0` keeps the model in RAM; anything else (incl. `-1`)
      is treated as fully offloaded to GPU(s).
    * `:split_mode` - `:none` (default) places everything on `:main_gpu`;
      `:layer`/`:row` splits by `:tensor_split`.
    * `:tensor_split` - per-GPU weights; when empty, an equal split across the
      `n_gpus` detected devices.
    * `:main_gpu` - target device for `:none` (default `0`).
    * `:offload_kqv` - whether the KV cache lives on GPU (default `true`).
    * `:n_ctx`, `:n_parallel` - feed the KV-cache estimate.
  """
  @spec distribute(non_neg_integer(), keyword(), non_neg_integer()) :: placement()
  def distribute(file_bytes, opts, n_gpus) when is_integer(file_bytes) and file_bytes >= 0 do
    kv = if Keyword.get(opts, :mode, :server) == :server, do: kv_cache_estimate(opts), else: 0
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 99)
    offloaded? = n_gpu_layers != 0 and n_gpus > 0

    if offloaded? do
      kqv_on_gpu? = Keyword.get(opts, :offload_kqv, true)
      vram_total = file_bytes + if(kqv_on_gpu?, do: kv, else: 0)
      ram_total = if kqv_on_gpu?, do: 0, else: kv
      weights = device_weights(opts, n_gpus)
      vram = Map.new(weights, fn {i, w} -> {i, round(vram_total * w)} end)
      %{ram: ram_total, vram: vram}
    else
      %{ram: file_bytes + kv, vram: %{}}
    end
  end

  # Returns %{gpu_index => fraction}, summing to 1.0.
  defp device_weights(opts, n_gpus) do
    case Keyword.get(opts, :split_mode, :none) do
      :none -> %{Keyword.get(opts, :main_gpu, 0) => 1.0}
      _layer_or_row -> split_weights(Keyword.get(opts, :tensor_split, []), n_gpus)
    end
  end

  defp split_weights([], n_gpus) do
    frac = 1.0 / n_gpus
    Map.new(0..(n_gpus - 1), fn i -> {i, frac} end)
  end

  defp split_weights(weights, _n_gpus) do
    sum = Enum.sum(weights)

    weights
    |> Enum.with_index()
    |> Enum.reject(fn {w, _i} -> w == 0 end)
    |> Map.new(fn {w, i} -> {i, w / sum} end)
  end

  # --- Fit check ---

  @doc "An empty usage accumulator."
  @spec empty_usage() :: placement()
  def empty_usage, do: %{ram: 0, vram: %{}}

  @doc "Folds a placement into a usage accumulator."
  @spec add_usage(placement(), placement()) :: placement()
  def add_usage(used, placement) do
    %{
      ram: used.ram + placement.ram,
      vram: Map.merge(used.vram, placement.vram, fn _k, a, b -> a + b end)
    }
  end

  @doc """
  Decides whether `placement` fits within `budget` given current `used`.

  Returns `:ok` or
  `{:error, {:insufficient_memory, device: device, required: r, available: a}}`,
  where `device` is `:total` (combined budget), `:ram`, or `{:gpu, index}`.
  """
  @spec check(budget(), placement(), placement()) ::
          :ok | {:error, {:insufficient_memory, keyword()}}
  def check(%{mode: :unlimited}, _placement, _used), do: :ok

  def check(%{mode: :combined, limit: limit}, placement, used) do
    required = placement.ram + sum_vram(placement)
    used_total = used.ram + sum_vram(used)
    fits(limit, required, used_total, :total)
  end

  def check(%{mode: :per_device, ram: ram_limit, vram: vram}, placement, used) do
    with :ok <- fits(ram_limit, placement.ram, used.ram, :ram) do
      Enum.reduce_while(placement.vram, :ok, fn {gi, required}, :ok ->
        check_gpu(vram, gi, required, used)
      end)
    end
  end

  defp check_gpu(vram, gi, required, used) do
    case fits(vram_limit(vram, gi), required, Map.get(used.vram, gi, 0), {:gpu, gi}) do
      :ok -> {:cont, :ok}
      error -> {:halt, error}
    end
  end

  defp vram_limit(:infinity, _gi), do: :infinity
  defp vram_limit(map, gi), do: Map.get(map, gi, :infinity)

  defp fits(:infinity, _required, _used, _device), do: :ok

  defp fits(limit, required, used, device) when is_integer(limit) do
    available = max(limit - used, 0)

    if required <= available do
      :ok
    else
      {:error, {:insufficient_memory, device: device, required: required, available: available}}
    end
  end

  defp sum_vram(%{vram: vram}), do: Enum.sum(Map.values(vram))

  # Coarse KV-cache size: 2 (K+V) * n_ctx * n_parallel * bytes-per-context-token.
  defp kv_cache_estimate(opts) do
    n_ctx = Keyword.get(opts, :n_ctx, 8192)
    n_parallel = max(Keyword.get(opts, :n_parallel, 4), 1)
    per_token = 2 * 1024
    n_ctx * per_token * n_parallel
  end

  # Best-effort total system memory in bytes, or nil. Shelled out once at budget
  # resolution; failures fall back to nil.
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
      {out, 0} -> parse_bytes(out)
      _ -> nil
    end
  end

  # `sysctl` output should be a plain integer, but parse defensively so unexpected
  # output (virtualized hosts, odd macOS builds) degrades to nil rather than raising.
  defp parse_bytes(out) do
    case out |> String.trim() |> Integer.parse() do
      {bytes, _rest} -> bytes
      :error -> nil
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
