# Multi-GPU Placement & VRAM Budgeting

How to place models across GPUs with `LlamaCppEx.ModelManager`, inspect devices,
and use the placement-aware memory budget — plus how to verify it on a real
multi-GPU box.

See also: the "Multiple Models (ModelManager)" section of the [README](../README.md)
and [ADR 009](adr/009-multi-model-manager.md).

## Device introspection

`LlamaCppEx.devices/0` lists every ggml backend device (GPUs, integrated GPUs,
accelerators, CPU):

```elixir
LlamaCppEx.devices()
#=> [%{index: 0, gpu_index: 0, type: :gpu, backend: "CUDA",
#      name: "NVIDIA RTX 4090", description: "...",
#      memory_total: 24_000_000_000, memory_free: 23_500_000_000}, ...]
```

- `:gpu_index` is 0-based across GPU/IGPU devices and **matches the index space
  of `:tensor_split`** (non-GPU devices have `gpu_index: nil`).
- `:memory_free`/`:memory_total` are bytes. Device order follows
  `CUDA_VISIBLE_DEVICES`.

## Placement options

These pass straight through `load/3` (per model) to `Model.load/2` /
`Server.start_link/1`:

| Option | Meaning |
|---|---|
| `:n_gpu_layers` | Layers to offload (`-1` = all, `0` = CPU only) |
| `:split_mode` | `:none` (single GPU), `:layer` (split layers), `:row` (split tensor rows) |
| `:tensor_split` | A **list of per-device proportions** — one float per GPU, indexed by device order. Zeros exclude a device. |
| `:main_gpu` | Primary device: the single GPU under `:none`, or the device holding non-split tensors under `:layer` |

`:tensor_split` is a weight per device (llama.cpp normalizes the values), **not**
a list of indices.

```elixir
# Pin a model to one GPU
LlamaCppEx.ModelManager.load("a", {:path, m}, n_gpu_layers: -1, split_mode: :none, main_gpu: 5)

# Spread one big model across all 8 GPUs equally
LlamaCppEx.ModelManager.load("big", {:path, m},
  n_gpu_layers: -1, split_mode: :layer, tensor_split: [1, 1, 1, 1, 1, 1, 1, 1])

# Use a subset — "big" on GPUs 0–3, "embed" on GPUs 4–7
LlamaCppEx.ModelManager.load("big", {:path, m1},
  n_gpu_layers: -1, split_mode: :layer, tensor_split: [1, 1, 1, 1, 0, 0, 0, 0])
LlamaCppEx.ModelManager.load("embed", {:path, m2},
  capabilities: [:embed], n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [0, 0, 0, 0, 1, 1, 1, 1])
```

## Placement-aware memory budget

`:memory_budget` knows whether a model lands in RAM or on specific GPUs and
checks each pool independently:

- `:infinity` (default) — no limit.
- an **integer** — a single combined pool (RAM + all VRAM count against one number).
- `:auto` — RAM ≈ 80% system memory, and **per-GPU VRAM from each card's free memory**.
- a map `%{ram: …, vram: …}` — explicit per-device limits. `vram` is a list
  `[b0, b1, …]` indexed by GPU, or a map `%{gpu_index => bytes}`. `ram`/`vram`
  may be `:auto` or `:infinity`.

Over-budget loads are refused, naming the device that didn't fit:

```elixir
{:error, {:insufficient_memory, device: {:gpu, 3}, required: r, available: a}} =
  LlamaCppEx.ModelManager.load("too-big", {:path, "70b.gguf"}, n_gpu_layers: -1, main_gpu: 3)
```

`device` is `:total` (combined budget), `:ram`, or `{:gpu, index}`. There is no
automatic eviction — unload a model to make room.

> **Coarse/advisory.** Footprint is estimated from GGUF byte size plus a coarse
> KV-cache estimate for `:server` mode. Partial offload
> (`0 < n_gpu_layers < n_layers`) is treated as fully offloaded; compute buffers
> and fragmentation aren't modeled, so real `nvidia-smi` usage runs somewhat
> higher than the estimate.

## Verifying on a multi-GPU box

### 1. Build from source

The `device_list` NIF only exists in a **source build** — the precompiled
release artifacts don't include it. Force a CUDA source build:

```bash
export LLAMA_BACKEND=cuda          # forces make_force_build = true
mix deps.get
mix compile                        # builds the NIF from source

# For the test suite, also build the test env:
LLAMA_BACKEND=cuda MIX_ENV=test mix compile
```

### 2. Confirm all GPUs are detected

```elixir
LlamaCppEx.devices()
|> Enum.filter(&(&1.type == :gpu))
|> Enum.each(fn d ->
  IO.puts("gpu_index=#{d.gpu_index} #{d.name} " <>
          "free=#{div(d.memory_free, 1024 * 1024)} MB / total=#{div(d.memory_total, 1024 * 1024)} MB")
end)
```

Expect one row per GPU with `gpu_index: 0..N-1` and real memory figures.

### 3. Verify placement

```elixir
{:ok, _} = LlamaCppEx.ModelSupervisor.start_link(memory_budget: :auto)

{:ok, "a"} = LlamaCppEx.ModelManager.load("a", {:path, "/models/m1.gguf"},
  n_gpu_layers: -1, split_mode: :none, main_gpu: 0)
{:ok, "b"} = LlamaCppEx.ModelManager.load("b", {:path, "/models/m2.gguf"},
  n_gpu_layers: -1, split_mode: :none, main_gpu: 1)

LlamaCppEx.ModelManager.list() |> Enum.each(&IO.inspect(&1.placement))
#=> %{ram: 0, vram: %{0 => ...}}  and  %{ram: 0, vram: %{1 => ...}}
```

Cross-check with `watch -n1 nvidia-smi` — the models should sit on the GPUs you
targeted.

### 4. Spread one model across all GPUs

```elixir
{:ok, "big"} = LlamaCppEx.ModelManager.load("big", {:path, "/models/70b.gguf"},
  n_gpu_layers: -1, split_mode: :layer, tensor_split: [1, 1, 1, 1, 1, 1, 1, 1])

LlamaCppEx.ModelManager.info("big") |> elem(1) |> Map.get(:placement)
#=> %{ram: 0, vram: %{0 => .., 1 => .., ... 7 => ..}}
```

### 5. Per-device refusal

```elixir
{:ok, _} = LlamaCppEx.ModelSupervisor.start_link(
  memory_budget: %{ram: :infinity, vram: %{3 => 1_000_000_000}})  # 1 GB cap on GPU 3

LlamaCppEx.ModelManager.load("x", {:path, "/models/big.gguf"},
  n_gpu_layers: -1, split_mode: :none, main_gpu: 3)
#=> {:error, {:insufficient_memory, device: {:gpu, 3}, required: _, available: 1_000_000_000}}
```

### 6. Run the suites

```bash
LLAMA_BACKEND=cuda MIX_ENV=test mix test
LLAMA_GEN_MODEL_PATH=/models/chat.gguf \
LLAMA_EMB_MODEL_PATH=/models/embed.gguf \
  mix run examples/model_manager.exs
```

## Notes

- **`free` vs `total`:** `:auto` budgets off `memory_free` at startup, so other
  processes sharing the GPUs are reflected at that moment.
- **Device ordering:** `gpu_index` follows ggml/CUDA order; remap with
  `CUDA_VISIBLE_DEVICES`.
- **Metal / Apple Silicon:** a single unified-memory device, so per-device VRAM
  is effectively one pool.
