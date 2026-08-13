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
  of `:tensor_split`** (non-GPU devices have `gpu_index: nil`) — *unless* an RPC
  device is registered, which reorders the placement list. See
  [Remote devices](#remote-devices-rpc).
- `:memory_free`/`:memory_total` are bytes. Device order follows
  `CUDA_VISIBLE_DEVICES`.

## Placement options

These pass straight through `load/3` (per model) to `Model.load/2` /
`Server.start_link/1`:

| Option | Meaning |
|---|---|
| `:n_gpu_layers` | Layers to offload (`-1` = all, `0` = CPU only) |
| `:split_mode` | `:none` (single device), `:layer` (split layers), `:row` (**throws on CUDA**), `:tensor` (in-process tensor parallelism) |
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

## Remote devices (RPC)

With a build that has `LLAMA_RPC=1`, `LlamaCppEx.RPC.add_server/1` registers
another host's devices into the same registry `LlamaCppEx.devices/0` reads, and
they take part in placement like any other device. That is how a model larger
than one machine loads at all.

```elixir
{:ok, 1} = LlamaCppEx.RPC.add_server("10.100.64.2:50052")

LlamaCppEx.devices()
#=> [%{index: 0, gpu_index: 0, type: :igpu, backend: "CUDA", name: "CUDA0",
#      description: "NVIDIA GB10", memory_total: 130_662_940_672, ...},
#    %{index: 1, gpu_index: nil, type: :cpu, backend: "CPU", ...},
#    %{index: 2, gpu_index: 1, type: :gpu,  backend: "RPC",  name: "RPC0",
#      description: "10.100.64.2:50052", ...}]
```

Three things about that list are not what you would guess.

### `gpu_index` stops indexing `:tensor_split`

**This is the one that will cost you an afternoon.** There are two device
orderings and they are not the same list:

| list | order | what reads it |
|---|---|---|
| the ggml **registry** | registration order — local backends first, RPC endpoints appended as they register | `LlamaCppEx.devices/0`, and therefore `gpu_index` |
| llama.cpp's **placement** list | **RPC first**, then discrete GPUs, then integrated GPUs | `:tensor_split`, `:main_gpu` |

llama.cpp rebuilds the second list at load time and inserts RPC devices at the
front of it, with the comment *"to minimize network transfers"*. So with one
local GPU and one endpoint:

```elixir
# devices/0 says RPC0 has gpu_index: 1 ...
# ... but placement order is [RPC0, CUDA0], so this puts 25% on the REMOTE node.
LlamaCppEx.Model.load(path, split_mode: :layer, tensor_split: [0.25, 0.75])

# And main_gpu: 0 selects the REMOTE node for the non-split tensors, which is
# almost certainly not what you want — the output tensor would then cross the
# network on every token. The local GPU is index 1.
LlamaCppEx.Model.load(path, split_mode: :layer, tensor_split: [0.5, 0.5], main_gpu: 1)
```

A backwards split still works, produces correct tokens, and simply benchmarks
badly, so nothing tells you. The placement order is not observable from Elixir;
it is derived, at load time, from what is registered. Two rules follow:

1. With RPC devices registered, **do not use `gpu_index` to build
   `:tensor_split`.** Count RPC devices first, then local GPUs.
2. Better: state the order. `LlamaCppEx.Model.load/2` accepts `:devices`, a list
   of device names used **verbatim** with no reordering — see below.

One local wrinkle worth knowing if you are on a DGX Spark or another
integrated-GPU box: the local GB10 reports as `:igpu`, and llama.cpp appends
integrated GPUs only when it found no discrete ones. RPC devices deliberately do
not count as discrete for that test, so the local iGPU is not dropped — the
placement list really is `[RPC0, CUDA0]`.

### Stating the device order explicitly

```elixir
# Verbatim: no reordering, no dedup, no CPU filtering.
LlamaCppEx.Model.load(path,
  devices: ["CUDA0", "RPC0"],
  split_mode: :layer,
  tensor_split: [0.6, 0.4])   # 60% local, 40% remote — and it says so
```

`:devices` names devices as `LlamaCppEx.devices/0` reports them, and
`:tensor_split` then indexes *that* list. This is the sanctioned way to make
placement deterministic, and with more than one device in play it is the only
way to make it obvious in review.

### The type is always `:gpu`

Even when the remote worker serves only a CPU device. Upstream hardcodes it with
a TODO. `:description` carries the endpoint string and is the only reliable way
to tell remote devices apart.

### The memory numbers are real, the budget's model of them is not

RPC devices report the remote host's actual free and total memory, so
`LlamaCppEx.devices/0` is accurate. But `ModelManager`'s `:memory_budget` derives
placement from `:split_mode` / `:tensor_split` / `:main_gpu` and has no concept
of a device being a network away — it will budget a remote device as if it were
local VRAM. The numbers are not nonsense, but do not read more into them than
that.

### It buys capacity, not speed

Pipeline parallelism is disabled whenever an RPC device participates: the RPC
backend reports `async = false, events = false`, and llama.cpp checks that before
enabling pipelining. The two nodes therefore execute **sequentially**. A model
that already fits on one machine gains nothing from a second one. See
[DGX Spark](dgx-spark.md) for the measurements.

> #### A peer failure aborts the VM {: .error}
>
> Every client-side RPC command checks its result with `GGML_ABORT`. A peer that
> crashes or a network that drops kills the OS process, BEAM included. There is
> nothing to rescue. Registration is checkable up front — use
> `LlamaCppEx.RPC.ping/1` — and after that the VM is the unit of restart. See
> `LlamaCppEx.RPC`.

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
