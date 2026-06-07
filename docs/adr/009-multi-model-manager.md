# ADR 009: Multi-Model Manager

## Status

Accepted

## Context

The library could load and serve one model at a time: `Model.load/2` returns a `%Model{}` handle, and `LlamaCppEx.Server` (ADR 006) wraps a single model + context for batched inference. Running two or more models simultaneously — e.g. a chat model and an embedding model, or hot-swapping models on a fixed-memory box — required the caller to track handles, pids, and memory by hand.

We wanted a layer that keeps several models resident, routes requests to them by id, and exposes a clean load/unload abstraction, while reusing what already exists: the `Hub` HuggingFace downloader/cache and the batching `Server`.

Key constraints from the existing design:

- A `%Model{}` is a refcounted, read-only, thread-shareable NIF resource. Inference NIFs run on dirty schedulers and do not block the BEAM. **Concurrency does not require one process per model** — many callers can hit one model at once.
- Model cleanup is **GC-only**: the RAII destructor (ADR 002) runs when the last reference is dropped. There is no explicit free NIF.
- `Server` loads its own model + context internally and has **no embedding path** (`Embedding.embed/3` builds its own `embeddings: true` context).
- The library has no Application/supervision tree, and shouldn't impose one on dependents that don't use this feature.

## Decision

Add an opt-in `LlamaCppEx.ModelManager` (a singleton GenServer) plus a `LlamaCppEx.ModelSupervisor` that starts a `Registry`, a `DynamicSupervisor`, and the manager.

### ETS ownership — writes serialize, reads bypass

The manager owns a named, `:protected`, `read_concurrency: true` ETS table of model entries. **Load/unload writes serialize through the GenServer; inference-time lookups (`generate`/`stream`/`chat`/`embed`) read the ETS table directly from the caller process.** This keeps the manager off the inference hot path — a generation request never enters the manager's mailbox, so the singleton is not a throughput bottleneck.

`last_used` (for diagnostics/LRU) is kept in a separate `:public` table so callers can bump it with `:ets.insert/2` without writing to the protected entries table.

### Two backing modes

- **`:server`** (default for generation/chat) — backs the model with a supervised `LlamaCppEx.Server` under the `DynamicSupervisor`, named in the `Registry` by id. Gets continuous batching, streaming, prefix caching, and telemetry for free.
- **`:direct`** — the manager holds the `%Model{}` and callers run stateless `LlamaCppEx.generate/3` / `Embedding.embed/2`. No standing KV-cache memory. **Auto-selected when `:embed` is in `:capabilities`**, because the server has no embedding path; this is a structural constraint, not a preference.

### Routing

By explicit id, or `:default` (set at load with `default: true` or via `set_default/1`). Capability gating is enforced for embeddings (`embed/3` requires an `:embed`-capable, `:direct` model).

### Memory budget — placement-aware, refuse, don't evict

Loads are checked against an advisory, **placement-aware** budget. Footprint is estimated from GGUF file size (plus a coarse KV-cache estimate for `:server` mode) and **distributed across RAM and GPUs** from the load's `:n_gpu_layers`/`:split_mode`/`:tensor_split`/`:main_gpu`. Three budget shapes: `:infinity`; an integer (a single combined RAM+VRAM pool, backward-compatible); and `:auto`/explicit map (per-device — RAM pool plus per-GPU VRAM pools checked independently). `:auto` reads each GPU's free VRAM via a new backend-agnostic `device_list` NIF (`ggml_backend_dev_*`, exposed as `LlamaCppEx.devices/0`). Over-budget loads are **refused**, naming the device: `{:error, {:insufficient_memory, device: :total | :ram | {:gpu, i}, required:, available:}}`. There is **no automatic eviction** — yanking a model another caller is mid-stream on is worse than a clear refusal.

Estimation is coarse and advisory: partial offload (`0 < n_gpu_layers < n_layers`) is treated as fully offloaded, and compute buffers/fragmentation aren't modeled.

### GC-based unload

`unload/1` stops the backing server (dropping its context and model refs), deletes the ETS entry, and forces a GC. Because reclamation is by garbage collection, a caller still holding a `%Model{}` (e.g. from `fetch_model/1`) keeps the model alive until that reference is dropped. This is documented; `list/0`/`info/1` return sanitized, ref-free views to avoid accidentally leaking handles.

### Opt-in supervision

`mix.exs` `application/0` is unchanged — there is no auto-started Application. Users add `{LlamaCppEx.ModelSupervisor, opts}` to their own tree (or call `start_link/1` for scripts/IEx). The supervisor starts the `Registry` and `DynamicSupervisor` before the manager.

## Consequences

- Zero impact on existing single-model usage — the manager is additive and opt-in, with no new dependencies.
- The singleton GenServer is not a bottleneck: inference bypasses it via direct ETS reads.
- A backing-server crash is isolated by the `DynamicSupervisor`; the manager monitors each server and marks the entry `:error` on `:DOWN` (it does **not** auto-reload, to avoid crash loops without backoff).
- Unload is eventual, not deterministic — adequate for swapping a few models, but a fixed-VRAM box needing hard, immediate reclamation would want an explicit `model_free` NIF (deferred; see below).
- Model I/O is behind an injectable `Backend` behaviour, so lifecycle tests run without real GGUF files.

## Alternatives Considered

### One process per model

Rejected — models are read-only and shareable and inference runs on dirty schedulers, so a process per model buys no concurrency. It would only add supervision overhead. The `:server` mode still uses a process per model, but for batching, not for safe concurrent access.

### Auto-started Application

Rejected — this is a library. Auto-starting a process tree would impose it on every dependent whether or not they use multi-model. Opt-in via `ModelSupervisor` keeps users in control.

### Automatic LRU eviction on over-budget

Rejected as the default — eviction can terminate a model another caller is actively streaming from (the busy-check is advisory and racy). Refusing is safer and predictable; the caller decides what to unload.

### Deterministic `model_free` NIF

Deferred. A NIF that nulls the `llama_model*` would make unload's memory reclamation immediate, but any live `llama_context` still references that model — freeing it out from under a context is a use-after-free that crashes the BEAM. It would need refcount gating in every accessor. GC-based reclamation plus the documented caveat is sufficient for now.
