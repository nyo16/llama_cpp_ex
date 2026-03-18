# Changelog

## v0.6.2

### Changed

- **llama.cpp submodule** — Updated from fdb17643d to 6729d4920 (113 commits).
  - model: add control vector support where missing, Mistral Small 4 support, Nemotron 3 Nano benchmarks
  - ggml: bump version to 0.9.8, AVX512-FP16 support, new GATED_DELTA_NET op, native IM2COL F16
  - CUDA: GDN hide memory latency, limit FA stream-k CUDA blocks, RDNA4 MMVQ table, cuBLAS V100 overflow fix
  - Vulkan: async/event fixes, GATED_DELTA_NET op support, AMD graphics queue, Intel GPU fixes
  - Metal: FA specialization HSK=320/HSV=256, fix l2 norm scale
  - HIP: soft error handling for hipMemAdviseSetCoarseGrain
  - SYCL: enhance UPSCALE, fix untransposed GDA recurrent state
  - OpenCL: fix l2_norm, add cumsum op, larger workgroup for get_rows
  - ggml-cpu: RVV vec dot kernels for quantization, fix RVV checks in quants/repacking
  - server: ctx checkpoint fix, test_cancel_requests fix, refusal content for Responses API
  - webui: model info dialog, MCP CORS proxy detection, duplicated messages fix
  - convert: mixed-precision ModelOpt with NVFP4/FP8, contiguous lora tensors
  - vendor: update cpp-httplib to 0.38.0
  - OpenVINO backend added

## v0.6.1

### Changed

- **llama.cpp submodule** — Updated from c5a778891 to fdb17643d (70 commits).
  - model: add support for Phi4ForCausalLMV, Nemotron 3 Super, Qwen3VL reranker text
  - ggml: add NVFP4 quantization type support
  - llama: chunked fused GDN path, dynamic head_dim and n_rot for SWA
  - metal: extend mul_mv_ext to BF16/Q2_K/Q3_K, fix q5_k register spill, add upscale, handle command buffer failures gracefully
  - CUDA/HIP: GDN shared mem for HIP, fix loop unrolling in ssm-conv, display VRAM capacity on init
  - Vulkan: add SGN and ELU ops, fix data races in coopmat1, skip zero size tensors in copies
  - SYCL: Flash Attention support for fp32/fp16/Q4/Q5/Q8
  - WebGPU: add REPEAT op, faster quant matrix operations
  - KleidiAI: concurrent SME and NEON kernel execution
  - ggml-cpu: add RVV repack GEMM/GEMV for quantization types
  - server: kill switch when stuck, fix checkpoints and OAI completion stream index
  - common: fix --n-cpu-moe/--cpu-moe for fused gate+up models, gracefully handle incomplete output
  - vendor: update cpp-httplib to 0.37.0, miniaudio to 0.11.25
  - llama-quant: fail early on missing imatrix, refactor type selection

## v0.6.0

### Added

- **Qwen 3.5 support** — llama.cpp updated to c5a778891 (35 commits since v0.5.0).
- **`reasoning_content` in ChatCompletion** — `chat_completion/3` now splits `<think>...</think>` blocks from the response when `enable_thinking: true`. The choice message includes `reasoning_content` (the thinking text) and `content` (the final answer). Returns `nil` when thinking is not enabled or no thinking block is present.
- **`reasoning_content` in ChatCompletionChunk** — `stream_chat_completion/3` emits chunks with `reasoning_content` in the delta while the model is thinking, then switches to `content` after `</think>`.
- **`LlamaCppEx.Thinking`** — New module with `parse/1` for one-shot parsing and `stream_parser/1` + `feed/2` for streaming token-boundary-safe parsing of think blocks. Handles the real-world Qwen3/3.5 template behavior where `<think>` is opened by the template itself.

### Changed

- **llama.cpp submodule** — Updated from 7f5ee54 to c5a778891.
  - ggml: add GATED_DELTA_NET op for Qwen 3.5 hybrid architecture
  - model: update Qwen 3.5 model type detection
  - convert: register Qwen 3.5 ForCausalLM for text only
  - CUDA: use shared mem for ssm_conv, improve performance via fewer synchronizations
  - Hexagon: add f32 ssm_conv, fp16 binary ops, Flash Attention optimizations
  - OpenCL: add l2_norm, neg, exp, diag ops
  - CPU: skip redundant ROPE cache updates, fix data race for debug asserts
  - quants: add memsets and other fixes for IQ quants
  - kv-cache: fix M-RoPE checkpoints, checkpoint every n tokens
  - server: preserve Anthropic thinking blocks in conversion

### Unchanged

- `chat/3` and `stream_chat/3` continue returning raw text (no breaking change).

## v0.5.0

### Added

- **Structured output via JSON Schema** — New `:json_schema` option on `generate/3`, `stream/3`, `chat/3`, `stream_chat/3`, `chat_completion/3`, and `stream_chat_completion/3`. Pass a JSON Schema map and the model output is automatically constrained to valid JSON matching the schema. Uses llama.cpp's built-in `json_schema_to_grammar()` under the hood.

  ```elixir
  schema = %{
    "type" => "object",
    "properties" => %{"name" => %{"type" => "string"}, "age" => %{"type" => "integer"}},
    "required" => ["name", "age"],
    "additionalProperties" => false
  }
  {:ok, json} = LlamaCppEx.chat(model, messages, json_schema: schema, temp: 0.0)
  ```

- **`LlamaCppEx.Grammar`** — New module for JSON Schema to GBNF conversion.
  - `from_json_schema/1` — returns `{:ok, gbnf_string}` or `{:error, reason}`
  - `from_json_schema!/1` — returns the GBNF string or raises

- **`LlamaCppEx.Schema`** — New module for converting Ecto schema modules to JSON Schema maps. Maps all standard Ecto types (`:string`, `:integer`, `:float`, `:boolean`, `:date`, `{:array, inner}`, etc.) and supports nested `embeds_one`/`embeds_many`. Automatically excludes `:id` and timestamp fields.

- **NIF: `json_schema_to_grammar_nif/1`** — Exposes llama.cpp's `json_schema_to_grammar()` via `nlohmann::ordered_json`.

### Changed

- **Elixir requirement** bumped to `~> 1.18` (for built-in `JSON.encode!/1`).
- **Dependencies** — added `{:ecto, "~> 3.0", optional: true}` for optional Ecto schema integration.

## v0.4.4

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8198).
  - ggml: fix `ggml_is_contiguous_n` for ne == 1
  - ggml: use simple `std::thread` in AMX without OpenMP
  - KleidiAI: add SME fp16 compute path for q4_0 GEMM on aarch64
  - OpenCL: add optimized q4_1 mm kernel for Adreno
  - Vulkan: tune MMVQ for Intel Windows
  - WebGPU: fix workgroup dispatch limit for large batch sizes
  - Fix locale-dependent float printing in GGUF metadata

## v0.4.3

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8185).
  - Vulkan: improve partial offloading performance on AMD
  - CUDA: cap grid.y at 65535 in non-contiguous dequantize/convert kernels
  - ggml-cpu: optimise s390x multiply extend instructions
  - Vendors: update cpp-httplib to 0.35.0, miniaudio to 0.11.24

## v0.4.2

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8179).

## v0.4.1

### Improved

- **Error handling** — `Chat.apply_template/3`, `Tokenizer.encode/3`, and `Tokenizer.decode/2` now return `{:error, reason}` instead of crashing when NIFs raise.
- **Telemetry documentation** — Server moduledoc documents all telemetry events, measurements, and metadata.
- **Typespecs** — Added `@spec` to `Server.start_link/1`.

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8157).

## v0.4.0

### Added

- **Full model loading params** — `main_gpu`, `split_mode`, `tensor_split` for multi-GPU placement; `use_mlock` and `use_direct_io` for memory control; `vocab_only` for cheap model introspection without loading weights.
- **Server GPU forwarding** — `Server.start_link/1` now forwards `main_gpu`, `split_mode`, `tensor_split`, `use_mlock`, and `use_direct_io` to `Model.load/2`.

## v0.3.0

### Added

- **Jinja chat templates** — switched from `llama_chat_apply_template()` C API to the full Jinja-based `common_chat_templates_apply()` engine from llama.cpp's common library.
- **`enable_thinking` option** — pass `enable_thinking: false` to `Chat.apply_template/3`, `chat/3`, `stream_chat/3`, `chat_completion/3`, and `stream_chat_completion/3` to disable CoT reasoning for models like Qwen3/3.5.
- **`chat_template_kwargs` option** — pass arbitrary key-value pairs to the Jinja template engine.
- **Penalty parameters** — `penalty_repeat`, `penalty_freq`, and `penalty_present` options for repetition/frequency/presence penalties in sampling.
- **OpenAI-compatible response format** — `chat_completion/3` and `stream_chat_completion/3` return `ChatCompletion` and `ChatCompletionChunk` structs.
- **Qwen3.5 benchmark results** in README — Qwen3.5-27B and Qwen3.5-35B-A3B on Apple M4 Max.

### Changed

- `Chat.apply_template/3` now uses the Jinja engine and takes the model ref directly (no longer accepts `:template` option for raw template strings).
- Linked `libcommon.a` from llama.cpp build (previously excluded).
- `LlamaModel` RAII wrapper now caches `common_chat_templates` at model load time.

## v0.2.0

### Added

- **Continuous batching server** (`LlamaCppEx.Server`) — GenServer with slot pool for concurrent multi-sequence inference. One forward pass per tick with decode tokens and prefill chunks mixed in a single batch.
- **Embeddings** (`LlamaCppEx.Embedding`) — `embed/3` and `embed_batch/3` with L2 normalization and configurable pooling type.
- **Grammar-constrained generation** — GBNF grammar support via `grammar` and `grammar_root` options in `Sampler.create/2` and `generate/3`.
- **Batched inference primitives** — `prefill/3`, `decode_batch/3`, `decode_token/4`, `batch_eval/2`, `sampler_sample_at/3` NIFs for building custom inference loops.
- **Streaming via Server** — `LlamaCppEx.Server.stream/3` for token-by-token streaming through the batched server.
- **Telemetry events** — `[:llama_cpp_ex, :server, :tick]` and `[:llama_cpp_ex, :server, :request, :done]` for observability.
- **Benchmark suite** (`bench/`) — Benchee-based benchmarks for single-sequence and server generation, plus a custom continuous batching harness measuring throughput scaling.

### Changed

- `Sampler.create/1` now requires the model as the first argument: `Sampler.create(model, opts)`.
- `Context.create/2` accepts new options: `:embeddings`, `:pooling_type`, `:n_seq_max`.

## v0.1.0

Initial release.

- Model loading and introspection
- Text generation with configurable sampling
- Streaming token generation via `Stream.resource/3`
- Chat template support
- Tokenization and detokenization
- Metal, CUDA, Vulkan, and CPU backends
- RAII resource management via `fine`
