# Changelog

## v0.6.8

### Changed

- **llama.cpp submodule** — Updated from 1772701f9 to 9f102a140 (15 commits).
  - **models**: move the token embedding norms to the first layer (#20943)
  - **ggml-backend**: re-enable graph reuse with pipeline parallelism (#20927)
  - **metal**: add FLOOR, CEIL, ROUND, TRUNC unary ops (#20930), add FA instantiations for HSK=512, HSV=512 (#20902)
  - **common**: add standard Hugging Face cache support (#20775), add a WARNING for HF cache migration (#20935), fix get_gguf_split_info (#20946), replace wrap_for_generation with a prefix convenience function (#20912)
  - **hexagon**: general DMA and Binary Op fixes for large strides (#20918)
  - **llama-fit**: fix regex pattern for gate_up tensors (#20910)
  - **vendor**: update cpp-httplib to 0.39.0 (#20933)

## v0.6.7

### Changed

- **llama.cpp submodule** — Updated from eac9c6ea8 to 1772701f9 (30 commits).
  - **rpc**: RCE patch (#20908), prevent division by zero in deserialize_tensor (#20712)
  - **memory**: fix seq_id bounds in llama_memory_recurrent::state_read_meta() (#20887)
  - **server**: use httplib dynamic threads (#20817), allow router to report child instances sleep status (#20849), fix Host header (#20843)
  - **metal**: add CONV_3D (#19927)
  - **common/autoparser**: detect reasoning markers when enable_thinking changes system prompt (#20859)
  - **common/grammar**: fix grammar parsing issues to prevent stack overflow and hangs (#18604)
  - **context**: use n_embd_out for pooled embedding extraction (#20840)
  - **jinja**: refactor token advancement (#20864)
  - **CUDA**: fix BF16 FA compilation (#20865), native bf16 flash attention for vec kernel (#20525), increase output elements per-thread block for small K-dimension (#20635)
  - **CANN**: add RoPE cache preload before ACL graph capture (#20747)
  - **opencl**: add q6_K gemm and gemv kernels for Adreno (#20089), add flattened Q4_K mv and general Q4_K mm (#20773)
  - **openvino**: explicit memset in buffer_context allocation (#20857)
  - **mtmd**: add dynamic high-resolution image preprocessing for InternVL model (#20847), fix LightOnOCR image preprocessing (#20877)
  - **ggml**: support bf16 and quantized type (#20803)
  - **webui**: improve chat form positioning (#20901), fix --webui-config-file settings not applied on load (#20823)

## v0.6.6

### Changed

- **llama.cpp submodule** — Updated from 6729d4920 to eac9c6ea8 (47 commits).
  - **context**: zero output buffer on allocation (#20781)
  - **model**: assert nextn_predict_layers to prevent underflow (#20783), fix Granite Hybrid type check for 7B.A1B (#20795)
  - **jinja**: fix heap OOB read in value equality comparison (#20782)
  - **common/parser**: fix nasty bug causing subtle corruption of generation prompt (#20825), fix out_of_range crash in throw path (#20777), add proper reasoning tag prefill reading (#20424), fix gpt-oss content removal (#20745)
  - **chat**: handle tool calls with no required args in TAG_WITH_TAGGED format (#20764)
  - **server**: fix router mode deadlock on child crash and TOCTOU race (#20763), add cached_tokens info to oaicompat responses (#19361), improve mtmd ctx checkpoints (#20726), become source of truth for sampling defaults (#20558)
  - **vulkan**: change gated_delta_net to shard across subgroup (#20662), dequantize iq4_xs 4 at a time (#20657)
  - **hip**: avoid compiler bug in RDNA code generation during debug builds on Windows (#20655)
  - **hexagon**: add Matrix Extensions (HMX) for NPU backend (#20693)
  - **CANN**: add BF16 support for core operators (#20152), handle in-place ROPE on non-contiguous f32 tensors (#20274), support flash attention for head dim not multiple of 16 (#20031)
  - **ggml-cpu**: add always_inline to tinyBLAS_PPC accumulator saves (#20791)
  - **ggml-webgpu**: ops support for qwen3.5 (SET, TRI_SOLVE, SSM_CONV, GATED_DELTA_NET) (#20687), add DIAG/TRI ops (#20664), update RMS_NORM/L2_NORM (#20665)
  - **vocab**: assert array size of scores and toktypes (#20737)
  - **convert**: support is_causal hyperparameter (#20746), make NVFP4/MXFP4 say correct type (#20730)
  - **cmake**: fix build warning when kleidiai is enabled (#20457), guard KleidiAI DOWNLOAD_EXTRACT_TIMESTAMP for cmake < 3.24 (#20767)

## v0.6.5

### Changed

- **llama.cpp submodule** — Updated from b6c83aad5 to 6729d4920 (26 commits).
  - **model**: add control vector support where missing (#20653)
  - **ggml**: bump version to 0.9.8 (ggml/1442), restore ggml_type_sizef() to avoid major version bump (ggml/1441)
  - **ggml-cpu**: fix RVV checks in quants and repacking (#20682), fix unused changemask warning in repack (#20692)
  - **ggml-blas**: set MKL threads from thread context (#20602)
  - **Vulkan**: async and event fixes (#20518), disable MMVQ on Intel Windows driver (#20672), allow graphics queue only through env var (#20599)
  - **HIP**: ignore return of hipMemAdvise (#20696)
  - **hexagon**: add neg, exp, sigmoid, softplus, cont, repeat ops (#20701)
  - **kleidiai**: fix MUL_MAT support for batched (3D) inputs (#20620)
  - **server**: fix ctx checkpoint invalidation (#20671)
  - **context**: fix graph not resetting when control vector changes (#20381)
  - **llama**: re-enable manual LoRA adapter free (#19983)
  - **common**: rework gpt-oss parser (#20393), add `--skip-chat-parsing` to force pure content parser (#20289)
  - **webui**: fix duplicated messages on q param (#20715), improve tooltip wording for attachment requirements (#20688)
  - **OpenCL**: no timeout for WaitAny in graph submission to avoid deadlocks on llvm-pipe backends (#20618)

## v0.6.4

### Changed

- **llama.cpp submodule** — Updated from 463b6a963 to b6c83aad5 (56 commits).
  - **model**: Mistral Small 4 support (#20649), Nemotron-H NVFP4 tensors (#20561), Qwen3.5/Qwen3.5MoE NVFP4 tensors (#20506)
  - **ggml**: OpenVINO backend (#15307), native AVX512-FP16 support for F16 operations (#20529), extend im2col f16 (#1434), guard against sumq2 being 0 in IQ4_NL (#20460)
  - **CUDA**: GDN shared mem latency hiding (#20537), limit FA stream-k block count (#20586), RDNA4-specific MMVQ for bs=1 decode (#19478), FP32 cuBLAS for V100 to avoid overflows (#19959), fix data race in cpy kernel (#20507), avoid creating CUDA context during device init (#20595)
  - **metal**: FA specialization for HSK=320, HSV=256 (#20549)
  - **Vulkan**: fix flash attention dot product precision (#20589), use graphics queue on AMD (#20551)
  - **HIP**: APU compatibility — soft error handling for hipMemAdviseSetCoarseGrain (#20536)
  - **SYCL**: fix untransposed GDA recurrent state (#20583), enhance UPSCALE to support all UT cases (#20637)
  - **OpenCL**: fix l2_norm (#20480)
  - **server**: support refusal content for Responses API (#20285), fix wait in test_cancel_requests() (#20601), fix model selector locked to first loaded model (#20580)
  - **tools/cli**: fix disable reasoning (#20606)
  - **convert**: support mixed-precision ModelOpt NVFP4/FP8 quantization (#20539), support contiguous method on lora tensors (#20489)
  - **kv-cache**: fix reading llama_kv_cell_ext during state read (#20273)
  - **common**: fix iterator::end() dereference (#20445)
  - **vendor**: cpp-httplib 0.37.2 → 0.38.0 (#20484, #20578)
  - **webui**: model information dialog (#20600), MCP CORS proxy detection (#20167), code preview iframe isolation (#20477)
  - **hexagon**: Q4_0 and MXFP4 repack fixes (#20527)

## v0.6.3

### Added

- **CI workflow** — New `.github/workflows/ci.yml` runs `mix compile --warnings-as-errors`, `mix format --check-formatted`, `mix test`, and `mix dialyzer` on push/PR to master.
- **Dialyzer** — Added `dialyxir` dependency for static analysis. All modules pass with zero warnings.
- **Example scripts** — New `examples/` directory with 6 runnable scripts: `basic_generation.exs`, `streaming.exs`, `chat.exs`, `structured_output.exs`, `embeddings.exs`, and `server.exs`.
- **Expanded test coverage** — New `test/schema_test.exs` covering `embeds_one`, `embeds_many`, additional Ecto types (`:date`, `:utc_datetime`, `:decimal`, `:map`), empty schemas, and end-to-end nested schema to GBNF conversion. Added edge case tests to `test/thinking_test.exs` for unicode content, nested/malformed tags, and very long content.

### Fixed

- **`Chat.apply_template/3`** — Now accepts string-keyed message maps (`%{"role" => ..., "content" => ...}`) in addition to atom-keyed maps and tuples.
- **`Schema.to_json_schema/1`** — Fixed Dialyzer opaque type warning (replaced `MapSet.member?/2` with `in` operator).
- **GitHub Actions Node.js 20 deprecation** — Updated `actions/checkout` to v5 and added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` env to precompile workflow, preparing for the June 2026 Node.js 24 migration.
- **Stream test reliability** — Fixed `stream with early halt` test to use a prompt compatible with instruction-tuned models.

### Changed

- **llama.cpp submodule** — Updated from fdb17643d to 463b6a963 (31 commits).
  - tools: enable kvu in perplexity for hellaswag, winogrande, multiple-choice (#19954)
  - graph: remove redundant GDN state transposes (#20443)
  - llama: fix pooling assertion crash in chunked GDN detection path (#20468), disable graph reuse with pipeline parallelism (#20463)
  - metal: fix l2 norm scale (#20493), avoid divisions in bin kernel (#20426)
  - Vulkan: add GATED_DELTA_NET op support (#20334), fix l2_norm epsilon handling (#20350), fix OOB check in flash_attn_mask_opt (#20296), fix ErrorOutOfHostMemory on Intel GPU with --no-mmap (#20059)
  - OpenCL: add cumsum op (#18981), use larger workgroup size for get_rows (#20316)
  - HIP: compile debug builds with -O2 to avoid compiler bug (#20392)
  - ggml-cpu: add RVV vec dot kernels for quantization types (#18859)
  - server: reset counter related to kill-switch on client error (#20513), auto-select first loaded model for new conversations (#20403)
  - common/parser: gracefully handle undetected tool parser (#20286), add GigaChatV3/3.1 models support (#19931)
  - grammar: fix root symbol check (#19761)
  - vendor: update cpp-httplib to 0.37.1 (#20390)
  - convert: better mtp check and fix return (#20419)

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
