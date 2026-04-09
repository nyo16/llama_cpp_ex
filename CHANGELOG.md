# Changelog

## v0.7.3

### Changed

- **llama.cpp submodule** — Updated from b8635075f to d12cc3d1c (55 commits).
  - **model**: add HunyuanOCR support (#21395), support step3-vl-10b (#21287)
  - **llama**: remove per-arch tensor name lists (#21531), correct platform-independent loading of BOOL metadata (#21428)
  - **server**: respect the ignore eos flag (#21203), fix model params not propagated (#21509), fix restore for checkpoints with `pos_min == 0` (#21510), handle unsuccessful sink.write in chunked stream provider (#21478), fix logging of build + system info (#21460)
  - **kv-cache**: extend cache quantization checks (#21586), support attention rotation for heterogeneous iSWA (#21513)
  - **vocab**: remove `</s>` eog token for gemma4 (#21492), add byte token handling to BPE detokenizer for Gemma4 (#21488)
  - **gemma**: perform per-layer projections in the first layer (#21612)
  - **unicode**: add custom Qwen2 regex handler to fix segfault on long input (#21257)
  - **parser**: fix MiniMax handling (#21573)
  - **convert**: set `add bos == True` for Gemma 4 (#21500), fix `block_ff_dim` retrieval for lfm2 (#21508)
  - **ggml**: add Q1_0 1-bit quantization support (CPU) (#21273), deprecate `GGML_OP_ADD1` (#21363), free `ctx_copy` in `ggml_opt_free` to plug per-training-session leak (#21592)
  - **metal**: Q1_0 backend (#21528)
  - **CUDA**: also store `node->src->data` ptrs for equality check (#21635), check for buffer overlap before fusing (#21566), make cuda graphs props check faster (#21472), write an optimized `flash_attn_stream_k_fixup` kernel (#21159), `ds_read_b128` for q4_0 and q4_1 mmq kernels (#21168), fix CDNA2 compute capability constant for gfx90a/MI210 (#21519)
  - **SYCL**: Add Q8_0 reorder optimization (~3x tg speedup on Intel Arc) (#21527), handle other FA case (#21377)
  - **Vulkan**: add FA dequant for q4_1, q5_0, q5_1, iq4_nl (#21029), Linux output error string for errno on fork failure (#20904)
  - **WebGPU**: query for adapter support when registering backend (#21579), parameterize submission size and add iOS specific limits (#21533), add support of `MUL_MAT_ID` (#21147)
  - **hexagon**: slight optimization for argsort output init (#21463)
  - **webui**: store reasoning_content so it is sent back in subsequent requests (#21249), fix syntax highlighting lost after streaming (#21206), detect streaming state in reasoning content blocks (#21549), fix RTL text rendering (#21382), send both `backend_sampling == false/true` (#18781)
  - **cli**: fix stripping of `\n` in multiline input (#21485)
  - **llama-bench**: add `-fitc` and `-fitt` arguments (#21304)
  - **devops/ci**: provide KleidiAI-enabled ARM release artifact (#21259), lower cuda12 floor to 12.8.1 for broader host compatibility (#21438), fix vulkan workflow referencing non-existent action (#21442), use default RISE RISC-V Runners (#21263)

## v0.7.2

### Fixed

- **NIF signature mismatch on precompiled builds** — When `LLAMA_BACKEND` is set, the build now forces compilation from source instead of downloading a precompiled NIF that may have a stale function signature. (#23)
- **Precompile workflow CI failures** — The CI Checks job in the precompile workflow used a stale cached NIF (arity 9 vs 10 for `model_load`) because the cache key didn't include C source hashes and `mix compile` ran under the wrong `MIX_ENV`. Aligned with `ci.yml` by adding `c_src/**` to the cache key, compiling for `MIX_ENV=test`, and running `mix clean` before compile.
- **Precompile archive version mismatch** — The precompile and checksum jobs now set `@version` from the git tag (via `sed`), matching what the publish job already did. Previously, archives were named with the old version from `mix.exs`, causing the publish job to fail when looking for archives matching the tag version.

## v0.7.1

### Added

- **Full llama.cpp optimization parameters** — Exposed 17 new context parameters and 1 model parameter:
  - KV cache quantization: `type_k`, `type_v` (f16, q8_0, q4_0, etc.) for 2-4x memory savings
  - Flash attention & GPU offload: `flash_attn`, `offload_kqv`, `op_offload`
  - RoPE scaling: `rope_scaling_type`, `rope_freq_base`, `rope_freq_scale`, YaRN parameters
  - Misc: `attention_type`, `no_perf`, `swa_full`, `check_tensors`

## v0.7.0

### Added

- **Prefix caching** — Same-slot KV cache reuse for multi-turn chat. When a new request shares a prefix with the slot's previous request, the common prefix is skipped during prefill. 1.23x faster for multi-turn conversations. Controlled by `cache_prompt` option (default `false`, opt-in). Includes prefix-affinity slot selection. See [ADR 007](docs/adr/007-prefix-caching.md).

- **Pluggable batching strategies** — Extracted batch building into `BatchStrategy` behaviour with three built-in strategies: `DecodeMaximal` (default, generation-latency optimized), `PrefillPriority` (throughput optimized), `Balanced` (fair split). Custom strategies can implement the behaviour. See [ADR 008](docs/adr/008-batching-strategies.md).

- **Pre-tokenized API** — `Server.generate_tokens/3`, `Server.stream_tokens/3`, and `Server.get_model/1` allow callers to tokenize outside the GenServer, reducing mailbox contention under concurrent load.

- **HuggingFace Hub integration** — New `LlamaCppEx.Hub` module with `search/2` (find GGUF models), `list_gguf_files/2` (with file sizes via tree API), `download/3` (with local caching, ETag support, offline mode via `LLAMA_OFFLINE=1`), and `get_model_info/2`. Authentication via `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars. New `LlamaCppEx.load_model_from_hub/3` convenience wrapper. Requires optional `:req` dependency.

- **Performance guide** — New `docs/performance.md` with server tuning, prefix caching patterns, strategy selection guide, and optimization recipes.

- **Benchee benchmarks** — New `bench/prefix_cache.exs`, `bench/strategies.exs`, `bench/tokenize_overhead.exs` for measuring prefix cache impact, strategy comparison, and tokenization overhead.

### Changed

- **Graceful batch_eval error handling** — The server now fails active slots with error replies instead of crashing the GenServer when `batch_eval` returns an error (e.g., KV cache overflow).

### Fixed

- **CI warning suppression** — Suppress `-Wunused-function` warnings from vendored llama.cpp jinja headers (`runtime.h`, `lexer.h`).

## v0.6.14

### Changed

- **llama.cpp submodule** — Updated from 50e0ad08f to b8635075f (7 commits).
  - **common**: add Gemma 4 specialized parser (#21418), respect specified tag fallback when tag is empty (#21413)
  - **llama-model**: read `final_logit_softcapping` for Gemma 4 (#21390)
  - **llama**: add custom newline split for Gemma 4 (#21406)
  - **server**: fix undefined timing measurement errors in server context (#21201)
  - **ggml-webgpu**: move from parameter buffer pool to single buffer with offsets (#21278)
  - **ci**: add Windows Vulkan backend testing on Intel (#21292)

## v0.6.13

### Changed

- **llama.cpp submodule** — Updated from 95a6ebabb to 50e0ad08f (32 commits).
  - **server**: save and clear idle slots on new task (`--clear-idle`) (#20993)
  - **common/parser**: fix call ID detection (Mistral parser mostly) + atomicity for tag-json parsers (#21230)
  - **common**: fix tool call type detection for nullable and enum schemas (#21327), add commentary rules for gpt-oss-20b (#21286)
  - **chat**: avoid including json in chat.h (#21306), add Granite 4.0 chat template (#20804), Gemma4 tool response support
  - **jinja**: coerce input for string-specific filters (#21370)
  - **vocab**: fix Gemma4 tokenizer (#21343)
  - **ggml**: bump to 0.9.11 (ggml/1456)
  - **ggml-webgpu**: add vectorized flash attention (#20709)
  - **ggml-zendnn**: add MUL_MAT_ID op support for MoE models (#21315)
  - **rpc**: reuse compute graph buffers (#21299)
  - **kv-cache**: do not quantize SWA KV cache (#21277)
  - **SYCL**: fix llama_kv_cache hang when kv_cache is huge: 5GB (#21283)
  - **hexagon**: add cumsum op support (#21246)
  - **model/mtmd**: fix gguf conversion for audio/vision mmproj (#21309)
  - **tests**: add unit test coverage for llama_tensor_get_type (#20112), allow exporting graph ops from HF file without downloading weights (#21182)
  - **fix**: remove stale assert (#21369), fix gemma 4 template (#21326)

## v0.6.12

### Changed

- **llama.cpp submodule** — Updated from 08f21453a to 95a6ebabb (37 commits).
  - **CUDA**: add FA support for head dim 512 (#20998), fix FA kernel selection logic (#21271), add generic NVFP4 MMQ kernel (#21074), fix kernel selection for mmvq mmid kernel (#21238)
  - **opencl**: fix leak in Adreno q8_0 path (#21212)
  - **ggml**: bump to 0.9.10 (ggml/1454), fix RWKV ops thread assignment (#21226)
  - **ggml-cpu**: fix fallback for RVV kernels without zvfh (#21157)
  - **ggml-webgpu**: quantized buffers to u32 + wider browser/device support (#21046), port AOT operators to JIT (#20728)
  - **kleidiai**: add CPU feature detection to CI run script (#20394)
  - **hexagon**: improve RMS_NORM and DIV accuracy (#21251)
  - **SYCL**: support nvfp4 in mul_mat (#21227), enhance fattn perf (#21185)
  - **CANN**: fix multi-thread set_tensor race conditions (#20151)
  - **memory**: respect unified KV cache in hybrid memory for eval tasks (#21224)
  - **llama**: rotate activations for better quantization (#21038), refactor llama_model_quantize_params to pure C interface (#20346)
  - **common**: gpt-oss handle builtin/unsolicited tool calls (#21213), cleanup logs and modernize progress bar (#21215), disable backend sampling if reasoning budget enabled (#21209), add bounds check to prevent segfault on failed model load (#21082), move up common_init() and fix Windows UTF-8 logs (#21176)
  - **server**: bypass API key validation for WebUI static assets (#21269), no more gzip compression for webui (#21073), cleanup dual representation to openai-compat (#21090)
  - **fix**: tool call parsing for LFM2/LFM2.5 (#21242), correct misspellings (#21217), use lower-case proxy headers (#21235), include API key in CORS proxy for MCP (#21193)
  - **vendor**: update BoringSSL to 0.20260327.0 (#21211)

## v0.6.11

### Changed

- **llama.cpp submodule** — Updated from 82b703f8b to 08f21453a (21 commits).
  - **opencl**: add q4_K gemm and gemv kernels for Adreno (#20919)
  - **CUDA**: fix CUB's argsort when nrows % block_size == 0 (#21181), optimize MOE GEMV kernel for BS > 1 (#20905)
  - **jinja**: handle empty expressions correctly (#20913)
  - **common/parser**: fix handling of tool definition with missing properties key (#21128), add reasoning_format = none support to gpt-oss (#21094)
  - **common/json-schema**: fix non-capturing groups in pattern converter (#21124)
  - **common**: add character class support to glob_match (#21111)
  - **server**: wrap headers for mcp proxy (#21072), fix processing of multiple back-to-back mtmd chunks (#21107)
  - **model**: add missing ROPE_FACTORS_LONG/SHORT for MiniCPM (#21150)
  - **llama-model-loader**: print warning when using overrides with mmap (#20978)
  - **hexagon**: dma optimizations (#21137)
  - **SYCL**: enhance build script to use half cores to avoid OS hang (#21093)
  - **rpc**: fix misleading error log (#21184)

## v0.6.10

### Changed

- **llama.cpp submodule** — Updated from 5c1a7b835 to 82b703f8b (7 commits).
  - **vendor**: update cpp-httplib to 0.40.0 (#21100)
  - **vulkan**: add noncontiguous GLU support (#21081)
  - **common/parser**: fix reasoning whitespace bugs + extra parser tests (#21085)
  - **cli**: add /glob command (#21084)
  - **webui**: conversation forking + branching improvements (#21021)
  - **docker**: fix and enable ARM64 image build (#20929)

## v0.6.9

### Changed

- **llama.cpp submodule** — Updated from 9f102a140 to 1743d9805 (38 commits).
  - **model**: F2LLM-v2 support, allow causal_attn and pooling_type on all architectures (#20973)
  - **convert**: register Qwen3Model architecture (#20967), support Qwen3.5/Qwen3.5 Moe NVFP4 and add input scales (#20505), add RuGPT3XL support (#21011)
  - **ggml-cuda**: add NVFP4 dp4a kernel (#20644), support F32 kernel type for CONV_TRANSPOSE_2D (#17094)
  - **hip**: use fnuz fp8 for conversion on CDNA3 (#21040)
  - **opencl**: allow large buffer for Adreno (#20997)
  - **jinja**: fix macro with kwargs (#20960)
  - **common**: make LLAMA_CACHE the one cache for everything (#21009), fix split model migration (#21019), fix verbosity setup (#20989), add getpwuid fallback for HF cache (#21035), filter out imatrix when finding models (#21023)
  - **llama**: fix llama-model-saver (#20503)
  - **mtmd**: add DeepSeekOCR support (#17400), refactor image preprocessing (#21031), fix quant and im2col ops on Metal for deepseek-ocr (#21027)
  - **imatrix**: fix crash with --show-statistics and zero counts (#19532)

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
