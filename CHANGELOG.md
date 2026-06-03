# Changelog

## v0.8.18

### Changed

- **llama.cpp submodule** — Updated from 0b7154066 to 166fe2949 (16 commits, tag b9495). No NIF changes were required. `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/common.h` are all unchanged. The only changed header the binding compiles against is `common/speculative.h`, which renames `common_speculative_need_embd_pre_norm` → `common_speculative_need_embd_nextn`; the NIF only calls `common_speculative_need_embd` (not the renamed variant), so the MTP/speculative paths are unaffected. The full test suite passes.
  - **model/convert**: add Mellum architecture (#23966); support Granite multilingual embeddings R2 (ibm-granite/granite-embedding-{97,311}m-multilingual-r2) (#22716); add StepFun 3.5 MTP (#23274); qwen35 — use the post-norm hidden state for MTP (#24025).
  - **mtmd**: enable non-causal vision for Gemma 4 unified (#24082); allow skipping `build_vit()` (#24077).
  - **CUDA**: reserve space for the quantized KV cache at startup (#23907); avoid PDL race conditions by disabling `__restrict__` when PDL is used (#24030).
  - **opencl**: use flat variants of q4_K and q6_K gemv for very large M (#24006).
  - **hexagon**: profiler output fix and script updates (#24042).
  - **ggml-cpu**: use the runtime SVE width in FWHT (#24059).
  - **common/arg**: skip the unnecessary mmproj download when `--no-mmproj` is passed (#23425).
  - **webui**: Mermaid diagrams in chat with interactive preview (#24032).
  - **tests**: add support for qwen3 SSM archs (#24031).
  - **build/vendor/CI**: update BoringSSL to 0.20260526.0 (#23794); disable ccache for MSVC Windows release jobs (#23911).

## v0.8.17

### Changed

- **llama.cpp submodule** — Updated from d4c8e2c29 to 0b7154066 (37 commits, tag b9479). No NIF changes were required. `common/chat.h` and `common/json-schema-to-grammar.h` are unchanged; the changes to `include/llama.h`, `common/sampling.h`, and `common/common.h` are additive or touch symbols the binding does not use. `include/llama.h` adds an `n_outputs_max` field to `llama_context_params` (the NIF initializes via `llama_context_default_params()` and sets fields by name, so it defaults to `0` = `n_batch`) and marks `llama_set_warmup` as `DEPRECATED` (not called by the NIF). `common/sampling.h` adds `common_sampler_reasoning_budget_force` and `common/common.h` adds `reasoning_control`/`n_outputs_max`/`sse_ping_interval` fields plus a signature change to `common_prompt_batch_decode` — none of which the binding uses. The full test suite passes.
  - **model/vocab/convert**: add EXAONE 4.5 implementations (#21733); support Step3.7-Flash conversion (#23845); add `normalizer.lowercase` support to WPM tokenizer (#23899).
  - **llama**: deprecate `llama_set_warmup` (#24009); limit max outputs of `llama_context` via `n_outputs_max` (#23861); SWA checkpoints store only non-masked cells (#23981); tensor-parallel quantized KV cache support (#23792); speculative — fix `n_outputs_max` and remove draft-simple auto-enable (#23988).
  - **common**: fix state save in `common_prompt_batch_decode` (#23468); support manually triggering the reasoning budget end sequence (#23949).
  - **server**: add SSE ping interval (#24013); real-time reasoning interruption via control endpoint (#23971); handle `If-None-Match` weak ETags (#23916); disable private security disclosures (#23963).
  - **vulkan**: don't hold the device mutex while compiling pipelines (#23641); reduce host memory lock contention (#23376); block-load Q3_K/Q6_K block data and subtract on 32-bit ints (#23056); remove unused functions (#23175).
  - **SYCL**: support Q4_1/Q5_0/Q5_1 in Flash-Attention (#23812); add more types in `GET_ROWS` op (#23710); optimize Q3_K `mul_mat` by reorder (#23725).
  - **metal**: template GLU kernels to support f16/f32 (#23882).
  - **hexagon**: MUL_MAT, MUL_MAT_ID, FLASH_ATTN and GDN cleanup and optimizations for latest models (#23989); add `gelu_quick` (#24007).
  - **opencl**: add basic support for q5_0 and q5_1 (#23548); fix compiler warnings for the non-Adreno path (#23922); revert to using `global_invocation_id` for the cpy shader (#23955).
  - **webui**: add a Thinking-mode toggle with reasoning effort levels and Chat Form "Add Action" UI improvements (#23434); simplify network error handling (#23431).
  - **build/vendor/CI/docs**: update cpp-httplib to 0.46.1 (#23980); add nix-nodejs facilities to build the Web UI (#23846); clean up unused-variable warnings (#23975); CI job trimming and runner-label fixes (#24012, #23958, #23927, #23938); update `HOWTO-add-model.md` (#23883).

## v0.8.16

### Changed

- **llama.cpp submodule** — Updated from 19e92c33e to d4c8e2c29 (40 commits, tag b9442). No public API changes; `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, and `common/sampling.h` are all unchanged. `common/common.h` has only additive/benign changes (a new `skip_download` field, the server `timeout_read` default bumped 600s→3600s, and a comment tweak) that don't touch any NIF binding. No NIF changes were required; the full test suite passes.
  - **model/vocab**: support for `DeepseekV32ForCausalLM` with a generic DeepSeek Sparse Attention (DSA) implementation (#23346); tokenizer support for jina-embeddings-v2-base-zh (#18756) and LFM2.5-8B-A1B (#23826).
  - **mtmd**: add DeepSeekOCR 2 support (#20975).
  - **llama**: use f16 mask for Flash Attention to save VRAM (#23764); only use one iGPU device by default (#23897) and don't skip iGPU when only RPC devices are present (#23868); add `llm_graph_input_mtp` (#23643).
  - **server**: in SSE mode, send HTTP headers when the slot starts (#23884); bump read timeout to 3600s (#23842); add speed-bench for speculative decoding (#23869); remove obsolete scripts (#23870).
  - **metal**: restore im2col implementation for large kernels (#23901).
  - **vulkan**: add Flash Attention support for BFloat16 KV cache (#23420).
  - **CUDA**: check PTX version on the host side to guard PDL dispatch (#23530); disable `launch_fattn` PDL enrollment due to a compiler bug (#23825).
  - **opencl**: support bf16 by converting to f16 (#23839).
  - **ggml**: add some LSX support (#23798); bump ggml version to 0.13.1; sync ggml; fix missing `buffer` set in the allreduce fallback (#23480).
  - **ggml-webgpu**: check earlier for required WebGPU features (#23879); add q4_0/q8_0 `SET_ROWS` (#23760).
  - **graph**: ensure DS32 `kq_mask_lid` is F32 (#23864).
  - **tensor-parallel**: fix granularity for Qwen 3.5/3.6 with 3 GPUs (#23843).
  - **download**: add option to skip downloading (#23059); **llama-bench**: support `-fa auto` (#23714).
  - **app/ui**: add `llama update` self-updater (#23865); move licences to llama-app (#23824); custom CSS injection via config (#23904); handle `audio/vnd.wave` as WAV (#23754); fix ETag truncation with MSVC (#23917); exclude generated build dirs from prettier/eslint (#23910).
  - **CI/docs/build**: macOS and iOS release jobs moved to macos-26 runners (#23878, #23906); fix s390x release job (#23898); cache-key fixes (#23895); update ZenDNN docs for Q8 support (#23791); `ngram-mod` missing include (#23857).

### Fixed

- **`LlamaCppEx.Server` double-accept.** The server's per-slot sampling called
  `sampler_accept` after `sampler_sample_at`, but `sampler_sample_at` (like
  `llama_sampler_sample`) already accepts the selected token. The redundant
  accept double-advanced sampler state — the same class of bug fixed in the
  direct generation loops in v0.8.15. It broke grammar-constrained server
  inference and double-counted repeat/frequency/presence penalties. Removed both
  redundant accepts (first-token-after-prefill and each decode step). Output for
  penalty-based sampling through the server changes (now correct); greedy/temp
  sampling is unaffected.

### Performance

- **Server batch loop — removed super-linear hot paths.** The three batching
  strategies (`DecodeMaximal`, `PrefillPriority`, `Balanced`) shared identical
  per-slot/per-token assembly code with several costly patterns: `batch_idx =
  length(entries)` inside the prefill token loop (**O(n²)** per prompt),
  `length(slot.prompt_tokens)` (O(n)) called twice per chunk, `Enum.slice/3` on a
  list (O(prefill_pos)) per chunk, and `accumulated_text <> piece` per decode
  token (**O(n²)** over a generation). The shared logic now lives in
  `LlamaCppEx.Server.Strategy.Batch` and: threads a running entry counter instead
  of `length/1`, uses the cached `slot.n_prompt_tokens`, slices from a
  `prompt_tokens_tuple` (O(1) indexing), and accumulates token pieces as iodata
  joined once at completion. Behavior is unchanged (guarded by
  `test/server_batch_test.exs`).
- **`common_prefix_length/2`** rewritten as a single-pass tail recursion (was
  `Enum.zip |> Enum.take_while |> length`, allocating an intermediate list). Runs
  on every prefix-cache lookup.
- **`embed_batch/3` no longer allocates a context per text.** It now packs texts
  into a single context as distinct sequences (greedy bin-packing within the
  context budget, capped by `:max_batch_sequences`, default 64) and decodes each
  group in one batch via the new `embed_batch_decode` NIF, retrieving pooled
  per-sequence embeddings. Falls back to one-context-per-text only for
  `:pooling_type: :none`. Equivalence with the per-text path is guarded by a smoke
  test.
- **Streaming NIF loops** (`generate_tokens`, MTP `generate_mtp_tokens`) intern
  the hot result atoms once instead of per token and reuse the detokenize fallback
  buffer across iterations.

## v0.8.15

### Changed

- **llama.cpp submodule** — Updated from 0d18aaa9d to 19e92c33e (51 commits). No public API changes; `include/llama.h`, `common/common.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, and `common/sampling.h` are all unchanged. The only header diffs are internal (`src/llama-chat.h`, `src/llama-vocab.h`, `ggml-cpu/vec.h`, the ggml-hexagon op headers, `clip-graph.h`, `server-http.h`, and the vendored `cpp-httplib`). No NIF binding changes were required.
  - **chat**: add Granite 4.1 chat template (#23518) — picked up automatically by `chat_apply_template`.
  - **model/mtmd**: fix gemma 4 projector pre_norm (#23822) and audio rms norm eps (#23815); `n_head_kv` defaults to `n_head` (#23782); mtmd-debug color and rainbow mode (#23829).
  - **convert**: add FP8 to Q8 conversion (#23250); add MiniCPM5 tokenizer support (#23384).
  - **arg/common**: add `LLAMA_ARG_API_KEY_FILE` env var for `--api-key-file` (#23167); fix env names to all have the `LLAMA_ARG_` prefix (#23778).
  - **server**: add support for HTTP ETags in llama-server (#23701); minor tweaks to use more cpp features (#23785); fix the log message when using SSL (#23393).
  - **ggml**: auto-apply iGPU flag for CUDA/HIP on integrated devices (#23007); fix Arm SVE usage bug in `vec.h`/`vec.cpp` (#22841).
  - **CUDA**: route batch>=4 quantized matmul to MMQ on AMD MFMA hardware (#23227); add `MMVQ_PARAMETERS_TURING` (#23729); fix KQ mask offset integer overflow in the fattn MMA kernel (#23610); restrict PDL to CTK >= 12.3 due to MSVC issues (#23742).
  - **vulkan**: fast path for Walsh–Hadamard transform (#23687); use `GL_NV_cooperative_matrix_decode_vector` for faster matmul (#23541); switch `MUL_MAT_VEC` to 4 K per iteration for F16/32 (#22887); add `REPEAT` op support for f16→f16 (#23298); avoid preferring transfer queue on AMD UMA devices (#22455); fix inner-loop index variable (#23665) and memory-logger unsafe iterator access (#23667).
  - **hexagon**: basic/generic op fusion + `RMS_NORM`+`MUL` fusion (#23835); `OP_GATED_DELTA_NET` K>1 support (#23531); add Q4_1 in `MUL_MAT`/`MUL_MAT_ID` (#23647); minor refresh for HMX FA and MM (#23796).
  - **opencl**: `OP_GATED_DELTA_NET` (#23312); move backend info printing into its own function (#23702).
  - **ggml-webgpu**: remove legacy constants (#23672); fix workgroup dispatch for some ops (#23750).
  - **ggml-zendnn**: fix naming of matmul function (#20964).
  - **ui**: fix audio and video modality detection (#23756).
  - **app**: improve help output (#23805).
  - **perplexity**: fix format specifier in `LOG_ERR` (#23788).
  - **vendor**: update `cpp-httplib` to 0.46.0 (#23650).
  - **docker**: add ZenDNN Dockerfile (#23716).
  - **pyproject**: add conversion folder and update dependencies (#23746).
  - **docs**: fix duplicated "the" in granitevision and model-conversion docs (#23767).
  - **CI**: numerous build/runner changes — UI publish on ubuntu-slim (#23818), releases use GitHub-hosted builds for the UI (#23823), Vulkan builds switched to Release (#23820), CI refactor (#23789), move ARM jobs to self-hosted (#23780), bump CUDA release to 13.3 (#23749), add ccache to server builds (#23763), fix windows ccaches (#23777), remove wasm test (#23733).

### Fixed

- **Grammar / structured output crash (double-accept)** — Constrained generation
  (`:json_schema` and `:grammar` options on `generate`/`stream`/`chat`/`stream_chat`)
  crashed on the **first** generated token with
  `RuntimeError: Unexpected empty grammar stack after accepting piece: ...`.
  The generation loops in the NIF called `llama_sampler_accept/2` after
  `llama_sampler_sample/3`, but `llama_sampler_sample/3` already accepts the
  selected token internally. The redundant accept advanced grammar state twice,
  so the grammar tried to match the just-consumed token against the *next*
  position and emptied its stack. For unconstrained sampling the double-accept
  was mostly harmless (it double-counted repeat/frequency/presence penalties),
  which is why it went unnoticed. Removed the redundant `llama_sampler_accept/2`
  from all five sampling sites: `generate`, `generate_tokens` (streaming),
  `decode_batch`, and both MTP/speculative loops. Structured output now returns
  schema-valid JSON, and penalty-based sampling is no longer double-applied.

### Testing

- **Added an end-to-end smoke test** (`test/smoke_test.exs`, tagged `:smoke` and
  **excluded by default**) covering generation, streaming, chat templating,
  structured output (JSON-schema + raw GBNF grammar — a regression guard for the
  double-accept bug), and embeddings against real GGUF models. Run with
  `LLAMA_SMOKE_GEN_MODEL=... [LLAMA_SMOKE_EMB_MODEL=...] mix test --include smoke`.

## v0.8.14

### Changed

- **llama.cpp submodule** — Updated from b22ff4b7b to 0d18aaa9d (52 commits). No public API changes; the only `include/llama.h` diff is a doc comment on `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` clarifying that getting a per-seq state with the flag invalidates prior on-device states for the same `seq_id`. `common/common.h` replaces `checkpoint_every_nt` with `checkpoint_min_step` (server-only field) and drops the `LLAMA_UI_DEFAULT_ENABLED` ifdef (UI still defaults to `true`); `common/chat.h` adds additive `common_chat_msg_span`/`common_chat_msg_delimiter` structs, a `message_spans` field on `common_chat_params`, an `is_continuation` field on `common_chat_parser_params`, and a new `common_chat_split_by_role` helper. None of these are used by the NIF, so no binding changes were required.
  - **llama**: document that only one on-device state can be saved per sequence (#23520).
  - **server**: fix checkpoints creation (#22929); MTP layer kv-cache should respect draft type ctk (#23646); expose prompt token counts in `/slots` (carried from v0.8.13 lineage); add margin for draft model for `fit` (#23485).
  - **convert**: support `Gemma4ForCausalLM` (#23682); add compressed-tensors NVFP4 support (#21095); minor fixes for numpy 2.x (#23571).
  - **model**: add support for `talkie-1930-13b` (#22596); tag `ffn_latent` as `MUL_MAT` to fix buft probe (#23664); attach Mistral3 NVFP4 weight scales (#23629).
  - **vocab**: fix `HybridDNA` tokenizer (#23466) (carried).
  - **ggml**: bump to 0.13.0 (ggml/1510) and 0.12.1 (ggml/1508); `gguf_init_from_callback` and `gguf_init_from_buffer` (#22341); parallelize quant LUT init (#23595); ggml-alloc out-of-bounds read fix in `ggml_dyn_tallocr_remove_block` (ggml/1492); TP fix ggml context size calculation (#22616); `ggml_silu_back` docstring fix (ggml/1500).
  - **metal**: add apple device id (#23566).
  - **CUDA**: add fast Walsh–Hadamard transform (#23615); missing PDL sync for FWHT + better fallback (#23690).
  - **vulkan**: optimize `conv2d` and implement `coopmat1` support (#22620).
  - **SYCL**: implement `ggml_sycl_pool_vmm` (#22862).
  - **hexagon**: add `CONCAT` op (#23648); flash-attn softmax repl optimization (#23455).
  - **ggml-webgpu**: add MMVQ path for Q4/Q8/Q2_K/Q4_K and clean up legacy `MUL_MAT` pipeline (#23594); check `batch_compute_passes` before sending passes when not GPU profiling (#23457).
  - **opencl**: batch profiling to improve speed and prevent memory leaks (#23495).
  - **perplexity**: fix even more integer overflows (#23623).
  - **TP**: fix entirely zero-sized slices per device (#23525).
  - **ui**: fix stop/continue during an agentic loop (#23356); media attachments before text (#23467).
  - **vendor**: update `cpp-httplib` to 0.45.1 (#23639).
  - **snapdragon**: bump toolchain docker to v0.7 to fix UI build issues (#23680); update windows toolchain to use `hsdk` v6.6.0.0 (#23552).
  - **cmake**: fix UI build (#23592).
  - **tests**: `test-backend-ops -j <N>` to run tests in parallel (#23637).
  - **CI**: many self-hosted runner migrations, `[no release]` keyword support, and macOS/apple workflow consolidation (#23705, #23713, #23715, #23718, #23721, #23728, #23730, #23734, #23619, #23616, #23675, #23642, #23651, #23630).

## v0.8.13

### Changed

- **llama.cpp submodule** — Updated from 52fb93a2b to b22ff4b7b (25 commits). No public API changes; `include/llama.h` is unchanged. `common/chat.h` adds an additive `is_continuation` field (default `false`) on `common_chat_parser_params`; `common/common.h` simplifies the `ui` default (removes the `LLAMA_UI_DEFAULT_ENABLED` ifdef, still defaults to `true`). No NIF changes required.
  - **model**: add NVFP4 MTP scale tensors (#23563).
  - **server**: only parse empty message if continuing an assistant message (#23506); expose prompt token counts in `/slots` endpoint (#23454).
  - **vocab**: fix HybridDNA tokenizer (#23466).
  - **perplexity**: fix integer overflow (#23496).
  - **ggml**: check the right iface method before using the fallback 2D get (#23514).
  - **flash-attn**: replace `f32` with `kv_type` and `q_type` (#23372).
  - **metal**: optimize concat kernel and fix `set` kernel threads (#23411).
  - **CUDA**: fix PDL CC check for JIT compilation (#23471).
  - **vulkan**: fuse snake activation `mul + sin + sqr + mul + add` (#22855); fix windows `find_package` of `SPIRV-Headers` (#23215).
  - **SYCL**: improve MoE prefill throughput (#23142); Level Zero detection in `ggml_sycl_init` (#23097); `gated_delta_net` K>1 (#23174); add BF16 to DMMV kernel path (~4x tg speedup on Intel Arc) (#21580).
  - **opencl**: generalize Adreno MoE kernels on M (#23449).
  - **ggml-zendnn**: add Q8_0 quantization support (#23414).
  - **cmake**: refactor UI build (#23352); add `install()` for impl libraries + fix Apple builds (#23511); remove `STATIC` from impl libraries, enable `LLAMA_BUILD_APP` by default (#23462); build router app only during standalone builds (#23521).
  - **tests**: move `save-load-state` from examples to tests (#23336).
  - **docs**: update documentation with Granite 4.0/4.1 (#23404); update WebGPU support and add link to blog/demo (#23483).
  - **requirements**: bump torch to 2.11.0 (#23503).

### Fixed

- **`mix.exs` `@version` drift** — `@version` is bumped from `0.8.11` to `0.8.13` to re-align with the published Hex/tag stream. Tag `v0.8.12` was cut against a `@version "0.8.11"` source tree, so this release skips `0.8.12` to avoid republishing under a stale version.

## v0.8.12

### Changed

- **llama.cpp submodule** — Updated from b28a2f372 to 52fb93a2b (30 commits). No public API changes; existing NIF and `LlamaCppEx.MTP` bindings continue to work unchanged.
  - **MTP / speculative**: move draft sampling to the backend (`backend_sampling` defaults `true` on the new `common_params_speculative_draft` field — additive) (#23287); skip logit computation via `inp_out_ids` (#23433); fix `nullptr` crash in `common_speculative_get_devices_str` (#23386); free draft/MTP resources on server slot sleep to fix a VRAM leak (#23461); doc typo (#23435).
  - **llama**: fix `llm_graph_input_attn_kv_iswa` null-buffer crash on SWA-only models (#23131).
  - **vocab**: add Carbon-3B `HybridDNATokenizer` support (#23410).
  - **server**: re-inject subcommand when the router spawns children under the unified binary (#23442).
  - **app**: introduce the `llama` unified executable (#23296); add `batched-bench`, `fit-params`, `quantize`, and `perplexity` subcommands (#23459); show version (#23426).
  - **mtmd**: merge HunyuanOCR into HunyuanVL and fix OCR vision precision (#23329); DeepSeek-OCR image-processing fixes + `img_tool::resize` padding refactor (#23345); `fit_params` now accounts for `mmproj` (#21489); WAV MIME-type variants and improved audio format detection (#23396).
  - **ggml**: check the right iface method before falling back to the 2D get (#23306).
  - **metal**: optimize `pad` + `cpy` (#23354).
  - **CUDA**: Programmatic Dependent Launch (PDL) for Hopper+ (#22522); tune RDNA3 Q6_K MMVQ nwarps (#23349).
  - **vulkan**: optimize `IM2COL` shader (#22685).
  - **opencl**: refactor backend initialization (#23318).
  - **hexagon**: `ssm-conv` fix for large prompts (#23307); HMX quantized matmul rework (#23368).
  - **snapdragon**: update toolchain to v0.6 (#23369).
  - **webui**: max image size option (#22849); reactive `isMobile` in viewport store (#23330); div-wrapper pointer-events fix on hidden (#23390); move text attachments before message content in chat-completions payload (#23406); improve UI dev git hooks (#23403).
  - **docker**: copy conversion files (#23370).

## v0.8.11

### Changed

- **llama.cpp submodule** — Updated from 0253fb21f to b28a2f372 (57 commits).
  - **llama**: MTP clean-up (#23269); initialize pre-norm embedding mask flag (#23256); avoid copying logits during prompt decode in MTP (#23198).
  - **common**: delegate assistant continuation to underlying template handlers (#23089) — new `common_chat_continuation` enum and `continue_final_message` field on `common_chat_templates_inputs` (default `COMMON_CHAT_CONTINUATION_NONE`, additive); enable streaming JSON argument values (#23173); remove hf cache migration (#23266); fix `--help` and `--fit` `--verbosity` output (#23278, #23282).
  - **server**: guarantee at least 1 token to decode in server-context (#23280); print graphs reused in slot timings (#23279); honor `--embd-normalize` CLI arg (#23125); router allocates tmp buffer on heap (#23159); skip device enumeration in router mode to avoid creating CUDA primary context (#23137).
  - **model**: clarify MTP layer comment in qwen35.cpp (#23338); update bid to match each layer's MTP source (#23237).
  - **vulkan**: add cpy bf16 → f32 pipelines (#22677); support unaligned tensors for ROPE (#22637); fuse `SSM_CONV + BIAS + SILU` (#22653); add `SPIRV-Headers` cmake check (#22009); remove duplicate `#include <memory>` (#23144).
  - **hexagon**: add MROPE and IMROPE in HTP rope op (#23317); enable NORM op (#23319); add TRI op (#22822); ggml-hexagon PAD op HVX kernel (#23078).
  - **opencl**: add MoE support for q4_k, q5_k, q6_k on Adreno (#23303).
  - **CUDA**: continue directly including `cuda/iterator` (#23102); support `d_conv=15` for `ssm-conv.cu` (#23017).
  - **SYCL**: add `GGML_SYCL_USE_ASYNC_MEM_OP` env toggle (#22153); scalar SWAR byte-subtract in Q6_K MMVQ dot product (#22156); route small f32 matmuls to oneMKL, bypass oneDNN (#22150); fix error when using `-mg 1` (#23140); performance reference in SYCL.md (#23315).
  - **ggml-webgpu**: extend GDN for K>1 (#23299).
  - **rpc**: keep `last_graph_uid` in the device context (#23273).
  - **webui**: chat screen UI refactor (#23333); bump packages + address build warnings (#23300); update KaTeX + clean `sass` warnings (#23275); scroll-to-bottom button + prevent forced scroll (#23270); refactor models store / MCP service / gate logs behind `VITE_DEBUG` (#23236); centralize monospace font styles (#23272); fix Tailwind v4 utility classes missing when built via cmake (#23253); support video files as input (#22830).
  - **convert**: update MTP-related help (#23334); filter LoRA tensor names (#23077).
  - **save-load-state**: refactor tests and improve readability (#23196).
  - **llama-eval**: add per-task summary stats (#23151).
  - **ngram**: reduce noisy logs (#23185).
  - **build/CI**: install libssl-dev (#23325); install server kleidiai runner dependencies (#23259); add kleidiai-server to server-self-hosted workflow (#22435); cmake — do not check for bin install dir (#23234), fix `LLAMA_BUILD_UI` logic (#23190), do not install conversion script (#23204); docker — add OCI image labels for version and build date (#21653).

## v0.8.8

### Fixed

- **Server prefix-cache crash on hybrid GDN models** (#38) — On hybrid Gated Delta Net architectures (Qwen 3.5 / 3.6) `llama_memory_seq_rm` silently no-ops on partial-range trims, leaving the KV cache at the old positions while the next prefill tried to write tokens at lower positions. This produced an M-RoPE positional-consistency abort (`X = 56 >= Y = 46 ... requires X < Y`). `LlamaCppEx.Server` now probes `common_context_can_seq_rm` once at init and falls back to a full reset (n_match=0) when the model only supports `:full` seq_rm. Includes a regression test for sequential same-prefix requests under `cache_prompt: true`.

### Added

- **`LlamaCppEx.NIF.context_can_seq_rm/1`** — exposes `common_context_can_seq_rm`, returning `:no | :part | :full | :rs`. Clears KV memory as a side effect, so call once before any decode.

## v0.8.7

### Added

- **Multi-Token Prediction (MTP) speculative decoding** (#37) — new `LlamaCppEx.MTP` module exposing `init/2`, `stream/3`, `stream_events/3`, `generate/3`, `stats/1`, and `print_stats/1`. Drives a target/draft speculative loop where the draft model is the MTP head embedded in the same GGUF (e.g. [`ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF), or the `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` UD-Q4_K_XL quant). On hybrid models (GDN + attention, e.g. Qwen 3.6) the loop wraps each iteration in a recurrent-state checkpoint save/restore so partial draft rejections are recoverable. See README "Speculative decoding (MTP)" and `examples/mtp_speculative.exs` / `examples/mtp_benchmark.exs`.

  **Performance status (Apple Silicon):** the lack of speedup on Metal is intrinsic to the hardware, not the binding. Direct comparison on M1 Max with upstream's own `llama-server --spec-type draft-mtp`: 39.80 tok/s MTP vs 39.14 tok/s plain (1.02×) on Qwen 3.6 35B-A3B. Pair this with `n_draft: 1` and our binding reaches 39.7 tok/s at 79% acceptance for a ~1.06× speedup — see upstream [#23011](https://github.com/ggml-org/llama.cpp/issues/23011) and the Metal MTP follow-up [#23114](https://github.com/ggml-org/llama.cpp/pull/23114). On NVIDIA, the upstream-quoted 2× should hold with `n_draft: 3`.
- **Live MTP statistics** — `MTP.stats/1` returns a lock-free snapshot of speculative counters (`iters`, `drafts_generated`, `drafts_accepted`, `acceptance_rate`, `tokens_emitted`, `tokens_per_sec`, per-stage `timing_us`). Safe to call mid-stream from any process; optional `:emit_stats_every` flag streams periodic snapshots over the token channel.
- **Context options for speculative decoding** — `LlamaCppEx.Context.create/2` accepts `:ctx_type` (`:default` / `:mtp`) and `:n_rs_seq` (rollback snapshot count), plus new `Context.n_rs_seq/1` getter.

### Changed

- **llama.cpp submodule** — Updated from 834a24366 to 0253fb21f (31 commits), pulling in MTP and related speculative-decoding work.
  - **llama + spec**: MTP Support (#22673) — multi-token prediction speculative decoding, new `llama_context_type` enum (`LLAMA_CONTEXT_TYPE_DEFAULT` / `LLAMA_CONTEXT_TYPE_MTP`), new `llama_context_params.ctx_type` and `n_rs_seq` fields, new `llama_n_rs_seq()` API, new `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.
  - **spec**: allow partial seq_rm for GDN models for speculative decoding (#22400).

## v0.8.6

### Changed

- **llama.cpp submodule** — Updated from 1e5ad35d5 to 834a24366 (63 commits).
  - **model**: fix model type check for granite/llama3 and deepseek2/glm4.7 lite (#22870).
  - **spec**: parallel drafting support (#22838); update CLI arguments for better consistency (#22964).
  - **server**: accept `continue_final_message` flag for vLLM API compat (#23012); support continue generation on reasoning models (#22727); expose modalities to `/v1/models` (#22952); print warning when HTTP timeout exceeded (#22907).
  - **mtmd**: add MiMo v2.5 vision (#22883).
  - **CUDA**: handle `OW > 65535` in `im2col` (2D and 3D) (#22944); snake fusion hardening (#22912); directly include `cuda/iterator` (#22936); internal AllReduce kernel for CUDA provider (#22299).
  - **SYCL**: fix multi-GPU system RAM exhaustion by using Level Zero allocations (#21597); add OP `im2col_3d` (#22903).
  - **vulkan**: fix matmul integer pipeline selection (#23005); fix Windows performance regression on Intel GPU BF16 for Xe2+ (#22461); check shared memory size for MMQ shaders (#22693); support asymmetric FA in scalar/MMQ/coopmat1 paths (#22589).
  - **hexagon**: add unary tanh op (#22999); eliminate scalar VTCM loads via HVX splat helpers (#22993).
  - **opencl**: add q5_0/q5_1 MoE for Adreno (#22985); fix crash when warming up MoE on Adreno (#22876); add opt-in Adreno xmem F16xF32 GEMM for prefill (#22755); add q4_1 MoE for Adreno (#22856).
  - **ggml-webgpu**: enable NVIDIA self-hosted CI (#22976); subgroup-aware flash attn vec path (#23040); restrict subgroup-matrix path to compatible head dims (#23020); enable running gpt-oss-20b (#22906); precision fixes for multimodal (#22808); cast intermediate results to float to avoid half+half ambiguity (#22994); flush GPU profile timestamp before queryset overflow (#22995).
  - **ggml-cpu**: add IME2 instruction support for the SpacemiT backend (#22863).
  - **ggml-zendnn**: adaptive fallback to CPU backend for small batch sizes (#22681).
  - **ggml-virtgpu**: add a GHA build check (#22943); include missing mutex header (#22810).
  - **ggml**: bump version to 0.11.1; sync ggml.
  - **metal**: promote `mul_mv`/`mul_mm` batch divisors to function constants (#22711).
  - **backend sampling**: support returning post-sampling probs (#22622).
  - **unicode**: add Qwen3.5 non-backtracking tokenizer handler and regression test (#22110).
  - **logs**: reduce verbosity (#23021).
  - **download**: do not `exit()` on error (#23008).
  - **convert**: fix Pixtral 12B `--mistral-format` conversion (3 bugs) (#22981); add `split()` to `LoraTorchTensor` in LoRA converter (#22832); add image break token fallback (#22914).
  - **webui**: move static build output from repo code to HF Bucket (#22937); deduplicate model aliases (#22979); preserve system message on edit cancel (#22911); fix chat screen form box disappearing + autoscroll issues on WebKit (#22977); autoscroll detection (#23026); propagate version tag to WebUI asset download in self-hosted CI (#23051).
  - **examples**: add `llama-eval` (#21152); enable type check in `llama-eval` (#22988); update speculative-simple README (#22938).
  - **model-conversion**: add `causal-convert-mmproj` target (#22969).
  - **vendor/deps**: update cpp-httplib to 0.44.0 (#22919, #22888).
  - **build/CI**: revert docker intel compute-runtime to stable (#22968); validate model naming convention (#22680); bump `ty` to 0.0.35 (#22961).
  - **docs**: update OPENVINO.md (#22959); fix metrics endpoint description in server README (#22879).

## v0.8.5

### Changed

- **llama.cpp submodule** — Updated from eff06702b to 1e5ad35d5 (68 commits).
  - **model**: add sarvam_moe architecture (#20275); support Gemma4_26B_A4B_NVFP4 (#22804); add Mimo v2.5 (#22493); support sarashina2.2-vision-3b (#22103); don't crash on unsupported architecture (#22742).
  - **llama**: add option to save memory in device buffers, with new `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` flag (#22679); fix device state save/load (#22805); remove unnecessary seq_id check during state restore (#22797); add missing `ggml_backend_load_all()` call (#22752).
  - **common**: do not wrap raw strings in schema parser for tagged parsers (#22827); revert reasoning budget +inf logit bias (#22740); preserve media markers for typed-content templates (#22634); do not fit to unknown device memory (#22614); only load backends when required (#22290); fix missing-noreturn warnings on clang 21 (#22702).
  - **server**: support Vertex AI compatible API (#22545); router exposes child model info from `/v1/models` (#22683); validate `--tools` CLI argument against known tool names (#22538).
  - **mtmd**: support MiniCPM-V 4.6 (#22529); add granite-speech support (#22101); fix whisper audio tail truncation by exposing padded buffer to FFT (#22770).
  - **CUDA**: fuse snake activation (#22667); batch `out_prod` inner loop with `cublasSgemmStridedBatched` (#22651); lower-case PCI bus id, standardize for ggml (#22820).
  - **SYCL**: reduce allocation overhead during flash attention (#22732); BF16 support in `GET_ROWS` (#21391); Q5_K reorder MMVQ/dequant + Q8_0 reorder MMVQ (#22152); Battlemage AOT build via `spir64_gen` (#22147); add FILL, CUMSUM, DIAG, SOLVE_TRI, SSM_SCAN, GATED_DELTA_NET (#22149); non-contiguous input in PAD op (#22148).
  - **vulkan**: flash attention MMA / Tiles for MiMo-V2.5 (#22812); fix spv shadowing (#22760).
  - **hexagon**: HTP kernel for `GGML_OP_GATED_DELTA_NET` (#22837); l2 norm (#22816); process M-tail rows on HMX instead of HVX (#22724).
  - **opencl**: q4_0 MoE GEMM for Adreno (#22731); refactor Adreno q4_0 (#22335); use `CL_DEVICE_GLOBAL_MEM_SIZE` for `--fit` memory estimate (#22688); add opfilter regex for debugging (#22782).
  - **ggml-cpu**: fuse `RMS_NORM + MUL` on CPU backend (#22423); optimized risc-v q1_0 dot.
  - **ggml**: fast Walsh-Hadamard transform for KV rotation (#22631); bump version to 0.11.0; update `SCHED_DEBUG` output to use `ggml_op_desc()` (#22825).
  - **graph**: handle non-contiguous Q/K/V in `mul_mat_aux` (#22630).
  - **rpc**: use graph uid instead of graph cache (#22701).
  - **convert**: fix RuntimeError when stripping FP8 KV-cache scales (#22818); ignore non-language tensors for Gemma4Model (#22753); add `filter_tensors` method (#22597).
  - **gguf-py**: bump to 0.19.0 (#22664); migrate to PEP 621 and add uv support (#21907).
  - **webui**: import/export of settings (#22803); LLM title generation for agentic conversations (#22840); fix `?model=` URL param race in router mode (#22771); remove Google favicons (#22719); accessibility fixes (#22699, #22773).
  - **build/deps**: update BoringSSL to 0.20260508.0 (#22839); cpp-httplib 0.43.3 (#22686); upgrade default intel compute-runtime in docker (#22567); update Nix systems (#22869).

## v0.8.4

### Changed

- **llama.cpp submodule** — Updated from e48034dfc to eff06702b (12 commits).
  - **model**: move `load_hparams` and `load_tensors` to per-model definition (#22004)
  - **server**: implement `/models?reload=1` (#21848); add a simple `get_datetime` server tool (#22649)
  - **CUDA**: use fastdiv for batch index split in `get_rows` (#22650)
  - **vulkan**: delete dead `GGML_VK_MAX_NODES` def (#22621)
  - **ggml-webgpu**: add layer norm ops (#22406)
  - **kleidiai**: update to v1.24.0 and use release archive (#22549)
  - **common/autoparser**: fixes for newline handling / forced tool calls (#22654)
  - **webui**: fix circular dependency between `chat.service.ts` and `models.svelte.ts` (#22625); restore missing settings (#22666)
  - **examples**: refactor diffusion generation (#22590)
  - **docs**: update speculative decoding parameters after refactor (#22539)

## v0.8.3

### Changed

- **llama.cpp submodule** — Updated from b97ebdc98 to e48034dfc (14 commits).
  - **common**: determine generation prompt using longest common prefix (#22657)
  - **convert**: Mistral format yarn `apply_scale` support (#22612); apply Q/K RoPE permutation in NVFP4 repack path (#22611); disable uint types (#18908)
  - **CUDA**: fix device PCI bus ID de-dupe OOMing (ignoring other 3 GPUs entirely) (#22533)
  - **server**: avoid checkpoint data host copies (#22558)
  - **ggml-virtgpu**: fix circular dependency in headers (#22557)
  - **opencl**: Adreno optimization for MoE - MxFP4 (#22301)
  - **hexagon**: HMX flash attention (#22347)
  - **ggml**: bump version to 0.10.2; sync ggml; try fix win32 build

## v0.8.2

### Changed

- **llama.cpp submodule** — Updated from d77599234 to b97ebdc98 (18 commits).
  - **llama-quant**: fix `--tensor-type` when default `qtype` is overriden (#22572); add fast matmul iquants (#22504)
  - **CUDA**: fix tile FA kernel on Pascal (#22541)
  - **vulkan**: support asymmetric FA in coopmat2 path (#21753); add get/set tensor 2d functions (#22514)
  - **ggml-webgpu**: fix vectorized handling in mul-mat and mul-mat-id (#22578); add the upscale shader (#22419); improve performance of mat-vec and mat-mat for `MUL_MAT_ID` (#22464)
  - **hexagon**: enable non-contiguous row tensor support for unary ops (#22574)
  - **llama-mmap**: use `ftello`/`fseeko` (#22497)
  - **spec**: fix draft model checkpoints (#22521); fix vocab compat checks in spec example (#22426); fix argument typo (#22552)
  - **common**: check for null `getpwuid` in hf-cache (#22550)
  - **webui**: Spring Cleaning Refactor v1 (#22505)
  - **vendor**: update cpp-httplib to 0.43.2 (#22548)
  - **ci**: bump ty to 0.0.33 (#22535)
  - **scripts**: add `wc2wt.sh` - create worktree from current HEAD (#22513)

## v0.8.1

### Changed

- **llama.cpp submodule** — Updated from 98dc1418e to d77599234 (49 commits).
  - **server**: use `pos_next` instead of `n_tokens` for m-rope (#22439); (router) forward form-data to model server (#22118)
  - **CUDA**: fuse SSM_CONV + ADD(bias) + SILU (#22478); refactor fusion code (#22468); Blackwell native NVFP4 support (#22196); flash-attn support for DKQ=320/DV=256 with `ncols2=32` (#22286); better coalesce data-access for contiguous concat (#22330)
  - **ggml-cpu**: disable tiled matmul on AIX to fix page boundary segfault (#22293); append `xsmtvdotii` march for SpacemiT IME (#22317); re-enable fast `gelu_quick_f16` (#22339); optimize avx2 q6_k (#22345); SVE-tuned `gemm_q8_0_4x8_q8_0` kernel (#21916)
  - **ggml-webgpu**: fix FlashAttention support check (#22492); fix buffer aliasing for `ssm_scan` (#22456); add Q1_0 support (#22374)
  - **vulkan**: coalesce Q4_K/Q5_K scale loads (#21751); add barrier after `writetimestamp` (#21865)
  - **ggml**: bump version to 0.10.1; use 64-byte aligned tile buffers (#21058); skip already-registered backends and devices (#22296); revert to `-lm` linking instead of `find_library` (#22355); improve SPIR-V headers detection with `__has_include` (#21918)
  - **hexagon**: make vmem and buffer-size configurable (#22487); guard HMX clock request for v75+ platforms (#22377)
  - **spec**: discard last drafted token with low prob (#22506); refactor params (#22397)
  - **common**: do not pass prompt tokens to reasoning budget sampler (#22488); re-arm reasoning budget after DONE on new `<think>` (#22323); intentionally leak logger instance to fix hanging on Windows (#22273); fix missing exports in `llama-common` (#22340)
  - **chat**: fix handling of space in reasoning markers (#22353); handle gemma4 parsing edge cases (#22420)
  - **convert**: add support for Nemotron Nano 3 Omni (#22481); remove `input_scale` for dequantized fp8 modelopt (#22356)
  - **model**: remove duplicate `wo_s` scale after `build_attn` (Qwen3, LLaMA) (#22421)
  - **opencl**: add iq4_nl support (#22272)
  - **CANN**: add new ops, optimize existing ops (#21204)
  - **TP**: fix delayed AllReduce + zero-sized slices (#22489)
  - **rpc**: fix rpc-server cache on Windows (#22394)
  - **download**: prefer q8_0 when q4_k not available (#22428)
  - **webui**: fix slow mic stop and WAV encode (#22480); add Server tools (#21237)

## v0.8.0

### Changed

- **llama.cpp submodule** — Updated from 550d684bd to 98dc1418e (30 commits).
  - **server**: fix swa-full logic (#22288); rename debug tags to match `--cache-idle-slots` (#22292); `convert_anthropic_to_oai` also copy `chat_template_kwargs` (#22154); fix heap-buffer-overflow from negative `n_discard` (CVE-2026-21869) (#22267); (anthropic API) fix prefix caching (#21793)
  - **CUDA**: reduce MMQ stream-k overhead (#22298)
  - **metal**: optimize Metal Tensor API usage for `GGML_OP_MUL_MAT` (#20962); print GPU description (#22318)
  - **SYCL**: optimize Q4_0 `mul_mat` for Arc770, add scripts (#22291); fix build number for SYCL release (#22283)
  - **hexagon**: bump HMX frequency to max corner (#22334); use DIRID 13 in `libggml-htp.inf` for modern InfVerif (#22306); add SOLVE_TRI op (#21974); add basic and extended op profiling (#22269)
  - **ggml-webgpu**: support for SSM_SCAN and disable `set_rows` error checking (#22327); enable `FLASH_ATTN_EXT` on browser without subgroup matrix (#22199)
  - **llama-quant**: default ftype param `Q5_1` → `Q8_0` (#20828)
  - **spec**: fix vocab compat checks (#22358)
  - **parser**: fix structured output bug (#22302)
  - **common**: fix jinja warnings with clang 21 (#22313)
  - **vendor**: update LibreSSL to 4.3.1 (#22285)

## v0.7.9

### Changed

- **llama.cpp submodule** — Updated from 45cac7ca7 to 550d684bd (69 commits).
  - **server**: Enable transcriptions API for LFM2-Audio (#22000); ignore reasoning content from transcription api (#21905); allow cancel loading model (#21814); fix hardcoded proxy connection timeout in router mode (#22003)
  - **metal**: fix event synchronization (#22260); workaround macOS GPU interactivity watchdog (#22216)
  - **ggml-base**: use `MATH_LIBRARY` variable instead of hardcoded `m` (#22239)
  - **ggml**: bump version to 0.10.0
  - **SYCL**: update oneapi 2025.3.3, separate SYCL build, release Ubuntu 24 package (#22078); fused MoE `mul_mat_vec_q` for TG (#21920); improve `mul_mat_id` memory efficiency and add BF16 fast path (#22119)
  - **CUDA**: fuse relu + sqr (#22249); flush legacy pool on OOM and retry (#22155)
  - **HIP**: flip `GGML_HIP_GRAPHS` to default on (#22254)
  - **ggml-webgpu**: add support for im2col (#22259); implement async tensor api and event api (#22099); fused RMS_NORM + MUL (#21983); conv2d kernels (#21964); reset CPU/GPU profiling time when freeing context (#22050)
  - **vulkan**: Support F16 OP_FILL (#22177)
  - **hexagon**: add support for FILL op (#22198); DAIG op (#22195); fix missing v79 entry in `libggml-htp.inf` (#22194)
  - **mtmd**: also support `LLAMA_ROPE_TYPE_NONE` (#22242); update HunyuanVL vision-language model support (#22037); correct `mtmd_decode_use_mrope()` (#22188); add support for Reka Edge 2603 (#21616)
  - **chat**: fix `parallel_tool_calls` default setting based on model capabilities, add tests for parallel tool calls and structured outputs (#22217)
  - **common**: refactoring sampler parameters (#22233); refactor, move all conversion functions to common, add tests (#20690)
  - **speculative**: add checkpoint support (#22227); reset `i_last` when low acceptance streak occurs (#22168); `--spec-default` arg (#22223)
  - **convert**: handle ModelOpt produced mixed precision model during convert to GGUF (#22247)
  - **openvino**: driver setup, CI split, thread safety, and NPU optimizations (#21944)
  - **llama-ext**: fix exports (#22202)
  - **vendor**: update cpp-httplib to 0.43.1 (#22143)

### Fixed

- **build**: Added `-DLLAMA_OPENSSL=OFF` to suppress upstream HTTPS dependency pulled in by the new `LLAMA_OPENSSL=ON` default.

## v0.7.8

### Changed

- **llama.cpp submodule** — Updated from 30dce2cf2 to 45cac7ca7 (7 commits).
  - **model**: Gemma4 model type detection (#22027)
  - **mtmd**: add missing struct tag (#22023)
  - **libs**: rename `libcommon` → `libllama-common` (#21936)
  - **CUDA**: use LRU based eviction for cuda graphs (#21611)
  - **OpenCL**: refactor q8_0 `set_tensor` and `mul_mat` host side dispatch for Adreno (#21938)
  - **ggml-webgpu**: fix compiler warnings and refactor FlashAttention encoding (#21052)
  - **ci**: add android arm64 build and release (#21647)

## v0.7.7

### Changed

- **llama.cpp submodule** — Updated from 408225bb1 to 30dce2cf2 (18 commits).
  - **model**: using single llm_build per arch (#21970), refactor QKV into common `build_qkv` and `create_tensor_qkv` helpers (#21245), support NVFP4 tensors for Gemma4 (#21971)
  - **cli**: use `get_media_marker` (#22017)
  - **server**: tests fetch random media marker via `/apply-template` (#21980)
  - **convert**: fix NemotronH config parsing (#21664)
  - **ggml**: add `graph_reused` (#21764)
  - **ggml-cpu**: 128-bit RVV implementation for Quantization Vector Dot (#20633), SIMD gemm kernel for RISC-V vector extension (#20627)
  - **Metal**: implement ROLL op (#21946)
  - **OpenCL**: add q5_K gemm and gemv kernels for Adreno (#21595)
  - **SYCL**: fix Q8_0 reorder garbage on 2nd prompt + crash on full VRAM (#21638)
  - **hexagon**: optimize HMX matmul operations (#21071)
  - **ggml-webgpu**: compute pass batching and remove profiling overhead (#21873)
  - **cmake**: use glob to collect `src/models` sources (#22005)
  - **ci**: use ggml-org/ccache-action on RISC-V (#21632)
  - **devops**: add spirv-headers to nix (#21965)

## v0.7.6

### Changed

- **llama.cpp submodule** — Updated from a8bad3842 to 408225bb1 (28 commits).
  - **server**: use random media marker (#21962), support OAI `/v1/audio/transcriptions` API (#21863)
  - **chat**: dedicated DeepSeek v3.2 parser + "official" template (#21785)
  - **autoparser**: support case of JSON_NATIVE with per-call markers (test case: Reka-Edge) (#21892)
  - **common**: handle gemma4 parsing edge cases (#21760), skip reasoning budget sampler when no budget is requested (#21870)
  - **mtmd**: add `mtmd_image_tokens_get_decoder_pos()` API (#21851)
  - **llama**: read `n_ctx` back after making `llama_context` (#21939)
  - **CUDA**: Q1_0 initial backend (#21629), require explicit opt-in for P2P access (#21910), manage NCCL communicators in context (#21891)
  - **Metal**: fix FA support logic (#21898), add XIELU unary op (#20802)
  - **Vulkan**: optimize im2col (#21713), support GGML_TYPE_NVFP4 (#21455), programmatically add RoundingModeRTE to all shaders when the device supports it (#21572)
  - **ggml-webgpu**: fix dequantization helpers to not pass in pointers (#21872), update register tiling matmul to use f32 accumulation (#21644)
  - **ggml**: remove `ggml-ext.h` (#21869), fix ARM NEON nvfp4 dot product on non-dotprod targets (#21559)
  - **hexagon**: optimization for HMX mat_mul (#21554)
  - **rpc**: add native RDMA transport for RPC backend (RoCEv2) (#20590)
  - **vendor**: update BoringSSL to 0.20260413.0 (#21881)
  - **cmake**: fix CMP0194 warning on Windows with MSVC (#21630)
  - **ci**: re-enable mac workflows (#21894), disable test-backend-ops on Vulkan llvmpipe run and restore default timeout (#21901)

## v0.7.5

### Changed

- **llama.cpp submodule** — Updated from 073bb2c20 to a8bad3842 (18 commits).
  - **mtmd**: add Gemma 4 audio conformer encoder support (#21421), qwen3 audio support (qwen3-omni and qwen3-asr) (#19441), use causal attn for gemma 4 audio (#21824), fix crash when sending image under 2x2 pixels (#21711)
  - **Vulkan**: Flash Attention DP4A shader for quantized KV cache (#20797)
  - **CUDA**: limit DeviceSegmentedSort to immediate mode (#21718), skip compilation of superfluous FA kernels (#21768)
  - **common**: add download cancellation and temp file cleanup (#21813)
  - **server**: expose build_info in router mode (#21835)
  - **convert**: force f16 or f32 on step3-vl conv weights (#21646)

## v0.7.4

### Changed

- **llama.cpp submodule** — Updated from d12cc3d1c to 073bb2c20 (42 commits).
  - **model**: make Gemma 4 shared-KV tail attn_k tensors optional on load (#21739), fix multimodal padding token for gemma3n/gemma4 (#21625)
  - **mtmd**: add MERaLiON-2 multimodal audio support (#21756), support dots.ocr (#17575)
  - **common**: better align to the updated official gemma4 template (#21704), enable reasoning budget sampler for gemma4 (#21697), add callback interface for download progress (#21735), fix when loading cached HF models with unavailable API (#21670), mark `--split-mode tensor` as experimental (#21684), add fluidity to the progress bar (#21671), fix ambiguous grammar rule in gemma4 (#21661), simplify autoparser tagged parser rules (#21216), skip non-primary GGUF split files when selecting model (#21633)
  - **server**: ignore `--alias` when using `--models-preset` (#21380), fix grammar commandline args (#21543)
  - **jinja**: support `ensure_ascii=true`, string repetition and int/float self-filtering (#21623)
  - **vocab**: add gemma4 tokenizer tests, fix edge case (#21534)
  - **structured output**: fix broken structured output when using `$refs` in json_schema (#21699)
  - **ggml**: backend-agnostic tensor parallelism (experimental) (#19378), fix missing GGML_TYPE_Q1_0 cases (#21716), check return value of CUB calls in argsort and top-k (#21676)
  - **CUDA**: fuse muls (#21665), also store `node->src` ne/nb for graph equality (#21736)
  - **Metal**: add missing mm-id specializations for q1_0 (#21662)
  - **Vulkan**: support Q1_0 (#21539), unify type macros to use Vx instead of _VECx (#21605)
  - **SYCL**: add flash-attn support for head size 512 (#21654)
  - **HIP**: add CDNA4 (gfx950) architecture support for MI350X/MI355X (#21570)
  - **OpenCL**: add basic support for q5_k (#21593)
  - **WebGPU**: support non-square subgroup matrix configs for Intel GPUs (#21669), address quantization precision and backend lifecycle management (#21521)
  - **hexagon**: add support for linux on snapdragon (#21707), improved Op queuing, buffer and cache management (#21705)
  - **TP**: fix Qwen 3 Next data split (#21732)
  - **webui**: static build output improvements (#21667), add "Send message on Enter" setting (#21577), add option to pre-encode conversation for faster next turns (#21034), fix Model Selector choice sync (#21628)

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
