# Tests that load real GGUF models and run inference are excluded by default,
# behind four opt-in tags. Each tag names the model it needs:
#
#   :smoke      — generation/chat/grammar/server paths; needs LLAMA_SMOKE_GEN_MODEL
#   :embeddings — embedding paths;                      needs LLAMA_SMOKE_EMB_MODEL
#   :mtp        — MTP speculative decoding;             needs LLAMA_SMOKE_MTP_MODEL
#   :slow       — long-running comparison matrices (F16 vs Q8_0 KV cache);
#                 needs LLAMA_SMOKE_GEN_MODEL
#
# `--include` beats `--exclude` in ExUnit, so the tags are independent: opt into
# exactly the ones whose model you have. The helper `LlamaCppEx.TestModels`
# raises with the env var name when an included test has no model to load.
#
#   GGML_METAL_NO_RESIDENCY=1 \
#   LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf \
#     mix test --include smoke
#
#   GGML_METAL_NO_RESIDENCY=1 \
#   LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf \
#   LLAMA_SMOKE_EMB_MODEL=/path/to/embedding-model.gguf \
#     mix test --include smoke --include embeddings
#
#   GGML_METAL_NO_RESIDENCY=1 \
#   LLAMA_SMOKE_MTP_MODEL=/path/to/mtp-model.gguf \
#     mix test --include mtp
#
# `GGML_METAL_NO_RESIDENCY=1` is only needed on Metal, and only to keep the VM
# from aborting *after* the suite has passed:
#
#   ggml-metal-device.m:622: GGML_ASSERT([rsets->data count] == 0) failed
#
# llama.cpp's Metal device is owned by a function-local `static std::vector`, so
# it is destroyed by `__cxa_finalize_ranges` after the BEAM calls `exit(3)`, and
# its destructor asserts that the global `MTLResidencySet` collection is empty.
# The BEAM does not promise that NIF resource destructors have run by then, so a
# model or context still holding Metal buffers trips the assert and the run exits
# 134 with a green result already printed (measured at 3 of 12 full-suite runs).
# Setting the variable stops the collection from being allocated at all, which
# removes the assert instead of racing it; residency sets are only an OS
# memory-residency hint, so nothing under test changes.
#
# It has to come from the shell: `System.put_env/2` does not reach the C
# `getenv` that ggml reads, and the library deliberately does not set it either
# — production keeps the upstream default.
Code.require_file("support/test_models.exs", __DIR__)
Code.require_file("support/test_slots.exs", __DIR__)

ExUnit.start(exclude: [:smoke, :embeddings, :slow, :mtp])
