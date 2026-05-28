# Smoke tests load real GGUF models and run inference; they are excluded by
# default. Run them explicitly with model paths set, e.g.:
#
#   LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf \
#   LLAMA_SMOKE_EMB_MODEL=/path/to/embedding-model.gguf \
#     mix test --include smoke
#
ExUnit.start(exclude: [:smoke])
