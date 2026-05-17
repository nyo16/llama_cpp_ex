Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("max_tokens:-1", fn -> LlamaCppEx.generate(m, "Hello", max_tokens: -1) end)
