Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("n_ctx:0", fn -> LlamaCppEx.Context.create(m, n_ctx: 0) end)
