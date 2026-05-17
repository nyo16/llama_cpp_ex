Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("sampler top_k:0", fn -> LlamaCppEx.Sampler.create(m, top_k: 0, temp: 1.0) end)
