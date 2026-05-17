Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("sampler temp:-1.0", fn -> LlamaCppEx.Sampler.create(m, temp: -1.0) end)
