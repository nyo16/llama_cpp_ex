Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("sampler temp:100", fn -> LlamaCppEx.Sampler.create(m, temp: 100.0) end)
