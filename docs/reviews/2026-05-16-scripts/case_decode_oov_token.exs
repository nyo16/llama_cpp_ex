Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("decode [10_000_000]", fn -> LlamaCppEx.Tokenizer.decode(m, [10_000_000]) end)
