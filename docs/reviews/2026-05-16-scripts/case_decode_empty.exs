Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("decode empty", fn -> LlamaCppEx.Tokenizer.decode(m, []) end)
