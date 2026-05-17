Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("decode [-1]", fn -> LlamaCppEx.Tokenizer.decode(m, [-1]) end)
