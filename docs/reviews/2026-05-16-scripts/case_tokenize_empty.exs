Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
EdgeCase.report("tokenize empty", fn -> LlamaCppEx.Tokenizer.encode(m, "") end)
