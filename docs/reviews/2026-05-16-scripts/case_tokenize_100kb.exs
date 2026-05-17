Code.require_file("/tmp/llama_review/_common.exs")
m = EdgeCase.model()
big = String.duplicate("a ", 50_000)
EdgeCase.report("tokenize 100KB", fn ->
  case LlamaCppEx.Tokenizer.encode(m, big) do
    {:ok, t} -> {:ok, "n_tokens=#{length(t)}"}
    other -> other
  end
end)
