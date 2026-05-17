Code.require_file("/tmp/llama_review/_common.exs")
case EdgeCase.mtp_model() do
  nil -> IO.puts("RESULT[skip]: MTP model not available")
  m -> EdgeCase.report("mtp n_draft:0", fn -> LlamaCppEx.MTP.init(m, n_draft: 0, n_ctx: 1024) end)
end
