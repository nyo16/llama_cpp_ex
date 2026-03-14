# Usage: LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/streaming.exs

model_path =
  System.get_env("LLAMA_MODEL_PATH") ||
    raise "Set LLAMA_MODEL_PATH to a .gguf model file"

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

IO.puts("Streaming tokens...")
IO.puts("---")

model
|> LlamaCppEx.stream("Once upon a time in a land of functional programming,",
  max_tokens: 256,
  temp: 0.8
)
|> Enum.each(&IO.write/1)

IO.puts("\n---")
IO.puts("Done.")
