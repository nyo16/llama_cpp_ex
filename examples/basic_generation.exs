# Usage: LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/basic_generation.exs

model_path =
  System.get_env("LLAMA_MODEL_PATH") ||
    raise "Set LLAMA_MODEL_PATH to a .gguf model file"

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

IO.puts("Model: #{LlamaCppEx.Model.desc(model)}")
IO.puts("---")

{:ok, text} =
  LlamaCppEx.generate(model, "Explain what Elixir is in one paragraph:",
    max_tokens: 256,
    temp: 0.7,
    seed: 42
  )

IO.puts(text)
