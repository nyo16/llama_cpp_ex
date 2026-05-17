defmodule EdgeCase do
  def model do
    model_path =
      System.get_env("MODEL_PATH") ||
        Path.expand("~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    if !File.exists?(model_path) do
      IO.puts(:stderr, "model not found: #{model_path}")
      System.halt(1)
    end

    :ok = LlamaCppEx.init()
    {:ok, model} = LlamaCppEx.Model.load(model_path, n_gpu_layers: 999)
    model
  end

  def mtp_model do
    p = System.get_env("LLAMA_MTP_MODEL_PATH") || Path.expand("~/Downloads/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf")
    if !File.exists?(p), do: nil, else: (
      :ok = LlamaCppEx.init()
      {:ok, m} = LlamaCppEx.Model.load(p, n_gpu_layers: 999)
      m
    )
  end

  def report(name, fun) do
    result =
      try do
        fun.()
      rescue
        e -> {:rescue, e.__struct__, Exception.message(e)}
      catch
        kind, reason -> {:catch, kind, reason}
      end

    IO.puts("RESULT[#{name}]: #{inspect(result, limit: 100)}")
  end
end
