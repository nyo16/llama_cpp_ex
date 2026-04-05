defmodule Bench.Helpers do
  @moduledoc false

  def setup(opts \\ []) do
    model_path = System.get_env("LLAMA_MODEL_PATH") || raise "Set LLAMA_MODEL_PATH"
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, -1)

    :ok = LlamaCppEx.init()
    {:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: n_gpu_layers)
    model
  end

  def start_server(opts \\ []) do
    model_path = System.get_env("LLAMA_MODEL_PATH") || raise "Set LLAMA_MODEL_PATH"
    n_parallel = Keyword.get(opts, :n_parallel, 4)

    server_opts = [
      model_path: model_path,
      n_gpu_layers: Keyword.get(opts, :n_gpu_layers, -1),
      n_parallel: n_parallel,
      n_ctx: Keyword.get(opts, :n_ctx, 4096),
      temp: 0.0
    ]

    # Pass through new options
    server_opts =
      server_opts
      |> maybe_add(opts, :cache_prompt)
      |> maybe_add(opts, :batch_strategy)

    {:ok, server} = LlamaCppEx.Server.start_link(server_opts)
    server
  end

  def prompts do
    %{
      "short" => "The capital of France is",
      "medium" => String.duplicate("The quick brown fox jumps over the lazy dog. ", 10),
      "long" => String.duplicate("The quick brown fox jumps over the lazy dog. ", 50)
    }
  end

  defp maybe_add(server_opts, source_opts, key) do
    case Keyword.fetch(source_opts, key) do
      {:ok, val} -> Keyword.put(server_opts, key, val)
      :error -> server_opts
    end
  end
end
