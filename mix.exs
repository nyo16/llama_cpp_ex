defmodule LlamaCppEx.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :llama_cpp_ex,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: &make_env/0,
      make_clean: ["clean"]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:fine, "~> 0.1", runtime: false}
    ]
  end

  defp make_env do
    env = %{"FINE_INCLUDE_DIR" => Fine.include_dir()}

    env =
      case System.get_env("LLAMA_BACKEND") do
        nil -> env
        backend -> Map.put(env, "LLAMA_BACKEND", backend)
      end

    case System.get_env("LLAMA_CMAKE_ARGS") do
      nil -> env
      args -> Map.put(env, "LLAMA_CMAKE_ARGS", args)
    end
  end
end
