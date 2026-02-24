defmodule LlamaCppEx.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/nyo16/llama_cpp_ex"

  def project do
    [
      app: :llama_cpp_ex,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: &make_env/0,
      make_clean: ["clean"],
      make_precompiler: {:nif, LlamaCppEx.Precompiler},
      make_precompiler_url:
        "https://github.com/nyo16/llama_cpp_ex/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "llama_cpp_ex_nif",
      make_precompiler_priv_paths: ["llama_cpp_ex_nif.so"],
      make_precompiler_nif_versions: [versions: ["2.17", "2.18"]],
      make_force_build: System.get_env("LLAMA_BACKEND") in ["cuda", "vulkan"],
      description: description(),
      package: package(),
      name: "LlamaCppEx",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:fine, "~> 0.1", runtime: false},
      {:telemetry, "~> 1.0"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:benchee, "~> 1.0", only: :bench, runtime: false}
    ]
  end

  defp description do
    "Elixir bindings for llama.cpp — run LLMs locally with Metal, CUDA, Vulkan, or CPU acceleration."
  end

  defp package do
    [
      name: "llama_cpp_ex",
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "llama.cpp" => "https://github.com/ggml-org/llama.cpp"
      },
      files: ~w(
        lib c_src Makefile mix.exs README.md CHANGELOG.md LICENSE .formatter.exs
        checksum.exs
      )
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "docs/architecture.md",
        "docs/cross-platform-builds.md",
        "docs/adr/001-cpp-nif-over-rustler.md",
        "docs/adr/002-fine-for-nif-ergonomics.md",
        "docs/adr/003-static-linking.md",
        "docs/adr/004-streaming-via-enif-send.md",
        "docs/adr/005-batching-architecture.md",
        "docs/adr/006-continuous-batching.md"
      ],
      groups_for_extras: [
        "Architecture Decision Records": ~r/docs\/adr\/.*/
      ],
      groups_for_modules: [
        "High-Level API": [LlamaCppEx],
        "Core Modules": [
          LlamaCppEx.Model,
          LlamaCppEx.Context,
          LlamaCppEx.Sampler,
          LlamaCppEx.Tokenizer,
          LlamaCppEx.Chat,
          LlamaCppEx.Embedding,
          LlamaCppEx.Server
        ],
        Internal: [LlamaCppEx.NIF]
      ]
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
