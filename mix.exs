defmodule LlamaCppEx.Precompiler do
  @moduledoc false

  @all_targets ["aarch64-apple-darwin", "x86_64-linux-gnu"]

  def all_supported_targets(:fetch), do: @all_targets

  def all_supported_targets(:compile) do
    case current_target() do
      {:ok, target} -> [target]
      _ -> []
    end
  end

  def current_target do
    system_arch = to_string(:erlang.system_info(:system_architecture))

    cond do
      system_arch =~ ~r/aarch64.*apple.*darwin/ -> {:ok, "aarch64-apple-darwin"}
      system_arch =~ ~r/x86_64.*linux.*gnu/ -> {:ok, "x86_64-linux-gnu"}
      true -> {:error, "unsupported target: #{system_arch}"}
    end
  end

  def build_native(args), do: ElixirMake.Precompiler.mix_compile(args)

  def precompile(args, _target) do
    case ElixirMake.Precompiler.mix_compile(args) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  def unavailable_target(_target), do: :compile
end

defmodule LlamaCppEx.MixProject do
  use Mix.Project

  @version "0.8.42"
  @source_url "https://github.com/nyo16/llama_cpp_ex"

  def project do
    [
      app: :llama_cpp_ex,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: Mix.compilers() ++ [:elixir_make],
      make_env: &make_env/0,
      make_clean: ["clean"],
      make_precompiler: {:nif, LlamaCppEx.Precompiler},
      make_precompiler_url:
        "https://github.com/nyo16/llama_cpp_ex/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "llama_cpp_ex_nif",
      make_precompiler_priv_paths: ["llama_cpp_ex_nif.so"],
      # Verified against erts/emulator/beam/erl_nif.h in the OTP source: OTP 25
      # reports NIF 2.16, OTP 26/27/28 report 2.17, and OTP 29 reports 2.18.
      # Only 2.17 and 2.18 artifacts are built, so the precompiled floor is
      # OTP 26. Declaring a "2.16" entry would be worse than omitting it: no
      # such artifact exists, so `mix elixir_make.checksum --all` could not
      # vouch for it. On OTP 25 elixir_make finds no matching artifact and
      # recovers with a source build (see compile.elixir_make.ex), which the
      # Makefile's llama.cpp clone fallback now makes possible.
      make_precompiler_nif_versions: [versions: ["2.17", "2.18"]],
      make_force_build: System.get_env("LLAMA_BACKEND") != nil,
      description: description(),
      package: package(),
      name: "LlamaCppEx",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),
      # Keep the PLT where CI caches it (priv/plts). Without this dialyxir
      # writes under _build and the cache never hits.
      dialyzer: [
        plt_local_path: "priv/plts",
        plt_core_path: "priv/plts"
      ],
      test_coverage: [summary: [threshold: 0]],
      # test/support/test_models.exs is a helper module that test_helper.exs
      # loads with Code.require_file/2, not a test file. Elixir 1.20 warns about
      # unmatched files under test/ unless they are filtered out here.
      test_ignore_filters: [~r{^test/support/}]
    ]
  end

  def application do
    [extra_applications: [:logger], mod: {LlamaCppEx.Application, []}]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:fine, "~> 0.1", runtime: false},
      {:telemetry, "~> 1.0"},
      {:ecto, "~> 3.0", optional: true},
      {:req, "~> 0.5 or ~> 0.6", optional: true},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.0", only: :bench, runtime: false},
      {:benchee_html, "~> 1.0", only: :bench, runtime: false}
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
        "Changelog" => "https://hexdocs.pm/llama_cpp_ex/changelog.html",
        "llama.cpp" => "https://github.com/ggml-org/llama.cpp"
      },
      # vendor/llama.cpp is deliberately not shipped: it would add hundreds of
      # megabytes to every release. .gitmodules is shipped instead so the
      # Makefile can read the upstream URL and clone the pinned commit when a
      # source build needs it.
      files: ~w(
        lib c_src Makefile mix.exs README.md CHANGELOG.md LICENSE .formatter.exs
        checksum.exs .gitmodules
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
        "docs/adr/006-continuous-batching.md",
        "docs/adr/007-prefix-caching.md",
        "docs/adr/008-batching-strategies.md",
        "docs/examples.md",
        "docs/performance.md",
        "docs/release-guide.md"
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
          LlamaCppEx.Grammar,
          LlamaCppEx.Schema,
          LlamaCppEx.Server,
          LlamaCppEx.Hub
        ],
        "Batching Strategies": [
          LlamaCppEx.Server.BatchStrategy,
          LlamaCppEx.Server.Strategy.DecodeMaximal,
          LlamaCppEx.Server.Strategy.PrefillPriority,
          LlamaCppEx.Server.Strategy.Balanced
        ],
        Internal: [LlamaCppEx.NIF]
      ]
    ]
  end

  # Environment variables forwarded to the Makefile. System.cmd merges :env into
  # the inherited environment anyway, but listing them keeps the build's input
  # contract in one place:
  #
  #   LLAMA_BACKEND    auto | metal | cuda | vulkan | cpu
  #   LLAMA_CMAKE_ARGS extra flags appended to the llama.cpp cmake invocation
  #   LLAMA_PORTABLE   1 to drop -march=native, set by the precompile workflow
  @make_env_passthrough ["LLAMA_BACKEND", "LLAMA_CMAKE_ARGS", "LLAMA_PORTABLE"]

  defp make_env do
    base = %{"FINE_INCLUDE_DIR" => Fine.include_dir()}

    Enum.reduce(@make_env_passthrough, base, fn key, env ->
      case System.get_env(key) do
        nil -> env
        value -> Map.put(env, key, value)
      end
    end)
  end
end
