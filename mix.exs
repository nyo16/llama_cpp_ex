defmodule LlamaCppEx.Precompiler do
  @moduledoc false

  # Linux CUDA artifacts are published per CUDA major version because the NIF
  # links libcudart, libcublas and libcublasLt dynamically and their sonames are
  # major-versioned: libcudart.so.12 against a CUDA 13 install does not resolve,
  # and there is no compatibility shim in either direction. So the variant has
  # to be part of the target name -- one "linux CUDA" artifact cannot exist.
  #
  # Newest first: a host with both toolkits installed should get the newer one.
  @cuda_majors ["13", "12"]

  # Only x86_64 Linux gets CUDA variants today. aarch64 Linux (DGX Spark and
  # friends) still resolves to no artifact and source-builds, which the
  # Makefile's toolkit discovery now handles; adding it here is a matrix entry
  # in .github/workflows/precompile.yml plus this list.
  @cuda_targets for major <- @cuda_majors, do: "x86_64-linux-gnu-cu#{major}"

  @all_targets ["aarch64-apple-darwin", "x86_64-linux-gnu"] ++ @cuda_targets

  # Set by each CUDA leg of the precompile workflow. Detection below deliberately
  # refuses to name a CUDA target on a machine with no driver, which is exactly
  # what a release runner is, so the build has to state its own variant.
  # Also the escape hatch for a host whose layout defeats the probe: "cu12",
  # "cu13", or "none" to force the CPU artifact.
  @variant_env "LLAMA_CUDA_VARIANT"

  # Where a CUDA runtime shows up when ldconfig has nothing to say -- a container
  # with no ldconfig cache, or an install that was never registered.
  @cuda_lib_globs [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-*/lib64",
    "/usr/local/cuda-*/targets/*/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64"
  ]

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
      system_arch =~ ~r/x86_64.*linux.*gnu/ -> {:ok, "x86_64-linux-gnu" <> cuda_suffix()}
      true -> {:error, "unsupported target: #{system_arch}"}
    end
  end

  @doc false
  # Exposed for tests: the probe is pure over the two facts it looks up, so the
  # interesting cases can be exercised without a CUDA install.
  def cuda_suffix(env \\ &System.get_env/1, present? \\ &library_present?/1) do
    case env.(@variant_env) do
      nil -> detect_cuda_suffix(present?)
      "" -> ""
      "none" -> ""
      "cu" <> major when major in @cuda_majors -> "-cu#{major}"
      other -> raise ArgumentError, bad_variant_message(other)
    end
  end

  defp bad_variant_message(value) do
    allowed = Enum.map_join(@cuda_majors, ", ", &"cu#{&1}")
    "#{@variant_env}=#{inspect(value)} is not a known CUDA variant (#{allowed}, none)"
  end

  # Two conditions, both required. The runtime libraries are what the artifact
  # links against, and the driver is what -lcuda resolves to at load time: a
  # machine with the toolkit but no driver cannot dlopen a CUDA build at all, so
  # handing it one would turn a working CPU install into a NIF that fails to
  # load. nvcc is deliberately not consulted -- running a CUDA build needs no
  # compiler, and plenty of GPU hosts have no toolkit installed.
  defp detect_cuda_suffix(present?) do
    if present?.("libcuda.so.1") do
      Enum.find_value(@cuda_majors, "", fn major ->
        if present?.("libcudart.so.#{major}"), do: "-cu#{major}"
      end)
    else
      ""
    end
  end

  defp library_present?(soname) do
    ldconfig_lists?(soname) or on_disk?(soname)
  end

  defp ldconfig_lists?(soname) do
    # ldconfig lives in /sbin, which is routinely off a non-root PATH.
    case Enum.find(
           ["ldconfig", "/sbin/ldconfig", "/usr/sbin/ldconfig"],
           &System.find_executable/1
         ) do
      nil ->
        false

      ldconfig ->
        case System.cmd(ldconfig, ["-p"], stderr_to_stdout: true) do
          {output, 0} -> String.contains?(output, soname)
          _ -> false
        end
    end
  catch
    # An ldconfig that is present but unusable is a "no", never a build failure.
    _, _ -> false
  end

  defp on_disk?(soname) do
    Enum.any?(@cuda_lib_globs, fn glob ->
      glob |> Path.join(soname) |> Path.wildcard() |> Enum.any?()
    end)
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

  @version "0.8.43"
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
        "docs/dgx-spark.md",
        "docs/adr/001-cpp-nif-over-rustler.md",
        "docs/adr/002-fine-for-nif-ergonomics.md",
        "docs/adr/003-static-linking.md",
        "docs/adr/004-streaming-via-enif-send.md",
        "docs/adr/005-batching-architecture.md",
        "docs/adr/006-continuous-batching.md",
        "docs/adr/007-prefix-caching.md",
        "docs/adr/008-batching-strategies.md",
        "docs/adr/009-multi-model-manager.md",
        "docs/examples.md",
        "docs/multi-gpu.md",
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
  #   LLAMA_BACKEND      auto | metal | cuda | vulkan | cpu
  #   LLAMA_CMAKE_ARGS   extra flags appended to the llama.cpp cmake invocation
  #   LLAMA_PORTABLE     1 to drop -march=native, set by the precompile workflow
  #   LLAMA_CUDA_NCCL    1 to build and link ggml's NCCL multi-GPU collectives,
  #                      which also makes libnccl.so.2 a load-time requirement
  #   LLAMA_CPU_ARM_ARCH the ARM architecture string for ggml's CPU backend, for
  #                      hosts where -mcpu=native degrades silently (GB10 on GCC
  #                      13.3). Requires LLAMA_CUDA_ARCH on a CUDA build; the
  #                      Makefile errors otherwise, because reaching this flag
  #                      needs GGML_NATIVE=OFF and that turns one CUDA arch into
  #                      seven.
  #   LLAMA_CUDA_ARCH    CMAKE_CUDA_ARCHITECTURES, e.g. 121a-real for GB10
  #   LLAMA_RPC          1 to build the ggml RPC backend, which lets a model's
  #                      layers live on another host. Off by default: it is a
  #                      networked surface and a protocol version coupling.
  #   LLAMA_RPC_RDMA     1 (default) to use RDMA for the RPC transport on Linux.
  #                      Declared rather than auto-detected, and paired with
  #                      -libverbs on the link line.
  #
  # None of these need to appear in make_force_build: each is part of the
  # Makefile's build-directory key, so changing one lands in a different tree
  # with its own CMakeCache.txt and rebuilds on its own. That is a better answer
  # than forcing a rebuild, because switching back is still a cache hit.
  @make_env_passthrough [
    "LLAMA_BACKEND",
    "LLAMA_CMAKE_ARGS",
    "LLAMA_PORTABLE",
    "LLAMA_CUDA_NCCL",
    "LLAMA_CPU_ARM_ARCH",
    "LLAMA_CUDA_ARCH",
    "LLAMA_RPC",
    "LLAMA_RPC_RDMA"
  ]

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
