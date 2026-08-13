defmodule LlamaCppEx.MakefileArchFlagsTest do
  use ExUnit.Case, async: true

  # LLAMA_CPU_ARM_ARCH and LLAMA_CUDA_ARCH are chained: reaching the ARM CPU flag
  # requires GGML_NATIVE=OFF, and GGML_NATIVE=OFF drops ggml-cuda out of `native`
  # into a seven-architecture fat binary. Setting one without the other is a
  # silent ~6x build-time regression, so the Makefile refuses. That refusal, and
  # the build-directory key that keeps a toggle from reusing a stale
  # CMakeCache.txt, are the two things worth pinning.
  #
  # `make print-<VAR>` echoes one fully expanded variable and runs no part of the
  # build. Hermetic on macOS and Linux alike as long as three inputs are pinned
  # rather than discovered: MIX_APP_PATH (the Makefile requires it),
  # LLAMA_BACKEND (otherwise `auto` answers differently per host), and CUDA_HOME
  # (the Linux link-flag block errors when it finds no toolkit libraries). The
  # fake toolkit below supplies the last one; nothing is ever executed from it.
  #
  # scripts/spark/verify-build-flags.sh covers what this cannot: whether the
  # flags survived cmake into the compiler command line and into the emitted
  # machine code. This file covers whether the Makefile emits them at all.

  @cpu_arch "armv9.2-a+dotprod+i8mm+fp16+bf16+sve2"
  @cuda_arch "121a-real"

  setup_all do
    root = Path.expand("..", __DIR__)

    tmp =
      Path.join(
        System.tmp_dir!(),
        "llama_cpp_ex_make_probe_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(Path.join(tmp, "cuda/bin"))
    File.mkdir_p!(Path.join(tmp, "cuda/lib64"))
    File.write!(Path.join(tmp, "cuda/bin/nvcc"), "#!/bin/sh\nexit 0\n")
    File.chmod!(Path.join(tmp, "cuda/bin/nvcc"), 0o755)

    on_exit(fn -> File.rm_rf!(tmp) end)

    %{root: root, cuda_home: Path.join(tmp, "cuda"), app_path: Path.join(tmp, "app")}
  end

  # {:ok, %{flags: [...], build: "..."}}, or {:error, output} when make refused
  # to parse — which is what $(error) does.
  defp probe(ctx, env) do
    env =
      Map.merge(
        %{
          "MIX_APP_PATH" => ctx.app_path,
          "CUDA_HOME" => ctx.cuda_home,
          "LLAMA_BACKEND" => "cuda",
          # Every variable the Makefile reads has to be pinned, not just the
          # ones under test: scripts/spark/remote.sh exports the whole Spark
          # build contract, so an unpinned one leaks in and the build-directory
          # assertions fail on a Spark while passing on a laptop.
          "LLAMA_CPU_ARM_ARCH" => "",
          "LLAMA_CUDA_ARCH" => "",
          "LLAMA_PORTABLE" => "",
          "LLAMA_CMAKE_ARGS" => "",
          "LLAMA_CUDA_NCCL" => "0",
          "LLAMA_RPC" => "0",
          "LLAMA_RPC_RDMA" => "0"
        },
        env
      )

    case System.cmd("make", ["print-CMAKE_FLAGS", "print-LLAMA_BUILD"],
           cd: ctx.root,
           env: env,
           stderr_to_stdout: true
         ) do
      {out, 0} ->
        [flags, build] = out |> String.trim() |> String.split("\n")
        {:ok, %{flags: String.split(flags, ~r/\s+/, trim: true), build: build}}

      {out, _} ->
        {:error, out}
    end
  end

  describe "the CPU/CUDA architecture pairing" do
    test "both unset leaves today's behaviour untouched", ctx do
      assert {:ok, %{flags: flags, build: build}} = probe(ctx, %{})

      refute "-DGGML_NATIVE=OFF" in flags
      refute Enum.any?(flags, &String.starts_with?(&1, "-DGGML_CPU_ARM_ARCH="))
      refute Enum.any?(flags, &String.starts_with?(&1, "-DCMAKE_CUDA_ARCHITECTURES="))

      assert String.ends_with?(build, "llama_build-cuda"),
             "an unflagged build must keep its existing build directory, got #{build}"
    end

    test "both set emits the paired flags exactly once", ctx do
      assert {:ok, %{flags: flags}} =
               probe(ctx, %{"LLAMA_CPU_ARM_ARCH" => @cpu_arch, "LLAMA_CUDA_ARCH" => @cuda_arch})

      assert Enum.count(flags, &(&1 == "-DGGML_NATIVE=OFF")) == 1
      assert "-DGGML_CPU_ARM_ARCH=#{@cpu_arch}" in flags
      assert "-DCMAKE_CUDA_ARCHITECTURES=#{@cuda_arch}" in flags
    end

    # The whole reason the pairing is enforced rather than documented.
    test "LLAMA_CPU_ARM_ARCH alone on a CUDA build is a hard error", ctx do
      assert {:error, out} = probe(ctx, %{"LLAMA_CPU_ARM_ARCH" => @cpu_arch})
      assert out =~ "LLAMA_CPU_ARM_ARCH requires GGML_NATIVE=OFF"
      assert out =~ "LLAMA_CUDA_ARCH"
    end

    # No CUDA, no fat-binary hazard, so no reason to demand a CUDA architecture.
    test "LLAMA_CPU_ARM_ARCH alone on a non-CUDA build is allowed", ctx do
      assert {:ok, %{flags: flags}} =
               probe(ctx, %{"LLAMA_BACKEND" => "cpu", "LLAMA_CPU_ARM_ARCH" => @cpu_arch})

      assert "-DGGML_NATIVE=OFF" in flags
      assert "-DGGML_CPU_ARM_ARCH=#{@cpu_arch}" in flags
      refute Enum.any?(flags, &String.starts_with?(&1, "-DCMAKE_CUDA_ARCHITECTURES="))
    end

    # LLAMA_PORTABLE sets GGML_NATIVE=OFF for a different reason — artifacts that
    # leave the machine, built on release runners with no GPU to pin an arch for.
    # The two sources must not double-emit, and portable alone must keep working
    # with no CUDA architecture named.
    test "LLAMA_PORTABLE and LLAMA_CPU_ARM_ARCH together emit GGML_NATIVE=OFF once", ctx do
      assert {:ok, %{flags: flags}} =
               probe(ctx, %{
                 "LLAMA_PORTABLE" => "1",
                 "LLAMA_CPU_ARM_ARCH" => @cpu_arch,
                 "LLAMA_CUDA_ARCH" => @cuda_arch
               })

      assert Enum.count(flags, &(&1 == "-DGGML_NATIVE=OFF")) == 1
    end

    test "LLAMA_PORTABLE alone still needs no CUDA architecture", ctx do
      assert {:ok, %{flags: flags}} = probe(ctx, %{"LLAMA_PORTABLE" => "1"})
      assert "-DGGML_NATIVE=OFF" in flags
    end
  end

  describe "the build-directory key" do
    # Without this, toggling the flags reuses the previous CMakeCache.txt and the
    # new configuration silently no-ops — the exact failure the key exists for.
    test "differs between flag sets and is stable within one", ctx do
      {:ok, a} = probe(ctx, %{"LLAMA_CPU_ARM_ARCH" => @cpu_arch, "LLAMA_CUDA_ARCH" => @cuda_arch})

      {:ok, a_again} =
        probe(ctx, %{"LLAMA_CPU_ARM_ARCH" => @cpu_arch, "LLAMA_CUDA_ARCH" => @cuda_arch})

      {:ok, b} =
        probe(ctx, %{"LLAMA_CPU_ARM_ARCH" => "armv8.2-a+dotprod", "LLAMA_CUDA_ARCH" => @cuda_arch})

      {:ok, c} = probe(ctx, %{"LLAMA_CPU_ARM_ARCH" => @cpu_arch, "LLAMA_CUDA_ARCH" => "90-real"})

      assert a.build == a_again.build
      assert a.build != b.build
      assert a.build != c.build

      # Portability is a separate axis and must stay one.
      {:ok, portable} =
        probe(ctx, %{
          "LLAMA_PORTABLE" => "1",
          "LLAMA_CPU_ARM_ARCH" => @cpu_arch,
          "LLAMA_CUDA_ARCH" => @cuda_arch
        })

      assert portable.build != a.build
    end

    # LLAMA_RPC adds a whole backend library and a public define, and the RDMA
    # toggle changes the code inside it. Both have to key the directory or a
    # toggle silently reuses the previous cmake tree.
    test "RPC and its RDMA toggle are part of the key", ctx do
      {:ok, off} = probe(ctx, %{})
      {:ok, on} = probe(ctx, %{"LLAMA_RPC" => "1", "LLAMA_RPC_RDMA" => "1"})
      {:ok, tcp} = probe(ctx, %{"LLAMA_RPC" => "1", "LLAMA_RPC_RDMA" => "0"})

      assert off.build != on.build
      assert String.ends_with?(tcp.build, "-rpc-tcp")

      case :os.type() do
        {:unix, :linux} ->
          # RDMA is a real option here, so the two RPC builds must not collide.
          assert String.ends_with?(on.build, "-rpc")
          assert on.build != tcp.build

        _ ->
          # ggml forces GGML_RPC_RDMA off anywhere but Linux, so asking for it is
          # a configure-time lie and the Makefile does not pass it on.
          assert String.ends_with?(on.build, "-rpc-tcp")
      end
    end
  end

  describe "the RPC flags" do
    # Stated in both directions, never left to ggml's default, so the cmake
    # configuration and the hand-assembled link line cannot disagree — the same
    # discipline the NCCL comment in the Makefile exists to enforce.
    test "GGML_RPC is always stated explicitly", ctx do
      assert {:ok, %{flags: off}} = probe(ctx, %{})
      assert "-DGGML_RPC=OFF" in off

      assert {:ok, %{flags: on}} = probe(ctx, %{"LLAMA_RPC" => "1"})
      assert "-DGGML_RPC=ON" in on
      refute "-DGGML_RPC=OFF" in on
    end

    test "GGML_RPC_RDMA is declared, never auto-detected", ctx do
      # ggml/src/ggml-rpc/CMakeLists.txt:11-22 turns RDMA on whenever libibverbs
      # happens to exist on the build host. Same source, different artifact per
      # machine — exactly what this pins shut.
      assert {:ok, %{flags: on}} = probe(ctx, %{"LLAMA_RPC" => "1", "LLAMA_RPC_RDMA" => "1"})
      assert {:ok, %{flags: off}} = probe(ctx, %{"LLAMA_RPC" => "1", "LLAMA_RPC_RDMA" => "0"})

      assert "-DGGML_RPC_RDMA=OFF" in off

      case :os.type() do
        {:unix, :linux} -> assert "-DGGML_RPC_RDMA=ON" in on
        _ -> assert "-DGGML_RPC_RDMA=OFF" in on
      end
    end
  end

  describe "the configuration stamp" do
    # File timestamps do not capture a flag change: toggling LLAMA_RPC alters
    # CXXFLAGS and the archive set while touching no source, so make kept a
    # previously linked .so and shipped a NIF missing functions it was built to
    # export. Silent, and it looks like an Elixir bug. The stamp's name carries
    # a hash of the configuration so the prerequisite disappears when it changes.
    defp config_stamp(ctx, env) do
      base = %{
        "MIX_APP_PATH" => ctx.app_path,
        "CUDA_HOME" => ctx.cuda_home,
        "LLAMA_BACKEND" => "cuda",
        "LLAMA_CPU_ARM_ARCH" => "",
        "LLAMA_CUDA_ARCH" => "",
        "LLAMA_PORTABLE" => "",
        "LLAMA_CMAKE_ARGS" => "",
        "LLAMA_CUDA_NCCL" => "0",
        "LLAMA_RPC" => "0",
        "LLAMA_RPC_RDMA" => "0"
      }

      {out, 0} =
        System.cmd("make", ["print-LLAMA_CONFIG_STAMP", "print-NIF_LINK_STAMP"],
          cd: ctx.root,
          env: Map.merge(base, env),
          stderr_to_stdout: true
        )

      [config, link] = out |> String.trim() |> String.split("\n")
      %{config: config, link: link}
    end

    test "changes with the compiler flags", ctx do
      # -DGGML_USE_RPC lands in CXXFLAGS and nowhere else observable.
      assert config_stamp(ctx, %{}).config != config_stamp(ctx, %{"LLAMA_RPC" => "1"}).config
    end

    test "changes with the linker and cmake flags", ctx do
      # NCCL changes LDFLAGS on Linux and CMAKE_FLAGS everywhere, while leaving
      # the build directory alone — exactly what a timestamp rule cannot see.
      assert config_stamp(ctx, %{}).config !=
               config_stamp(ctx, %{"LLAMA_CUDA_NCCL" => "1"}).config
    end

    test "is stable for an unchanged configuration", ctx do
      assert config_stamp(ctx, %{}) == config_stamp(ctx, %{})
    end

    # The .so lives in priv/, which Mix symlinks into every MIX_ENV's build tree,
    # so dev, test and bench share ONE artifact while keeping separate objects
    # and separate llama.cpp trees. Two ways that goes wrong, both observed:
    # another MIX_ENV linking a different configuration over it, and Mix copying
    # a downloaded precompiled artifact over it with a fresh mtime. Neither is
    # visible to a timestamp rule, so the marker records what the artifact IS —
    # config hash plus a digest of the linked bytes — and lives beside it.
    test "the link marker sits beside the shared artifact", ctx do
      %{link: link} = config_stamp(ctx, %{})

      assert String.ends_with?(link, "/priv/.llama_cpp_ex_nif.built"),
             "the marker must live beside the artifact it describes, got #{link}"
    end

    test "the marker is shared, not per-configuration", ctx do
      # Deliberately one filename: a per-config *name* would let two
      # environments' markers coexist beside a single artifact, which is exactly
      # the ambiguity being removed. Discrimination is by content.
      assert config_stamp(ctx, %{}).link ==
               config_stamp(ctx, %{"LLAMA_RPC" => "1"}).link
    end

    test "`all` runs the artifact check before deciding to link", ctx do
      # Without this wiring the digest is recorded and never consulted.
      {out, 0} =
        System.cmd("make", ["-n", "all"],
          cd: ctx.root,
          env: %{
            "MIX_APP_PATH" => ctx.app_path,
            "CUDA_HOME" => ctx.cuda_home,
            "LLAMA_BACKEND" => "cuda",
            "LLAMA_CPU_ARM_ARCH" => "",
            "LLAMA_CUDA_ARCH" => "",
            "LLAMA_PORTABLE" => "",
            "LLAMA_CMAKE_ARGS" => "",
            "LLAMA_CUDA_NCCL" => "0",
            "LLAMA_RPC" => "0",
            "LLAMA_RPC_RDMA" => "0"
          },
          stderr_to_stdout: true
        )

      assert out =~ ".llama_cpp_ex_nif.built",
             "`make all` must consult the link marker"
    end
  end
end
