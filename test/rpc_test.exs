defmodule LlamaCppEx.RPCTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.{Model, RPC, Server}

  # The RPC backend is opt-in at build time. The contract worth pinning on a
  # default build is that it degrades *cleanly*: every entry point reports an
  # error tuple, nothing raises, nothing aborts, and the option plumbing accepts
  # the same keys either way — so a build flag is the only difference between the
  # two configurations.
  #
  # These tests run on BOTH configurations and assert the *exact* refusal for the
  # build they are on. The earlier version accepted either atom
  # (`reason in [:rpc_unsupported, :unreachable]`), which meant an RPC build
  # answering `:rpc_unsupported` was indistinguishable from a correct non-RPC
  # build. `RPC.supported?/0` makes each assertion exact.
  #
  # Be precise about what that does and does not buy, because it is tempting to
  # overclaim: `supported?/0` reads the *same* artifact as `add_server/1`, so a
  # stale or cross-environment `.so` makes the two agree and these tests pass.
  # Detecting *that* needs a source of truth outside the artifact — the requested
  # build flag — which is the last test in this block.
  #
  # Tests that need a reachable worker are tagged `:rpc_live`; see
  # test/test_helper.exs.

  # The refusal this build must give for an endpoint that cannot serve devices.
  # On an RPC build the port is genuinely probed and comes back :unreachable; on a
  # non-RPC build the call never leaves the NIF.
  defp expected_refusal do
    if RPC.supported?(), do: :unreachable, else: :rpc_unsupported
  end

  describe "supported?/0" do
    test "is a boolean and cannot disagree with the error path" do
      supported = RPC.supported?()
      assert is_boolean(supported)

      assert {:error, reason} = RPC.add_server("127.0.0.1:1")
      assert reason == :rpc_unsupported == not supported
    end

    # The one assertion here that can catch a stale or cross-environment
    # artifact, because `LLAMA_RPC` is evidence from outside the `.so`. Every
    # MIX_ENV shares one `llama_cpp_ex_nif.so` (Mix symlinks `priv/`), so
    # building test with `LLAMA_RPC=1` and bench without it used to leave
    # whichever ran last in place — twice during development. The Makefile's link
    # marker is the fix; this is the tripwire that says the fix stopped working.
    #
    # A no-op when the variable is unset, which is the common case and cannot be
    # helped: nothing else in the running VM knows what was asked for.
    test "the loaded NIF matches the build that was requested" do
      case System.get_env("LLAMA_RPC") do
        flag when flag in ["1", "true", "yes"] ->
          assert RPC.supported?(),
                 "LLAMA_RPC=#{flag} is set, but the loaded NIF reports no RPC backend. " <>
                   "The artifact is stale or came from another MIX_ENV — check " <>
                   "priv/.llama_cpp_ex_nif.built against `make print-LLAMA_CONFIG_HASH`."

        _ ->
          assert is_boolean(RPC.supported?())
      end
    end
  end

  describe "add_server/1" do
    test "an endpoint that cannot serve devices is an error, never a crash" do
      # Port 1 on loopback: nothing listens. Neither build may raise, and neither
      # may succeed: upstream's ggml_backend_register silently no-ops on a null
      # registration, so a vanished endpoint would leave the model loading onto
      # local devices while the caller believed otherwise.
      assert RPC.add_server("127.0.0.1:1") == {:error, expected_refusal()}
    end

    test "ping/1 reports the same refusal" do
      assert RPC.ping("127.0.0.1:1") == {:error, expected_refusal()}
    end
  end

  describe "add_servers/1" do
    test "stops at the first failure and names the endpoint" do
      # A partially registered set would place tensors somewhere nobody intended,
      # so the whole call fails and says which endpoint did it.
      assert RPC.add_servers(["127.0.0.1:1", "127.0.0.1:2"]) ==
               {:error, {"127.0.0.1:1", expected_refusal()}}
    end

    test "an empty list is a no-op" do
      assert RPC.add_servers([]) == {:ok, 0}
    end
  end

  describe "devices/0" do
    test "reports only RPC-backed devices" do
      # Nothing registered in this VM, so this is empty — and on a machine where
      # something *is* registered it must still never include the local backend.
      assert Enum.all?(RPC.devices(), &(&1.backend == "RPC"))
    end
  end

  describe "Model.load/2 with :rpc_servers" do
    test "surfaces the registration failure instead of loading" do
      assert {:error, message} =
               Model.load("/nonexistent/model.gguf", rpc_servers: ["127.0.0.1:1"])

      assert message =~ "127.0.0.1:1"

      # Registration happens before the load, so this must fail at the endpoint
      # and never reach the (also missing) file. Placement is computed from the
      # devices that exist at load time; registering afterwards would be useless.
      refute message =~ "/nonexistent/model.gguf"
    end

    test "an empty list skips registration entirely" do
      assert {:error, message} = Model.load("/nonexistent/model.gguf")
      assert message =~ "/nonexistent/model.gguf"
    end
  end

  # Needs a reachable worker, not just an RPC build — so it is tagged separately
  # and test_helper.exs excludes it unless LLAMA_RPC_ENDPOINT is set. The
  # "a closed port is :unreachable" case that used to live here is now covered
  # unconditionally by add_server/1 above, via RPC.supported?/0.
  describe "against a live worker" do
    @describetag :rpc_live

    # `--include rpc_live` beats the exclusion in test_helper.exs, so the tag
    # alone cannot stop this running on a machine with no worker. A compile-time
    # `skip:` can: module attributes are evaluated when the file is compiled,
    # which for a test file is during the run, so this sees the real environment.
    # The result is that asking for the tag without a worker skips with a reason
    # instead of failing on a missing variable.
    if System.get_env("LLAMA_RPC_ENDPOINT") in [nil, ""] do
      @describetag skip: "set LLAMA_RPC_ENDPOINT to a reachable RPC worker"
    end

    test "a live worker registers, is idempotent, and appears in devices/0" do
      endpoint = System.fetch_env!("LLAMA_RPC_ENDPOINT")

      assert {:ok, n} = RPC.add_server(endpoint)
      assert n >= 1

      # Upstream memoizes per endpoint, so a repeat adds nothing.
      assert {:ok, 0} = RPC.add_server(endpoint)

      devices = RPC.devices()
      assert Enum.any?(devices, &(&1.description == endpoint))

      # Hardcoded upstream even when the worker serves only a CPU device.
      assert Enum.all?(devices, &(&1.type == :gpu))

      # There are TWO device orderings and they disagree. This one — the ggml
      # registry, which is what devices/0 enumerates — is *registration* order,
      # so a locally-detected backend comes first and RPC endpoints are appended
      # as they are registered.
      all = LlamaCppEx.devices()
      refute hd(all).backend == "RPC"
      assert List.last(Enum.filter(all, &(&1.backend == "RPC"))).description == endpoint

      # llama.cpp builds a *different* list for placement and inserts RPC
      # devices at the FRONT of it (src/llama.cpp:263-273, "to minimize network
      # transfers"), so `tensor_split[0]` and `main_gpu: 0` address the remote
      # node even though devices/0 shows it last. `gpu_index` is derived from
      # registry order and therefore does NOT index tensor_split once an RPC
      # device exists. Nothing in this VM can observe the placement list, so the
      # invariant pinned here is the one that misleads: the two differ.
      rpc_gpu_index = Enum.find(all, &(&1.backend == "RPC")).gpu_index
      assert rpc_gpu_index > 0, "registry order put RPC first; docs/multi-gpu.md needs revisiting"
    end
  end

  describe "split mode encoding" do
    # Upstream's llama_split_mode enum (llama.h). A silent drift here places
    # tensors somewhere nobody asked for and nothing downstream would notice.
    test "maps every mode to its upstream value" do
      assert Model.encode_split_mode(:none) == 0
      assert Model.encode_split_mode(:layer) == 1
      assert Model.encode_split_mode(:row) == 2
      assert Model.encode_split_mode(:tensor) == 3
    end

    # There was no fallback clause, so `split_mode: :tensor` raised
    # FunctionClauseError from inside load/2 with no indication of what was
    # wrong. A typo should say what the accepted set is.
    test "an unknown mode raises with the accepted set" do
      assert_raise ArgumentError, ~r/unknown split_mode :diagonal/, fn ->
        Model.encode_split_mode(:diagonal)
      end

      assert_raise ArgumentError, ~r/:none, :layer, :row or :tensor/, fn ->
        Model.encode_split_mode("layer")
      end
    end
  end

  describe ":rpc_servers option plumbing" do
    test "is a forwardable model tuning option" do
      assert :rpc_servers in Model.tuning_option_keys()
      refute :rpc_servers in Model.structural_option_keys()
    end

    test "Server.start_link/1 accepts it" do
      assert :rpc_servers in Server.start_option_keys()
    end

    # Registration mutates a process-global device registry and must precede a
    # load, so it cannot be a per-request option: a request cannot move layers
    # onto another machine mid-flight.
    test "request-level calls reject it" do
      refute :rpc_servers in Server.request_option_keys()
    end
  end

  describe ":devices option" do
    test "is a forwardable model tuning option" do
      assert :devices in Model.tuning_option_keys()
      assert :devices in Server.start_option_keys()
      refute :devices in Server.request_option_keys()
    end

    test "an unknown device name is refused, and says what exists" do
      # llama.cpp uses params.devices verbatim with no validation of its own, so
      # a typo would otherwise become a null entry in a NULL-terminated array —
      # a silently truncated device list.
      assert {:error, message} = Model.load("/nonexistent/model.gguf", devices: ["GPU42"])

      assert message =~ "unknown device: GPU42"
      assert message =~ "available:"
      refute message =~ "failed to load model"
    end

    test "an empty list leaves llama.cpp to build the placement list" do
      assert {:error, message} = Model.load("/nonexistent/model.gguf", devices: [])
      assert message =~ "failed to load model"
    end

    test "a real device name gets past placement and fails on the file" do
      name = hd(LlamaCppEx.devices()).name
      assert {:error, message} = Model.load("/nonexistent/model.gguf", devices: [name])
      assert message =~ "failed to load model"
    end
  end
end
