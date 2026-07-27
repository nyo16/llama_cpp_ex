defmodule LlamaCppEx.OptionForwardingTest do
  @moduledoc """
  Drift alarm for the option key lists.

  `Context.create/2`, `Sampler.create/2` and `Model.load/2` each read a set of
  options, and several callers forward user options into them. Those callers used
  to keep their own copies of the key lists, and the copies drifted: the one in
  `LlamaCppEx.Server` was missing `:n_threads`, `:n_threads_batch` and
  `:n_ubatch`, so those options were silently dropped rather than rejected.

  Two things are checked, from two directions:

    * **Source.** Every option any of the three modules reads, anywhere in the
      file, must be classified. Scanning the whole module rather than one
      function body is deliberate — a key read by a private helper used to be
      invisible, and "reads it but does not declare it" is the direction that
      actually breaks callers.
    * **Behaviour.** Every declared key must actually be *accepted* by the public
      entry points, which is now observable: `LlamaCppEx.Options.validate!/3`
      lists the accepted set in its error, and `@gen_opt_keys`, `@server_opt_keys`,
      `@start_opt_keys` and `@call_opt_keys` are assembled from
      `Sampler.option_keys/0`, `Context.tuning_option_keys/0` and
      `Model.tuning_option_keys/0` at compile time. The previous version of this
      check grepped the source for the string `"Context.tuning_option_keys()"`,
      which a comment mentioning the function satisfied.

  Neither needs a model, so both stay in the default suite.
  """
  use ExUnit.Case, async: true

  alias LlamaCppEx.{Context, Model, Options, Sampler, Server}
  alias LlamaCppEx.ModelManager.ModelIO

  # --- Source scanning ---

  # Extracts the body of the option-reading function so sibling functions don't
  # pollute the scan — Context.generate/4 also reads :max_tokens and :cancel,
  # which are not Context.create/2 options.
  defp function_body(path, marker) do
    [_, rest] = String.split(File.read!(path), marker, parts: 2)

    ~r/\n  (@doc|defp )/
    |> Regex.split(rest, parts: 2)
    |> hd()
  end

  # Every shape in which this codebase reads an option out of `opts`. The old
  # pattern matched only `Keyword.get/get_lazy`, so `opts[:key]`,
  # `Keyword.fetch!/2` and `Keyword.pop/2` would all have slipped through.
  @read_forms ~r/(?:Keyword\.(?:get|get_lazy|fetch|fetch!|pop|pop!|has_key\?)\(opts,\s*|opts\[)\s*:([a-z_0-9]+)/

  # Returns the option names as strings, not atoms: String.to_existing_atom/1
  # would fail here when a setup block runs before the owning module has been
  # loaded, since the atom would not exist yet.
  defp scan(source) do
    @read_forms
    |> Regex.scan(source)
    |> Enum.map(fn [_, key] -> key end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp opts_read_in(path, marker), do: path |> function_body(marker) |> scan()
  defp opts_read_anywhere_in(path), do: path |> File.read!() |> scan()

  defp declared(keys), do: keys |> Enum.map(&Atom.to_string/1) |> Enum.sort()

  # --- Behavioural probing ---

  # A key no entry point will ever own, used to provoke the unknown-option error.
  @unknown_key :definitely_not_an_option

  # The accepted key set is not a module attribute anything can read at runtime —
  # it is baked into the `Options.validate!/3` call site. It *is* observable,
  # though: the error lists it. Reading it back is a far stronger assertion than
  # grepping lib source for the name of the function that built it.
  defp accepted_keys(fun) do
    message =
      try do
        fun.()
        flunk("expected an unknown-option ArgumentError, got a normal return")
      rescue
        e in ArgumentError -> Exception.message(e)
      end

    assert message =~ "#{inspect(@unknown_key)}",
           "the error should name the offending key, got:\n#{message}"

    [_, list] = Regex.run(~r/^Known options: (.*)$/m, message)

    list
    |> String.split(", ", trim: true)
    |> Enum.map(&(&1 |> String.trim_leading(":") |> String.to_existing_atom()))
    |> MapSet.new()
  end

  defp assert_accepts(accepted, keys, label) do
    missing = Enum.reject(keys, &MapSet.member?(accepted, &1))

    assert missing == [],
           """
           #{label} rejects options owned by the module that consumes them: #{inspect(missing)}

           The key set is assembled from Sampler.option_keys/0,
           Context.tuning_option_keys/0 and Model.tuning_option_keys/0 — if a key
           is missing here, the assembly has drifted and users get an
           "unknown option" error for an option that works.
           """
  end

  describe "Context.create/2" do
    setup do
      %{read: opts_read_in("lib/llama_cpp_ex/context.ex", "def create(")}
    end

    test "every option it reads is classified as tuning or structural", %{read: read} do
      declared = declared(Context.tuning_option_keys() ++ Context.structural_option_keys())

      assert read -- declared == [],
             """
             Context.create/2 reads options that are in neither
             tuning_option_keys/0 nor structural_option_keys/0: #{inspect(read -- declared)}

             Classify each new option. Tuning keys are forwarded by callers
             automatically; structural keys must be set explicitly by each caller.
             """
    end

    test "every declared key is actually read by create/2", %{read: read} do
      declared = declared(Context.tuning_option_keys() ++ Context.structural_option_keys())

      assert declared -- read == [],
             "declared but never read by Context.create/2: #{inspect(declared -- read)}"
    end

    # `Context.generate/4` legitimately reads options of its own. Listing them
    # explicitly is what lets the whole-file scan below be exhaustive: a new
    # unclassified read anywhere in context.ex — including in a private helper
    # that create/2 calls, which the function-body scan cannot see — fails until
    # it is either declared or added here.
    @context_non_create_keys [:max_tokens, :cancel]

    test "no option is read anywhere in context.ex without being classified" do
      read = opts_read_anywhere_in("lib/llama_cpp_ex/context.ex")

      declared =
        declared(
          Context.tuning_option_keys() ++
            Context.structural_option_keys() ++ @context_non_create_keys
        )

      assert read -- declared == [],
             """
             context.ex reads unclassified options: #{inspect(read -- declared)}

             Either add them to tuning_option_keys/0 or structural_option_keys/0,
             or — if they belong to a function other than create/2 — to
             @context_non_create_keys in this test.
             """
    end

    test "tuning and structural sets are disjoint" do
      # A -- (A -- B) is the intersection.
      overlap =
        Context.tuning_option_keys() --
          (Context.tuning_option_keys() -- Context.structural_option_keys())

      assert overlap == [],
             "keys classified as both tuning and structural: #{inspect(overlap)}"
    end

    test "context-defining options are never forwardable" do
      for key <- [:embeddings, :pooling_type, :ctx_type, :n_ctx] do
        refute key in Context.tuning_option_keys(),
               """
               #{inspect(key)} must stay structural. Forwarding it would let a caller
               change what the context is — e.g. embeddings: true would turn a
               generation server into an embedding context.
               """
      end
    end
  end

  describe "Sampler.create/2" do
    test "option_keys/0 matches exactly the options it reads" do
      read = opts_read_in("lib/llama_cpp_ex/sampler.ex", "def create(")

      assert read == declared(Sampler.option_keys())
    end

    test "no option is read anywhere in sampler.ex without being declared" do
      # Sampler owns exactly one option-reading function, so the whole-file scan
      # must agree with option_keys/0 exactly.
      assert opts_read_anywhere_in("lib/llama_cpp_ex/sampler.ex") ==
               declared(Sampler.option_keys())
    end
  end

  describe "Model.load/2" do
    setup do
      %{read: opts_read_in("lib/llama_cpp_ex/model.ex", "def load(")}
    end

    test "every option it reads is classified as tuning or structural", %{read: read} do
      declared = declared(Model.tuning_option_keys() ++ Model.structural_option_keys())

      assert read -- declared == [],
             "Model.load/2 reads unclassified options: #{inspect(read -- declared)}"
    end

    test "every declared key is actually read by load/2", %{read: read} do
      declared = declared(Model.tuning_option_keys() ++ Model.structural_option_keys())

      assert declared -- read == [],
             "declared but never read by Model.load/2: #{inspect(declared -- read)}"
    end

    test "no option is read anywhere in model.ex without being classified" do
      declared = declared(Model.tuning_option_keys() ++ Model.structural_option_keys())

      assert opts_read_anywhere_in("lib/llama_cpp_ex/model.ex") -- declared == []
    end

    test "vocab_only is never forwardable" do
      refute :vocab_only in Model.tuning_option_keys(),
             ":vocab_only must stay structural — forwarding it into a server would " <>
               "load a model with no weights."
    end

    test "all three load-mode flags are forwardable together" do
      # They collapse into llama.cpp's single load_mode enum with
      # dio > mlock > mmap > none precedence. Forwarding a subset (Server used to
      # omit :use_mmap) leaves callers unable to select some modes.
      for key <- [:use_mmap, :use_mlock, :use_direct_io] do
        assert key in Model.tuning_option_keys()
      end
    end
  end

  describe "callers do not keep their own copies" do
    @callers [
      "lib/llama_cpp_ex.ex",
      "lib/llama_cpp_ex/server.ex",
      "lib/llama_cpp_ex/mtp.ex"
    ]

    # Matches a literal multi-atom list passed to Keyword.take/2 — `[:a, :b]` —
    # which is how every copy drifted. Deliberately does not match the cons form
    # `[:n_ctx | Context.tuning_option_keys()]` (MTP's one extra key), nor the
    # option names that legitimately appear in @doc prose.
    @inline_list ~r/Keyword\.take\([a-z_]+,\s*\[\s*:[a-z_0-9]+\s*,/

    test "no caller passes an inline literal key list to Keyword.take/2" do
      for path <- @callers do
        refute Regex.match?(@inline_list, File.read!(path)),
               """
               #{path} passes a literal key list to Keyword.take/2. That is how the
               copies drifted before — select the keys from the module that owns
               them: Context.tuning_option_keys/0, Sampler.option_keys/0 or
               Model.tuning_option_keys/0.
               """
      end
    end
  end

  # These probe the assembled key sets themselves rather than the source that
  # builds them. Validation runs before anything touches the NIF, so a model with
  # a nil ref and a server name that resolves to nothing are enough.
  describe "the public entry points accept every key their owners declare" do
    @unowned %Model{ref: nil}

    test "LlamaCppEx.generate/3 (@gen_opt_keys)" do
      accepted = accepted_keys(fn -> LlamaCppEx.generate(@unowned, "x", [{@unknown_key, 1}]) end)

      assert_accepts(accepted, Sampler.option_keys(), "LlamaCppEx.generate/3")
      assert_accepts(accepted, Context.tuning_option_keys(), "LlamaCppEx.generate/3")

      assert_accepts(
        accepted,
        [:max_tokens, :n_ctx, :timeout, :grammar, :json_schema],
        "generate/3"
      )

      # Structural context options stay out; :n_ctx is the deliberate exception,
      # since generate/3 sizes its own throwaway context.
      for key <- Context.structural_option_keys() -- [:n_ctx] do
        refute MapSet.member?(accepted, key),
               "#{inspect(key)} is structural and must not be accepted by generate/3"
      end

      # Model options are not generation options: the model is already loaded.
      refute MapSet.member?(accepted, :n_gpu_layers)
      refute MapSet.member?(accepted, :vocab_only)
    end

    test "LlamaCppEx.stream/3 and chat/3 accept the same set as generate/3" do
      base = accepted_keys(fn -> LlamaCppEx.generate(@unowned, "x", [{@unknown_key, 1}]) end)

      for {label, fun} <- [
            {"stream/3", fn -> LlamaCppEx.stream(@unowned, "x", [{@unknown_key, 1}]) end},
            {"chat/3", fn -> LlamaCppEx.chat(@unowned, [], [{@unknown_key, 1}]) end},
            {"stream_chat/3",
             fn -> LlamaCppEx.stream_chat(@unowned, [], [{@unknown_key, 1}]) end},
            {"chat_completion/3",
             fn -> LlamaCppEx.chat_completion(@unowned, [], [{@unknown_key, 1}]) end}
          ] do
        assert accepted_keys(fun) == base, "#{label} accepts a different key set to generate/3"
      end
    end

    # The invariant that actually broke: `@server_opt_keys` in the facade was a
    # hand-written `@gen_opt_keys ++ [:cache_prompt, :session]` and lagged behind
    # `Server`'s per-request options by exactly `:cache_scope`, so the whole
    # tenant-scoping feature was unreachable through `chat_completion/3` — the
    # route most callers take. It is now assembled from
    # `Server.request_option_keys/0`; this asserts the assembly, not the list.
    test "the server-routed entry points accept every Server request option" do
      for {label, fun} <- [
            {"chat_completion/3",
             fn -> LlamaCppEx.chat_completion(:no_such_server, [], [{@unknown_key, 1}]) end},
            {"stream_chat_completion/3",
             fn ->
               LlamaCppEx.stream_chat_completion(:no_such_server, [], [{@unknown_key, 1}])
             end}
          ] do
        accepted = accepted_keys(fun)
        assert_accepts(accepted, Server.request_option_keys(), "LlamaCppEx.#{label} (server)")

        # And nothing else: the server-routed set is the request options plus what
        # this module handles itself. It used to be `@gen_opt_keys ++
        # request_option_keys()`, which accepted `:n_ctx` and all 20
        # `Context.tuning_option_keys()` — 21 keys a running server cannot honour,
        # each of which passed this gate and then raised inside
        # `Server.complete_tokens/3`. The containment test below pins the
        # relationship from the server's side; this pins the list.
        extra = MapSet.difference(accepted, MapSet.new(Server.request_option_keys()))

        assert Enum.sort(extra) == [
                 :add_assistant,
                 :chat_template_kwargs,
                 :enable_thinking,
                 :json_schema,
                 :max_tokens,
                 :timeout
               ],
               "#{label} accepts something that is neither a generation nor a request option"
      end
    end

    test "Server.request_option_keys/0 is the union of sampler and per-request keys" do
      request = Server.request_option_keys()

      assert_accepts(MapSet.new(request), Sampler.option_keys(), "Server.request_option_keys/0")

      assert Enum.sort(request -- Sampler.option_keys()) ==
               [:cache_prompt, :cache_scope, :session]
    end

    test "LlamaCppEx.Server.start_link/1 (@start_opt_keys)" do
      accepted = accepted_keys(fn -> Server.init([{@unknown_key, 1}]) end)

      assert_accepts(accepted, Sampler.option_keys(), "Server.start_link/1")
      assert_accepts(accepted, Context.tuning_option_keys(), "Server.start_link/1")
      assert_accepts(accepted, Model.tuning_option_keys(), "Server.start_link/1")

      assert_accepts(
        accepted,
        [:model_path, :n_parallel, :n_ctx, :n_batch, :chunk_size, :max_queue, :batch_strategy],
        "Server.start_link/1"
      )

      # The drift that motivated this file: Server used to omit these three.
      assert_accepts(accepted, [:n_threads, :n_threads_batch, :n_ubatch], "Server.start_link/1")

      refute MapSet.member?(accepted, :vocab_only)
      refute MapSet.member?(accepted, :embeddings)
    end

    test "LlamaCppEx.Server request calls (@call_opt_keys) take sampler options only" do
      for {label, fun} <- [
            {"generate/3", fn -> Server.generate(:none, "x", [{@unknown_key, 1}]) end},
            {"stream/3", fn -> Server.stream(:none, "x", [{@unknown_key, 1}]) end},
            {"generate_tokens/3",
             fn -> Server.generate_tokens(:none, [1], [{@unknown_key, 1}]) end},
            {"complete_tokens/3",
             fn -> Server.complete_tokens(:none, [1], [{@unknown_key, 1}]) end},
            {"stream_tokens/3", fn -> Server.stream_tokens(:none, [1], [{@unknown_key, 1}]) end}
          ] do
        accepted = accepted_keys(fun)

        assert_accepts(accepted, Sampler.option_keys(), "Server.#{label}")

        assert_accepts(
          accepted,
          [:max_tokens, :timeout, :cache_prompt, :session, :cache_scope],
          "Server.#{label}"
        )

        # Model and context options are fixed at start_link/1: a request cannot
        # resize the KV cache or move layers to the GPU mid-flight.
        for key <-
              (Model.tuning_option_keys() ++ Context.tuning_option_keys()) --
                Sampler.option_keys() do
          refute MapSet.member?(accepted, key),
                 "Server.#{label} must not accept the start-time option #{inspect(key)}"
        end
      end
    end
  end

  # The invariant every widened-contract bug in this codebase violated: a key an
  # outer gate accepts must be accepted by every inner gate it can route to,
  # except the keys the outer consumes itself.
  #
  # Manual review missed this class four times — `:cache_scope` missing from the
  # facade's copy of the server's request options, `Sampler.create/2`'s widened
  # return type with three of four call sites updated, `@server_opt_keys`
  # accepting 21 keys `Server`'s own gate rejects, and `ModelIO.native_opts/1`'s
  # denylist forwarding `:vocab_only` into `Server.start_link/1` — three of them
  # introduced by the change set that was fixing the first. Every other test in
  # this file pins one instance. These pin the invariant, so instance five fails
  # here rather than in a user's ArgumentError.
  describe "no outer gate accepts a key its inner gate rejects" do
    # Keys the facade reads and consumes before forwarding: chat templating is
    # split off by `split_chat_opts/1` and `:json_schema` is rewritten into
    # `:grammar` by `resolve_grammar_opts/1`. Everything else must survive the
    # inner gate intact.
    @facade_consumed [
      :add_assistant,
      :chat_template_kwargs,
      :enable_thinking,
      :json_schema
    ]

    # Held in a function rather than a module attribute: a `fn` cannot be escaped
    # into an attribute.
    defp server_routes do
      [
        {"LlamaCppEx.chat_completion/3", "LlamaCppEx.Server.complete_tokens/3",
         fn -> LlamaCppEx.chat_completion(:no_such_server, [], [{@unknown_key, 1}]) end,
         fn -> Server.complete_tokens(:none, [1], [{@unknown_key, 1}]) end},
        {"LlamaCppEx.stream_chat_completion/3", "LlamaCppEx.Server.subscribe_stream_tokens/4",
         fn -> LlamaCppEx.stream_chat_completion(:no_such_server, [], [{@unknown_key, 1}]) end,
         fn -> Server.subscribe_stream_tokens(:none, [1], make_ref(), [{@unknown_key, 1}]) end}
      ]
    end

    test "the facade's server-routed sets contain exactly what the server accepts" do
      for {outer_label, inner_label, outer, inner} <- server_routes() do
        outer_set = accepted_keys(outer)
        inner_set = accepted_keys(inner)

        escaping = MapSet.difference(outer_set, inner_set)

        assert MapSet.equal?(escaping, MapSet.new(@facade_consumed)),
               """
               #{outer_label} does not agree with #{inner_label}.

               Accepted by the outer gate and not the inner: #{inspect(Enum.sort(escaping))}
               Expected exactly the keys the facade consumes: #{inspect(@facade_consumed)}

               An extra key here reaches the inner gate and raises an
               ArgumentError naming #{inner_label} — a function the caller never
               called. A missing one is an option the facade advertises and then
               silently drops.
               """

        unreachable = MapSet.difference(inner_set, outer_set)

        assert MapSet.equal?(unreachable, MapSet.new([])),
               """
               #{inner_label} accepts #{inspect(Enum.sort(unreachable))}, which
               #{outer_label} rejects — the feature is unreachable through the
               route most callers take. This is the `:cache_scope` bug verbatim.
               """
      end
    end

    test "ModelManager.load/3's backend routes every key it accepts somewhere" do
      backend =
        accepted_keys(fn ->
          ModelIO.start_server("probe", "/nonexistent", [{@unknown_key, 1}])
        end)

      assert_accepts(backend, Server.start_option_keys(), "ModelIO (server route)")
      assert_accepts(backend, Model.structural_option_keys(), "ModelIO (direct route)")
      assert_accepts(backend, [:cache_dir, :token, :revision, :force, :progress], "ModelIO (hub)")
      assert_accepts(backend, [:mode, :capabilities, :default, :memory_budget, :io], "ModelIO")

      # The bug: `:vocab_only` is a legitimate `Model.load/2` option that must
      # never be forwarded into a server, and the denylist forwarded it. The
      # backend accepts it (above) and routes it to `Model.load/2` only.
      #
      # `:n_gpu_layers` is the other structural Model option and is deliberately
      # NOT in this list: the server reads it in `handle_continue/2` to load its
      # own model, so it is a legitimate `start_link/1` option too.
      refute :vocab_only in Server.start_option_keys(),
             ":vocab_only must not be forwarded into a server — it produces a " <>
               "model with no weights, and Server.init/1 rejects it"
    end

    test "every key LlamaCppEx.generate/3 accepts is read by someone" do
      accepted = accepted_keys(fn -> LlamaCppEx.generate(@unowned, "x", [{@unknown_key, 1}]) end)

      # A key earns its place by being read out of an `*opts` keyword list in the
      # facade or the chat layer, or by being declared by a downstream owner.
      # Anything else is advertised and then ignored — the mirror image of the
      # containment failure above, and exactly how `:template` came to be
      # documented on `chat/3` for a release while `apply_template/3` never read
      # it.
      #
      # `:timeout` is read through `Options.timeout/2` rather than a literal
      # `Keyword.get(opts, :timeout, ...)`, which is the point of that function —
      # the scanner cannot see it, so it is named here.
      consumed =
        MapSet.new(
          ["timeout"] ++
            opts_read_anywhere_in("lib/llama_cpp_ex.ex") ++
            opts_read_anywhere_in("lib/llama_cpp_ex/chat.ex") ++
            declared(Sampler.option_keys() ++ Context.tuning_option_keys())
        )

      orphans =
        accepted
        |> Enum.map(&Atom.to_string/1)
        |> Enum.reject(&MapSet.member?(consumed, &1))
        |> Enum.sort()

      assert orphans == [],
             """
             LlamaCppEx.generate/3 accepts options nothing reads: #{inspect(orphans)}

             Either wire the option up or drop it from the accepted set. An
             accepted-but-unread option is indistinguishable from a working one
             at the call site.
             """
    end
  end

  describe "Options.validate!/3" do
    test "passes a known key set through unchanged" do
      opts = [temp: 0.1, top_k: 5]
      assert Options.validate!(opts, [:temp, :top_k, :seed], "probe") == opts
      assert Options.validate!([], [:temp], "probe") == []
    end

    test "names the offending key and the entry point" do
      err =
        assert_raise ArgumentError, fn ->
          Options.validate!([temp: 0.1, bogus: 1], [:temp], "LlamaCppEx.probe/1")
        end

      message = Exception.message(err)
      assert message =~ "LlamaCppEx.probe/1"
      assert message =~ "unknown option:"
      assert message =~ ":bogus"
      refute message =~ ":temp (did"
    end

    test "lists every unknown key, not just the first" do
      message =
        assert_raise(ArgumentError, fn ->
          Options.validate!([a: 1, b: 2], [:temp], "probe")
        end)
        |> Exception.message()

      assert message =~ "unknown options:"
      assert message =~ ":a"
      assert message =~ ":b"
    end

    test "suggests a near miss" do
      # The two typos that motivated the check.
      assert_raise ArgumentError, ~r/:temperature \(did you mean :temp\?\)/, fn ->
        Options.validate!([temperature: 0.1], [:temp, :top_k, :seed], "probe")
      end

      assert_raise ArgumentError, ~r/:n_paralell \(did you mean :n_parallel\?\)/, fn ->
        Options.validate!([n_paralell: 8], [:n_parallel, :n_ctx], "probe")
      end
    end

    test "does not invent a suggestion for an unrelated key" do
      message =
        assert_raise(ArgumentError, fn ->
          Options.validate!([elephant: 1], [:temp, :top_k], "probe")
        end)
        |> Exception.message()

      assert message =~ ":elephant"
      refute message =~ "did you mean"
    end

    test "rejects anything that is not a keyword list" do
      for bad <- [%{temp: 0.1}, [1, 2, 3], "temp=0.1", [{"temp", 1}]] do
        assert_raise ArgumentError, ~r/must be a keyword list/, fn ->
          Options.validate!(bad, [:temp], "probe")
        end
      end
    end

    test "a duplicated unknown key is reported once" do
      message =
        assert_raise(ArgumentError, fn ->
          Options.validate!([bogus: 1, bogus: 2], [:temp], "probe")
        end)
        |> Exception.message()

      assert message =~ "unknown option:"
      assert length(String.split(message, ":bogus")) == 2
    end
  end

  describe "a typo is rejected end to end" do
    test "generate/3 refuses :temperature instead of silently running at the default temp" do
      # The motivating bug: Keyword.take/2 *is* the routing mechanism, so an
      # unknown key was structurally indistinguishable from another module's key
      # and got dropped. Validation runs before the model ref is touched, so a
      # nil ref never reaches the NIF.
      assert_raise ArgumentError, ~r/:temperature \(did you mean :temp\?\)/, fn ->
        LlamaCppEx.generate(%Model{ref: nil}, "x", temperature: 0.1)
      end
    end

    test "Server.start_link/1 refuses :n_paralell instead of silently running 4 slots" do
      assert_raise ArgumentError, ~r/:n_paralell \(did you mean :n_parallel\?\)/, fn ->
        Server.init(model_path: "/nonexistent.gguf", n_paralell: 8)
      end
    end

    test "a valid option list gets past validation" do
      # Proof the probes above fail for the right reason. With no unknown key,
      # validate! returns and the call proceeds until the nil model ref reaches
      # the NIF — which also raises ArgumentError, so the message is what
      # distinguishes "rejected the options" from "ran with them".
      for opts <- [[temp: 0.1], [temp: 0.1, max_tokens: 4, n_threads: 1], []] do
        refute reached_validation_error?(fn ->
                 LlamaCppEx.generate(%Model{ref: nil}, "x", opts)
               end),
               "generate/3 rejected the valid option list #{inspect(opts)}"
      end
    end
  end

  # Whether `fun` failed the unknown-option check, as opposed to failing later.
  defp reached_validation_error?(fun) do
    fun.()
    false
  rescue
    e in ArgumentError -> Exception.message(e) =~ "unknown option"
    _ -> false
  catch
    _, _ -> false
  end
end
