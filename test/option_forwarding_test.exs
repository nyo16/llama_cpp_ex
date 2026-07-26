defmodule LlamaCppEx.OptionForwardingTest do
  @moduledoc """
  Drift alarm for the option key lists.

  `Context.create/2`, `Sampler.create/2` and `Model.load/2` each read a set of
  options, and several callers forward user options into them. Those callers used
  to keep their own copies of the key lists, and the copies drifted: the one in
  `LlamaCppEx.Server` was missing `:n_threads`, `:n_threads_batch` and
  `:n_ubatch`, so those options were silently dropped rather than rejected.

  These tests fail if a new option is added to one of those functions without
  being classified as tuning or structural, and if any caller re-inlines its own
  copy of a list. They read source rather than calling the NIF, so they need no
  model and stay in the default suite.
  """
  use ExUnit.Case, async: true

  alias LlamaCppEx.{Context, Model, Sampler}

  # Extracts the body of the option-reading function so sibling functions don't
  # pollute the scan — Context.generate/4 also reads :max_tokens and :cancel,
  # which are not Context.create/2 options.
  defp function_body(path, marker) do
    [_, rest] = String.split(File.read!(path), marker, parts: 2)

    ~r/\n  (@doc|defp )/
    |> Regex.split(rest, parts: 2)
    |> hd()
  end

  # Returns the option names as strings, not atoms: String.to_existing_atom/1
  # would fail here when a setup block runs before the owning module has been
  # loaded, since the atom would not exist yet.
  defp opts_read_in(path, marker) do
    path
    |> function_body(marker)
    |> then(&Regex.scan(~r/Keyword\.get(?:_lazy)?\(opts, :([a-z_0-9]+)/, &1))
    |> Enum.map(fn [_, key] -> key end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp declared(keys), do: keys |> Enum.map(&Atom.to_string/1) |> Enum.sort()

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

    test "Server selects its option sets from the owning modules" do
      source = File.read!("lib/llama_cpp_ex/server.ex")

      for expected <- [
            "Context.tuning_option_keys()",
            "Model.tuning_option_keys()",
            "Sampler.option_keys()"
          ] do
        assert String.contains?(source, expected),
               "server.ex should select options via #{expected}"
      end
    end
  end
end
