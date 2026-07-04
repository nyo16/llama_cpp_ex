# Credo configuration. Runs on top of Credo's default check set.
%{
  configs: [
    %{
      name: "default",
      files: %{
        included: ["lib/", "test/"],
        excluded: []
      },
      checks: %{
        extra: [
          # NIF entry points mirror the positional C ABI in c_src/llama_cpp_ex —
          # their arity is fixed by the native signatures, not a style choice.
          # The keyword-based wrappers around them keep the public API ergonomic.
          {Credo.Check.Refactor.FunctionArity, files: %{excluded: ["lib/llama_cpp_ex/nif.ex"]}}
        ]
      }
    }
  ]
}
