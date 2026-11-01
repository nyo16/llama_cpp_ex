defmodule LlamaCppEx.VMExitTest do
  @moduledoc """
  The VM must exit with its own status while models are still referenced
  (#116). The BEAM does not run NIF resource destructors at halt, and on Metal
  llama.cpp's device destructor asserted that no residency set was left, so a
  halt with a live model aborted with exit 134 instead.

  Each case runs a child VM, because the failure is the VM's exit status. The
  child runs without `GGML_METAL_NO_RESIDENCY`, which turns residency sets off
  and would hide the bug. Off Metal these pass trivially: the abort was
  Metal-only.

      LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf \\
        mix test test/vm_exit_test.exs --include smoke
  """
  use ExUnit.Case, async: false

  @moduletag :smoke
  @moduletag timeout: 300_000

  # Runs `code` in a fresh VM on this build's code paths and returns its exit
  # status and output.
  defp run_child(code) do
    paths =
      for path <- :code.get_path(),
          path = List.to_string(path),
          String.contains?(path, "/_build/"),
          do: ["-pa", path]

    System.cmd(
      System.find_executable("elixir"),
      List.flatten(paths) ++ ["-e", code],
      env: [
        {"GGML_METAL_NO_RESIDENCY", nil},
        {"LLAMA_SMOKE_GEN_MODEL", LlamaCppEx.TestModels.path!(:gen)}
      ],
      stderr_to_stdout: true
    )
  end

  test "halting with a model and context referenced by a live process keeps the status" do
    {output, status} =
      run_child("""
      {:ok, _} = Application.ensure_all_started(:llama_cpp_ex)
      {:ok, model} = LlamaCppEx.load_model(System.fetch_env!("LLAMA_SMOKE_GEN_MODEL"), n_gpu_layers: -1)
      {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512)
      {:ok, _} = LlamaCppEx.generate(model, "The sky is", max_tokens: 4, temp: 0.0)
      spawn(fn -> receive do :never -> {model, ctx} end end)
      System.halt(7)
      """)

    assert status == 7, "child exited #{status}:\n#{output}"
  end

  # A first model on the CPU only never builds the Metal buffer types, so a
  # later Metal context used to build them after the exit hook was registered,
  # and exit destroyed them before the hook freed that context (SIGBUS).
  test "halting after a CPU-only model and then a GPU model keeps the status" do
    {output, status} =
      run_child("""
      {:ok, _} = Application.ensure_all_started(:llama_cpp_ex)
      path = System.fetch_env!("LLAMA_SMOKE_GEN_MODEL")
      {:ok, cpu_model} = LlamaCppEx.load_model(path, devices: ["CPU"], n_gpu_layers: 0)
      {:ok, cpu_ctx} = LlamaCppEx.Context.create(cpu_model, n_ctx: 512)
      {:ok, gpu_model} = LlamaCppEx.load_model(path, n_gpu_layers: -1)
      {:ok, gpu_ctx} = LlamaCppEx.Context.create(gpu_model, n_ctx: 512)
      spawn(fn -> receive do :never -> {cpu_model, cpu_ctx, gpu_model, gpu_ctx} end end)
      System.halt(5)
      """)

    assert status == 5, "child exited #{status}:\n#{output}"
  end

  test "halting while a server is generating exits cleanly" do
    {output, status} =
      run_child("""
      {:ok, _} = Application.ensure_all_started(:llama_cpp_ex)
      {:ok, server} =
        LlamaCppEx.Server.start_link(
          model_path: System.fetch_env!("LLAMA_SMOKE_GEN_MODEL"),
          n_gpu_layers: -1,
          n_parallel: 1
        )

      parent = self()

      spawn(fn ->
        server
        |> LlamaCppEx.Server.stream("Write an endless story:", max_tokens: 4000)
        |> Stream.each(fn _ -> send(parent, :token) end)
        |> Stream.run()
      end)

      receive do
        :token -> :ok
      end

      System.halt(0)
      """)

    assert status == 0, "child exited #{status}:\n#{output}"
  end
end
