# Fetch the benchmark models onto a Spark.
#
#   scripts/spark/remote.sh spark-1 mix run scripts/spark/fetch_models.exs
#   scripts/spark/remote.sh spark-1 mix run scripts/spark/fetch_models.exs 8b 30b
#   scripts/spark/remote.sh spark-1 mix run scripts/spark/fetch_models.exs --list
#
# Uses LlamaCppEx.Hub.download/3 rather than curl or the huggingface CLI, so the
# download path this library ships is the one that gets exercised — including
# the SHA-256 verification, which is fail-closed by default.
#
# Files land at $LLAMA_CACHE_DIR/<repo>/<revision>/<filename>; remote.sh sets
# LLAMA_CACHE_DIR to ~/models.

defmodule FetchModels do
  # Sizes are the exact byte counts HuggingFace reports, resolved 2026-08-12.
  # 121 GiB of unified memory is 130.0 GB, so:
  #   - gpt-oss-120b at 63.4 GB fits one node with room for KV. It is the A/B
  #     control for measuring pure RPC overhead: same model, one node vs two.
  #   - Qwen3-235B-A22B Q4_K_M is 142.1 GB across three shards. It does NOT fit
  #     one node, which is the entire justification for the second Spark. The
  #     plan estimated ~133 GB; the real number is 142.1 GB, which only makes
  #     the case stronger.
  @models [
    %{
      label: "8b",
      repo: "Qwen/Qwen3-8B-GGUF",
      files: ["Qwen3-8B-Q4_K_M.gguf"],
      bytes: 5_027_782_656,
      note: "dense sanity check; external reference 43.7 t/s tg, 3167 t/s pp512"
    },
    %{
      label: "30b",
      repo: "Qwen/Qwen3-30B-A3B-GGUF",
      files: ["Qwen3-30B-A3B-Q4_K_M.gguf"],
      bytes: 18_565_509_120,
      note: "MoE; external reference 89.3 t/s tg, 2541 t/s pp512"
    },
    %{
      label: "120b",
      repo: "ggml-org/gpt-oss-120b-GGUF",
      files: ["gpt-oss-120b-MXFP4.gguf"],
      bytes: 63_390_146_560,
      note: "big but fits one node — the controlled A/B for RPC overhead"
    },
    %{
      label: "235b",
      repo: "unsloth/Qwen3-235B-A22B-GGUF",
      files: [
        "Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf",
        "Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00002-of-00003.gguf",
        "Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00003-of-00003.gguf"
      ],
      bytes: 142_100_000_000,
      note: "142.1 GB across 3 shards — does not fit 121 GiB; the two-node headline"
    },
    # Qwen3.6, the current generation. Two shapes (dense 27B, MoE 35B-A3B) and
    # each in a plain and an MTP build. The MTP repos are the same weights plus
    # the Multi-Token Prediction head, which is why they are a few hundred MB
    # larger — llama.cpp reads those layers only when the model is loaded with
    # `load_mtp: true`, so the plain and MTP files are a clean A/B for what
    # speculative decoding buys on this hardware.
    %{
      label: "q36-27b",
      repo: "unsloth/Qwen3.6-27B-GGUF",
      files: ["Qwen3.6-27B-Q4_K_M.gguf"],
      bytes: 16_820_000_000,
      note: "Qwen3.6 dense 27B"
    },
    %{
      label: "q36-27b-mtp",
      repo: "unsloth/Qwen3.6-27B-MTP-GGUF",
      files: ["Qwen3.6-27B-Q4_K_M.gguf"],
      bytes: 17_110_000_000,
      note: "Qwen3.6 dense 27B with the MTP head — needs load_mtp: true"
    },
    %{
      label: "q36-35b",
      repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
      files: ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],
      bytes: 22_130_000_000,
      note: "Qwen3.6 MoE 35B-A3B"
    },
    %{
      label: "q36-35b-mtp",
      repo: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
      files: ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],
      bytes: 22_660_000_000,
      note: "Qwen3.6 MoE 35B-A3B with the MTP head — needs load_mtp: true"
    }
  ]

  def models, do: @models

  def gb(bytes), do: Float.round(bytes / 1_000_000_000, 1)

  def run(labels) do
    selected =
      case labels do
        [] -> @models
        _ -> Enum.filter(@models, &(&1.label in labels))
      end

    if selected == [] do
      IO.puts(:stderr, "no model matches #{inspect(labels)}; try --list")
      System.halt(2)
    end

    total = selected |> Enum.map(& &1.bytes) |> Enum.sum()
    IO.puts("fetching #{length(selected)} model(s), #{gb(total)} GB total\n")

    results = Enum.flat_map(selected, &fetch/1)

    IO.puts("")

    Enum.each(results, fn
      {:ok, path} -> IO.puts("  ok    #{path}")
      {:error, what, reason} -> IO.puts("  FAIL  #{what}: #{reason}")
    end)

    if Enum.any?(results, &match?({:error, _, _}, &1)), do: System.halt(1)
  end

  defp fetch(model) do
    Enum.map(model.files, fn file ->
      IO.puts("#{model.label}  #{model.repo}/#{file}")
      started = System.monotonic_time(:millisecond)

      # verify_checksum defaults to true and fails closed. Left at the default
      # deliberately: a 142 GB download that silently truncates is exactly the
      # failure this check exists for.
      case LlamaCppEx.Hub.download(model.repo, file) do
        {:ok, path} ->
          elapsed = (System.monotonic_time(:millisecond) - started) / 1000
          size = File.stat!(path).size

          IO.puts(
            "        #{gb(size)} GB in #{Float.round(elapsed, 1)}s" <>
              if(elapsed > 1,
                do: " (#{Float.round(size / elapsed / 1_000_000, 1)} MB/s)",
                else: " (cached)"
              )
          )

          {:ok, path}

        {:error, reason} ->
          IO.puts("        FAILED: #{reason}")
          {:error, "#{model.repo}/#{file}", reason}
      end
    end)
  end

  def list do
    IO.puts("cache dir: #{System.get_env("LLAMA_CACHE_DIR") || "~/.cache/llama_cpp_ex/models"}\n")

    Enum.each(@models, fn m ->
      IO.puts(
        "#{String.pad_trailing(m.label, 6)} #{String.pad_leading("#{gb(m.bytes)} GB", 9)}  #{m.repo}"
      )

      IO.puts("       #{m.note}")
      Enum.each(m.files, &IO.puts("       - #{&1}"))
      IO.puts("")
    end)
  end
end

case System.argv() do
  ["--list"] -> FetchModels.list()
  labels -> FetchModels.run(labels)
end
