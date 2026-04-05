# HuggingFace Hub Download Example
#
# Demonstrates searching, listing, and downloading GGUF models from HuggingFace Hub.
# Requires the optional :req dependency.
#
# Usage:
#   mix run examples/hub_download.exs

IO.puts("=== HuggingFace Hub Integration ===\n")

# --- Search for GGUF models ---

IO.puts("Searching for 'qwen3 gguf' models...\n")

{:ok, models} = LlamaCppEx.Hub.search("qwen3 gguf", limit: 5)

for m <- models do
  IO.puts("  #{m.id}")
  IO.puts("    Downloads: #{m.downloads} | Likes: #{m.likes} | Gated: #{m.gated}")
end

IO.puts("")

# --- List GGUF files in a repository ---

repo = "Qwen/Qwen3-0.6B-GGUF"
IO.puts("Listing GGUF files in #{repo}...\n")

{:ok, files} = LlamaCppEx.Hub.list_gguf_files(repo)

for f <- files do
  size_mb = Float.round(f.size / 1_000_000, 1)
  IO.puts("  #{f.filename} (#{size_mb} MB)")
end

IO.puts("")

# --- Download and load a model ---

filename = hd(files).filename
IO.puts("Downloading #{repo}/#{filename}...\n")

{:ok, path} = LlamaCppEx.Hub.download(repo, filename)
IO.puts("  Cached at: #{path}\n")

# Load and generate
:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(path, n_gpu_layers: -1)
{:ok, text} = LlamaCppEx.generate(model, "Elixir is", max_tokens: 32, temp: 0.0)
IO.puts("Generated: #{text}")

IO.puts("\n--- Or use the convenience wrapper ---\n")

# One-liner: download + load
{:ok, model2} = LlamaCppEx.load_model_from_hub(repo, filename, n_gpu_layers: -1)
{:ok, text2} = LlamaCppEx.generate(model2, "The BEAM VM", max_tokens: 32, temp: 0.0)
IO.puts("Generated: #{text2}")
