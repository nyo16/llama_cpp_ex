# LlamaCppEx

Elixir bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) — run LLMs locally with Metal, CUDA, Vulkan, or CPU acceleration.

Built with C++ NIFs using [fine](https://github.com/elixir-nx/fine) for ergonomic resource management and [elixir_make](https://hex.pm/packages/elixir_make) for the build system.

## Features

- Load and run GGUF models directly from Elixir
- GPU acceleration: Metal (macOS), CUDA (NVIDIA), Vulkan, or CPU
- Streaming token generation via lazy `Stream`
- Chat template support (ChatML, Llama, etc.)
- RAII resource management — models, contexts, and samplers are garbage collected by the BEAM
- Configurable sampling: temperature, top-k, top-p, min-p, repetition penalty

## Installation

Add `llama_cpp_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llama_cpp_ex, "~> 0.1.0"}
  ]
end
```

### Prerequisites

- C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.14+
- Git (for the llama.cpp submodule)

### Backend Selection

```bash
mix compile                        # Auto-detect (Metal on macOS, CUDA if nvcc found, else CPU)
LLAMA_BACKEND=metal mix compile    # Apple Silicon GPU
LLAMA_BACKEND=cuda mix compile     # NVIDIA GPU
LLAMA_BACKEND=vulkan mix compile   # Vulkan
LLAMA_BACKEND=cpu mix compile      # CPU only
```

Power users can pass arbitrary CMake flags:

```bash
LLAMA_CMAKE_ARGS="-DGGML_CUDA_FORCE_CUBLAS=ON" mix compile
```

## Quick Start

```elixir
# Initialize the backend (once per application)
:ok = LlamaCppEx.init()

# Load a GGUF model (use n_gpu_layers: -1 to offload all layers to GPU)
{:ok, model} = LlamaCppEx.load_model("path/to/model.gguf", n_gpu_layers: -1)

# Generate text
{:ok, text} = LlamaCppEx.generate(model, "Once upon a time", max_tokens: 200, temp: 0.8)

# Stream tokens
model
|> LlamaCppEx.stream("Tell me a story", max_tokens: 500)
|> Enum.each(&IO.write/1)

# Chat with template
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "system", content: "You are a helpful assistant."},
  %{role: "user", content: "What is Elixir?"}
], max_tokens: 200)

# Stream a chat response
model
|> LlamaCppEx.stream_chat([
  %{role: "user", content: "Explain pattern matching in Elixir."}
], max_tokens: 500)
|> Enum.each(&IO.write/1)
```

## Lower-level API

For fine-grained control over the inference pipeline:

```elixir
# Tokenize
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello world")
{:ok, text} = LlamaCppEx.Tokenizer.decode(model, tokens)

# Create context and sampler separately
{:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 4096)
{:ok, sampler} = LlamaCppEx.Sampler.create(temp: 0.7, top_p: 0.9)

# Run generation with your own context
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "The answer is")
{:ok, text} = LlamaCppEx.Context.generate(ctx, sampler, tokens, max_tokens: 100)

# Model introspection
LlamaCppEx.Model.desc(model)          # "llama 7B Q4_K - Medium"
LlamaCppEx.Model.n_params(model)      # 6_738_415_616
LlamaCppEx.Model.chat_template(model) # "<|im_start|>..."
LlamaCppEx.Tokenizer.vocab_size(model) # 32000
```

## Architecture

```
Elixir API (lib/)
    │
LlamaCppEx.NIF (@on_load, stubs)
    │
C++ NIF layer (c_src/) — fine.hpp for RAII + type encoding
    │
llama.cpp static libs (vendor/llama.cpp, built via CMake)
    │
Hardware (CPU / Metal / CUDA / Vulkan)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

llama.cpp is licensed under the MIT License.
