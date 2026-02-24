# LlamaCppEx

Elixir bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) — run LLMs locally with Metal, CUDA, Vulkan, or CPU acceleration.

Built with C++ NIFs using [fine](https://github.com/elixir-nx/fine) for ergonomic resource management and [elixir_make](https://hex.pm/packages/elixir_make) for the build system.

## Features

- Load and run GGUF models directly from Elixir
- GPU acceleration: Metal (macOS), CUDA (NVIDIA), Vulkan, or CPU
- Streaming token generation via lazy `Stream`
- Jinja chat templates with `enable_thinking` support (Qwen3, Qwen3.5, etc.)
- RAII resource management — models, contexts, and samplers are garbage collected by the BEAM
- Configurable sampling: temperature, top-k, top-p, min-p, repetition penalty, frequency & presence penalty
- Embedding generation with L2 normalization
- Grammar-constrained generation (GBNF)
- Continuous batching server for concurrent inference
- Telemetry integration for observability

## Installation

Add `llama_cpp_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llama_cpp_ex, "~> 0.2.0"}
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

# Chat with thinking disabled (Qwen3/3.5 and similar models)
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "What is 2+2?"}
], max_tokens: 64, enable_thinking: false)

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
{:ok, sampler} = LlamaCppEx.Sampler.create(model, temp: 0.7, top_p: 0.9)

# Run generation with your own context
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "The answer is")
{:ok, text} = LlamaCppEx.Context.generate(ctx, sampler, tokens, max_tokens: 100)

# Model introspection
LlamaCppEx.Model.desc(model)          # "llama 7B Q4_K - Medium"
LlamaCppEx.Model.n_params(model)      # 6_738_415_616
LlamaCppEx.Model.chat_template(model) # "<|im_start|>..."
LlamaCppEx.Tokenizer.vocab_size(model) # 32000
```

## Server (Continuous Batching)

For concurrent inference, `LlamaCppEx.Server` manages a shared model/context with a slot pool and continuous batching:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_gpu_layers: -1,
  n_parallel: 4,
  n_ctx: 8192
)

# Synchronous
{:ok, text} = LlamaCppEx.Server.generate(server, "Once upon a time", max_tokens: 100)

# Streaming
LlamaCppEx.Server.stream(server, "Tell me a story", max_tokens: 200)
|> Enum.each(&IO.write/1)
```

Multiple callers are batched into a single forward pass per tick, improving throughput under load.

## Benchmarks

Measured on Apple M4 Max (64 GB) with Qwen3-4B Q4_K_M, Metal backend (`n_gpu_layers: -1`).

### Single-sequence generation

| Prompt | 32 tokens | 128 tokens |
|--------|-----------|------------|
| short (6 tok) | 0.31s (3.19 ips) | 1.01s (0.98 ips) |
| medium (100 tok) | 0.36s (2.79 ips) | 1.06s (0.94 ips) |
| long (500 tok) | 0.65s (1.53 ips) | 1.29s (0.77 ips) |

### Continuous batching throughput

```
max_tokens: 32, prompt: "short"
──────────────────────────────────────────────────────────────────────────────
Concurrency  Wall time    Total tok/s  Per-req tok/s  Speedup  Avg batch
1            318ms        100.6        100.6          1.00x    1.1
2            440ms        145.5         72.7          1.45x    2.2
4            824ms        155.3         38.8          1.54x    4.5
```

Run benchmarks yourself:

```bash
MIX_ENV=bench mix deps.get
LLAMA_MODEL_PATH=path/to/model.gguf MIX_ENV=bench mix run bench/single_generate.exs
LLAMA_MODEL_PATH=path/to/model.gguf MIX_ENV=bench mix run bench/server_concurrent.exs
```

## Running Qwen3.5-35B-A3B

[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GGUF) is a Mixture-of-Experts model with 35B total parameters but only 3B active per token. It supports 256K context and both thinking (CoT) and non-thinking modes.

### Hardware requirements

| Quantization | RAM / VRAM | File size |
|-------------|------------|-----------|
| Q4_K_M | ~20 GB | ~19 GB |
| Q8_0 | ~37 GB | ~36 GB |
| BF16 | ~70 GB | ~67 GB |

### Download

```bash
# Install the HuggingFace CLI if needed: pip install huggingface-hub
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir models/
```

### Thinking mode (general)

```elixir
:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model("models/Qwen3.5-35B-A3B-Q4_K_M.gguf", n_gpu_layers: -1)

# Qwen3.5 recommended: temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Explain the birthday paradox."}
], max_tokens: 2048, temp: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, penalty_present: 1.5)
```

### Thinking mode (math/code)

```elixir
# For math and code, lower temperature without presence penalty
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Write a function to find the longest palindromic substring."}
], max_tokens: 4096, temp: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0)
```

### Non-thinking mode

```elixir
# Disable thinking via enable_thinking option (uses Jinja chat template kwargs)
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "What is the capital of France?"}
], max_tokens: 256, enable_thinking: false, temp: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, penalty_present: 1.5)
```

### Streaming with Server

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  n_gpu_layers: -1,
  n_parallel: 2,
  n_ctx: 16384,
  temp: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, penalty_present: 1.5
)

LlamaCppEx.Server.stream(server, "Explain monads in simple terms", max_tokens: 1024)
|> Enum.each(&IO.write/1)
```

### Qwen3.5 enable_thinking benchmarks

Measured on **MacBook Pro, Apple M4 Max (16-core, 64 GB)**, Metal backend, `n_gpu_layers: -1`, 512 output tokens, `temp: 0.6`.

| Metric | Qwen3.5-27B (Q4_K_XL) | Qwen3.5-35B-A3B (Q6_K) |
|---|---|---|
| | Think ON / Think OFF | Think ON / Think OFF |
| **Prompt tokens** | 65 / 66 | 65 / 66 |
| **Output tokens** | 512 / 512 | 512 / 512 |
| **TTFT** | 599 ms / 573 ms | 554 ms / 191 ms |
| **Prompt eval** | 108.5 / 115.2 t/s | 117.3 / 345.5 t/s |
| **Gen speed** | 17.5 / 17.3 t/s | 56.0 / 56.0 t/s |
| **Total time** | 29.77 / 30.10 s | 9.69 / 9.33 s |

The MoE model (35B-A3B) is ~3.2x faster at generation since only 3B parameters are active per token despite the 35B total. Thinking mode only affects the prompt template, not inference speed.

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
