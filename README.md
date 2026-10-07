# LlamaCppEx

[![Precompile NIFs](https://github.com/nyo16/llama_cpp_ex/actions/workflows/precompile.yml/badge.svg)](https://github.com/nyo16/llama_cpp_ex/actions/workflows/precompile.yml)
[![CI](https://github.com/nyo16/llama_cpp_ex/actions/workflows/ci.yml/badge.svg)](https://github.com/nyo16/llama_cpp_ex/actions/workflows/ci.yml)

Elixir bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) — run LLMs locally with Metal, CUDA, Vulkan, or CPU acceleration.

Built with C++ NIFs using [fine](https://github.com/elixir-nx/fine) for ergonomic resource management and [elixir_make](https://hex.pm/packages/elixir_make) for the build system.

## Features

- Load and run GGUF models directly from Elixir
- **HuggingFace Hub integration** — search, list, and download GGUF models
- GPU acceleration: Metal (macOS), CUDA (NVIDIA), Vulkan, or CPU
- Streaming token generation via lazy `Stream`
- Jinja chat templates with `enable_thinking` support (Qwen3, Qwen3.5, etc.)
- RAII resource management — models, contexts, and samplers are garbage collected by the BEAM
- Configurable sampling: temperature, top-k, top-p, min-p, repetition penalty, frequency & presence penalty
- Embedding generation with L2 normalization
- Grammar-constrained generation (GBNF)
- Structured output via JSON Schema (auto-converted to GBNF grammar)
- Optional Ecto schema to JSON Schema conversion
- **Decision models** — typed choice/score/yes-no answers with probabilities and confidence, in one forward pass (llama.cpp's `/v1/systemone` API, in-process)
- Continuous batching server for concurrent inference
- **Multi-model manager** — keep several models resident, route requests by id, with a placement-aware (per-GPU VRAM) memory budget
- **Device introspection** — `LlamaCppEx.devices/0` lists GPUs/accelerators with per-device VRAM
- **Multi-Token Prediction (MTP) speculative decoding** — ~2x token-generation speedup on Qwen 3.6 with live acceptance-rate stats
- **Prefix caching** — cross-slot KV reuse with session affinity and an optional RAM prompt cache (TTFT 115 → 35 ms at ~75% hit ratio under concurrency)
- **Pluggable batching strategies** — DecodeMaximal, PrefillPriority, Balanced
- **Pre-tokenized API** — tokenize outside the GenServer for lower contention
- **Request lifecycle controls** — per-request sampling params, `:max_queue` backpressure, and cancellation (dead or halted stream consumers free their slot immediately)
- Telemetry integration for observability

## Installation

Add `llama_cpp_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llama_cpp_ex, "~> 0.7.5"}
  ]
end
```

### Prerequisites

Elixir `~> 1.18`, enforced by `mix.exs`. The Erlang/OTP floor depends on how the
NIF is obtained:

Precompiled NIFs are published at NIF versions 2.17 and 2.18 — that is
**Erlang/OTP 26 or newer** (OTP 26, 27 and 28 report NIF 2.17; OTP 29 reports
2.18) — for:

| Target | Backend | Selected when |
|---|---|---|
| `aarch64-apple-darwin` | Metal | Apple Silicon |
| `x86_64-linux-gnu` | CPU | no usable CUDA install found |
| `x86_64-linux-gnu-cu12` | CUDA | driver plus a CUDA 12 runtime |
| `x86_64-linux-gnu-cu13` | CUDA | driver plus a CUDA 13 runtime |

On those platforms `mix deps.get` and `mix compile` download a binary and none
of the build tooling below is needed.

The CUDA variants are separate artifacts rather than one because the NIF links
`libcudart`/`libcublas`/`libcublasLt` dynamically and those sonames are
major-versioned: a cu13 build cannot load against a CUDA 12 install. Selection
is automatic and deliberately conservative — it requires **both** a CUDA runtime
and a driver (`libcuda.so.1`), because a CUDA artifact on a machine with no
driver cannot be loaded at all, and falls back to the CPU artifact otherwise.
Override with `LLAMA_CUDA_VARIANT=cu12|cu13|none` if the probe gets it wrong.

No CUDA toolkit is needed to *run* these; only the runtime libraries they link.

Everything else builds from source: OTP 25 (NIF 2.16), other architectures
(including aarch64 Linux — DGX Spark and friends), musl, Windows, Vulkan, and
any CUDA major version without a published artifact. A source build needs

- a C++17 compiler (GCC, Clang, or MSVC),
- CMake 3.14+,
- Git — the Hex package ships `.gitmodules` rather than the llama.cpp tree, so
  the Makefile clones the pinned commit on demand.

### Backend Selection

A downloaded artifact never runs the Makefile, so nothing is auto-detected at
install time — the backend is whatever that artifact was built with. Vulkan, and
CUDA on any platform without a published variant, always mean an explicit source
build:

```bash
mix compile                        # Precompiled artifact when one matches this
                                   # OS/arch/NIF version, else a source build
LLAMA_BACKEND=metal mix compile    # Apple Silicon GPU
LLAMA_BACKEND=cuda mix compile     # NVIDIA GPU, needs the CUDA toolkit
LLAMA_BACKEND=vulkan mix compile   # Vulkan, needs the Vulkan SDK
LLAMA_BACKEND=cpu mix compile      # CPU only
```

Setting `LLAMA_BACKEND` to anything forces a source build and bypasses the
precompiled artifact. When a source build runs with `LLAMA_BACKEND` unset it
picks Metal on macOS, CUDA if a toolkit is found, and CPU otherwise.

CUDA is located by `CUDA_HOME`, then `CUDA_PATH`, then `nvcc` on `PATH`, then
`/usr/local/cuda`, `/opt/cuda` and the versioned `/usr/local/cuda-*` directories.
`PATH` alone is not enough to rely on: DGX OS, environment modules and most CI
shells leave `nvcc` off a non-login `PATH`, which used to mean a silent CPU-only
build on a machine with a perfectly good toolkit.

Power users can pass arbitrary CMake flags:

```bash
LLAMA_CMAKE_ARGS="-DGGML_CUDA_FORCE_CUBLAS=ON" mix compile
```

More build variables:

- `LLAMA_PORTABLE=1` drops `-march=native`. ggml turns it on by default, which
  tunes the binary to the exact CPU it was built on; the release workflow sets
  this so published artifacts run on every machine of that architecture. Leave
  it unset locally, where the native flags are free performance.
- `LLAMA_CUDA_NCCL=1` builds and links ggml's NCCL collectives, which speed up
  multi-GPU work. Off by default: ggml would otherwise enable NCCL silently
  whenever the build host happens to have it, and since the Makefile assembles
  the link line by hand rather than through cmake, that produced a NIF that
  failed to load with `undefined symbol: ncclAllReduce`. Turning it on also
  makes `libnccl.so.2` a load-time requirement.
- `LLAMA_CPU_ARM_ARCH=<arch>` names the architecture for ggml's CPU backend
  instead of letting `-mcpu=native` probe for it. Needed wherever that probe
  gives the wrong answer *silently* — on DGX Spark (GB10) with GCC 13.3 it
  degrades to base ARMv8-A behind a soft warning, and the emitted
  `libggml-cpu.a` loses every `sdot`, `smmla` and SVE instruction, i.e. the Q4/Q8
  quantized matmul kernels. On a CUDA build this **requires** `LLAMA_CUDA_ARCH`
  and errors without it.
- `LLAMA_CUDA_ARCH=<arch>` sets `CMAKE_CUDA_ARCHITECTURES`, e.g. `121a-real` for
  GB10. Reaching the CPU flag above needs `GGML_NATIVE=OFF`, which otherwise
  turns one CUDA architecture into a seven-architecture fat binary — a ~6× build
  with no runtime benefit.
- `LLAMA_RPC=1` builds the ggml RPC backend, which lets a model's layers live on
  another machine. Off by default: it is a networked surface and a protocol
  version coupling. `LLAMA_RPC_RDMA` (default `1` on Linux) declares whether the
  transport may use RDMA, rather than letting ggml enable it based on whether
  the build host happens to have `libibverbs`. See `LlamaCppEx.RPC`.
- `LLAMA_COMMIT=<sha>` overrides the pinned llama.cpp commit used when
  `vendor/llama.cpp` has to be cloned.

On a DGX Spark, all of the above is `scripts/spark/remote.sh spark-1 mix compile`
— see [docs/dgx-spark.md](docs/dgx-spark.md) for the one- and two-node runbook.

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

## HuggingFace Hub

Download GGUF models directly from HuggingFace Hub. Requires the optional `:req` dependency:

```elixir
{:req, "~> 0.5 or ~> 0.6"}
```

```elixir
# Search for GGUF models
{:ok, models} = LlamaCppEx.Hub.search("qwen3 gguf", limit: 5)

# List GGUF files in a repository
{:ok, files} = LlamaCppEx.Hub.list_gguf_files("Qwen/Qwen3-0.6B-GGUF")

# Download (cached locally in ~/.cache/llama_cpp_ex/models/)
{:ok, path} = LlamaCppEx.Hub.download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")

# Or download + load in one step
{:ok, model} = LlamaCppEx.load_model_from_hub(
  "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf",
  n_gpu_layers: -1
)

# Private or gated repo — pass a HuggingFace token explicitly
{:ok, model} = LlamaCppEx.load_model_from_hub(
  "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf",
  token: "hf_xxx",
  n_gpu_layers: -1
)
```

For private/gated models, set `HF_TOKEN` or pass `token: "hf_..."`. Set `LLAMA_OFFLINE=1` for offline-only cached access.

## Embeddings

`LlamaCppEx.embed/3` and `LlamaCppEx.embed_batch/3` run an embedding GGUF and
return one L2-normalized vector per text (`normalize: -1` for raw, `format:
:binary` for a zero-copy f32 binary that `Nx.from_binary(bin, :f32)` loads).
`embed_batch/3` packs the texts into one context as separate sequences and
decodes them in as few batches as fit. The vector length is
`LlamaCppEx.Model.n_embd_out/1` — not `n_embd/1`, which is the hidden width
and differs for models with an output projection.

### EmbeddingGemma 2

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) is
a 270M-parameter text embedder (768-dimensional output from a 512-wide
backbone, 100+ languages and code, Apache 2.0). ggml-org publishes it as
[`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)
in Q8_0 (310 MB) and BF16 (558 MB); the `mmproj-*` files beside them hold the
vision and audio encoders, which need libmtmd and are not used by this
binding — text input only. Needs llama.cpp `b11452` or later (#30054), so
`llama_cpp_ex` 0.8.56+; the full 768 dimensions also need 0.8.56's
`n_embd_out` fix.

```elixir
:ok = LlamaCppEx.init()

{:ok, model} =
  LlamaCppEx.load_model_from_hub("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf",
    n_gpu_layers: -1
  )

LlamaCppEx.Model.n_embd_out(model)
# => 768

# Retrieval is asymmetric: one prefix for the query, another for the documents.
# `title: none` when a document has no title.
{:ok, query} = LlamaCppEx.embed(model, "task: search result | query: What causes the northern lights?")

{:ok, [aurora, elixir]} =
  LlamaCppEx.embed_batch(model, [
    "title: none | text: The northern lights are caused by charged particles from the sun colliding with the atmosphere.",
    "title: none | text: Elixir is a dynamic, functional language for building scalable applications."
  ])

# Vectors are unit length, so the dot product is the cosine similarity.
dot = fn a, b -> Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum() end
dot.(query, aurora)   # => 0.8755
dot.(query, elixir)   # => 0.5924
```

The model is trained with a short task prefix on every text; leaving it off
still works but loses precision. Asymmetric tasks pair a query prefix with the
document form, symmetric tasks put the same prefix on every side:

| Task | Query / input prefix | Document form |
|---|---|---|
| Search | `task: search result \| query: {q}` | `title: {title} \| text: {content}` |
| Question answering | `task: question answering \| query: {q}` | `title: {title} \| text: {passage}` |
| Fact checking | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| Code search | `task: code retrieval \| query: {q}` | `title: {filename} \| text: {code}` |
| Classification | `task: classification \| query: {text}` | — |
| Clustering | `task: clustering \| query: {text}` | — |
| Sentence similarity | `task: sentence similarity \| query: {text}` | — |

```elixir
{:ok, [a, b, c]} =
  LlamaCppEx.embed_batch(model, [
    "task: sentence similarity | query: The cat sleeps on the sofa.",
    "task: sentence similarity | query: A cat is napping on the couch.",
    "task: sentence similarity | query: Quarterly revenue grew by 12%."
  ])

dot.(a, b)   # => 0.9853
dot.(a, c)   # => 0.7010
```

**Matryoshka truncation.** The 768 dimensions are ordered by importance, so a
vector can be cut to its first 512, 256 or 128 and re-normalized (slicing a
unit vector does not keep it unit length). Queries and documents must share a
dimension. Google reports near-lossless quality down to 256 for text;
the retrieval example above keeps its ranking at 256 (0.899 vs 0.611):

```elixir
truncate = fn vec, dim ->
  head = Enum.take(vec, dim)
  norm = :math.sqrt(Enum.reduce(head, 0.0, &(&2 + &1 * &1)))
  Enum.map(head, &(&1 / norm))
end

q256 = truncate.(query, 256)
dot.(q256, truncate.(aurora, 256))   # => 0.8994
dot.(q256, truncate.(elixir, 256))   # => 0.6111
```

Through `LlamaCppEx.ModelManager` the same model is `capabilities: [:embed]`
(see [Multiple Models](#multiple-models-modelmanager)); the task prefixes are
part of the text you pass to `ModelManager.embed/2`.

## Structured Output (JSON Schema)

Constrain model output to valid JSON matching a schema. Pass `:json_schema` to any generate or chat function — the schema is automatically converted to a GBNF grammar via llama.cpp's built-in converter.

```elixir
schema = %{
  "type" => "object",
  "properties" => %{
    "name" => %{"type" => "string"},
    "age" => %{"type" => "integer"},
    "hobbies" => %{"type" => "array", "items" => %{"type" => "string"}}
  },
  "required" => ["name", "age", "hobbies"],
  "additionalProperties" => false
}

# Works with generate
{:ok, json} = LlamaCppEx.generate(model, "Generate a person:",
  json_schema: schema, temp: 0.0)
# => "{\"name\": \"Alice\", \"age\": 30, \"hobbies\": [\"reading\", \"hiking\"]}"

# Works with chat
{:ok, json} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Generate a person named Bob who is 25."}
], json_schema: schema, temp: 0.0)

# Works with streaming
model
|> LlamaCppEx.stream("Generate a person:", json_schema: schema, temp: 0.0)
|> Enum.each(&IO.write/1)

# Works with chat completions
{:ok, completion} = LlamaCppEx.chat_completion(model, [
  %{role: "user", content: "Generate a person."}
], json_schema: schema, temp: 0.0)
```

> **Tip:** Set `"additionalProperties" => false` in your schema to produce a tighter grammar
> that avoids potential issues with the grammar sampler.

### Manual Grammar Conversion

You can also convert the schema to GBNF manually for more control:

```elixir
{:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
IO.puts(gbnf)
# root ::= "{" space name-kv "," space age-kv "," space hobbies-kv "}" space
# ...

# Use the grammar directly
{:ok, json} = LlamaCppEx.generate(model, "Generate a person:", grammar: gbnf, temp: 0.0)
```

### Ecto Schema Integration

Convert Ecto schema modules to JSON Schema automatically (requires `{:ecto, "~> 3.0"}` — optional dependency):

```elixir
defmodule MyApp.Person do
  use Ecto.Schema

  embedded_schema do
    field :name, :string
    field :age, :integer
    field :active, :boolean
    field :tags, {:array, :string}
  end
end

# Ecto schema -> JSON Schema -> constrained generation
schema = LlamaCppEx.Schema.to_json_schema(MyApp.Person)
# => %{"type" => "object", "properties" => %{"name" => %{"type" => "string"}, ...}, ...}

{:ok, json} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Generate a person."}
], json_schema: schema, temp: 0.0)
```

Supported Ecto types: `:string`, `:integer`, `:float`, `:decimal`, `:boolean`, `:map`, `{:array, inner}`, `:date`, `:utc_datetime`, `:naive_datetime`, and embedded schemas (`embeds_one`/`embeds_many`). Fields `:id`, `:inserted_at`, and `:updated_at` are excluded automatically.

## Decision Models

A decision model answers typed questions about a `state` in one forward pass, with no token generated: each answer is a probability distribution, so it comes with a confidence instead of free text. `LlamaCppEx.Decision` is llama-server's [`/v1/systemone`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api) endpoint (TypeSafe-compatible System One API) ported into the NIF. It builds the same prompts and returns the same answers; checked against upstream `llama-server` on tinylaya, tinyopenjev, Julia-1 and Clef-Flash, with identical token counts and probabilities equal to 1e-6.

```elixir
{:ok, model} = LlamaCppEx.load_model("Laya-Q8_0.gguf", n_gpu_layers: -1)
{:ok, decision} = LlamaCppEx.Decision.new(model)

{:ok, %{answers: answers}} =
  LlamaCppEx.Decision.decide(
    decision,
    "Customer message: I was charged twice for my order last week.",
    route: [
      type: :choice,
      instructions: "Which team should handle this?",
      criteria: [billing: "payments and refunds", shipping: nil, technical: nil]
    ],
    angry: [type: :noul, instructions: "Is the customer angry?"],
    urgency: [
      type: :score,
      instructions: "How urgent is this?",
      criteria: ["can wait", "this week", "today", "right now"]
    ]
  )

# Laya-Q8_0, rounded:
answers.route
# => %{type: :choice, choice: :billing, confidence: 0.988,
#      probabilities: %{billing: 0.992, shipping: 0.005, technical: 0.003}}
answers.angry   # => %{type: :noul, noul: 0.812}
answers.urgency # => %{type: :score, score: 1.153, confidence: 0.811,
                #      legend: %{0 => "can wait", 1 => "this week", ...},
                #      probabilities: %{0 => 0.018, 1 => 0.888, 2 => 0.018, 3 => 0.077}}

# One-off: creates the decision context for this call only
{:ok, result} = LlamaCppEx.decide(model, state, questions)
```

The model file declares its decision type; `LlamaCppEx.Decision.model_type/1` reads it. All eight upstream types are supported: `openjev`, `lev`, `nimble`, `pplx-decider`, `lfm2-d1` (label logits), `kev` (hidden-state dot product), `laya` (ModernBERT encoder) and `clef` (all questions decided jointly in one prompt). ggml-org publishes them in its "Decision models" Hugging Face collection, e.g. [`ggml-org/Laya-GGUF`](https://huggingface.co/ggml-org/Laya-GGUF), [`ggml-org/Clef-Flash-GGUF`](https://huggingface.co/ggml-org/Clef-Flash-GGUF), [`ggml-org/OpenJev-GGUF`](https://huggingface.co/ggml-org/OpenJev-GGUF); LiquidAI publishes `d1-3B`.

`lfm2-d1` is [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (3B, LFM2.5-VL-based, accepts a `nil` state). Upstream renamed its type from `d1` to `lfm2-d1` in the second commit of the PR that added it (#30110); a GGUF converted before that loads as `model_type/1 == :unknown` and `Decision.new/2` returns `{:error, "unsupported decision model type: d1"}` — the same refusal `llama-server` gives. Re-convert, or rewrite the `lfm2.decision.type` key to `"lfm2-d1"` (one `gguf-py` call); checked against upstream on such a file, every probability and token count agrees.

Limits: text only (upstream's image input needs libmtmd, which this build does not link), laya and clef evaluate the whole prompt in one batch (`:n_batch`, 2048 by default), and a `%Decision{}` is single-process like a `Context`.

### Clef Flash

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) is a 9B
Qwen3.5-based clef model: one prompt carries the state *and every question*,
and the model decides them jointly, so the state is paid for once per request
rather than once per question. GGUFs:
[`bartowski/Cloudflare_clef-flash-GGUF`](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF)
(Q2_K through Q8_0 and bf16; static quants, because an imatrix cannot be
calibrated on a model that generates nothing, so prefer Q4_K_M or larger — the
decision head is Q8_0 in every file) and
[`ggml-org/Clef-Flash-GGUF`](https://huggingface.co/ggml-org/Clef-Flash-GGUF).
The `mmproj-*` files in both repos are for image input, which this binding
does not link.

```elixir
{:ok, model} =
  LlamaCppEx.load_model_from_hub(
    "bartowski/Cloudflare_clef-flash-GGUF", "Cloudflare_clef-flash-Q4_K_M.gguf",
    n_gpu_layers: -1
  )

LlamaCppEx.Decision.model_type(model)
# => :clef

# Clef reads the embeddings output, so Decision.new/2 builds a context with
# embeddings on, no pooling, and n_batch = n_ubatch = min(n_ctx, 2048): the
# whole prompt (state + all questions + options) is evaluated in one batch and
# has to fit. Raise :n_ctx and :n_batch together for long states.
{:ok, decision} = LlamaCppEx.Decision.new(model, n_ctx: 8192, n_batch: 8192)

questions = [
  route: [
    type: :choice,
    instructions: "Which team should handle this?",
    criteria: [billing: nil, shipping: nil, technical: nil]
  ],
  angry: [type: :noul, instructions: "Is the customer angry?"],
  urgency: [
    type: :score,
    instructions: "How urgent is this?",
    criteria: ["can wait", "this week", "today", "right now"]
  ]
]

{:ok, %{answers: answers, usage: usage}} =
  LlamaCppEx.Decision.decide(
    decision,
    "Customer message: I was charged twice for my order last week and nobody has replied.",
    questions
  )

# Cloudflare_clef-flash-Q8_0, rounded:
usage           # => %{input_tokens: 324, output_tokens: 0}  — one prompt for all three
answers.route   # => %{type: :choice, choice: :billing, confidence: 0.966,
                #      probabilities: %{billing: 0.977, shipping: 0.014, technical: 0.008}}
answers.angry   # => %{type: :noul, noul: 0.420}
answers.urgency # => %{type: :score, score: 2.209, confidence: 0.209,
                #      probabilities: %{0 => 0.029, 1 => 0.156, 2 => 0.393, 3 => 0.423}, ...}

# The state can be a chat transcript; the same engine serves every request.
{:ok, %{answers: answers}} =
  LlamaCppEx.Decision.decide(
    decision,
    %{"messages" => [%{"role" => "user", "content" => "The app crashes when I open the settings page on Android 15."}]},
    questions
  )

answers.route.choice   # => :technical  (0.990)
answers.angry.noul     # => 0.022
```

#### More clef examples

All numbers below are `Cloudflare_clef-flash-Q8_0` on an M1 Max (Metal),
rounded; each request is one forward pass over one prompt.

**A record as the state.** A map is given to the model as JSON text, so a
database row or an API payload works as is:

```elixir
order = %{
  "order_id" => "A-10492",
  "items" => [%{"sku" => "KB-77", "qty" => 2, "price" => 129.0}],
  "shipping_country" => "DE",
  "payment" => %{"method" => "card", "attempts" => 3, "last_error" => "insufficient_funds"},
  "customer_note" => "Please ship before Friday, it's a birthday gift."
}

{:ok, %{answers: a, usage: %{input_tokens: 408}}} =
  LlamaCppEx.Decision.decide(decision, order,
    payment_ok: [type: :noul, instructions: "Did the payment succeed?"],
    risk: [
      type: :score,
      instructions: "How likely is this order to be fraudulent?",
      criteria: ["very unlikely", "unlikely", "possible", "likely"]
    ],
    action: [
      type: :choice,
      instructions: "What should happen next?",
      criteria: [
        ship: "ship the order as is",
        retry_payment: "ask the customer to retry the payment",
        manual_review: "hold for a human to review"
      ]
    ]
  )

a.payment_ok.noul    # => 0.022
a.risk.score         # => 1.198  (legend 1 = "unlikely"; probabilities 0.17 / 0.55 / 0.19 / 0.09)
a.action.choice      # => :retry_payment  (0.954; ship 0.007, manual_review 0.039)
```

**Moderation as one `choice`.** Describe the categories; a single described
choice reads a veiled threat that a bare yes/no question does not (the same
first message scores `noul` 0.02 on "Does the message contain harassment,
threats or hate speech?"):

```elixir
category = [
  type: :choice,
  instructions: "Classify the message.",
  criteria: [
    ok: "normal message",
    spam: "spam or advertising",
    threat: "threat or intimidation",
    hate: "hate speech"
  ]
]

for msg <- messages do
  {:ok, %{answers: %{category: c}}} = LlamaCppEx.Decision.decide(decision, msg, category: category)
  {c.choice, c.probabilities[c.choice]}
end
```

| message | choice |
|---|---|
| "If I see you at the office tomorrow you'll regret it." | `:threat` 0.68 |
| "I know where you live. Watch your back." | `:threat` 0.96 |
| "You people are subhuman and should be wiped out." | `:hate` 0.87 |
| "Super Produkt, schnelle Lieferung. Danke!" | `:ok` 0.93 |
| "BUY CHEAP WATCHES NOW!!! visit w4tch-deals.example" | `:spam` 0.98 |

**RAG verification.** Grade a generated answer against the passage it was
supposed to come from. Prose with a natural shape is better given as text than
as a map: "does the passage contain the information needed to answer the
question?" on `"Question: ...\n\nPassage: ..."` scores 0.92, the same content
as `%{"question" => ..., "passage" => ...}` 0.29, and an unrelated passage
0.02.

```elixir
passage = "Physical goods can be returned within 30 days of delivery. Digital purchases are final and cannot be refunded once downloaded."
question = "What is the refund window for digital purchases?"

grade = fn draft ->
  state = "Question: #{question}\n\nPassage: #{passage}\n\nDraft answer: #{draft}"

  {:ok, %{answers: a}} =
    LlamaCppEx.Decision.decide(decision, state,
      grounded: [type: :noul, instructions: "Is the draft answer supported by the passage?"],
      quality: [
        type: :score,
        instructions: "How good is the draft answer?",
        criteria: ["wrong", "partly wrong", "mostly right", "correct"]
      ]
    )

  {a.grounded.noul, a.quality.score}
end

grade.("You can get a refund on digital purchases within 30 days.")
# => {0.019, 0.141}   — "wrong" at 0.91

grade.("Digital purchases cannot be refunded once downloaded; only physical goods have a 30-day window.")
# => {0.956, 2.714}   — "correct" at 0.83
```

**Route or escalate.** `confidence` is the answer's margin, so a threshold turns
the model into a classifier that knows when to hand off:

```elixir
route = [
  type: :choice,
  instructions: "Which team should handle this?",
  criteria: [billing: nil, shipping: nil, technical: nil]
]

assign = fn ticket ->
  {:ok, %{answers: %{route: r}}} = LlamaCppEx.Decision.decide(decision, ticket, route: route)
  if r.confidence >= 0.8, do: r.choice, else: {:human, r.choice, r.confidence}
end

assign.("I was charged twice for my order last week and nobody has replied.")
# => :billing                     (0.94)
assign.("Where is my package? Tracking has not updated in 6 days.")
# => :shipping                    (0.94)
assign.("The app crashes when I open the settings page on Android 15.")
# => :technical                   (0.98)
assign.("I want to change my shipping address and also my card was declined, not sure which is the issue.")
# => {:human, :shipping, 0.495}   — genuinely ambiguous, so it goes to a person
```

Clef specifics: a `choice` takes up to 255 options and shows them to the model
sorted by key (answers still come back under the keys you gave); the questions
of one request are decided together, so their order is visible to the model —
pass a keyword list, not a map, when it matters — and so are each other: in
the RAG example, the relevance question above scores 0.06 *in the same
request* as the bad draft, against 0.92 on its own, because the wrong draft is
in the prompt the model reads for every question. Put a
judgement that must not see the others in its own request. On an M1 Max the
three questions of the first example take ~0.7 s on Metal and ~5.6 s on the
CPU build at Q8_0 (one 324-token prompt). The opt-in
`test/decision_clef_test.exs` runs against this model
(`LLAMA_SMOKE_DECISION_CLEF_MODEL`, tag `:decision_clef`).

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

### Prefix Caching

The server caches KV state between requests (on by default) and shares it **across slots**: with unified KV (`kv_unified: true`, the default) a system prompt prefilled by any slot is adopted by every other slot via a metadata-only copy, so it is computed once, ever. Requests carrying a `:session` term stick to their slot under concurrency, keeping conversations on their cached prefix:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 4,
  cache_prompt: true,        # default: true; also overridable per request
  prompt_cache_ram_mb: 1024  # optional level-2 RAM cache for evicted prefixes (default: 0 = off)
)

{:ok, text} = LlamaCppEx.Server.generate(server, prompt, session: "conversation-42")
```

Benchmark (8 interleaved conversations × 4 turns on 4 slots, shared system prompt): **TTFT median 115 → 35.5 ms (3.2x)** at a ~75% prefix-cache hit ratio.

Notes:

- Hybrid GDN models (e.g. Qwen 3.5/3.6) only hit on exact-prefix continuations; dense-attention models additionally get partial-prefix and cross-slot hits.
- If a chat template rewrites history (e.g. stripping thinking blocks), cache hits silently degrade — the server emits `[:llama_cpp_ex, :server, :prefix_instability]` telemetry when it detects this.

### Chat Completions via the Server

`chat_completion/3` and `stream_chat_completion/3` accept a running server in place of a `%Model{}` — templating and tokenization happen in the caller, and the multi-turn prompt benefits from the prefix cache (**1.6x faster** than the stateless path over a 4-turn conversation):

```elixir
{:ok, completion} =
  LlamaCppEx.chat_completion(server, messages, max_tokens: 200, session: "conversation-42")
```

### Backpressure & Cancellation

- `max_queue: n` bounds the request queue; overflow returns `{:error, :queue_full}` immediately and streams emit a single `{:error, :queue_full}` element (default: `0`, unlimited).
- Halting a stream early (`Enum.take/2`, consumer exit) cancels generation and frees the slot right away instead of decoding to `max_tokens`; `LlamaCppEx.Server.cancel/2` is also available explicitly.
- Sampling options (`:temp`, `:seed`, `:grammar`, ...) can be set per request, overriding the server defaults.

### Batching Strategies

Choose how the token budget is split between generation and prompt processing:

```elixir
# Default: generation latency optimized
batch_strategy: LlamaCppEx.Server.Strategy.DecodeMaximal

# Throughput optimized (batch processing)
batch_strategy: LlamaCppEx.Server.Strategy.PrefillPriority

# Fair split (mixed workloads)
batch_strategy: LlamaCppEx.Server.Strategy.Balanced
```

### Pre-Tokenized API

Tokenize outside the GenServer to reduce contention under concurrent load:

```elixir
model = LlamaCppEx.Server.get_model(server)
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)
{:ok, text} = LlamaCppEx.Server.generate_tokens(server, tokens, max_tokens: 100)
```

### llama.cpp Optimizations

Pass llama.cpp optimization parameters directly:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 8,
  n_ctx: 32768,

  # KV cache quantization — 2x memory savings, identical output
  type_k: :q8_0,
  type_v: :q8_0,

  # Flash attention — faster prefill
  flash_attn: :enabled
)
```

These also work with the high-level API:

```elixir
{:ok, text} = LlamaCppEx.generate(model, "Hello",
  max_tokens: 256,
  type_k: :q8_0,
  type_v: :q8_0,
  flash_attn: :enabled
)
```

See [Performance Guide](docs/performance.md) for all available parameters including RoPE context extension, GPU offload control, attention type, and more.

## Multiple Models (ModelManager)

`LlamaCppEx.ModelManager` keeps several models resident at once and routes requests to them by id. It reuses the HuggingFace Hub downloader and the batching `Server`, and adds named load/unload, capability-based routing, and an advisory memory budget.

Add `LlamaCppEx.ModelSupervisor` to your application's supervision tree (it starts a `Registry`, a `DynamicSupervisor`, and the manager):

```elixir
children = [
  {LlamaCppEx.ModelSupervisor,
   memory_budget: :auto,
   models: [
     # Server-backed (batching + streaming), marked as the default route
     {"chat", {:hub, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"},
      n_gpu_layers: -1, default: true},
     # Embedding model — :embed capability auto-selects :direct mode
     {"embed", {:path, "/models/nomic-embed.gguf"}, capabilities: [:embed]}
   ]}
]
```

For scripts or IEx, start it directly and load at runtime:

```elixir
{:ok, _sup} = LlamaCppEx.ModelSupervisor.start_link([])

# Download from the Hub (cached in ~/.cache/llama_cpp_ex/models/) and keep resident
{:ok, "chat"} = LlamaCppEx.ModelManager.load(
  "chat", {:hub, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"}, n_gpu_layers: -1
)
# Or from a local path
{:ok, "embed"} = LlamaCppEx.ModelManager.load(
  "embed", {:path, "/models/nomic-embed.gguf"}, capabilities: [:embed]
)

# Route by id
{:ok, text} = LlamaCppEx.ModelManager.generate("chat", "Once upon a time", max_tokens: 100)
LlamaCppEx.ModelManager.stream("chat", "Tell me a story") |> Enum.each(&IO.write/1)
{:ok, reply} = LlamaCppEx.ModelManager.chat("chat", [%{role: "user", content: "Hi!"}])
{:ok, vector} = LlamaCppEx.ModelManager.embed("embed", "text to embed")

# Route to the default model
{:ok, text} = LlamaCppEx.ModelManager.generate(:default, "Hello")

# Inspect and manage
LlamaCppEx.ModelManager.list()        # sanitized views, no raw refs
LlamaCppEx.ModelManager.loaded?("chat")
LlamaCppEx.ModelManager.unload("chat")  # stops the backing server, frees memory
```

### Loading and concurrency

`ModelManager` is a **node-wide singleton** — run one `ModelSupervisor` per node. The client API targets the manager by module name, and the backing `Registry`/`DynamicSupervisor` use fixed names, so a second instance is refused at startup.

`load/3` blocks the *calling* process until the model is ready (returning `{:ok, id}` or `{:error, reason}`), but the slow work — the Hub download and the native model load — runs in a supervised `Task`, **not** on the manager process. So a long load never blocks other lifecycle calls: a concurrent `load/3`, an `unload/1`, a `set_default/1`, or reads like `list/0`/`info/1` all proceed while it runs. A model in flight shows `status: :loading`, and re-loading the same id returns `{:error, :already_loaded}`. The memory-budget check and the ETS commit are serialized on the manager, so resident models are always accounted for.

### Backing modes

- **`:server`** (default for generation/chat) — backs the model with a supervised `LlamaCppEx.Server`, so you get continuous batching, streaming, prefix caching, and telemetry.
- **`:direct`** (auto-selected when `:embed` is in `:capabilities`) — holds the model and runs stateless calls. Required for embeddings, since the server has no embedding path.

Override with `mode: :server | :direct`.

### GPU placement

All of llama.cpp's placement options pass straight through `load/3` (per model) to `Model.load/2`/`Server.start_link/1`:

| Option | Meaning |
|---|---|
| `:n_gpu_layers` | Layers to offload (`-1` = all, `0` = CPU only) |
| `:split_mode` | `:none` (single GPU), `:layer` (split layers across GPUs), `:row` (split tensor rows) |
| `:tensor_split` | A **list of per-device proportions** — one float per GPU, indexed by device order. Zeros exclude a device. |
| `:main_gpu` | Primary device: the single GPU under `:none`, or the device holding non-split tensors under `:layer` |

`:tensor_split` is the "array of GPUs": it's a weight per device (llama.cpp normalizes the values), **not** a list of indices. Device order follows `CUDA_VISIBLE_DEVICES`. See [docs/multi-gpu.md](docs/multi-gpu.md) for a full multi-GPU guide and verification steps.

```elixir
# Pin a model to one specific GPU
LlamaCppEx.ModelManager.load("a", {:path, m}, n_gpu_layers: -1, split_mode: :none, main_gpu: 5)

# Spread one big model across all 8 GPUs equally
LlamaCppEx.ModelManager.load("big", {:path, m},
  n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [1, 1, 1, 1, 1, 1, 1, 1]
)

# Use only a subset — e.g. "big" on GPUs 0–3, "embed" on GPUs 4–7
LlamaCppEx.ModelManager.load("big", {:path, m1},
  n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [1, 1, 1, 1, 0, 0, 0, 0]
)

LlamaCppEx.ModelManager.load("embed", {:path, m2},
  capabilities: [:embed], n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [0, 0, 0, 0, 1, 1, 1, 1]
)
```

> On a multi-GPU box, `memory_budget: :auto` reads each card's free VRAM and tracks placement per device — `:tensor_split` and `:main_gpu` are accounted for (see Memory budget below).

### Memory budget

`:memory_budget` is **placement-aware** — it knows whether a model lands in RAM or on specific GPUs (from `:n_gpu_layers`/`:split_mode`/`:tensor_split`/`:main_gpu`) and checks each pool independently. It accepts:

- `:infinity` (default) — no limit.
- an **integer** — a single combined pool (RAM + all VRAM count against one number).
- `:auto` — RAM ≈ 80% of system memory, and **per-GPU VRAM from each card's free memory** (via `LlamaCppEx.devices/0`).
- a map `%{ram: …, vram: …}` — explicit per-device limits. `vram` is a list `[b0, b1, …]` indexed by GPU, or a map `%{gpu_index => bytes}`; `ram`/`vram` may be `:auto` or `:infinity`.

The manager estimates footprint from GGUF size (plus a coarse KV-cache estimate for `:server` mode) and **refuses** over-budget loads, naming the device that didn't fit:

```elixir
# combined (integer) budget
{:error, {:insufficient_memory, device: :total, required: r, available: a}} = ...

# per-device (:auto / map) budget — e.g. GPU 3 is full
{:error, {:insufficient_memory, device: {:gpu, 3}, required: r, available: a}} =
  LlamaCppEx.ModelManager.load("too-big", {:path, "70b.gguf"}, n_gpu_layers: -1, main_gpu: 3)
```

`device` is `:total` (combined), `:ram`, or `{:gpu, index}`. There is no automatic eviction — unload a model yourself to make room. `LlamaCppEx.devices/0` lists each GPU's `:memory_total`/`:memory_free` and its `:gpu_index` (the same index space as `:tensor_split`).

> **Coarse estimation:** footprint is advisory. Partial offload (`0 < n_gpu_layers < n_layers`) is treated as fully on GPU; compute buffers and fragmentation aren't modeled.

### Unloading and memory reclamation

Model cleanup is garbage-collection based. `unload/1` stops the backing server (dropping its context and model references) and forces a GC. Because reclamation is by GC, **any caller still holding a `%LlamaCppEx.Model{}` obtained via `fetch_model/1` keeps the model alive** past `unload/1` — prefer id-based routing and avoid holding raw refs.

## Speculative decoding (MTP)

Multi-Token Prediction speculative decoding (upstream PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)) drafts several tokens at once via a head shipped inside the same GGUF as the target model. Upstream llama-server reports ~2x speedup at ~75% draft acceptance on Qwen 3.6.

> **Performance note: Apple Silicon.** The upstream 2× claim is from NVIDIA datacenter GPUs, where a batched verify decode costs ~1.2× a single-token decode. On Apple Silicon (Metal), a 4-wide verify costs ~2.4× a single decode, which cancels MTP's iteration savings. We measured upstream's own `llama-server --spec-type draft-mtp` on M1 Max: **39.80 tok/s with MTP vs 39.14 tok/s plain** on Qwen 3.6 35B-A3B (1.02×) — i.e. effectively zero speedup from the reference implementation itself. This matches the pattern in upstream [#23011](https://github.com/ggml-org/llama.cpp/issues/23011); a Metal MTP optimization is tracked in [#23114](https://github.com/ggml-org/llama.cpp/pull/23114).
>
> **Tuning for Apple Silicon:** use `n_draft: 1`. With one draft per iteration the verify batch is only 2-wide (much cheaper on Metal) and acceptance jumps to ~79% on Qwen 3.6 35B-A3B. Our measurements on M1 Max with `n_draft: 1`:
> - Qwen 3.6 35B-A3B-MTP (hybrid MoE): plain 39.5 → MTP **44.0 tok/s (1.11×)**
> - Qwen 3.6 27B (dense): plain 10.7 → MTP **10.6 tok/s (~1.0×, neutral)**
>
> Larger `n_draft` hurts on Metal because verify cost grows faster than acceptance benefit.

> **Performance note: NVIDIA GB10 (DGX Spark).** MTP does pay here, but nothing
> like the upstream 2×, and the best `n_draft` is not 3. Qwen3.6-35B-A3B
> UD-Q4_K_XL from the `-MTP` build, 128-token greedy generations, plain and MTP
> interleaved in one process so drift hits both arms equally (n=11 each):
>
> | | median tok/s | range | vs plain |
> |---|---|---|---|
> | plain | 61.4 | 61.2–62.4 | — |
> | MTP `n_draft: 2` | **71.2** | 62.5–75.5 | **+16%** |
>
> The ranges do not overlap — MTP's slowest run beat plain's fastest — so the
> gain is real despite MTP being the noisier arm by an order of magnitude.
>
> Sweeping `n_draft` on the same model, though, puts the optimum at 2 rather
> than 3, and the engine's own counters say why:
>
> | `n_draft` | acceptance | tokens/iteration | tok/s |
> |---|---|---|---|
> | 2 | 68.5% | 2.38 | 63.0 |
> | 3 | 57.2% | 2.73 | 52.8 |
>
> Going from 2 to 3 buys 15% more tokens per iteration and pays 31% more drafting
> plus 10% more verify for them, because the third draft position is the one
> least likely to be accepted. Marginal acceptance decays faster than marginal
> cost, so the extra draft loses money. In a five-run sweep `n_draft: 3` came out
> 2% *below* plain and `n_draft: 4` 14% below.
>
> So the shape matches Apple Silicon even though the cause differs: on Metal the
> wide verify is expensive, while GB10 is a unified-memory part whose MoE decode
> is memory-bandwidth bound, and a wider verify reads more expert weights per
> step. Both end up wanting a narrower draft than a datacenter GPU does. Treat
> `n_draft: 3` as the datacenter default the upstream 2× assumes, not as a value
> that transfers.

> **Performance note: Qwen 3.8 27B (hybrid SSM, sidecar head).** This one is
> shaped by a cost the models above do not pay. Qwen 3.8 puts 48 SSM layers
> beside 16 attention ones, and a recurrent layer cannot be rolled back to an
> arbitrary position, so every speculative iteration snapshots and restores the
> whole recurrent state — 150 MiB of it at these sizes. `stats/1` reports that
> separately as `timing_us.ckpt`. Q4_K_M target + Q4_0 sidecar head, 120-token
> greedy generations:
>
> | `n_draft` | acceptance | M1 Max (Metal) | GB10 (DGX Spark) |
> |---|---|---|---|
> | 1 | 75.0% | 0.89× | **1.24×** |
> | 2 | 54–59% | 0.66× | 1.17× |
> | 3 | 40–44% | 0.68× | 1.09× |
> | 4 | 32% | — | 0.96× |
> | 5 | 30% | 0.56× | — |
>
> On Metal MTP is a net loss at every draft length: `ckpt` alone was 1.8 s of a
> 12.5 s run at `n_draft: 1` and 6.9 s of 16.3 s at `n_draft: 3`. On GB10 the
> same snapshot is cheap enough that `n_draft: 1` wins, and — unlike the MoE
> above, whose optimum was 2 — the optimum here is 1, monotonically decreasing
> after it. Measure `ckpt` against `total` before trusting speculation on any
> hybrid model.
>
> Both arms of the GB10 column are warm-cache numbers. A first run off cold page
> cache reads ~19 GB and reported a 4.18 tok/s baseline against 10.83 tok/s warm,
> which inverts the comparison entirely.

### Other speculative types (EAGLE-3, DFlash, n-gram)

Upstream llama.cpp implements more speculative types behind the same `common_speculative` API — `draft-eagle3`, `draft-dflash` (block-diffusion drafting via a separate drafter GGUF), and several n-gram self-speculation modes, plus `--spec-default`, which stacks n-gram speculation on top of a model-based drafter. **This binding currently exposes only MTP**: `MTP.init/2` pins `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`, so the other types and the combinations are not reachable from here. The draft *model* is no longer tied to the target, though — see `:draft_model` below for the target/sidecar split.

> **DFlash status (July 2026, llama.cpp b9932).** DFlash runs end-to-end on Metal via upstream `llama-cli`/`llama-server`, but we measured it *slower* than plain decoding on Apple Silicon at small target sizes: Qwen3.5-4B target + z-lab 0.6B drafter on M4 Max reached 42 tok/s with DFlash vs 85 tok/s plain (greedy sampling; 30% draft acceptance, mean accepted run 2.8 — and stochastic sampling at `temp 0.8` collapses acceptance to ~7%). The Metal economics are the same as the MTP note above (wide verify batches are expensive), and the community drafter-GGUF conversions are still churning: of three third-party Qwen 4B drafter repos tested, only one loads with current upstream (the others hit the `dflash-draft` arch mismatch [#25116](https://github.com/ggml-org/llama.cpp/issues/25116) or lack the `target_layers` metadata added by the conversion refactor [#25110](https://github.com/ggml-org/llama.cpp/pull/25110)). Worth revisiting when the drafter format settles; the natural entry point is a `spec_type` + drafter-model option on `speculative_init`.

### Models with MTP heads

- [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) — **sidecar layout**: the target (`Qwen3.8-27B-Q4_K_M.gguf`, ~18 GB) carries *no* head, and `mtp-Qwen3.8-27B-Q4_0.gguf` (~1.6 GB) carries nothing else. Load both and pass the head as `draft_model:`.
- [`ggml-org/gemma-4-E4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) — **sidecar layout**: the target (`gemma-4-E4B-it-Q4_K_M.gguf`) is `gemma4` with *no* nextn head; `mtp-gemma-4-E4B-it-Q8_0.gguf` is a separate `gemma4-assistant` sidecar. Same `draft_model:` API. llama.cpp requires the target as `ctx_other` for `gemma4-assistant`; `MTP.init/2` always passes it (Qwen's constructor ignores it).
- [`ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF) (recommended: `Q4_K_M`, ~21 GB)
- [`ggml-org/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-27B-MTP-GGUF)
- [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)

For trying MTP out, or for running the `:mtp` test suite, the small Qwen 3.5 MTP
builds are far cheaper than any of the above and carry the same `nextn` layers:

- [`unsloth/Qwen3.5-0.8B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF) (`Q8_0`, ~0.8 GB — what this repo's MTP suite runs against)
- [`unsloth/Qwen3.5-4B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF)

Acceptance on a 0.8B target is not representative of production throughput —
drafting is nearly as expensive as decoding at that size — so use these to
exercise the path, not to measure it.

A regular (non-MTP) quant will fail at `LlamaCppEx.MTP.init/2` — some GGUF in the pair must contain the MTP head's tensors. To check a file before loading it, look for a `*.nextn_predict_layers` key and `blk.N.nextn.*` tensors in its metadata. The E4B sidecar also has those, and is additionally a `gemma4-assistant` architecture. When the publisher ships the head separately, that sidecar is the file with the head tensors and the target legitimately has none; pass it as `draft_model:` rather than looking for a combined build.

The model must also be loaded with `load_mtp: true` (see below). Upstream gates those tensors behind a load-time flag that defaults to off, and they cannot be attached afterwards, so `MTP.init/2` refuses a model loaded without it rather than letting the omission surface later as `verify decode failed: code=-1`.

> **Do not reuse a session straight after abandoning a stream.** Cancellation is asynchronous and unacknowledged, and a session's two contexts are shared by every call on it, so starting the next `generate/3` immediately can put a second writer on a KV cache the cancelled draft loop has not released — which aborts the VM. Let an abandoned stream reach a terminal event, or build a fresh session. Tracked in the v0.8.42 changelog.

### Usage

#### Minimal: stream a single response

```elixir
:ok = LlamaCppEx.init()

{:ok, model} =
  LlamaCppEx.load_model(
    Path.expand("~/Downloads/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"),
    n_gpu_layers: 999,
    # Required. Upstream defaults this off so non-speculative callers do not pay
    # for the MTP head's tensors, and they cannot be attached to an
    # already-loaded model, so MTP.init/2 refuses a model loaded without it.
    load_mtp: true
  )

# Build the speculative session once — it owns a target context and a
# separate MTP draft context on the *same* model file (no extra download).
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 8192)

mtp
|> LlamaCppEx.MTP.stream("Write a haiku about the sea:", max_tokens: 256)
|> Stream.each(&IO.write/1)
|> Stream.run()

# Final stats (also returned via the {:done, stats} stream event)
stats = LlamaCppEx.MTP.stats(mtp)
IO.puts("\nacceptance: #{Float.round(stats.acceptance_rate * 100, 1)}%  " <>
        "throughput: #{Float.round(stats.tokens_per_sec, 1)} tok/s")
```

#### Sidecar head: Qwen 3.8 (`-hf` target + `-hfd` draft)

Same session API, two files. This is what upstream's
`llama serve -hf ggml-org/Qwen3.8-27B-GGUF --spec-type draft-mtp` resolves to
once it has downloaded the pair — the flag makes upstream fetch the `mtp-*`
sidecar and build the draft context against it instead of against the target.

```elixir
:ok = LlamaCppEx.init()

# The target carries no MTP head at all: Model.n_layer_nextn/1 returns 0 for it.
{:ok, target} =
  LlamaCppEx.load_model(
    Path.expand("~/Downloads/Qwen3.8-27B-Q4_K_M.gguf"),
    n_gpu_layers: 999,
    load_mtp: true
  )

# The sidecar carries nothing *but* the head — ~1.6 GB against the target's 18.
{:ok, head} =
  LlamaCppEx.load_model(
    Path.expand("~/Downloads/mtp-Qwen3.8-27B-Q4_0.gguf"),
    n_gpu_layers: 999,
    load_mtp: true
  )

# n_draft: 1 — see the Qwen 3.8 performance note above. Acceptance is 75% here
# and falls off fast, and every iteration pays a recurrent-state snapshot.
{:ok, mtp} = LlamaCppEx.MTP.init(target, draft_model: head, n_draft: 1, n_ctx: 8192)

{:ok, text} = LlamaCppEx.MTP.generate(mtp, "Explain MTP in one paragraph.", max_tokens: 200)

stats = LlamaCppEx.MTP.stats(mtp)
IO.puts("acceptance: #{Float.round(stats.acceptance_rate * 100, 1)}%  " <>
        "ckpt: #{div(stats.timing_us.ckpt, 1000)}ms of #{div(stats.timing_us.total, 1000)}ms")
```

`MTP.init/2` refuses the mismatched pairings before building anything: a sidecar
loaded without `load_mtp: true`, an ordinary model passed as `:draft_model`, and
a head whose hidden width does not match the target's — that last one because
upstream compares the two with a `GGML_ASSERT`, which aborts the VM rather than
failing the call.

#### Sidecar head: Gemma4 E4B (`gemma4` target + `gemma4-assistant` draft)

Same `draft_model:` API as Qwen 3.8, two files. llama.cpp requires the target
as `ctx_other` for `gemma4-assistant`. `MTP.init/2` always passes it.

```elixir
:ok = LlamaCppEx.init()

{:ok, target} =
  LlamaCppEx.load_model(
    Path.expand("~/Downloads/gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers: 999,
    load_mtp: true
  )

{:ok, head} =
  LlamaCppEx.load_model(
    Path.expand("~/Downloads/mtp-gemma-4-E4B-it-Q8_0.gguf"),
    n_gpu_layers: 999,
    load_mtp: true
  )

{:ok, mtp} = LlamaCppEx.MTP.init(target, draft_model: head, n_draft: 3, n_ctx: 8192)

{:ok, text} = LlamaCppEx.MTP.generate(mtp, "Explain MTP in one paragraph.", max_tokens: 200)
IO.puts(text)
```

#### Synchronous generate (collect to a string)

```elixir
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 4096)

{:ok, text} =
  LlamaCppEx.MTP.generate(mtp, "Explain monads to a Go programmer:",
    max_tokens: 200,
    temp: 0.7,
    top_p: 0.95,
    seed: 42
  )

IO.puts(text)
```

#### Reuse a session across multiple prompts

`MTP.init/2` allocates two `llama_context`s and the speculative state. It's the expensive bit. Reuse the same `%MTP{}` value across calls — KV caches are cleared at the start of each `stream/3` / `generate/3`:

```elixir
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 8192)

for q <- ["What is Elixir?", "What is OTP?", "What is BEAM?"] do
  IO.puts("\n> #{q}")
  mtp |> LlamaCppEx.MTP.stream(q, max_tokens: 150) |> Stream.each(&IO.write/1) |> Stream.run()
end

# Counters are cumulative across all calls on this session.
LlamaCppEx.MTP.stats(mtp) |> IO.inspect(label: "cumulative")
```

#### Watch stats live from a separate process

`MTP.stats/1` is lock-free, so a sibling process can poll it while a stream is in flight — handy for Phoenix LiveView dashboards:

```elixir
parent = self()

gen_task =
  Task.async(fn ->
    mtp
    |> LlamaCppEx.MTP.stream("Generate a 500-line Python implementation of A*:",
      max_tokens: 1024,
      temp: 0.7
    )
    |> Enum.into("")
    |> then(&send(parent, {:done, &1}))
  end)

# Sample every 200 ms while the generation runs.
Stream.repeatedly(fn ->
  Process.sleep(200)
  s = LlamaCppEx.MTP.stats(mtp)
  IO.puts(
    "iters=#{s.iters}  emitted=#{s.tokens_emitted}  " <>
      "accept=#{Float.round(s.acceptance_rate * 100, 1)}%  " <>
      "tok/s=#{Float.round(s.tokens_per_sec, 1)}"
  )
end)
|> Stream.take_while(fn _ -> not Task.yield(gen_task, 0) |> match?({:ok, _}) end)
|> Stream.run()

Task.await(gen_task, :infinity)
```

For in-band progress events (no separate process), use `stream_events/3` with `emit_stats_every`:

```elixir
mtp
|> LlamaCppEx.MTP.stream_events("Write a sonnet:",
  max_tokens: 400,
  emit_stats_every: 32
)
|> Enum.each(fn
  {:token, _id, text} -> IO.write(text)
  {:stats, s}        -> IO.puts("\n[stats] accept=#{Float.round(s.acceptance_rate * 100, 1)}%")
  {:done, _final}    -> IO.puts("\n[done]")
  {:eog, _}          -> IO.puts("\n[eog]")
end)
```

### Options

`LlamaCppEx.MTP.init/2`:

  * `:n_draft` — draft tokens proposed per iteration (default `3`). The optimum
    is hardware-specific and worth measuring rather than assuming: `1` on Apple
    Silicon, `2` on GB10 (where `3` measured *slower* than no speculation at
    all), `2–4` on datacenter NVIDIA. See the two performance notes above.
  * `:n_ctx`, `:n_threads`, `:flash_attn`, `:type_k`/`:type_v`, `:offload_kqv`, … — any `LlamaCppEx.Context` option; applied to both target and draft contexts.

`LlamaCppEx.MTP.stream/3`:

  * `:max_tokens` (default `256`), plus all sampling options (`:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`, `:penalty_*`, `:grammar`).
  * `:emit_stats_every` — when set, periodic `{:stats, _}` events become available via `stream_events/3`.

### Caveats

- Upstream currently requires `n_parallel = 1` for MTP; this binding mirrors that. Use `LlamaCppEx.Server` for concurrent non-MTP inference, or stick to a single MTP session at a time.
- Prompt prefill is somewhat slower with MTP than without (the MTP head also processes the prompt). The win shows up at decode time.

See [`examples/mtp_speculative.exs`](examples/mtp_speculative.exs) for a runnable demo with full timing breakdown.

## Benchmarks

Each subsection names its own hardware and backend — the numbers below span
Apple Silicon (Metal) and NVIDIA (CUDA) and are not comparable across sections
unless they say so. Unless noted otherwise, `n_gpu_layers: -1`.

### Single-model generation speed

Apple M4 Max (64 GB), Metal backend.

| Model | Quantization | Tokens/sec |
|-------|-------------|------------|
| Llama 3.2 3B Instruct | Q4_K_XL | 125.6 |
| Ministral 3 3B Reasoning | Q4_K_XL | 113.0 |
| Ministral 3 3B Instruct | Q4_K_XL | 104.3 |
| GPT-OSS 20B | Q4_K_XL | 79.4 |
| Qwen3.5-35B-A3B | Q6_K | 56.0 |
| Qwen3.5-27B | Q4_K_XL | 17.5 |

### Qwen3.6-35B-A3B (v0.7.8)

New `qwen35moe` architecture with Gated Delta Net (hybrid linear/full attention). Measured on Apple M1 Max (64 GB) with v0.7.8 bindings — not directly comparable to the M4 Max numbers above.

| Model | Quantization | Tokens/sec (M1 Max) |
|-------|-------------|---------------------|
| Qwen3.6-35B-A3B | Q4_K_XL | 43.8 |

128-token generation, `temp: 0.0`, 3-run average (43.3 / 44.1 / 44.0 t/s).

### CUDA: NVIDIA DGX Spark (GB10)

Same model and quantization as the M1 Max row above, so the two are directly
comparable. GB10 (`sm_121a`, aarch64, 128 GB unified), CUDA 13.0.2, driver
580.173.02, llama.cpp b10280, source build with `LLAMA_BACKEND=cuda`.

| Model | Quantization | Tokens/sec (GB10) | Tokens/sec (M1 Max) |
|-------|-------------|-------------------|---------------------|
| Qwen3.6-35B-A3B | UD-Q4_K_XL | **62.1** | 43.8 |

128-token generation, `temp: 0.0`, `n_gpu_layers: -1`. Median of 5 runs after a
discarded warm-up: 61.7 / 62.0 / 62.1 / 62.2 / 62.2 t/s — a 0.9% spread, so the
1.42x over M1 Max is well outside the noise. All 41 layers offload; the model
takes 20 799 MiB of device memory.

Two notes on method, both learned the hard way:

- **Each run uses a distinct prompt.** Repeating one prompt hits the context
  reuse path and reports a throughput the engine never achieved.
- **Tokens are counted by re-encoding the output**, not by counting stream
  chunks — a chunk is not a token, and under speculative decoding it can carry
  several.

The first call after load is discarded: it pays CUDA graph capture and the
allocator's first-touch layout, and is not representative of steady state.

With MTP speculative decoding, using the separate `-MTP` build of the same model
(the plain UD-Q4_K_XL carries no MTP head — `MTP.init/2` now says so rather than
failing with a bare context error):

| | median tok/s | vs plain |
|---|---|---|
| plain | 61.4 | — |
| `n_draft: 2` | **71.2** | +16% |
| `n_draft: 3` | 61.1 | −2% |
| `n_draft: 4` | 53.8 | −14% |

`n_draft: 2` is the optimum on this hardware, not the documented default of 3.
See the GB10 performance note under [Speculative decoding
(MTP)](#speculative-decoding-mtp) for the acceptance and timing counters behind
that.

### Single-sequence generation (Qwen3-4B Q4_K_M)

| Prompt | 32 tokens | 128 tokens |
|--------|-----------|------------|
| short (6 tok) | 0.31s (3.19 ips) | 1.01s (0.98 ips) |
| medium (100 tok) | 0.36s (2.79 ips) | 1.06s (0.94 ips) |
| long (500 tok) | 0.65s (1.53 ips) | 1.29s (0.77 ips) |

### Continuous batching throughput (Qwen3-4B Q4_K_M)

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

## Examples

The `examples/` directory contains runnable scripts demonstrating key features:

```bash
# Basic text generation
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/basic_generation.exs

# Streaming tokens to terminal
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/streaming.exs

# Interactive multi-turn chat
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/chat.exs

# JSON Schema constrained generation + Ecto integration
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/structured_output.exs

# Embedding generation and cosine similarity (any embedding GGUF, e.g. embeddinggemma-2)
LLAMA_EMBEDDING_MODEL_PATH=~/Downloads/embeddinggemma-2-Q8_0.gguf mix run examples/embeddings.exs

# Continuous batching server with concurrent requests
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/server.exs
```

## Architecture

```
Elixir API (lib/)
    │
LlamaCppEx.NIF (@on_load, stubs)
    │
C++ NIF layer (c_src/) — fine.hpp for RAII + type encoding;
    │                    decision.cpp ports llama-server's /v1/systemone
    │
llama.cpp static libs (vendor/llama.cpp, built via CMake)
    │
Hardware (CPU / Metal / CUDA / Vulkan)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

llama.cpp is licensed under the MIT License.
