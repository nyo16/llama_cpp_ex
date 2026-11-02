# DGX Spark

A runbook for running `llama_cpp_ex` on NVIDIA DGX Spark (GB10), single node and
across two nodes. Everything here is measured on the hardware rather than
inferred from spec sheets; where a number differs from the marketing figure, the
measurement wins and the reason is given.

## The machine

| | |
|---|---|
| SoC | GB10 Grace Blackwell, compute capability **12.1** (`sm_121a`) |
| Memory | **121 GiB unified** — simultaneously host RAM and GPU memory |
| CPU | 20 cores: Cortex-X925 ×10, Cortex-A725 ×10 |
| OS | Ubuntu 24.04 LTS, kernel 6.17 (DGX OS) |
| CUDA | 13.0 at `/usr/local/cuda`, driver 580.173.02 |
| Toolchain | GCC 13.3, cmake 3.28 |

Three things about this machine are counter-intuitive enough to cost a day each:

**`nvidia-smi` cannot report GPU memory.** It prints `memory.total = [N/A]`,
because the addressing mode is ATS and there is no separate GPU pool to report.
Use `free -h`. `LlamaCppEx.devices/0` reports it correctly (130.7 GB total on
`CUDA0`) because it asks ggml, not `nvidia-smi`. The device type comes back as
`:igpu` for the same reason.

**The CPU clusters are interleaved.** The performance cores are **5-9, 15-19**;
the efficiency cores are **0-4, 10-14**. `taskset -c 0-9` therefore pins a job
entirely to little cores. Measured on a 1-byte RDMA ping-pong: p50 21.5 µs on
cpu19 versus 29.1 µs on cpu0.

**`ping` lies by three orders of magnitude.** ICMP RTT over the 200 Gb/s direct
link reads ~1.2 ms. That is cpuidle exit latency (LPI-3 is 433 µs), not the wire
— the 1 GbE LAN measures 1.34 ms, which is the tell. Hold the cores out of deep
idle and ICMP drops to 0.028 ms. The real numbers are **1.39 µs RTT over RDMA**
and **~19 µs p50 over TCP**.

### The fabric

Two point-to-point ConnectX-7 links, no switch:

| link | spark-1 | spark-2 | MTU | measured |
|---|---|---|---|---|
| 0 | 10.100.64.1 | 10.100.64.2 | 9000 | 13.98 GB/s |
| 1 | 10.100.65.1 | 10.100.65.2 | 9000 | 13.98 GB/s |

RoCE v2, GID 3, `active_mtu` 4096. 13.98 GB/s is 89% of the theoretical
15.75 GB/s, and the ceiling is **PCIe Gen5 x4** per port (`max_link_width = 4`,
so this is the design and not a training failure) — not the 200 Gb/s wire rate.
Both links together do reach 24.5 GB/s, but a single `ggml-rpc` peer connection
cannot use both: the transport opens one queue pair per socket and picks the HCA
by matching a GID against the socket's local address. Budget 13.98 GB/s.

## The remote development loop

The control node is a Mac; the Sparks are `spark-1` and `spark-2` over ssh with
`ControlMaster` configured. Nothing below needs `sudo` on either Spark — the
boxes require a password for it, so no package installs, no `sysctl`, no kernel
cmdline changes and no system units are available. Where root *would* help it is
called out explicitly rather than done silently.

```bash
scripts/spark/bootstrap.sh            # once per node: toolchain, directories, fact sheet
scripts/spark/sync.sh                 # push the working tree to both nodes
scripts/spark/remote.sh spark-1 mix test
```

### `bootstrap.sh`

Idempotent per-node provisioning. Creates `~/models`, `~/.cache/llama.cpp/rpc`
and `~/src/llama_cpp_ex`, prints a fact sheet, and provisions the Elixir
toolchain.

The toolchain is the only interesting part. `spark-2` has no asdf and we cannot
`apt install` the OTP build dependencies, so `asdf install erlang` is not a plan.
The boxes are identical — same distro, kernel, architecture — and a kerl-built
OTP links only against libraries present on both, so bootstrap copies
`spark-1:~/.asdf` across the fabric instead. Measured at **8 seconds**.

The nodes have no ssh keys for each other and this script does not add any:
authentication rides a forwarded agent (`ssh -o ControlPath=none -A`, because
`-A` on a session that reuses an existing multiplexed master does nothing). If
no agent is forwardable it relays the 431 MB through the control node instead.

Ubuntu's own Elixir is **not** a fallback: 24.04 ships 1.14 / OTP 25 and
`mix.exs` requires `~> 1.18`. If the copy fails, bootstrap prints the
`asdf install` path and the exact `apt` line for the user to run with their
password.

Note that nothing puts asdf's **shims** on `PATH`. asdf 0.19's shims are
`exec asdf exec <tool>` and the `asdf` binary lives in `/usr/bin` on `spark-1`
and nowhere on `spark-2`. The install directories go on `PATH` directly instead,
which works identically on both nodes and skips a process per invocation.

### `sync.sh`

`rsync -az --delete` of the *working* tree, uncommitted changes included, which
is why this is rsync and not `git pull`. About 36 s cold, a few seconds warm.

The exclusions are in `scripts/spark/lib.sh` with a reason each. One is
non-obvious: **`vendor/llama.cpp/.git`** is excluded, 36 MB of history the build
does not need. `Makefile` falls back to `LLAMA_COMMIT` when that directory is
absent, so `LLAMA_SHA` still resolves — and `sync.sh` verifies exactly that on
every run rather than trusting it.

After every sync both nodes are proven **byte-identical to the control node**
over the synced file set, by content digest. This is load-bearing rather than
hygiene: the ggml RPC `HELLO` handshake compares only `major`/`minor` and
**ignores `patch`**, so two builds from drifted trees connect happily and then
misinterpret each other's op codes.

### `remote.sh`

Runs a command on a node under the environment contract in
`scripts/spark/lib.sh`: `PATH` for the toolchain and CUDA, `CUDA_HOME`, the
build flags below, `LLAMA_CACHE_DIR`, and `LLAMA_SMOKE_GEN_MODEL` /
`LLAMA_MODEL_PATH` when those files exist.

The default is a **non-login** shell, deliberately. DGX OS puts `nvcc` on the
login `PATH` only, via `/etc/profile.d/nv_paths.sh`, which `ssh host cmd`, a
systemd unit and every CI shell never source. Rather than paper over that with
`bash -lc`, the contract sets `CUDA_HOME` and the Makefile's toolkit discovery
takes it from there. `--login` exists for the rare case that genuinely needs the
profile; if a *build* needs it, the bug is in the contract.

Useful flags: `--big-cores` wraps the command in `taskset -c 5-9,15-19`,
`--env K=V` adds one variable, `--print` shows the remote script without running
it.

## Build flags

```bash
LLAMA_BACKEND=cuda \
LLAMA_CPU_ARM_ARCH=armv9.2-a+dotprod+i8mm+fp16+bf16+sve2 \
LLAMA_CUDA_ARCH=121a-real \
  mix compile
```

`remote.sh` exports all three, so on a Spark this is just `mix compile`.

### Why `LLAMA_CPU_ARM_ARCH` matters, and when it does not

ggml probes the host with `-mcpu=native` when `GGML_NATIVE` is on, which is the
default. On GB10 with GCC 13.3 that probe **fails silently**: the compiler
predates Cortex-X925/A725 and rejects `-mcpu=cortex-x925`, cmake emits a soft
warning, and the build exits 0 with the CPU backend compiled at base ARMv8-A.

Measured on the emitted `libggml-cpu.a`:

| | `sdot` | `smmla` | SVE |
|---|---|---|---|
| `-mcpu=native` (default) | 0 | 0 | none |
| `-march=armv9.2-a+dotprod+i8mm+fp16+bf16+sve2` | 1134 | 370 | 10678 operands |

Those are the Q4/Q8 quantized matmul kernels.

They run only when the CPU does the math: CPU-only builds, `n_gpu_layers`
below the layer count, MoE experts kept on the CPU. With every layer on the
GPU the flag changes nothing measurable. A fresh project depending on the
published package (no `LLAMA_*` variables, so `-mcpu=native`, CUDA picked up
as `121a-real` automatically) against this repo's tuned build, Qwen3.6-35B-A3B
fully offloaded on spark-2, median of 5:

| build | TTFT, ~2k-token prompt | decode |
|---|---|---|
| `{:llama_cpp_ex, "0.8.66"}` from Hex, defaults | 698.0 ms | 66.78 tok/s |
| `LLAMA_CPU_ARM_ARCH` + `LLAMA_CUDA_ARCH` | 702.8 ms | 66.49 tok/s |

So a Hex user on a Spark gets full GPU speed with no flags; set them for
CPU-side work.

### Why `LLAMA_CUDA_ARCH` must accompany it

Naming the CPU architecture requires `GGML_NATIVE=OFF`, and with native off
ggml-cuda stops compiling for the GPU it can see and emits a **seven-architecture
fat binary** instead — a ~6× build-time regression, silently. So the Makefile
makes `LLAMA_CPU_ARM_ARCH` without `LLAMA_CUDA_ARCH` a hard `$(error)` on a CUDA
build. Both variables are part of the build-directory key, so toggling either
gets a clean `CMakeCache.txt` and switching back is still a cache hit.

`LLAMA_PORTABLE=1` also sets `GGML_NATIVE=OFF`, for the unrelated reason that
published artifacts must not carry `-march=native`. The two never double-emit
it, and portable builds still need no CUDA architecture — the release runners
have no GPU.

### Verifying a build

```bash
scripts/spark/remote.sh spark-1 scripts/spark/verify-build-flags.sh
```

```
==> verifying _build/dev/lib/llama_cpp_ex/obj/llama_build-cuda-6b5f34be
  PASS  ggml-cuda: one architecture, compute_121a
  PASS  ggml-cpu: -march=armv9.2-a+dotprod+i8mm+fp16+bf16+sve2
  PASS  libggml-cpu.a: 1134 sdot, 370 smmla, 10678 SVE operands
==> all checks passed
```

Three assertions, because each catches a different silent failure: the fat
binary, the flag not reaching the compiler, and the flag reaching the compiler
but producing nothing. The CUDA architecture is asserted from
`ggml-cuda.dir/flags.make` and **not** from `CMakeCache.txt` —
`CMAKE_CUDA_ARCHITECTURES` is an ordinary variable and never appears in the
cache.

`test/makefile_arch_flags_test.exs` covers the Makefile side hermetically on both
macOS and Linux, including the `$(error)` and the build-directory key.

A full clean CUDA build takes **2m14s** at `-j20` (1m47s before the flags; the
difference is the wider CPU code generation, not the fat binary). There is no
`ccache` on these boxes and installing one needs a password.

## Models

```bash
scripts/spark/remote.sh spark-1 mix run scripts/spark/fetch_models.exs --list
scripts/spark/remote.sh spark-1 mix run scripts/spark/fetch_models.exs 8b 30b
```

Downloads go through `LlamaCppEx.Hub.download/3` — the library's own path,
SHA-256 verified fail-closed — into `~/models/<repo>/<revision>/<file>`.

| label | size | role |
|---|---|---|
| `8b` | 5.0 GB | Qwen3-8B Q4_K_M — dense sanity check |
| `30b` | 18.6 GB | Qwen3-30B-A3B Q4_K_M — the MoE case this chip is good at |
| `120b` | 63.4 GB | gpt-oss-120b MXFP4 — big but **fits** one node |
| `235b` | 142.1 GB | Qwen3-235B-A22B Q4_K_M, 3 shards — does **not** fit one node |

121 GiB is 130.0 GB, so `120b` is the controlled A/B for measuring pure RPC
overhead (same model, one node versus two) and `235b` at 142.1 GB is the case
that justifies a second Spark at all.

### Qwen3.6 and the MTP variants

| label | size | role |
|---|---|---|
| `q36-27b` | 16.8 GB | Qwen3.6-27B Q4_K_M — current-generation dense |
| `q36-27b-mtp` | 17.1 GB | the same weights **plus the MTP head** |
| `q36-35b` | 22.1 GB | Qwen3.6-35B-A3B UD-Q4_K_M — current-generation MoE |
| `q36-35b-mtp` | 22.7 GB | the same weights **plus the MTP head** |

The MTP repos are not different quantizations; they are the same model with
Multi-Token Prediction layers included. llama.cpp reads those layers only when
the model is loaded with `load_mtp: true`, which makes the plain and MTP files a
clean A/B for what speculative decoding buys on this hardware. See
`LlamaCppEx.MTP`.

## Running on ONE Spark

Full numbers and methodology in
`bench/results/v0.8.43-dgx-spark-baseline.md`, and the two-node numbers in
`bench/results/v0.8.43-dgx-spark-two-node.md`.
The short version:

| model | prefill (pp) | decode (tg) |
|---|---|---|
| Qwen3-8B Q4_K_M | 3500–4331 t/s | 40.5 t/s |
| Qwen3-30B-A3B Q4_K_M | 3287 t/s | 90.9 t/s |

Three of the four figures beat the published single-Spark references; the fourth
is 7% under. **Prefill is this machine's strong suit** and decode is bandwidth-
bound, so quote them separately or you describe neither.

### Settings that matter

```elixir
LlamaCppEx.Server.start_link(
  model_path: path,
  n_gpu_layers: 99,      # always. n_gpu_layers: 0 costs 57% of decode
  n_parallel: 8,         # ~8x aggregate throughput for ~0 per-request cost
  n_ctx: 4096 * 8
)
```

And the settings that do **not** matter, each of which looks like it should:

| knob | verdict |
|---|---|
| `flash_attn` | leave `:auto` — it is already on, and `:disabled` costs 24% of prefill |
| `type_k` / `type_v` | leave f16. Quantizing the KV cache **loses** 2–7% and buys nothing: there is no separate VRAM to free |
| `use_mmap`, `use_mlock`, `use_direct_io` | **leave all three unset.** Upstream's `auto` then reads the file instead of mapping it on this iGPU. A forced `use_mmap: true` — the binding's default before this run — left most gpt-oss-120b loads decoding at 48.5 tok/s against 57 with `auto`, `use_mmap: false` or `use_direct_io: true`; about one forced-mmap load in five came out fast, so single-load comparisons from before (this row used to say mlock and dio cost 3–4%) are not reliable |
| `n_batch`, `n_ubatch` | no measurable effect; the default is fine |
| cpuidle / `idle=poll` | **no effect**, single-node or two-node. See below |
| `taskset -c 5-9,15-19` | hygiene, not speed. But never use `-c 0-9` — those are the little cores |

### The cpuidle story, and why you can ignore it

cpuidle exit latency is the largest measured effect on this machine — LPI-3 exit
is 433 µs, and it is why `ping` reads 1.2 ms on a link whose real RTT is 1.39 µs.
It is natural to assume it also costs inter-token latency.

It does not. Every condition tested — a `nice -19` poller on every core, BEAM
busy-wait tuning, X925 pinning, and the same again across two nodes with a
network wake on every token — landed within 2% of doing nothing, and the poller
made TTFT *worse*. A decode loop keeps the CPU busy, so it never enters a deep
C-state and there is no exit latency to avoid.

**So do not go asking for `idle=poll` on the kernel cmdline.** It was the one
thing this work expected to need root for, and the measurement retired it.

## The binding costs nothing measurable

Same model, same llama.cpp commit (`785322d17`), upstream's own tools against
`bench/spark_overhead.exs`. Qwen3-8B Q4_K_M, 128 greedy tokens, decode timed
from the first token to the last:

| path | streams | upstream | llama_cpp_ex |
|---|---|---|---|
| single stream (`llama-bench` tg128 / `LlamaCppEx.stream/3`) | 1 | 44.0 tok/s | 43.5 tok/s |
| server (`llama-server -np 8` / `Server`, `n_parallel: 8`) | 8 | 286 tok/s aggregate | 286 tok/s aggregate |

`llama-batched-bench` reaches 311 tok/s at 8 streams because it never samples:
every server has to wait for the logits, sample on the CPU and submit the next
step, and that serial dependency is the 8%. Upstream's `--backend-sampling`
(sampling on the GPU) measured within 1% of CPU sampling here, so it is not
exposed. Tuning on this machine is therefore model and drafter choice, not the
binding.

## Qwen3.8-27B NVFP4

`esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF`, `MID-HIGH` tier: every attention and
FFN weight in native NVFP4 (GGML type 40), Q8_0 heads, the MTP head embedded.
Against the Q4_K_M build of the same model, `llama-bench`:

| | size | pp512 | pp2048 | tg128 |
|---|---|---|---|---|
| Q4_K_M | 17.7 GiB | 1015 | 1023 | 11.21 |
| NVFP4 MID-HIGH | 15.7 GiB | 1236 | 1265 | **14.19** |

NVFP4 is +22% prefill and +27% decode. Decode reads 15.7 GiB per token at
14.2 tok/s — 240 GB/s, 88% of GB10's 273 GB/s — so it is bandwidth-bound and
there is little left to take from the kernels. Prefill runs W4A4 on the
Blackwell FP4 MMA (`mma.sync ... kind::mxf4nvf4`); an `nsys` profile of pp2048
puts 53% of GPU time in that matmul, 9% in activation quantization and ~23% in
element-wise ops (SwiGLU, gating muls, adds, the GDN conv-state concat).

## Speculative decoding: MTP, DFlash 2, DSpark

256-token greedy chat completions (thinking off), four prompts run twice,
median decode rate. Upstream is `llama-server` with `--spec-type`; the binding
is `LlamaCppEx.MTP` (`bench/spark_spec.exs`):

| Qwen3.8-27B NVFP4 | upstream | llama_cpp_ex | acceptance |
|---|---|---|---|
| plain | 14.17 | 14.28 | — |
| MTP `n_draft: 1` | 21.22 | — | 83% |
| MTP `n_draft: 2` | 27.27 | 26.76 | 76% |
| MTP `n_draft: 3` | 29.16 | 28.63 | 65% |
| MTP `n_draft: 4` | 29.71 | — | 56% |
| DSpark `n_draft: 7` | 25.74 | — | 22% |
| **DFlash 2 `n_draft: 7`** | **46.60** | **44.22** | 42% |

DFlash 2 (`z-lab/Qwen3.8-27B-DFlash2-GGUF`, a 1.1 GB drafter) is **3.1×**
plain, against 2.0× for the checkpoint's own MTP head. Its `n_draft` is capped
at the drafter's block size minus one; asking for 10 measured the same as 7.
DSpark (`magnitudedev/Qwen3.8-27B-DSpark-GGUF`) lost to MTP here.

| Qwen3.6-35B-A3B UD-Q4_K_M (MoE) | llama_cpp_ex | vs plain | acceptance |
|---|---|---|---|
| plain | 68.27 | — | — |
| MTP `n_draft: 1` | 86.86 | 1.27× | 86% |
| MTP `n_draft: 2` | **96.69** | **1.42×** | 77% |
| MTP `n_draft: 3` | 91.58 | 1.34× | 66% |

Speculation spends compute (a wider verify) to save bandwidth (fewer
sequential decodes), and this chip has compute to spare: it pays on the dense
model and still pays on the MoE, where only ~3B parameters move per token, but
the MoE's optimum is narrower.

> #### Earlier numbers on this page were held down by the binding {: .info}
>
> Before this run the MoE measured *neutral* (0.95–1.03×) and the dense 27B
> 1.6×. Both models are hybrid (GDN layers beside attention), and the binding
> rolled a rejected draft back by saving the whole recurrent state every
> iteration, then restoring it and re-decoding the accepted prefix on every
> partial accept. Upstream's server instead gives the target context one
> recurrent snapshot per draft position (`n_rs_seq`) and rolls back natively;
> `LlamaCppEx.MTP.init/2` now does the same. Qwen3.8-27B NVFP4 at
> `n_draft: 2` went from 21.1 to 26.8 tok/s.

Greedy speculative output is not byte-identical to plain greedy decode over 256
tokens, and that holds for upstream too: `llama-server` with DFlash 2 or MTP
diverges from its own plain run on two of the four prompts, where the verify
batch's wider matmul rounds a near-tie the other way.

```bash
scripts/spark/remote.sh --env MIX_ENV=bench --big-cores spark-1 \
  mix run bench/spark_spec.exs <target.gguf> plain mtp:2 dflash:7:<drafter.gguf>
```

> #### Qwen3.6 instruct checkpoints need the chat template {: .warning}
>
> `Qwen3.6-35B-A3B` emits end-of-generation **immediately** when handed a bare
> completion prompt — zero tokens, from both the plain and the MTP path — while
> the identical prompt inside the chat template generates normally. The 27B
> tolerates raw completion, which is exactly the kind of difference that becomes
> a mystery if your harness does not template. Use `LlamaCppEx.chat/3`, or
> `LlamaCppEx.Chat.apply_template/3` when you need the prompt as a string (as
> `LlamaCppEx.MTP` does).

---

# Running on TWO Sparks

## What two nodes actually buy you

Read this before building anything on it, because the honest answer is narrower
than the marketing:

- **Capacity: yes.** A model that does not fit in 130 GB runs. Nothing else on
  this pair will run it at all.
- **Speed: prefill yes, decode no.** Since llama.cpp #18626 the RPC backend
  implements async compute and events, so llama.cpp enables **pipeline
  parallelism** across an RPC split (it used to refuse, and the nodes ran
  sequentially). A prompt's micro-batches now overlap across the two nodes:
  prefill measured **+16–26%** for gpt-oss-120b and +22% for Qwen3.8-27B NVFP4
  against one node. A single decode stream has nothing to overlap and stays
  within noise of one node (see B1 below).
- **Tensor parallelism ("tp=2"): technically yes, practically no.** It runs and
  it is correct, and it is 2.7× slower on decode. See the verdict section.

## The mechanism, in one paragraph

One node runs a *worker* (`LlamaCppEx.RPC.Server`) exposing its GPU on a TCP
endpoint. The other node is the *client*: it registers that endpoint with
`LlamaCppEx.RPC.add_server/1`, at which point the remote GPU appears in
`LlamaCppEx.devices/0` as `RPC0` and can hold part of a model like any other
device. `split_mode: :layer` then gives each device a contiguous range of layers
and its own KV cache. On Linux the transport auto-negotiates RDMA over the
ConnectX-7 link; on this pair it always does.

## Runbook

```bash
# 1. Worker on spark-2, bound to the fabric address, tensor cache on.
scripts/spark/rpc-worker.sh start spark-2

# 2. Prove the whole chain before believing any number.
scripts/spark/remote.sh --env LLAMA_RPC=1 --env MIX_ENV=test spark-1 \
  mix run scripts/spark/rpc_check.exs 10.100.64.2:50052

# 3. Use it.
scripts/spark/remote.sh --env LLAMA_RPC=1 --env MIX_ENV=bench spark-1 \
  mix run bench/spark_two_node.exs b1

# 4. Stop it between runs. The worker leaks; see below.
scripts/spark/rpc-worker.sh stop spark-2
```

Other subcommands: `status`, `rss`, `logs [n]`, `restart`. Useful flags:
`--debug` (`GGML_RPC_DEBUG=1` on the worker), `--tcp` (force TCP for an A/B),
`--upstream` (run upstream's `ggml-rpc-server` instead of ours, as a reference),
`--no-cache`, `--threads N`, `--port N`.

The worker needs an RPC build, which `rpc-worker.sh` arranges by exporting
`LLAMA_RPC=1` into the unit. The client needs one too — pass
`--env LLAMA_RPC=1` to `remote.sh`.

In code:

```elixir
{:ok, _} = LlamaCppEx.RPC.add_server("10.100.64.2:50052")

{:ok, server} =
  LlamaCppEx.Server.start_link(
    model_path: path,
    n_gpu_layers: 99,
    devices: ["CUDA0", "RPC0"],      # name them; see the ordering trap below
    split_mode: :layer,
    tensor_split: [0.5, 0.5]
  )
```

`:rpc_servers` does the registration for you, in the right order:

```elixir
LlamaCppEx.Server.start_link(
  model_path: path,
  rpc_servers: ["10.100.64.2:50052"],
  devices: ["CUDA0", "RPC0"],
  split_mode: :layer,
  tensor_split: [0.5, 0.5]
)
```

## Supervision

`loginctl enable-linger` succeeds without a password on these boxes, so the
worker runs as a **`systemd --user` transient unit** started with `systemd-run`.
That gets journald capture, `systemctl --user restart`, and survival across
logout, with no root and no unit files to install. `rpc-worker.sh` also starts an
RSS sampler alongside it.

`LlamaCppEx.RPC.Server` is a GenServer that owns the native server thread, but it
**cannot stop it**: upstream's accept loop is `while (true)` with no shutdown
hook, so the thread and its port outlive the process. `terminate/2` says so
rather than pretending. The VM is the unit of restart.

## Five things that will cost you an afternoon

### 1. A peer failure kills the VM

Every client-side RPC command checks its result with `RPC_STATUS_ASSERT`, which
is `GGML_ABORT`. A worker that crashes, a link that drops, or a malformed
response **terminates the OS process — the BEAM with it**. There is no error
return, no retry, no reconnect, and nothing to `rescue`.

This is upstream's design, not this binding's, and the API is shaped around it:
*registration* is the one operation that reports instead of aborting, so
`LlamaCppEx.RPC.ping/1` before a load turns "the model silently landed on the
wrong devices" into `{:error, :unreachable}`. After the load, treat the VM as the
unit of restart. Real fault isolation means putting the RPC client in a separate
OS process, which is a different architecture.

Upstream's registration aborts too since #18626 — an unreachable endpoint, a
listener that is not an RPC worker, a version mismatch and a malformed
`host:port` all reach `GGML_ABORT` in `rpc_dispatcher::start`. So the NIF
connects and exchanges the protocol HELLO itself before registering, with a
5-second timeout. That is what keeps `ping/1` a check rather than a crash on
this llama.cpp; the matching upstream fix is branch `rpc-add-server-no-abort`
on `nyo16/llama.cpp`.

### 2. `devices/0` order is NOT `tensor_split` order

Two device lists exist and they disagree:

| list | order | read by |
|---|---|---|
| ggml **registry** | registration order — local first, RPC appended | `LlamaCppEx.devices/0`, and so `gpu_index` |
| llama.cpp **placement** | **RPC first**, then GPUs, then iGPUs | `:tensor_split`, `:main_gpu` |

Measured on spark-1 with one endpoint registered:

```
[0] CUDA0   CUDA   igpu  gpu_index=0   NVIDIA GB10
[1] CPU     CPU    cpu   gpu_index=nil CPU
[2] RPC0    RPC    gpu   gpu_index=1   10.100.64.2:50052
```

…yet placement is `[RPC0, CUDA0]`, so `tensor_split: [0.25, 0.75]` puts 25% on
the **remote** node and `main_gpu: 0` selects it. A backwards split produces
correct tokens and merely benchmarks badly, so nothing warns you.

**Always pass `:devices`.** It is used verbatim — no reordering, no dedup, no CPU
filtering — and then `:tensor_split` indexes the list you wrote down.

### 3. The worker leaks, but it plateaus

Measured across repeated runs against one worker serving a 30 GB share:
349 → 595 MiB on the first client, then 595 → 595 for every subsequent client.
So it retains roughly 245 MiB per model share and never gives it back, but it
does **not** accumulate per run within one worker lifetime. Restart between
experiments anyway; `rpc-worker.sh` runs an RSS sampler that stops the worker at
92% of RAM rather than letting the node OOM.

### 4. A model that does not fit does not degrade — it OOMs the box

Loading Qwen3-235B-A22B Q4_K_M (142.1 GB) on **one** node with mmap did not
produce a slow number. It produced this:

```
oom-kill: constraint=CONSTRAINT_NONE, global_oom
Out of memory: Killed process 1599 (avahi-daemon)
NVRM: Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal
```

The machine survived, but the OOM killer took out an unrelated system service —
`avahi-daemon`, so the box stopped resolving over mDNS and became unreachable by
name from the control node while remaining perfectly healthy. Unified memory is
the reason: there is no separate VRAM to spill into, so "offload everything" and
"keep it in page cache" compete for the same 130 GB and `mmap` cannot save you.

That is why `scripts/spark/lib.sh` has `SPARK_HOST_SPARK_1` / `SPARK_HOST_SPARK_2`
overrides — a name-resolution failure should be a one-variable fix:

```bash
export SPARK_HOST_SPARK_1=192.168.0.164
```

### 5. Both nodes must be byte-identical

The RPC `HELLO` handshake compares only `major`/`minor` and **ignores `patch`**,
so two builds from drifted trees connect happily and then misinterpret each
other's op codes. `sync.sh` proves byte-identity by content digest on every run.
This is load-bearing, not hygiene.

## Measured

### The headline: a model that does not fit on one node

Qwen3-235B-A22B Q4_K_M, **142.1 GB** of weights against 130.0 GB of unified
memory per node. 32 decode steps.

| run | prompt | load s | TTFT ms | prefill t/s | decode t/s | worker RSS |
|---|---|---|---|---|---|---|
| single-node, mmap overflow | — | — | — | — | — | **global OOM, killed `avahi-daemon`** |
| two-node 50/50, RDMA, b10362 (sequential) | 512 | 538.4 | 1281.9 | 423.5 | 13.69 | 344 → 580 MiB |
| two-node 50/50, RDMA, `785322d17` (pipelined) | 1024 | 575.2 | 1432.5 | **760.3** | 11.67 | 348 → 628 MiB |

There is no percentage to quote here, and that is the point: on one Spark this
model does not run slowly, it takes the machine's memory out from under the OOM
killer. On two it runs at **12–14 tokens/s**, which is a usable interactive speed
for a 235B model, and with pipelining its prefill is 1.8× what it was. (The
decode gap between the two rows is within the load-to-load spread described
under the control below.)

The cold load is nine minutes, because ~71 GB of weights cross the fabric. Budget
for it, keep the worker's tensor cache on, and do not restart casually.

**This is the entire argument for the second Spark.** If your model fits in
130 GB, the numbers below say one node is the answer.

### The control: a model that does fit

gpt-oss-120b MXFP4 (63.4 GB — fits one node, so this isolates RPC cost from
memory benefit), 1024-token prompt, 128 decode steps, median of 7, RDMA, one
session at `785322d17`:

| run | load s | TTFT ms | prefill t/s | decode t/s |
|---|---|---|---|---|
| single-node | 50.9 | 515.2 | 2075.9 | 45.62 |
| two-node 50/50 | 154.2 | 447.9 | 2418.0 | 40.94 |
| two-node 50/50, fresh worker | 80.6 | 411.8 | 2625.7 | 45.81 |

**Prefill is now 16–26% faster on two nodes** and TTFT 13–20% lower: with the
RPC backend async (#18626), llama.cpp pipelines the prompt's micro-batches
across the nodes, where it used to run them one node after the other. Decode
is a single stream with nothing to overlap; across four RDMA runs it landed
between 36.5 and 45.8 tok/s against 45.6 on one node, so call it **no gain,
and noisier**. Qwen3.8-27B NVFP4 says the same: prefill 1410 → 1725 tok/s,
decode 14.02 → 13.77.

Load time is still the cost — 1.6–3× one node — because ~30 GB of weights
cross the network and the client still reads and hashes every tensor to check
the worker's cache.

### RDMA versus TCP, and the 10 GbE LAN

`GGML_RPC_NO_RDMA=1` on either peer keeps the connection on TCP
(`rpc-worker.sh start spark-2 --tcp`); `--link lan` binds the worker to the
10 GbE management NIC instead of the fabric. Same 120b model; load is from a
run with the worker's tensor cache off (every byte crosses the link), the rest
from a 7-sample run per transport on a freshly restarted worker:

| transport | load s | TTFT ms | prefill t/s | decode t/s (runs) |
|---|---|---|---|---|
| RDMA, 200 GbE fabric | 54.5 | 411.8 | 2625.7 | 45.8 (36.5–45.8 over 4 runs) |
| TCP, 200 GbE fabric | 59.5 | 426.6 | 2545.6 | 41.1 (41.1–42.8 over 3) |
| TCP, 10 GbE LAN | 75.8 | 426.3 | 2546.8 | 41.2 (40.0–41.2 over 3) |

**What you lose without RDMA is now small:** ~3% of prefill, decode within
RDMA's own run-to-run spread, and ~10% of load time on the same link. The 10
GbE LAN costs the same steady-state and **~40% more load time**, because a
load is the one phase that moves gigabytes. The −27% prefill / −11% decode TCP
penalty measured at b10362 is gone: per-token traffic across a layer split is
one activation row, and the async dispatcher now hides the round trip.
Keep RDMA for loads and as the default, but TCP — even over the LAN — is a
working fallback rather than a cliff.

To confirm which one you got, look at the worker's log with
`GGML_RPC_DEBUG=1`:

```bash
scripts/spark/rpc-worker.sh start spark-2 --debug
scripts/spark/rpc-worker.sh logs spark-2 200 | grep -E 'RDMA|transport'
# RDMA probed: dev=rocep1s0f1 gid=3 RoCEv2 qpn=33437 inline=316
# RDMA activated: qpn=33437->33437 mtu=4096 rx_depth=24
```

`RDMA activate failed, staying on TCP`, or no `RDMA activated` line at all, is
TCP.

### Concurrency across two nodes

| `n_parallel` | decode t/s per request |
|---|---|
| 1 | 49.48 |
| 4 | 44.71 |
| 8 | 44.25 |

Per-token RTT is fixed per *graph*, not per token, so batching amortises it
well: 8 concurrent requests cost 11% of per-request decode for 8× the aggregate.

### The decode fast path is holding

Worth checking whenever anything about batching changes. A repeated graph
collapses to a 4-byte `GRAPH_RECOMPUTE`; a miss re-serialises every tensor
descriptor on every token. Measured over a two-node generation:
**92 `graph_recompute`, zero `graph_compute`**. `LlamaCppEx.Server`'s per-tick
batch composition does not break the cache.

```bash
scripts/spark/rpc-worker.sh logs spark-2 4000 | grep -c graph_recompute
```

---

# "tp=2" — what it means here, and the verdict

"tp=2" is vLLM vocabulary. llama.cpp at b10362 has four split modes and only one
of them is what people mean by it:

| mode | value | status |
|---|---|---|
| `:none` | 0 | single device |
| `:layer` | 1 | contiguous layer ranges, one KV cache per device. **The only working cross-host mode** |
| `:row` | 2 | **dead for CUDA.** ggml-cuda no longer exports `ggml_backend_split_buffer_type`, so the load throws `device CUDA0 does not support split buffers`. Only SYCL still declares one |
| `:tensor` | 3 | real tensor parallelism via a Meta device (#19378, Apr 2026) |

`:tensor` forces flash attention on, refuses a handful of architectures
(`llm_arch_supports_sm_tensor` is a blocklist, so qwen3 and gpt-oss both pass),
and disables backend sampling.

## Does `-sm tensor` work across two hosts?

**Yes — and you should not use it.** No prior report of this combination exists;
here is one.

Single node first, to price the Meta device itself (Qwen3-8B, 512-token prompt):

| configuration | prefill t/s | decode t/s |
|---|---|---|
| `:none`, flash on | 3700.1 | 38.88 |
| `:tensor`, one local GPU | 3865.5 | 38.46 (−1.1%) |

Essentially free. So the Meta device is not the problem. Now two nodes:

| configuration | prefill t/s | decode t/s | vs layer split |
|---|---|---|---|
| `:layer`, 2 nodes | 3262.4 | 36.58 | — |
| `:tensor`, 2 nodes | 140.9 | 13.30 | **−63.6%** |
| `:tensor`, `GGML_CUDA_ALLREDUCE=none` | 137.4 | 13.59 | −63.2% |
| `:tensor`, `GGML_CUDA_ALLREDUCE=internal` | 277.6 | 10.33 | −71.8% |

It runs, and the output is byte-identical to the layer-split reference. It is
**2.7× slower on decode and 23× slower on prefill**.

## Why, and why the comm-mode knob cannot help

`ggml_backend_cuda_comm_init` returns `nullptr` the moment **any** member backend
is not CUDA. An RPC device is not CUDA, so the CUDA all-reduce — NCCL or the
internal pipeline — never engages at all, which is why all three
`GGML_CUDA_ALLREDUCE` settings land in the same place.

What runs instead is the meta backend's generic butterfly, which moves data with
`ggml_backend_tensor_{set,get}_2d`. The RPC backend leaves both 2-D hooks `NULL`,
and ggml then falls back to a **loop of `n_copies` separate 1-D transfers**. That
is the cliff: not a failure, just every all-reduce turned into a burst of
individual network round trips, once per layer, per token.

Note this contradicts the obvious reading of the source, which is that NULL 2-D
hooks would abort. They do not — `ggml-backend.cpp` degrades to the 1-D loop.
That is why this was worth measuring rather than reasoning about.

## The verdict

**On two DGX Sparks, use `split_mode: :layer` over the RPC backend.** It is the
only cross-host mode that is both correct and fast, and it buys capacity.

- `:row` throws at load on CUDA. Do not build on it.
- `:tensor` is in-process tensor parallelism. Its CUDA all-reduce is
  `ncclCommInitAll` — single-process, one distinct physical GPU per rank — so it
  cannot span hosts as designed. Across hosts it silently falls back to a generic
  path that is 2.7× slower. It is the right tool for several GPUs in **one** box,
  which a Spark does not have.
- Layer split over RPC buys **capacity, not speed**. If your model fits in
  130 GB, one Spark is the answer.

Re-check these claims against a future llama.cpp bump using upstream commits
`d6f303004` (`-sm tensor`), `adb541a6a` and `91fef9536`.
