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

### Why `LLAMA_CPU_ARM_ARCH` is not optional

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
| `use_mlock`, `use_direct_io` | leave off. Both cost 3–4%; "pinned in RAM" and "resident on the GPU" are the same DRAM here |
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

## Qwen3.6 and speculative decoding (MTP)

Two shapes of the current generation, each measured with and without the
Multi-Token Prediction head. The MTP repos are the *same weights* plus the head,
so this is a clean A/B rather than a comparison across quantizations.

256-token greedy generations through the chat template, **median of 5** with the
range alongside. Five samples matter here: MTP is the noisier arm, and a single
run per setting is not enough to tell a real 1.6× from a lucky draw. The plain
model is freed before the MTP arm runs — leaving ~20 GB of unrelated weights
resident cost the MTP arm about 10% on this unified-memory part, which is
exactly the kind of confound that flips a conclusion.

### Qwen3.6-27B Q4_K_M — dense

| config | decode t/s | range | vs baseline | acceptance |
|---|---|---|---|---|
| no MTP head | 11.59 | 11.5–11.6 | — | — |
| MTP `n_draft: 1` | 16.88 | 16.9–16.9 | 1.46× | 86.9% |
| MTP `n_draft: 2` | 18.38 | 18.2–18.4 | 1.59× | 76.4% |
| **MTP `n_draft: 3`** | **18.65** | 18.4–18.7 | **1.61×** | 68.2% |
| MTP `n_draft: 4` | 17.42 | 17.3–17.5 | 1.50× | 57.1% |

No range overlaps the baseline: **MTP is worth 1.6× on the dense model**, and
`n_draft` 2 and 3 are within noise of each other.

### Qwen3.6-35B-A3B UD-Q4_K_M — MoE

| config | decode t/s | range | vs baseline | acceptance |
|---|---|---|---|---|
| no MTP head | 65.36 | 65.1–65.8 | — | — |
| MTP `n_draft: 1` | 67.56 | 67.0–67.7 | 1.03× | 81.0% |
| MTP `n_draft: 2` | 62.29 | 61.4–62.3 | 0.95× | 67.2% |
| MTP `n_draft: 3` | 63.28 | 63.0–63.5 | 0.97× | 64.9% |
| MTP `n_draft: 4` | 47.66 | 44.8–48.6 | 0.73× | 40.9% |

Essentially neutral at best, and a loss past `n_draft: 1`.

> The README reports **+16%** at `n_draft: 2` for Qwen3.6-35B-A3B on GB10, from
> an interleaved n=11 run on **UD-Q4_K_XL**. This measurement is UD-Q4_K_**M**,
> and the draft acceptance rates agree closely (67.2% here versus 68.5% there at
> `n_draft: 2`) while the throughput economics do not. Take the quantization as
> the likely difference and measure your own before relying on either number.

### The rule this gives you

**MTP pays on dense models and roughly breaks even on sparse MoE.** The mechanism
is the one that makes this chip interesting: speculative decoding spends compute
(a batched verification pass) to save memory bandwidth (sequential decode steps).
On the dense 27B every token reads all 27B of weights, so that trade is strongly
favourable — 1.6×. On the 35B-A3B only ~3B parameters move per token, decode is
already cheap, and the draft-and-verify overhead eats the gain.

Two more things the numbers say:

- **The best `n_draft` is model-shaped, so measure it.** On the dense model 3 is
  best and 2 is within noise; on the MoE anything above 1 loses. There is no
  single default that transfers, which is also the README's conclusion for
  Metal.
- **Acceptance decays fast with draft depth** — 87% → 76% → 68% → 57% on the
  dense model — so past the sweet spot you pay twice: wasted draft compute and a
  longer verification batch.

```elixir
{:ok, model} = LlamaCppEx.Model.load(mtp_path, n_gpu_layers: 99, load_mtp: true)
{:ok, session} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 4096)
{:ok, text} = LlamaCppEx.MTP.generate(session, prompt, max_tokens: 256)
```

Reproduce:

```bash
scripts/spark/remote.sh --env MIX_ENV=bench --big-cores spark-1 \
  mix run bench/spark_mtp.exs <plain.gguf> <mtp.gguf>
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
- **Speed: no, but also not the loss you would expect.** Pipeline parallelism is
  *disabled* whenever an RPC device participates — the RPC backend reports
  `async = false, events = false` and llama.cpp checks that before enabling
  pipelining — so the two nodes execute **sequentially**. For a model that fits
  on one node, the second node measured within a few percent either way
  (see B1 below), because the sequential penalty and the halved per-node
  bandwidth pressure roughly cancel.
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
memory per node. 512-token prompt, 32 decode steps.

| run | load s | TTFT ms | prefill t/s | decode t/s | worker RSS |
|---|---|---|---|---|---|
| single-node, mmap overflow | — | — | — | — | **global OOM, killed `avahi-daemon`** |
| two-node 50/50, RDMA | 538.4 | 1281.9 | 423.5 | **13.69** | 344 → 580 MiB |

There is no percentage to quote here, and that is the point: on one Spark this
model does not run slowly, it takes the machine's memory out from under the OOM
killer. On two it runs at **13.7 tokens/s**, which is a usable interactive speed
for a 235B model.

The cold load is nine minutes, because ~71 GB of weights cross the fabric. Budget
for it, keep the worker's tensor cache on, and do not restart casually.

**This is the entire argument for the second Spark.** If your model fits in
130 GB, the numbers below say one node is the answer.

### The control: a model that does fit

gpt-oss-120b MXFP4 (63.4 GB — fits one node, so this isolates RPC cost from
memory benefit), 1024-token prompt, 64 decode steps.

| run | load s | TTFT ms | prefill t/s | decode t/s |
|---|---|---|---|---|
| single-node | 64.8 | 563.6 | 1889.0 | 46.50 |
| two-node 50/50, RDMA | 151.3 | 590.4 | 1797.6 | **48.14** |

Decode came out **+3.5% on two nodes** for a model that fits on one. That is not
what "the nodes run sequentially" predicts, and it is worth understanding before
reading too much into it: splitting halves each node's per-token weight traffic,
and on a bandwidth-bound part that relief roughly cancels the sequential
penalty. Repeat runs put both configurations in the 45–49 t/s band, so the
honest summary is **"no material difference"**, not "two nodes are faster".

Load time is the real cost: **2.3× worse cold**, because ~30 GB of weights cross
the network. The worker's content-addressed tensor cache (`-c`, on by default in
`rpc-worker.sh`) took a warm load from 151 s to 139 s — much less than hoped,
because the client still reads and hashes every tensor locally to check the
cache; only the transfer is skipped.

### RDMA versus TCP

There is no runtime switch. Transport selection is silent auto-negotiation with
no env var and no endpoint scheme; the only levers are `GGML_RDMA_DEV` pointing
at a device that does not exist, or a `LLAMA_RPC_RDMA=0` build.

| transport | TTFT ms | prefill t/s | decode t/s |
|---|---|---|---|
| RDMA | 595.7 | 1784.0 | 46.05 |
| TCP (forced) | 805.8 | 1310.5 | 40.90 |
| | **+35%** | **−27%** | **−11%** |

So RDMA is worth real tokens, and if RDMA ever wedges (upstream issue #24813,
closed as stale one week before our pinned commit), TCP is a working fallback
that costs about a tenth of decode.

To confirm which one you got, there is exactly one signal — the worker's log with
`GGML_RPC_DEBUG=1`:

```bash
scripts/spark/rpc-worker.sh start spark-2 --debug
scripts/spark/rpc-worker.sh logs spark-2 200 | grep -E 'RDMA|transport'
# RDMA probed: dev=rocep1s0f1 gid=3 RoCEv2 qpn=33437 inline=316
# RDMA activated: qpn=33437->33437 mtu=4096 rx_depth=24
```

`RDMA activate failed, staying on TCP` is the line that means you are measuring
the slow path.

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
