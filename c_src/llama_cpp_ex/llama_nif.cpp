#include "llama_nif.h"
#include <fine.hpp>
#include <llama.h>
#include <ggml-backend.h>
#include <nlohmann/json.hpp>
#include "json-schema-to-grammar.h"
#include "speculative.h"
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cerrno>

// The ggml RPC backend is opt-in at build time (LLAMA_RPC=1). GGML_USE_RPC is
// set by the Makefile, not inherited from cmake: ggml puts it on the `ggml`
// target as a PUBLIC definition, and this translation unit is compiled by hand
// with only -I flags.
#ifdef GGML_USE_RPC
#include <ggml-rpc.h>
#include <netdb.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#endif

using namespace llama_cpp_ex;

// --- Resource registrations ---

FINE_RESOURCE(LlamaModel);
FINE_RESOURCE(LlamaContext);
FINE_RESOURCE(LlamaSampler);
FINE_RESOURCE(LlamaSpeculative);
FINE_RESOURCE(CancelFlag);

// --- Error atoms ---
//
// Declared at namespace scope so fine interns the terms during NIF load.
namespace atoms {
inline auto invalid_seq_id  = fine::Atom("invalid_seq_id");
inline auto invalid_index   = fine::Atom("invalid_index");
inline auto invalid_grammar = fine::Atom("invalid_grammar");
inline auto invalid_state   = fine::Atom("invalid_state");
inline auto unsupported     = fine::Atom("unsupported");
inline auto rpc_unsupported = fine::Atom("rpc_unsupported");
inline auto unreachable     = fine::Atom("unreachable");
inline auto no_devices      = fine::Atom("no_devices");
inline auto bind_timeout    = fine::Atom("bind_timeout");
} // namespace atoms

// --- Input validation at the NIF boundary ---
//
// llama.cpp's GGML_ASSERT is never NDEBUG-gated, so it calls ggml_abort() ->
// abort(). That takes down the whole BEAM: no exception, no supervisor, no
// crash report. Every value crossing this boundary comes from Elixir, so every
// one has to be bounded — but by exactly one of two mechanisms, and knowing
// which is why the guard set below is what it is:
//
//   1. Values that reach a `llama_memory_*` or `llama_state_seq_*` call
//      DIRECTLY are bounded here. Those functions assert on their arguments,
//      and nothing upstream sees the value first.
//
//   2. Values carried INSIDE a `llama_batch` are bounded by upstream's
//      `llama_batch_allocr::init` (vendor/llama.cpp/src/llama-batch.cpp:61-64),
//      which range-checks batch seq ids and positions and returns false rather
//      than asserting — so `llama_decode` returns non-zero and the NIF returns
//      `{:error, _}`. That check is load-bearing and invisible from this file,
//      which is part of why the three unguarded sites in category 1
//      (embed_decode, embed_batch_decode, batch_eval_sample's purge list)
//      looked closed: their neighbours really were safe, for a reason nobody
//      had written down.
//
// `test/nif_guards_test.exs` enumerates both categories and is checked against
// LlamaCppEx.NIF's own source, so a new seq_id-taking NIF cannot be added
// without classifying it.

// A sequence id is valid iff 0 <= seq_id < llama_n_seq_max(ctx). Out of range
// trips GGML_ASSERT in the KV cache implementation.
static bool valid_seq_id(llama_context* ctx, int64_t seq_id) {
    return seq_id >= 0 && seq_id < static_cast<int64_t>(llama_n_seq_max(ctx));
}

// Raising variant of the above, for the NIFs whose success value is a bare
// integer or boolean. An out-of-range seq_id is a caller bug, not a recoverable
// condition, so it raises `** (ErlangError) :invalid_seq_id` rather than
// widening a hot-path return type into a tuple — `sampler_sample/2` runs once
// per generated token. Either way the VM survives, which is the whole point:
// the unguarded path was GGML_ASSERT -> abort().
static void check_seq_id(ErlNifEnv* env, llama_context* ctx, int64_t seq_id) {
    if (!valid_seq_id(ctx, seq_id)) {
        fine::raise(env, atoms::invalid_seq_id);
    }
}

// String form of the same predicate, for the NIFs whose error type is a message
// rather than an atom. One definition of "valid", two error shapes — the shape
// follows the NIF's existing return type so adding a guard is never a breaking
// change.
static std::string invalid_seq_id_message(llama_context* ctx, int64_t seq_id) {
    return "invalid seq_id " + std::to_string(seq_id) +
           " (must be 0 <= seq_id < " + std::to_string(llama_n_seq_max(ctx)) + ")";
}

// Single choke point for every llama_decode in this file, so the recorded batch
// shape cannot drift from reality as decode paths are added. Only a successful
// decode publishes a new shape; a failed one leaves the previous logits (and
// therefore the previous shape) in place.
//
// This claim was false for seven direct `llama_decode` calls (the two streaming
// generators, the blocking `generate`, and the MTP prefill/verify/rollback), and
// false in the *permissive* direction: `sampler_sample_at`'s bound was computed
// from a stale batch, so it accepted indices upstream would abort on. A
// permissively-stale bound is worse than no bound because it reads as safe.
// Every `llama_decode` in this translation unit now goes through here; the only
// decodes that do not are inside upstream's `common_speculative_*`, which run on
// the draft context and are never sampled through this boundary.
static int decode_tracked(LlamaContext& c, const llama_batch& batch) {
    int ret = llama_decode(c.ctx, batch);
    if (ret == 0) {
        c.record_batch(batch);
    }
    return ret;
}

// Same rationale as check_seq_id: an out-of-range logits index is a caller bug,
// and the alternative to raising is abort().
static void check_logits_idx(ErlNifEnv* env, const LlamaContext& c, int64_t idx) {
    if (!c.valid_logits_idx(idx)) {
        fine::raise(env, atoms::invalid_index);
    }
}

// Bounds for untrusted grammar and JSON-schema text. Both of llama.cpp's
// parsers are recursive descent over the input text, so the text's nesting
// depth bounds the C stack depth — unbounded nesting is a stack overflow
// (SIGSEGV), which no amount of exception handling recovers from.
static constexpr size_t kMaxGrammarBytes = 1u << 20; // 1 MiB
static constexpr int    kMaxGrammarDepth = 64;
static constexpr size_t kMaxSchemaBytes  = 1u << 20; // 1 MiB
static constexpr int    kMaxSchemaDepth  = 64;

// Max `(`-grouping nesting depth in a GBNF grammar. Grouping is the only
// construct llama.cpp's parse_sequence() recurses on. Quoted literals, `[...]`
// character classes and `#` comments are skipped so a grammar like
// `root ::= "((("` is not mistaken for depth 3.
static int gbnf_group_depth(const std::string& s) {
    int depth = 0;
    int max_depth = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (c == '#') {
            while (i < s.size() && s[i] != '\n') i++;
        } else if (c == '"' || c == '\'') {
            const char quote = c;
            for (i++; i < s.size() && s[i] != quote; i++) {
                if (s[i] == '\\') i++;
            }
        } else if (c == '[') {
            for (i++; i < s.size() && s[i] != ']'; i++) {
                if (s[i] == '\\') i++;
            }
        } else if (c == '(') {
            if (++depth > max_depth) max_depth = depth;
        } else if (c == ')') {
            if (depth > 0) depth--;
        }
    }
    return max_depth;
}

// Max `{`/`[` nesting depth in JSON text, skipping string literals. Bounds the
// depth of both nlohmann's parser and json_schema_to_grammar's recursion.
static int json_nesting_depth(const std::string& s) {
    int depth = 0;
    int max_depth = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (c == '"') {
            for (i++; i < s.size() && s[i] != '"'; i++) {
                if (s[i] == '\\') i++;
            }
        } else if (c == '{' || c == '[') {
            if (++depth > max_depth) max_depth = depth;
        } else if (c == '}' || c == ']') {
            if (depth > 0) depth--;
        }
    }
    return max_depth;
}

// RAII guard for an ErlNifEnv allocated inside a streaming NIF. The streaming
// generators throw C++ exceptions from several call sites; without this the
// unwind leaks the env (and fine's exception translation never sees the
// enif_free_env call that the happy path performs).
struct MsgEnvGuard {
    ErlNifEnv* env;

    MsgEnvGuard() : env(enif_alloc_env()) {}
    ~MsgEnvGuard() {
        if (env) enif_free_env(env);
    }

    MsgEnvGuard(const MsgEnvGuard&) = delete;
    MsgEnvGuard& operator=(const MsgEnvGuard&) = delete;

    operator ErlNifEnv*() const { return env; }
};

// Frees a llama_batch obtained from llama_batch_init on scope exit. Same
// rationale as MsgEnvGuard: several statements between the init and the free
// can throw (fine decoders, std::bad_alloc, detokenize failures), and an unwind
// past a bare llama_batch_init leaks the batch's token/pos/seq_id buffers.
// Binds to an existing local so the surrounding code keeps using `batch`
// directly.
struct BatchFreeGuard {
    llama_batch& batch;

    explicit BatchFreeGuard(llama_batch& b) : batch(b) {}
    ~BatchFreeGuard() { llama_batch_free(batch); }

    BatchFreeGuard(const BatchFreeGuard&) = delete;
    BatchFreeGuard& operator=(const BatchFreeGuard&) = delete;
};

// --- Cancellation ---

fine::ResourcePtr<CancelFlag> cancel_flag_new(ErlNifEnv* env) {
    return fine::make_resource<CancelFlag>();
}
FINE_NIF(cancel_flag_new, 0);

fine::Ok<> request_cancel(ErlNifEnv* env, fine::ResourcePtr<CancelFlag> flag) {
    flag->cancelled.store(true, std::memory_order_relaxed);
    return fine::Ok();
}
FINE_NIF(request_cancel, 0);

static bool cancel_abort_cb(void* data) {
    return static_cast<CancelFlag*>(data)->cancelled.load(std::memory_order_relaxed);
}

// Installs the flag as the context's abort callback for the duration of a
// generation loop, so a cancel interrupts even a long prefill decode
// (llama_decode returns 2). Cleared on scope exit.
struct AbortCallbackScope {
    llama_context* ctx;

    AbortCallbackScope(llama_context* c, CancelFlag* flag) : ctx(c) {
        llama_set_abort_callback(ctx, cancel_abort_cb, flag);
    }
    ~AbortCallbackScope() {
        llama_set_abort_callback(ctx, nullptr, nullptr);
    }
};

// --- Backend ---

fine::Ok<> backend_init(ErlNifEnv* env) {
    llama_backend_init();
    return fine::Ok();
}
FINE_NIF(backend_init, 0);

fine::Ok<> backend_free(ErlNifEnv* env) {
    llama_backend_free();
    return fine::Ok();
}
FINE_NIF(backend_free, 0);

// --- RPC ---
//
// The ggml RPC backend puts a model's layers on another host. It is compiled in
// only when LLAMA_RPC=1; without it every function here reports
// {:error, :rpc_unsupported} so a CPU or Metal build keeps loading unchanged.
//
// Two upstream properties shape this API and neither is ours to fix:
//
//   1. RPC_STATUS_ASSERT is GGML_ABORT (ggml-rpc.cpp:30). Any peer crash,
//      network failure or malformed response terminates the OS process — the
//      whole BEAM. There is no error return, no retry, no reconnect.
//   2. Registration, by contrast, is safe: an unreachable endpoint makes
//      ggml_backend_rpc_add_server return nullptr. So failures are *detectable
//      before load* and *fatal during it*, and that asymmetry is why
//      rpc_add_server reports rather than logs.

#ifdef GGML_USE_RPC

namespace {

// "host:port", where host may be an IPv4 literal or a name. The RPC transport
// parses the same string itself; this copy exists only for the pre-flight bind.
bool rpc_split_endpoint(const std::string& endpoint, std::string& host, std::string& port) {
    auto colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == endpoint.size()) {
        return false;
    }
    host = endpoint.substr(0, colon);
    port = endpoint.substr(colon + 1);
    return true;
}

// Bind the endpoint, then immediately give it back. ggml_backend_rpc_start_server
// returns void and never returns at all on success, so it can report neither a
// bad host nor a port already in use — it prints to stderr and the thread just
// sits there. Doing the bind here first turns the common failures into an
// errno we can hand back to Elixir. It is a TOCTOU window, which is why the
// caller still waits for the real listener afterwards.
std::string rpc_preflight_bind(const std::string& host, const std::string& port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* res = nullptr;
    int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
    if (rc != 0) {
        return std::string("cannot resolve ") + host + ": " + gai_strerror(rc);
    }

    std::string error = "no usable address for " + host;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            close(fd);
            error.clear();
            break;
        }
        error = std::string("cannot bind ") + host + ":" + port + ": " + std::strerror(errno);
        close(fd);
    }
    freeaddrinfo(res);
    return error;
}

// Wait for something to accept a connection on the endpoint. The proof that the
// detached thread actually got as far as listen(), rather than printing to
// stderr and returning. The probe is closed before any HELLO is sent; the
// server's read then fails and it loops back to accept, which is harmless.
bool rpc_wait_until_listening(const std::string& host, const std::string& port, int timeout_ms) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    for (int waited = 0; waited < timeout_ms; waited += 20) {
        addrinfo* res = nullptr;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) == 0) {
            bool up = false;
            for (addrinfo* ai = res; ai && !up; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                up = connect(fd, ai->ai_addr, ai->ai_addrlen) == 0;
                close(fd);
            }
            freeaddrinfo(res);
            if (up) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

} // namespace
#endif // GGML_USE_RPC

// Whether this build has the RPC backend compiled in.
//
// Exists so a test can assert the *exact* refusal for the build it is running on
// rather than accepting either atom. Without it, the natural way to write one
// test for both configurations is to allow {:rpc_unsupported, :unreachable} —
// which stays green when an RPC build reports :rpc_unsupported, i.e. precisely
// the stale/shared-artifact failure the Makefile's link marker exists to
// eliminate. A capability probe turns that into an exact assertion on both
// builds.
bool rpc_supported(ErlNifEnv* env) {
#ifdef GGML_USE_RPC
    return true;
#else
    return false;
#endif
}
FINE_NIF(rpc_supported, 0);

// Registers a remote endpoint's devices in the global ggml device registry.
//
// Deliberately a backend-level NIF next to backend_init/device_list rather than
// a twelfth model_load argument: registration mutates process-global state, not
// one model, and it has to happen before a load so that tensor placement can see
// the remote devices.
//
// Returns the number of devices the endpoint contributed. Upstream memoizes per
// endpoint, so a repeat call is a no-op and returns 0 added.
std::variant<fine::Ok<int64_t>, fine::Error<fine::Atom>>
rpc_add_server(ErlNifEnv* env, std::string endpoint) {
#ifndef GGML_USE_RPC
    (void)endpoint;
    return fine::Error(atoms::rpc_unsupported);
#else
    size_t before = ggml_backend_dev_count();

    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint.c_str());
    if (!reg) {
        // Both an unreachable endpoint and a HELLO major/minor mismatch collapse
        // to nullptr here, and ggml_backend_register silently no-ops on nullptr
        // — so without this check a dead or mismatched node simply vanishes and
        // the model loads onto the wrong devices.
        return fine::Error(atoms::unreachable);
    }

    // ggml_backend_rpc_add_server only *builds* the registration. Without this
    // second call the devices never enter ggml_backend_dev_count() and
    // device_list will not see them.
    ggml_backend_register(reg);

    return fine::Ok(static_cast<int64_t>(ggml_backend_dev_count() - before));
#endif
}
// Dirty IO: a blocking TCP connect plus the HELLO round trip.
FINE_NIF(rpc_add_server, ERL_NIF_DIRTY_JOB_IO_BOUND);

// Starts the worker-side RPC server. `device_names` empty means every non-CPU
// device, falling back to the CPU device, matching tools/rpc/rpc-server.cpp.
//
// Returns the device names actually being served. The server itself runs on a
// detached std::thread and never comes back: ggml_backend_rpc_start_server's
// accept loop is `while (true)` and the cleanup after it is dead code.
//
// Detaching that thread is exactly what decouples it from the scheduler that
// called us, so THIS NIF returns normally and the never-returning loop costs no
// scheduler at all. What the NIF does spend is real though: a blocking
// getaddrinfo in the pre-flight bind, another per poll iteration, and up to
// 5000 ms of connect-polling before it can honestly claim to be listening. That
// is thousands of times the ~1 ms a normal scheduler expects, and
// RPC.Server.start_link/1 is the sort of call made during application start, so
// it belongs on a dirty IO scheduler like rpc_add_server.
std::variant<fine::Ok<std::vector<std::string>>, fine::Error<fine::Atom>,
             fine::Error<std::string>>
rpc_start_server(ErlNifEnv* env, std::string endpoint, std::string cache_dir,
                 int64_t n_threads, std::vector<std::string> device_names) {
#ifndef GGML_USE_RPC
    (void)endpoint; (void)cache_dir; (void)n_threads; (void)device_names;
    return fine::Error(atoms::rpc_unsupported);
#else
    std::string host, port;
    if (!rpc_split_endpoint(endpoint, host, port)) {
        return fine::Error(std::string("endpoint must be \"host:port\", got: " + endpoint));
    }

    std::vector<ggml_backend_dev_t> devices;
    if (!device_names.empty()) {
        for (const auto& name : device_names) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) {
                return fine::Error(std::string("unknown device: " + name));
            }
            devices.push_back(dev);
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                devices.push_back(dev);
            }
        }
        if (devices.empty()) {
            if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
                devices.push_back(cpu);
            }
        }
    }

    if (devices.empty()) {
        return fine::Error(atoms::no_devices);
    }

    if (std::string error = rpc_preflight_bind(host, port); !error.empty()) {
        return fine::Error(error);
    }

    std::vector<std::string> served;
    served.reserve(devices.size());
    for (auto dev : devices) served.emplace_back(ggml_backend_dev_name(dev));

    // Copied into the thread: the NIF's locals are gone the moment it returns,
    // and the server outlives every one of them.
    std::thread([endpoint, cache_dir, n_threads, devices]() mutable {
        ggml_backend_rpc_start_server(endpoint.c_str(),
                                      cache_dir.empty() ? nullptr : cache_dir.c_str(),
                                      static_cast<size_t>(n_threads),
                                      devices.size(), devices.data());
    }).detach();

    // Spawning a thread is not the same as serving, and reporting :ok from the
    // spawn alone would make every misconfiguration look like a success until
    // the first client hangs.
    if (!rpc_wait_until_listening(host, port, 5000)) {
        return fine::Error(atoms::bind_timeout);
    }

    return fine::Ok(served);
#endif
}
// Dirty IO: a blocking bind, two blocking getaddrinfo calls, and a poll that can
// run for 5 s. Bounded, but nowhere near a normal scheduler's budget.
FINE_NIF(rpc_start_server, ERL_NIF_DIRTY_JOB_IO_BOUND);

// --- Devices ---

// Enumerates ggml backend devices for VRAM-aware placement and budgeting.
// GPU/IGPU devices receive a `gpu_index` (0-based, in device order) matching
// the index space of llama.cpp's `tensor_split`; other devices get -1.
fine::Term device_list(ErlNifEnv* env) {
    auto make_binary = [&](const char* s) -> ERL_NIF_TERM {
        size_t len = s ? std::strlen(s) : 0;
        ERL_NIF_TERM bin;
        unsigned char* data = enif_make_new_binary(env, len, &bin);
        if (len) std::memcpy(data, s, len);
        return bin;
    };

    size_t n = ggml_backend_dev_count();
    std::vector<ERL_NIF_TERM> devices;
    devices.reserve(n);

    int gpu_index = 0;
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char* backend = reg ? ggml_backend_reg_name(reg) : "";

        const char* type_atom;
        int this_gpu_index = -1;
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   type_atom = "cpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type_atom = "gpu";  this_gpu_index = gpu_index++; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type_atom = "igpu"; this_gpu_index = gpu_index++; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type_atom = "accel"; break;
            default:                             type_atom = "other"; break;
        }

        ERL_NIF_TERM keys[8] = {
            enif_make_atom(env, "index"),
            enif_make_atom(env, "gpu_index"),
            enif_make_atom(env, "name"),
            enif_make_atom(env, "description"),
            enif_make_atom(env, "type"),
            enif_make_atom(env, "backend"),
            enif_make_atom(env, "memory_total"),
            enif_make_atom(env, "memory_free"),
        };
        ERL_NIF_TERM vals[8] = {
            enif_make_int64(env, (int64_t)i),
            enif_make_int64(env, (int64_t)this_gpu_index),
            make_binary(ggml_backend_dev_name(dev)),
            make_binary(ggml_backend_dev_description(dev)),
            enif_make_atom(env, type_atom),
            make_binary(backend),
            enif_make_uint64(env, (uint64_t)total_mem),
            enif_make_uint64(env, (uint64_t)free_mem),
        };

        ERL_NIF_TERM map;
        enif_make_map_from_arrays(env, keys, vals, 8, &map);
        devices.push_back(map);
    }

    return fine::Term(enif_make_list_from_array(env, devices.data(), (unsigned)devices.size()));
}
FINE_NIF(device_list, ERL_NIF_DIRTY_JOB_IO_BOUND);

// --- Model ---

std::variant<fine::Ok<fine::ResourcePtr<LlamaModel>>, fine::Error<std::string>>
model_load(ErlNifEnv* env, std::string path, int64_t n_gpu_layers, bool use_mmap,
           int64_t main_gpu, int64_t split_mode, std::vector<double> tensor_split,
           bool use_mlock, bool use_direct_io, bool vocab_only, bool check_tensors,
           bool load_mtp, std::vector<std::string> device_names) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = static_cast<int32_t>(n_gpu_layers);
    params.main_gpu = static_cast<int32_t>(main_gpu);
    params.split_mode = static_cast<enum llama_split_mode>(split_mode);
    // Upstream collapsed the use_mmap/use_mlock/use_direct_io booleans into a
    // single llama_load_mode enum. Direct I/O still wins over everything else,
    // but mlock no longer implies mmap: b10173 redefined LLAMA_LOAD_MODE_MLOCK
    // as mlock *without* mmap and added LLAMA_LOAD_MODE_MMAP_MLOCK for the
    // combination, so :use_mlock and :use_mmap are both honoured rather than
    // the former silently forcing a memory map.
    params.load_mode = use_direct_io         ? LLAMA_LOAD_MODE_DIRECT_IO
                     : use_mlock && use_mmap ? LLAMA_LOAD_MODE_MMAP_MLOCK
                     : use_mlock             ? LLAMA_LOAD_MODE_MLOCK
                     : use_mmap              ? LLAMA_LOAD_MODE_MMAP
                                             : LLAMA_LOAD_MODE_NONE;
    params.vocab_only = vocab_only;
    params.check_tensors = check_tensors;
    // Upstream defaults this to false so that non-speculative callers do not pay
    // for the MTP head's tensors (#26296). It has to be set at *load* time: the
    // layers are either read off disk here or they are absent for the model's
    // whole lifetime, and `common_speculative_init` still succeeds without them
    // — the failure surfaces much later as `verify decode failed: code=-1` from
    // the first MTP draft. `LlamaCppEx.MTP.init/2` therefore refuses a model
    // loaded without this flag rather than letting that error escape.
    params.load_mtp = load_mtp;

    std::vector<float> ts_float;
    if (!tensor_split.empty()) {
        ts_float.reserve(tensor_split.size());
        for (auto v : tensor_split) ts_float.push_back(static_cast<float>(v));
        params.tensor_split = ts_float.data();
    }

    // llama_model_params.devices is used verbatim: no reordering, no dedup, no
    // CPU filtering (src/llama.cpp:152-176). That is the point. The default
    // path instead rebuilds the list with RPC devices at the FRONT
    // (src/llama.cpp:263-273), which does not match the ggml registry order
    // device_list reports — so with a remote device registered, tensor_split
    // silently indexes a different list than the caller was looking at.
    // Naming the devices is the only way to make placement deterministic.
    //
    // NULL-terminated, so it must outlive the load call.
    std::vector<ggml_backend_dev_t> devs;
    if (!device_names.empty()) {
        devs.reserve(device_names.size() + 1);
        for (const auto& name : device_names) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) {
                std::string available;
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    if (i) available += ", ";
                    available += ggml_backend_dev_name(ggml_backend_dev_get(i));
                }
                return fine::Error("unknown device: " + name + " (available: " + available + ")");
            }
            devs.push_back(dev);
        }
        devs.push_back(nullptr);
        params.devices = devs.data();
    }

    llama_model* model = llama_model_load_from_file(path.c_str(), params);
    if (!model) {
        return fine::Error(std::string("failed to load model from: " + path));
    }

    return fine::Ok(fine::make_resource<LlamaModel>(model));
}
FINE_NIF(model_load, ERL_NIF_DIRTY_JOB_IO_BOUND);

int64_t model_n_ctx_train(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_model_n_ctx_train(model->model);
}
FINE_NIF(model_n_ctx_train, 0);

int64_t model_n_embd(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_model_n_embd(model->model);
}
FINE_NIF(model_n_embd, 0);

// Output-side embedding width, which is what the MTP draft head consumes. It is
// `n_embd` for every architecture in tree today, but upstream reads this one
// (speculative.cpp: "MTP input row width must match the target h_nextn width")
// and enforces the target/draft match with a GGML_ASSERT — an unconditional
// ggml_abort that takes the whole VM down rather than failing the call. A
// separate drafter GGUF is the only way to reach that assert, so MTP.init/2
// compares this across the two models before it builds anything.
int64_t model_n_embd_out(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_model_n_embd_out(model->model);
}
FINE_NIF(model_n_embd_out, 0);

// Number of MTP / "next-N" prediction layers the checkpoint carries. Zero means
// the GGUF has no MTP head at all, which is a different situation from a model
// loaded with load_mtp: false: no flag can recover it, only a different file.
// Without this, asking for an MTP context on such a model surfaces as a bare
// "failed to create context" while the real reason is one line above it in
// llama.cpp's own log.
int64_t model_n_layer_nextn(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_model_n_layer_nextn(model->model);
}
FINE_NIF(model_n_layer_nextn, 0);

std::string model_desc(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    char buf[256];
    llama_model_desc(model->model, buf, sizeof(buf));
    return std::string(buf);
}
FINE_NIF(model_desc, 0);

uint64_t model_size(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_model_size(model->model);
}
FINE_NIF(model_size, 0);

uint64_t model_n_params(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_model_n_params(model->model);
}
FINE_NIF(model_n_params, 0);

std::string model_chat_template(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    const char* tmpl = llama_model_chat_template(model->model, nullptr);
    if (tmpl) {
        return std::string(tmpl);
    }
    return std::string();
}
FINE_NIF(model_chat_template, 0);

// --- Vocab ---

int64_t vocab_n_tokens(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_vocab_n_tokens(model->vocab());
}
FINE_NIF(vocab_n_tokens, 0);

int64_t vocab_bos(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_vocab_bos(model->vocab());
}
FINE_NIF(vocab_bos, 0);

int64_t vocab_eos(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model) {
    return llama_vocab_eos(model->vocab());
}
FINE_NIF(vocab_eos, 0);

bool vocab_is_eog(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model, int64_t token) {
    return llama_vocab_is_eog(model->vocab(), static_cast<llama_token>(token));
}
FINE_NIF(vocab_is_eog, 0);

// --- Tokenization ---

std::vector<int64_t> tokenize(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaModel> model,
    std::string text,
    bool add_special,
    bool parse_special)
{
    const auto* vocab = model->vocab();

    // First call: get required token count (returns negative)
    int n = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0,
                           add_special, parse_special);

    std::vector<llama_token> tokens(std::abs(n));
    n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(),
                       add_special, parse_special);

    if (n < 0) {
        throw std::runtime_error("tokenization failed");
    }

    tokens.resize(n);

    // Convert llama_token (int32_t) to int64_t for Elixir
    return std::vector<int64_t>(tokens.begin(), tokens.end());
}
// Dirty: measured 862 µs at 1k tokens, 8.56 ms at 10k, 137 ms at 160k — it
// crosses the 1 ms normal-scheduler budget at ~1160 prompt tokens, and it runs
// in the *caller's* process (Server.generate/3, LlamaCppEx.generate/3).
FINE_NIF(tokenize, ERL_NIF_DIRTY_JOB_CPU_BOUND);

std::string detokenize(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaModel> model,
    std::vector<int64_t> token_ids)
{
    const auto* vocab = model->vocab();

    // Convert int64_t to llama_token
    std::vector<llama_token> tokens(token_ids.begin(), token_ids.end());

    // First call to get required buffer size
    int n = llama_detokenize(vocab, tokens.data(), tokens.size(), nullptr, 0, false, false);

    std::vector<char> buf(std::abs(n));
    n = llama_detokenize(vocab, tokens.data(), tokens.size(), buf.data(), buf.size(), false, false);

    if (n < 0) {
        throw std::runtime_error("detokenization failed");
    }

    return std::string(buf.data(), n);
}
// Dirty: same class as tokenize — measured 1.38 ms at 160k tokens.
FINE_NIF(detokenize, ERL_NIF_DIRTY_JOB_CPU_BOUND);

std::string token_to_piece(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model, int64_t token) {
    const auto* vocab = model->vocab();
    char buf[1024];
    int n = llama_token_to_piece(vocab, static_cast<llama_token>(token),
                                  buf, sizeof(buf), 0, false);

    if (n < 0) {
        // Buffer too small, allocate larger
        std::vector<char> large_buf(-n);
        n = llama_token_to_piece(vocab, static_cast<llama_token>(token),
                                  large_buf.data(), large_buf.size(), 0, false);
        return std::string(large_buf.data(), std::max(0, n));
    }

    return std::string(buf, n);
}
FINE_NIF(token_to_piece, 0);

// --- Context ---

std::variant<fine::Ok<fine::ResourcePtr<LlamaContext>>, fine::Error<std::string>>
context_create(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaModel> model,
    int64_t n_ctx,
    int64_t n_batch,
    int64_t n_ubatch,
    int64_t n_threads,
    int64_t n_threads_batch,
    bool embeddings,
    int64_t pooling_type,
    int64_t n_seq_max,
    // KV cache quantization
    int64_t type_k,
    int64_t type_v,
    // Flash attention & GPU offload
    int64_t flash_attn,
    bool offload_kqv,
    bool op_offload,
    // RoPE scaling
    int64_t rope_scaling_type,
    double rope_freq_base,
    double rope_freq_scale,
    double yarn_ext_factor,
    double yarn_attn_factor,
    double yarn_beta_fast,
    double yarn_beta_slow,
    int64_t yarn_orig_ctx,
    // Misc
    int64_t attention_type,
    bool no_perf,
    bool swa_full,
    bool kv_unified,
    // Speculative decoding / MTP
    int64_t ctx_type,
    int64_t n_rs_seq)
{
    auto params = llama_context_default_params();
    params.n_ctx           = static_cast<uint32_t>(n_ctx);
    params.n_batch         = static_cast<uint32_t>(n_batch);
    params.n_ubatch        = static_cast<uint32_t>(n_ubatch);
    params.n_threads       = static_cast<int32_t>(n_threads);
    params.n_threads_batch = static_cast<int32_t>(n_threads_batch);
    params.embeddings      = embeddings;
    params.pooling_type    = static_cast<enum llama_pooling_type>(pooling_type);

    if (n_seq_max > 0) {
        params.n_seq_max = static_cast<uint32_t>(n_seq_max);
    }

    // KV cache quantization
    params.type_k = static_cast<enum ggml_type>(type_k);
    params.type_v = static_cast<enum ggml_type>(type_v);

    // Flash attention & GPU offload
    params.flash_attn_type = static_cast<enum llama_flash_attn_type>(flash_attn);
    params.offload_kqv     = offload_kqv;
    params.op_offload      = op_offload;

    // RoPE scaling
    params.rope_scaling_type = static_cast<enum llama_rope_scaling_type>(rope_scaling_type);
    params.rope_freq_base    = static_cast<float>(rope_freq_base);
    params.rope_freq_scale   = static_cast<float>(rope_freq_scale);
    params.yarn_ext_factor   = static_cast<float>(yarn_ext_factor);
    params.yarn_attn_factor  = static_cast<float>(yarn_attn_factor);
    params.yarn_beta_fast    = static_cast<float>(yarn_beta_fast);
    params.yarn_beta_slow    = static_cast<float>(yarn_beta_slow);
    params.yarn_orig_ctx     = static_cast<uint32_t>(yarn_orig_ctx);

    // Misc
    params.attention_type = static_cast<enum llama_attention_type>(attention_type);
    params.no_perf        = no_perf;
    params.swa_full       = swa_full;
    // Unified KV: all sequences share one buffer/stream, making cross-seq
    // llama_memory_seq_cp a metadata-only tag copy for ANY position range.
    // In split mode (false), partial cross-stream seq_cp aborts the process.
    params.kv_unified     = kv_unified;

    // Speculative decoding / MTP
    params.ctx_type = static_cast<enum llama_context_type>(ctx_type);
    params.n_rs_seq = static_cast<uint32_t>(n_rs_seq);

    // For embedding models, n_ubatch must equal n_batch
    if (embeddings) {
        params.n_ubatch = params.n_batch;
    }

    llama_context* ctx = llama_init_from_model(model->model, params);
    if (!ctx) {
        return fine::Error(std::string("failed to create context"));
    }

    auto res = fine::make_resource<LlamaContext>(ctx, model);
    res->kv_unified = kv_unified;
    return fine::Ok(std::move(res));
}
FINE_NIF(context_create, ERL_NIF_DIRTY_JOB_CPU_BOUND);

int64_t context_n_ctx(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx) {
    return llama_n_ctx(ctx->ctx);
}
FINE_NIF(context_n_ctx, 0);

int64_t context_n_rs_seq(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx) {
    return static_cast<int64_t>(llama_n_rs_seq(ctx->ctx));
}
FINE_NIF(context_n_rs_seq, 0);

// --- Sampler ---

std::variant<fine::Ok<fine::ResourcePtr<LlamaSampler>>, fine::Error<fine::Atom>>
sampler_init(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaModel> model,
    int64_t seed,
    double temp,
    int64_t top_k,
    double top_p,
    double min_p,
    double penalty_repeat,
    double penalty_freq,
    double penalty_present,
    std::string grammar_str,
    std::string grammar_root)
{
    // Validate the grammar BEFORE allocating the chain, so a rejection cannot
    // leak it.
    if (!grammar_str.empty()) {
        if (grammar_str.size() > kMaxGrammarBytes ||
            gbnf_group_depth(grammar_str) > kMaxGrammarDepth) {
            return fine::Error(atoms::invalid_grammar);
        }
    }

    auto chain_params = llama_sampler_chain_default_params();
    auto* chain = llama_sampler_chain_init(chain_params);

    // Grammar sampler goes first (before penalties/temperature).
    //
    // A parse failure must NOT be swallowed: silently omitting the grammar
    // stage turns "give me JSON" into unconstrained generation, which is a
    // validation bypass rather than a cosmetic degradation.
    if (!grammar_str.empty()) {
        const auto* vocab = model->vocab();
        auto* grammar = llama_sampler_init_grammar(
            vocab, grammar_str.c_str(), grammar_root.c_str());
        if (!grammar) {
            llama_sampler_free(chain);
            return fine::Error(atoms::invalid_grammar);
        }
        llama_sampler_chain_add(chain, grammar);
    }

    // Add samplers in recommended order: penalties -> top_k -> top_p -> min_p -> temp -> dist/greedy
    if (penalty_repeat != 1.0 || penalty_freq != 0.0 || penalty_present != 0.0) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_penalties(llama_vocab_n_tokens(model->vocab()), 64,
                static_cast<float>(penalty_repeat),
                static_cast<float>(penalty_freq), static_cast<float>(penalty_present)));
    }

    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(static_cast<int32_t>(top_k)));
    }

    if (top_p < 1.0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(static_cast<float>(top_p), 1));
    }

    if (min_p > 0.0) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(static_cast<float>(min_p), 1));
    }

    if (temp > 0.0) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(static_cast<float>(temp)));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(static_cast<uint32_t>(seed)));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    // The model term is captured on the resource because the grammar sampler
    // holds the raw `const llama_vocab*` from model->vocab() for its whole
    // lifetime. See the comment on LlamaSampler in llama_nif.h.
    return fine::Ok(fine::make_resource<LlamaSampler>(chain, std::move(model)));
}
// Dirty: compiles GBNF eagerly. Measured 190 µs on a 245-rule schema (1.4 µs
// without a grammar) and linear in rule count, so it crosses the 1 ms
// normal-scheduler budget at ~1.3k rules — and it runs per request inside
// Server.handle_call/3.
FINE_NIF(sampler_init, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// Validate-only grammar entry point.
//
// `sampler_init` compiles the grammar as a side effect of building the chain,
// which is too late for `LlamaCppEx.Server`: by then the request has been
// admitted to a slot inside the GenServer that owns the model, and a compile
// failure took the whole server down with it. This lets the *calling* process
// reject a bad `:grammar` synchronously, before the request is queued, without
// allocating a chain it would immediately throw away.
//
// It runs exactly the two checks `sampler_init` runs, in the same order, and
// nothing else — the admission check and `sampler_init` must agree on what
// "valid" means or a request would pass admission and still fail in the slot.
std::variant<fine::Ok<>, fine::Error<fine::Atom>>
grammar_validate(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaModel> model,
    std::string grammar_str,
    std::string grammar_root)
{
    if (grammar_str.empty()) {
        return fine::Ok();
    }

    if (grammar_str.size() > kMaxGrammarBytes ||
        gbnf_group_depth(grammar_str) > kMaxGrammarDepth) {
        return fine::Error(atoms::invalid_grammar);
    }

    auto* grammar = llama_sampler_init_grammar(
        model->vocab(), grammar_str.c_str(), grammar_root.c_str());
    if (!grammar) {
        return fine::Error(atoms::invalid_grammar);
    }

    llama_sampler_free(grammar);
    return fine::Ok();
}
// Dirty for the same reason as sampler_init: it runs the same GBNF parse, whose
// cost is linear in rule count (190 µs on a 245-rule schema).
FINE_NIF(grammar_validate, ERL_NIF_DIRTY_JOB_CPU_BOUND);

fine::Ok<> sampler_accept(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler, int64_t token) {
    llama_sampler_accept(sampler->sampler, static_cast<llama_token>(token));
    return fine::Ok();
}
FINE_NIF(sampler_accept, 0);

fine::Ok<> sampler_reset(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler) {
    llama_sampler_reset(sampler->sampler);
    return fine::Ok();
}
// Dirty: llama_sampler_reset re-parses the entire grammar from scratch
// (llama-sampler.cpp:2472-2490), so a reset costs the same as an init —
// measured 177 µs. It is called per request on the Server's slot samplers.
FINE_NIF(sampler_reset, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// Dirty: the sampler chain runs a softmax over the full vocab (100k+ entries)
// and grammar samplers can take multiple ms — too slow for a normal scheduler.
//
// idx -1 means "the last logits row", which only exists if the previous decode
// on this context asked for any. Sampling with none reaches
// GGML_ASSERT(logits != nullptr) -> abort().
int64_t sampler_sample(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler,
                       fine::ResourcePtr<LlamaContext> ctx) {
    check_logits_idx(env, *ctx, -1);
    return llama_sampler_sample(sampler->sampler, ctx->ctx, -1);
}
FINE_NIF(sampler_sample, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Decode ---

std::variant<fine::Ok<>, fine::Error<std::string>>
decode(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx, std::vector<int64_t> token_ids) {
    int n_tokens = static_cast<int>(token_ids.size());
    if (n_tokens == 0) {
        return fine::Ok();
    }

    // Explicit batch instead of llama_batch_get_one: get_one requests logits
    // for the LAST TOKEN OF EVERY CHUNK, computing (and copying) a full-vocab
    // logit row per n_batch tokens of prompt for nothing. Only the very last
    // token needs logits. Positions continue seq 0 from wherever the context
    // is, preserving get_one's append semantics for repeated decode calls.
    llama_pos start =
        llama_memory_seq_pos_max(llama_get_memory(ctx->ctx), 0) + 1;

    int n_batch = llama_n_batch(ctx->ctx);
    llama_batch& batch = ctx->reserve_batch(std::min(n_tokens, n_batch));

    for (int i = 0; i < n_tokens; i += n_batch) {
        int n = std::min(n_tokens - i, n_batch);
        bool is_last_chunk = (i + n >= n_tokens);

        batch.n_tokens = n;

        for (int j = 0; j < n; j++) {
            batch.token[j]      = static_cast<llama_token>(token_ids[i + j]);
            batch.pos[j]        = start + static_cast<llama_pos>(i + j);
            batch.n_seq_id[j]   = 1;
            batch.seq_id[j][0]  = 0;
            batch.logits[j]     = (is_last_chunk && j == n - 1);
        }

        int ret = decode_tracked(*ctx, batch);
        if (ret != 0) {
            return fine::Error(std::string("llama_decode failed with code: " + std::to_string(ret)));
        }
    }

    return fine::Ok();
}
FINE_NIF(decode, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Memory management ---

fine::Ok<> memory_clear(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx) {
    llama_memory_clear(llama_get_memory(ctx->ctx), false);
    ctx->forget_batch();
    return fine::Ok();
}
FINE_NIF(memory_clear, 0);

// Returns llama_memory_seq_rm's boolean verbatim, which callers must read
// according to the range they asked for. This is the convention every call site
// in this file and in lib/llama_cpp_ex/server{,/prompt_cache}.ex follows, and it
// is the reason some of them discard the result and others do not:
//
//   * A FULL clear (`p0 <= 0 && p1 < 0`) succeeds for every memory module, so
//     the return value carries no information and is discarded — the bare calls
//     in embed_decode, embed_batch_decode and bes_decode_range's purge loop, and
//     the `_ = ...` matches on the Elixir side.
//   * A PARTIAL trim can be refused: `llama_memory_recurrent::seq_rm` honours a
//     rollback of at most `n_rs_seq` positions
//     (vendor/llama.cpp/src/llama-memory-recurrent.cpp:181-187) and returns
//     false beyond that, without raising. Every partial trim therefore has to
//     check, and fall back to a full clear when refused.
//
// A `true = ...` on a partial trim was a MatchError inside the Server's tick.
bool memory_seq_rm(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
                   int64_t seq_id, int64_t p0, int64_t p1) {
    check_seq_id(env, ctx->ctx, seq_id);
    return llama_memory_seq_rm(
        llama_get_memory(ctx->ctx),
        static_cast<llama_seq_id>(seq_id),
        static_cast<llama_pos>(p0),
        static_cast<llama_pos>(p1));
}
FINE_NIF(memory_seq_rm, 0);

// Reports what kinds of seq_rm the context supports — `:part` (any position
// range), `:full` (whole sequence only — hybrid GDN models), `:rs` (partial
// bounded by n_rs_seq snapshots), or `:no` (no memory module). NOTE: calling
// this clears the context's KV memory as a side effect (upstream behavior).
// Only call once at init time, before any decode work has been done.
fine::Term context_can_seq_rm(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx) {
    ctx->forget_batch();
    switch (common_context_can_seq_rm(ctx->ctx)) {
        case COMMON_CONTEXT_SEQ_RM_TYPE_NO:   return fine::Term(enif_make_atom(env, "no"));
        case COMMON_CONTEXT_SEQ_RM_TYPE_PART: return fine::Term(enif_make_atom(env, "part"));
        case COMMON_CONTEXT_SEQ_RM_TYPE_FULL: return fine::Term(enif_make_atom(env, "full"));
        case COMMON_CONTEXT_SEQ_RM_TYPE_RS:   return fine::Term(enif_make_atom(env, "rs"));
    }
    return fine::Term(enif_make_atom(env, "unknown"));
}
FINE_NIF(context_can_seq_rm, 0);

// --- Memory seq_cp ---

// Guards the hybrid-KV abort path in addition to the seq_id range.
//
// `llama_kv_cache::seq_cp` (llama-kv-cache.cpp:447-502) takes a metadata-only
// fast path when both sequences map to the same stream, which is always true
// under a unified KV cache. In split mode the two sequences live in different
// streams, so the buffer data has to be copied, and upstream only implements
// that for a *whole* sequence: `GGML_ASSERT(is_full && "seq_cp() is only
// supported for full KV buffers")`. GGML_ASSERT is never NDEBUG-gated, so a
// partial cross-sequence copy in split mode calls ggml_abort() and takes the
// process image with it. Reproduced while benchmarking.
//
// `Server` avoids this with its `cross_slot_sharing` flag, but that is an
// Elixir-side convention and this NIF is callable directly, so the boundary
// guards it too. "Full" is read conservatively as p0 <= 0 && p1 < 0 — the only
// range upstream's `is_full` is guaranteed to accept regardless of KV size.
std::variant<fine::Ok<>, fine::Error<fine::Atom>>
memory_seq_cp(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
              int64_t seq_id_src, int64_t seq_id_dst,
              int64_t p0, int64_t p1) {
    if (!valid_seq_id(ctx->ctx, seq_id_src) || !valid_seq_id(ctx->ctx, seq_id_dst)) {
        return fine::Error(atoms::invalid_seq_id);
    }

    const bool full = p0 <= 0 && p1 < 0;
    if (!ctx->kv_unified && !full && seq_id_src != seq_id_dst) {
        return fine::Error(atoms::unsupported);
    }

    llama_memory_seq_cp(
        llama_get_memory(ctx->ctx),
        static_cast<llama_seq_id>(seq_id_src),
        static_cast<llama_seq_id>(seq_id_dst),
        static_cast<llama_pos>(p0),
        static_cast<llama_pos>(p1));
    return fine::Ok();
}
FINE_NIF(memory_seq_cp, 0);

// --- Memory seq_keep ---

fine::Ok<> memory_seq_keep(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx, int64_t seq_id) {
    check_seq_id(env, ctx->ctx, seq_id);
    llama_memory_seq_keep(
        llama_get_memory(ctx->ctx),
        static_cast<llama_seq_id>(seq_id));
    return fine::Ok();
}
FINE_NIF(memory_seq_keep, 0);

// --- Memory seq_pos_max ---

int64_t memory_seq_pos_max(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx, int64_t seq_id) {
    check_seq_id(env, ctx->ctx, seq_id);
    return llama_memory_seq_pos_max(
        llama_get_memory(ctx->ctx),
        static_cast<llama_seq_id>(seq_id));
}
FINE_NIF(memory_seq_pos_max, 0);

// --- Context n_seq_max ---

int64_t context_n_seq_max(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx) {
    return llama_n_seq_max(ctx->ctx);
}
FINE_NIF(context_n_seq_max, 0);

// --- Embeddings ---

std::variant<fine::Ok<>, fine::Error<std::string>>
embed_decode(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    std::vector<int64_t> token_ids,
    int64_t seq_id)
{
    int n_tokens = static_cast<int>(token_ids.size());
    if (n_tokens == 0) {
        return fine::Error(std::string("empty token list"));
    }

    // `seq_id` goes to llama_memory_seq_rm below, which is OUTSIDE any batch, so
    // upstream's llama_batch_allocr::init never sees it and
    // GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && ...)) at
    // llama-kv-cache.cpp:385 aborts the VM. Batch-carried seq ids are bounded by
    // upstream; ones that reach a memory op first are not.
    if (!valid_seq_id(ctx->ctx, seq_id)) {
        return fine::Error(invalid_seq_id_message(ctx->ctx, seq_id));
    }

    // Fresh decode for THIS sequence only — clearing the whole memory would
    // clobber every other sequence if the context is ever shared.
    llama_memory_seq_rm(llama_get_memory(ctx->ctx),
                        static_cast<llama_seq_id>(seq_id), -1, -1);

    // Build batch with explicit seq_id and position tracking
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    BatchFreeGuard batch_guard(batch);
    batch.n_tokens = n_tokens;

    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]      = static_cast<llama_token>(token_ids[i]);
        batch.pos[i]        = static_cast<llama_pos>(i);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = static_cast<llama_seq_id>(seq_id);
        batch.logits[i]     = true; // all tokens get embeddings
    }

    int ret = decode_tracked(*ctx, batch);

    if (ret != 0) {
        return fine::Error(std::string("embed_decode failed with code: " + std::to_string(ret)));
    }

    return fine::Ok();
}
FINE_NIF(embed_decode, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// Returns the embedding as a raw little-endian f32 binary (native order on
// all supported targets): one 16 KB refc binary for a 4096-dim vector instead
// of ~100 KB of boxed floats + list cells per call. The Elixir wrapper
// decodes to a list by default and passes the binary through for Nx users.
std::variant<fine::Ok<fine::Term>, fine::Error<std::string>>
get_embeddings(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    int64_t seq_id,
    int64_t normalize)
{
    int n_embd = llama_model_n_embd(llama_get_model(ctx->ctx));
    enum llama_pooling_type ptype = llama_pooling_type(ctx->ctx);

    const float* embd = nullptr;

    if (ptype == LLAMA_POOLING_TYPE_NONE) {
        // No pooling: get embeddings for the last token
        embd = llama_get_embeddings_ith(ctx->ctx, -1);
    } else {
        // Pooled: get embeddings for the sequence
        embd = llama_get_embeddings_seq(ctx->ctx, static_cast<llama_seq_id>(seq_id));
    }

    if (!embd) {
        return fine::Error(std::string("failed to get embeddings (null pointer)"));
    }

    ERL_NIF_TERM bin;
    float* out = reinterpret_cast<float*>(
        enif_make_new_binary(env, static_cast<size_t>(n_embd) * sizeof(float), &bin));

    if (normalize == 2) {
        // L2 normalization
        double sum = 0.0;
        for (int i = 0; i < n_embd; i++) sum += (double)embd[i] * (double)embd[i];
        float norm = sum > 0.0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0f;
        for (int i = 0; i < n_embd; i++) out[i] = embd[i] * norm;
    } else if (normalize == 0) {
        // Max-abs normalization
        double max_abs = 0.0;
        for (int i = 0; i < n_embd; i++) {
            double a = std::abs((double)embd[i]);
            if (a > max_abs) max_abs = a;
        }
        float norm = max_abs > 0.0 ? static_cast<float>(1.0 / max_abs) : 0.0f;
        for (int i = 0; i < n_embd; i++) out[i] = embd[i] * norm;
    } else {
        // No normalization
        std::memcpy(out, embd, static_cast<size_t>(n_embd) * sizeof(float));
    }

    return fine::Ok(fine::Term(bin));
}
FINE_NIF(get_embeddings, 0);

// --- Batched embeddings: decode many sequences in a single batch ---
//
// Each {seq_id, token_ids} sequence is laid out at its own positions (0..len-1)
// under its own seq_id, so one llama_decode populates per-sequence pooled
// embeddings retrievable via get_embeddings(ctx, seq_id, ...). The caller must
// size the context with embeddings=true and n_seq_max >= number of sequences,
// and keep the total token count within n_batch/n_ubatch.
std::variant<fine::Ok<>, fine::Error<std::string>>
embed_batch_decode(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    std::vector<std::tuple<int64_t, std::vector<int64_t>>> sequences)
{
    if (sequences.empty()) {
        return fine::Error(std::string("empty sequence list"));
    }

    int total = 0;
    for (auto& [seq_id, tokens] : sequences) {
        total += static_cast<int>(tokens.size());
    }
    if (total == 0) {
        return fine::Error(std::string("no tokens to decode"));
    }

    // Every seq_id is validated before the first memory op, for the reason on
    // embed_decode above: the loop below calls llama_memory_seq_rm outside any
    // batch. Validating all of them first also means a bad id in the middle of
    // the list cannot clear the sequences before it and then fail.
    for (auto& [seq_id, tokens] : sequences) {
        if (!valid_seq_id(ctx->ctx, seq_id)) {
            return fine::Error(invalid_seq_id_message(ctx->ctx, seq_id));
        }
    }

    // Fresh decode for the sequences in THIS batch only (successive groups
    // reuse the same seq ids) — never clear the whole memory.
    for (auto& [seq_id, tokens] : sequences) {
        llama_memory_seq_rm(llama_get_memory(ctx->ctx),
                            static_cast<llama_seq_id>(seq_id), -1, -1);
    }

    llama_batch batch = llama_batch_init(total, 0, 1);
    BatchFreeGuard batch_guard(batch);
    batch.n_tokens = total;

    int idx = 0;
    for (auto& [seq_id, tokens] : sequences) {
        int len = static_cast<int>(tokens.size());
        for (int i = 0; i < len; i++) {
            batch.token[idx]     = static_cast<llama_token>(tokens[i]);
            batch.pos[idx]       = static_cast<llama_pos>(i);
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = static_cast<llama_seq_id>(seq_id);
            batch.logits[idx]    = true; // all tokens get embeddings
            idx++;
        }
    }

    int ret = decode_tracked(*ctx, batch);

    if (ret != 0) {
        return fine::Error(std::string("embed_batch_decode failed with code: " + std::to_string(ret)));
    }

    return fine::Ok();
}
FINE_NIF(embed_batch_decode, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Prefill (batched inference) ---

std::variant<fine::Ok<int64_t>, fine::Error<std::string>>
prefill(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    std::vector<int64_t> token_ids,
    int64_t seq_id)
{
    int n_tokens = static_cast<int>(token_ids.size());
    if (n_tokens == 0) {
        return fine::Error(std::string("empty token list"));
    }

    int n_batch = llama_n_batch(ctx->ctx);
    llama_batch& batch = ctx->reserve_batch(std::min(n_tokens, n_batch));

    for (int i = 0; i < n_tokens; i += n_batch) {
        int n = std::min(n_tokens - i, n_batch);
        bool is_last_chunk = (i + n >= n_tokens);

        batch.n_tokens = n;

        for (int j = 0; j < n; j++) {
            batch.token[j]      = static_cast<llama_token>(token_ids[i + j]);
            batch.pos[j]        = static_cast<llama_pos>(i + j);
            batch.n_seq_id[j]   = 1;
            batch.seq_id[j][0]  = static_cast<llama_seq_id>(seq_id);
            // Only request logits for the last token of the last chunk
            batch.logits[j]     = (is_last_chunk && j == n - 1);
        }

        int ret = decode_tracked(*ctx, batch);

        if (ret != 0) {
            return fine::Error(std::string("prefill decode failed with code: " + std::to_string(ret)));
        }
    }

    return fine::Ok(static_cast<int64_t>(n_tokens));
}
FINE_NIF(prefill, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Decode batch (batched inference) ---

std::variant<
    fine::Ok<std::vector<std::tuple<int64_t, int64_t, std::string>>>,
    fine::Error<std::string>
>
decode_batch(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    fine::ResourcePtr<LlamaSampler> sampler,
    std::vector<std::tuple<int64_t, int64_t, int64_t>> entries)
{
    // entries: [{seq_id, token_id, position}, ...]
    int n = static_cast<int>(entries.size());
    if (n == 0) {
        return fine::Error(std::string("empty entries list"));
    }

    const auto* vocab = ctx->model->vocab();

    // Build a single batch with all entries
    llama_batch batch = llama_batch_init(n, 0, 1);
    BatchFreeGuard batch_guard(batch);
    batch.n_tokens = n;

    for (int i = 0; i < n; i++) {
        auto& [seq_id, token_id, pos] = entries[i];
        batch.token[i]      = static_cast<llama_token>(token_id);
        batch.pos[i]        = static_cast<llama_pos>(pos);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = static_cast<llama_seq_id>(seq_id);
        batch.logits[i]     = true; // need logits for all entries to sample
    }

    int ret = decode_tracked(*ctx, batch);

    if (ret != 0) {
        return fine::Error(std::string("decode_batch failed with code: " + std::to_string(ret)));
    }

    // Sample next token for each entry
    std::vector<std::tuple<int64_t, int64_t, std::string>> results;
    results.reserve(n);

    for (int i = 0; i < n; i++) {
        auto& [seq_id, token_id, pos] = entries[i];

        llama_sampler_reset(sampler->sampler);
        // llama_sampler_sample() already accepts the token internally.
        llama_token new_token = llama_sampler_sample(sampler->sampler, ctx->ctx, i);

        // Detokenize
        std::string piece;
        if (!llama_vocab_is_eog(vocab, new_token)) {
            char buf[1024];
            int pn = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
            if (pn < 0) {
                std::vector<char> large_buf(-pn);
                pn = llama_token_to_piece(vocab, new_token,
                    large_buf.data(), large_buf.size(), 0, false);
                if (pn > 0) piece.assign(large_buf.data(), pn);
            } else if (pn > 0) {
                piece.assign(buf, pn);
            }
        }

        results.emplace_back(seq_id, static_cast<int64_t>(new_token), piece);
    }

    return fine::Ok(results);
}
FINE_NIF(decode_batch, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Decode single token with seq_id (for Server) ---

std::variant<fine::Ok<>, fine::Error<std::string>>
decode_token(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    int64_t token_id,
    int64_t pos,
    int64_t seq_id)
{
    llama_batch& batch = ctx->reserve_batch(1);
    batch.n_tokens     = 1;
    batch.token[0]     = static_cast<llama_token>(token_id);
    batch.pos[0]       = static_cast<llama_pos>(pos);
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = static_cast<llama_seq_id>(seq_id);
    batch.logits[0]    = true;

    int ret = decode_tracked(*ctx, batch);

    if (ret != 0) {
        return fine::Error(std::string("decode_token failed with code: " + std::to_string(ret)));
    }

    return fine::Ok();
}
FINE_NIF(decode_token, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Batch eval (forward pass only, no sampling) ---

std::variant<fine::Ok<>, fine::Error<std::string>>
batch_eval(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    std::vector<std::tuple<int64_t, int64_t, int64_t, bool>> entries)
{
    int n = static_cast<int>(entries.size());
    if (n == 0) {
        return fine::Error(std::string("empty entries list"));
    }

    llama_batch& batch = ctx->reserve_batch(n);
    batch.n_tokens = n;

    for (int i = 0; i < n; i++) {
        auto& [token_id, pos, seq_id, logits] = entries[i];
        batch.token[i]      = static_cast<llama_token>(token_id);
        batch.pos[i]        = static_cast<llama_pos>(pos);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = static_cast<llama_seq_id>(seq_id);
        batch.logits[i]     = logits;
    }

    int ret = decode_tracked(*ctx, batch);

    if (ret != 0) {
        return fine::Error(std::string("batch_eval failed with code: " + std::to_string(ret)));
    }

    return fine::Ok();
}
FINE_NIF(batch_eval, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Fused batch eval + sample (Server hot loop) ---
//
// One NIF call per Server tick: builds a batch from `entries`
// ({token, pos, seq_id, wants_logits}), runs llama_decode, then samples every
// wants_logits entry whose seq_id has a sampler registered in `samplers`,
// returning {seq_id, new_token, piece, is_eog}. The piece is empty for EOG
// tokens. Unlike decode_batch above, samplers are per-sequence resources owned
// by the caller — their grammar/penalty state advances across ticks and is
// never reset or shared here.
//
// KV-pressure policy (llama_decode == 1, "no KV slot found"), mirroring
// llama-server's update_slots(): first drop whole sequences listed in
// `purgeable_seq_ids` (idle slots whose cache the caller is willing to lose),
// retrying after each purge; once the purge list is exhausted, recursively
// halve the batch — each half's logits entries are sampled right after its
// sub-decode, before the next decode invalidates the logits buffer. A
// single-token batch that still fails means THAT sequence is out of KV
// budget: it is added to `failed` and skipped for the rest of the call so the
// other sequences keep going (the caller fails just that request). Purged and
// failed seq ids plus the split count are returned so the caller can fix up
// its bookkeeping and emit telemetry. Purgeable seqs must not have entries in
// the batch.

static int bes_decode_range(
    LlamaContext& lctx,
    const llama_vocab* vocab,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, bool>>& entries,
    size_t begin, size_t end,
    const std::vector<std::pair<int64_t, llama_sampler*>>& samplers,
    const std::vector<int64_t>& purgeable,
    size_t& purge_idx,
    std::vector<int64_t>& purged,
    int64_t& n_splits,
    std::vector<int64_t>& failed,
    std::vector<std::tuple<int64_t, int64_t, std::string, bool>>& results,
    llama_batch& batch) // reserved by the caller for >= entries.size() tokens
{
    // Skip entries whose sequence already failed this call — decoding past a
    // failed (missing) position would leave a hole in that sequence's KV.
    std::vector<size_t> idxs;
    idxs.reserve(end - begin);
    for (size_t i = begin; i < end; i++) {
        int64_t seq_id = std::get<2>(entries[i]);
        if (std::find(failed.begin(), failed.end(), seq_id) == failed.end()) {
            idxs.push_back(i);
        }
    }

    size_t n = idxs.size();
    if (n == 0) {
        return 0;
    }

    batch.n_tokens = static_cast<int32_t>(n);

    for (size_t i = 0; i < n; i++) {
        const auto& [token_id, pos, seq_id, logits] = entries[idxs[i]];
        batch.token[i]      = static_cast<llama_token>(token_id);
        batch.pos[i]        = static_cast<llama_pos>(pos);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = static_cast<llama_seq_id>(seq_id);
        batch.logits[i]     = logits;
    }

    int ret = decode_tracked(lctx, batch);

    // Purge donatable idle caches one at a time while the KV cache is full.
    while (ret == 1 && purge_idx < purgeable.size()) {
        auto victim = static_cast<llama_seq_id>(purgeable[purge_idx++]);
        llama_memory_seq_rm(llama_get_memory(lctx.ctx), victim, -1, -1);
        purged.push_back(victim);
        ret = decode_tracked(lctx, batch);
    }

    if (ret == 0) {
        // Sample now: these logits belong to THIS decode call and the next
        // sub-decode would overwrite them.
        for (size_t i = 0; i < n; i++) {
            const auto& [token_id, pos, seq_id, logits] = entries[idxs[i]];
            if (!logits) continue;

            llama_sampler* smpl = nullptr;
            for (const auto& [sid, s] : samplers) {
                if (sid == seq_id) { smpl = s; break; }
            }
            if (!smpl) continue; // logits-only entry, nothing to sample

            // llama_sampler_sample() already accepts the selected token —
            // calling llama_sampler_accept() again would double-advance
            // grammar state.
            llama_token new_token =
                llama_sampler_sample(smpl, lctx.ctx, static_cast<int32_t>(i));
            bool is_eog = llama_vocab_is_eog(vocab, new_token);

            std::string piece;
            if (!is_eog) {
                char buf[1024];
                int pn = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
                if (pn < 0) {
                    std::vector<char> large_buf(-pn);
                    pn = llama_token_to_piece(vocab, new_token,
                        large_buf.data(), large_buf.size(), 0, false);
                    if (pn > 0) piece.assign(large_buf.data(), pn);
                } else if (pn > 0) {
                    piece.assign(buf, pn);
                }
            }

            results.emplace_back(seq_id, static_cast<int64_t>(new_token),
                                 std::move(piece), is_eog);
        }
        return 0;
    }

    if (ret == 1 && n == 1) {
        // A single token still can't fit: this sequence is out of KV budget.
        // Fail it and let the rest of the batch proceed.
        failed.push_back(std::get<2>(entries[idxs[0]]));
        return 0;
    }

    if (ret == 1) {
        // Halve and retry — explicit positions/seq_ids make any split valid,
        // and per-seq entries stay in position order across the halves.
        n_splits++;
        size_t mid = begin + (end - begin) / 2;
        int rc = bes_decode_range(lctx, vocab, entries, begin, mid, samplers,
                                  purgeable, purge_idx, purged, n_splits,
                                  failed, results, batch);
        if (rc != 0) return rc;
        return bes_decode_range(lctx, vocab, entries, mid, end, samplers,
                                purgeable, purge_idx, purged, n_splits,
                                failed, results, batch);
    }

    return ret;
}

std::variant<
    fine::Ok<std::vector<std::tuple<int64_t, int64_t, std::string, bool>>,
             std::vector<int64_t>,
             int64_t,
             std::vector<int64_t>>,
    fine::Error<std::string>
>
batch_eval_sample(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx,
    std::vector<std::tuple<int64_t, int64_t, int64_t, bool>> entries,
    std::vector<std::tuple<int64_t, fine::ResourcePtr<LlamaSampler>>> samplers,
    std::vector<int64_t> purgeable_seq_ids)
{
    if (entries.empty()) {
        return fine::Error(std::string("empty entries list"));
    }

    const auto* vocab = ctx->model->vocab();

    // The ResourcePtr arguments keep the samplers alive for the whole call.
    std::vector<std::pair<int64_t, llama_sampler*>> smpls;
    smpls.reserve(samplers.size());
    for (auto& [sid, s] : samplers) {
        smpls.emplace_back(sid, s->sampler);
    }

    std::vector<std::tuple<int64_t, int64_t, std::string, bool>> results;
    std::vector<int64_t> purged;
    std::vector<int64_t> failed;
    int64_t n_splits = 0;
    size_t purge_idx = 0;

    // `purgeable_seq_ids` reaches llama_memory_seq_rm inside bes_decode_range's
    // KV-pressure loop, outside any batch — same abort as embed_decode's. The
    // entries' seq ids are batch-carried and bounded by upstream's
    // llama_batch_allocr::init; these are not, so they are rejected here rather
    // than filtered: a purge id the caller cannot name is a bookkeeping bug in
    // the caller, not a condition to paper over.
    for (int64_t seq_id : purgeable_seq_ids) {
        if (!valid_seq_id(ctx->ctx, seq_id)) {
            return fine::Error(invalid_seq_id_message(ctx->ctx, seq_id));
        }
    }

    llama_batch& batch = ctx->reserve_batch(static_cast<int32_t>(entries.size()));

    int rc = bes_decode_range(*ctx, vocab, entries, 0, entries.size(),
                              smpls, purgeable_seq_ids, purge_idx, purged,
                              n_splits, failed, results, batch);

    if (rc != 0) {
        return fine::Error(std::string(
            "batch_eval_sample failed with code: " + std::to_string(rc)));
    }

    return fine::Ok(std::move(results), std::move(purged), n_splits,
                    std::move(failed));
}
FINE_NIF(batch_eval_sample, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Sampler sample at batch index ---

// Dirty for the same reason as sampler_sample: full-vocab softmax + optional
// grammar evaluation exceed the ~1 ms normal-scheduler guideline.
//
// `idx` is a *batch token* index (Server passes back the row it built for a
// slot), which upstream's output_resolve_row translates through `output_ids`.
// It aborts on a non-negative index past the batch's token count, and on one
// whose batch token never requested logits — llama_get_logits_ith only degrades
// to nullptr when llama.cpp itself was built with NDEBUG, and the following
// GGML_ASSERT(logits != nullptr) aborts either way. check_logits_idx reproduces
// upstream's accept set against the last recorded batch.
int64_t sampler_sample_at(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaSampler> sampler,
    fine::ResourcePtr<LlamaContext> ctx,
    int64_t idx)
{
    check_logits_idx(env, *ctx, idx);
    return llama_sampler_sample(sampler->sampler, ctx->ctx, static_cast<int32_t>(idx));
}
FINE_NIF(sampler_sample_at, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Sequence state save/restore (RAM prompt cache) ---

// Size of one sequence's serialized KV state. Cheap metadata walk — used by
// the caller to enforce a byte budget BEFORE paying for the copy.
int64_t state_seq_get_size(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
                           int64_t seq_id) {
    check_seq_id(env, ctx->ctx, seq_id);
    return static_cast<int64_t>(
        llama_state_seq_get_size(ctx->ctx, static_cast<llama_seq_id>(seq_id)));
}
FINE_NIF(state_seq_get_size, 0);

// Serializes a sequence's KV state into a binary. Dirty: the state is
// KV-sized (potentially hundreds of MB on long contexts).
std::variant<fine::Ok<fine::Term>, fine::Error<std::string>>
state_seq_get_data(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
                   int64_t seq_id) {
    check_seq_id(env, ctx->ctx, seq_id);
    size_t size = llama_state_seq_get_size(ctx->ctx, static_cast<llama_seq_id>(seq_id));
    if (size == 0) {
        return fine::Error(std::string("sequence has no state"));
    }

    ERL_NIF_TERM bin;
    unsigned char* data = enif_make_new_binary(env, size, &bin);
    size_t written = llama_state_seq_get_data(
        ctx->ctx, data, size, static_cast<llama_seq_id>(seq_id));

    if (written != size) {
        return fine::Error(std::string("state serialization size mismatch"));
    }

    return fine::Ok(fine::Term(bin));
}
FINE_NIF(state_seq_get_data, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// Restores a previously serialized sequence state into dest_seq_id (which
// must be empty). Returns bytes read. Dirty: KV-sized memcpy.
//
// The blob is untrusted: llama_state_seq_set_data parses it as a KV-cache
// serialization, and the only thing standing between a crafted binary and the
// cache is upstream's own bounds checking. Reject what can be rejected cheaply
// first — an empty blob, one that cannot possibly hold a header, and one larger
// than this context's own state could ever be — and treat a 0 return as an
// error rather than proceeding as if the restore had worked.
std::variant<fine::Ok<int64_t>, fine::Error<fine::Atom>>
state_seq_set_data(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
                   fine::Term state_bin, int64_t dest_seq_id) {
    if (!valid_seq_id(ctx->ctx, dest_seq_id)) {
        return fine::Error(atoms::invalid_seq_id);
    }

    ErlNifBinary bin;
    if (!enif_inspect_binary(env, state_bin, &bin)) {
        return fine::Error(atoms::invalid_state);
    }

    // Absolute floor: upstream's reader consumes a uint32 magic followed by a
    // llama_seq_id before anything else (llama-context.cpp:2985-2995), so a
    // blob shorter than that cannot be a state at all. Deliberately NOT
    // compared against this sequence's current state size — a restore target is
    // normally empty, and tying the floor to it would reject valid blobs.
    // Everything past the header is upstream's business: it validates the magic
    // and its host reader throws on overrun, which the surrounding try/catch
    // turns into a 0 return.
    constexpr size_t kMinStateBytes = sizeof(uint32_t) + sizeof(llama_seq_id);
    if (bin.size < kMinStateBytes) {
        return fine::Error(atoms::invalid_state);
    }

    size_t read = llama_state_seq_set_data(
        ctx->ctx, bin.data, bin.size, static_cast<llama_seq_id>(dest_seq_id));

    if (read == 0) {
        return fine::Error(atoms::invalid_state);
    }

    return fine::Ok(static_cast<int64_t>(read));
}
FINE_NIF(state_seq_set_data, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Chat template ---

static ERL_NIF_TERM make_binary_term(ErlNifEnv* env, const char* data, size_t len) {
    ERL_NIF_TERM bin;
    unsigned char* buf = enif_make_new_binary(env, len, &bin);
    memcpy(buf, data, len);
    return bin;
}

std::string chat_apply_template(
    ErlNifEnv* env,
    std::string tmpl,
    std::vector<std::tuple<std::string, std::string>> messages,
    bool add_assistant)
{
    // Build llama_chat_message array - keep strings alive
    std::vector<llama_chat_message> chat_messages;
    chat_messages.reserve(messages.size());
    for (const auto& msg : messages) {
        chat_messages.push_back({std::get<0>(msg).c_str(), std::get<1>(msg).c_str()});
    }

    // First call to get required buffer size
    int n = llama_chat_apply_template(
        tmpl.c_str(), chat_messages.data(), chat_messages.size(),
        add_assistant, nullptr, 0);

    if (n < 0) {
        throw std::runtime_error("failed to apply chat template");
    }

    std::vector<char> buf(n + 1);
    n = llama_chat_apply_template(
        tmpl.c_str(), chat_messages.data(), chat_messages.size(),
        add_assistant, buf.data(), buf.size());

    if (n < 0) {
        throw std::runtime_error("failed to apply chat template");
    }

    return std::string(buf.data(), n);
}
FINE_NIF(chat_apply_template, 0);

// --- Jinja chat template (via common library) ---

std::string chat_apply_template_jinja(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaModel> model,
    std::vector<std::tuple<std::string, std::string>> messages,
    bool add_assistant,
    bool enable_thinking,
    std::vector<std::tuple<std::string, std::string>> extra_kwargs)
{
    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = add_assistant;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;

    // Build messages
    for (const auto& msg : messages) {
        common_chat_msg m;
        m.role = std::get<0>(msg);
        m.content = std::get<1>(msg);
        inputs.messages.push_back(std::move(m));
    }

    // Extra kwargs
    for (const auto& kv : extra_kwargs) {
        inputs.chat_template_kwargs[std::get<0>(kv)] = std::get<1>(kv);
    }

    auto result = common_chat_templates_apply(model->chat_templates.get(), inputs);
    return result.prompt;
}
// Dirty: minja template rendering allocates and walks a full AST per call —
// multi-ms for large templates/histories.
FINE_NIF(chat_apply_template_jinja, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Speculative decoding (MTP) ---

std::variant<fine::Ok<fine::ResourcePtr<LlamaSpeculative>>, fine::Error<std::string>>
speculative_init(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx_tgt,
    fine::ResourcePtr<LlamaContext> ctx_dft,
    int64_t n_draft)
{
    if (n_draft <= 0) {
        return fine::Error(std::string("n_draft must be > 0"));
    }

    // Probe partial-rollback support on the target context BEFORE
    // common_speculative_init. Two reasons:
    //   1. common_context_can_seq_rm clears the context's KV memory as a
    //      side effect (see common.h:904).
    //   2. common_speculative_init's MTP impl calls
    //      llama_set_embeddings_pre_norm(ctx_tgt, true) in its constructor.
    //      Probing afterwards would clobber that flag and the MTP head
    //      would see garbage hidden states (acceptance drops to ~5%).
    // Dense attention-only models report PART → partial seq_rm is native,
    // skip the per-iter checkpoint path. Hybrid models (Qwen 3.6 MoE with
    // GDN layers) report FULL → checkpoint every iteration.
    const auto rm_type_tgt = common_context_can_seq_rm(ctx_tgt->ctx);
    const bool needs_ckpt = (rm_type_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    common_params_speculative params;
    params.types        = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.draft.n_max  = static_cast<int32_t>(n_draft);
    params.draft.ctx_tgt = ctx_tgt->ctx;
    params.draft.ctx_dft = ctx_dft->ctx;

    common_speculative* spec = nullptr;
    try {
        spec = common_speculative_init(params, /*n_seq=*/1);
    } catch (const std::exception& e) {
        return fine::Error(std::string("common_speculative_init threw: ") + e.what());
    }

    if (!spec) {
        return fine::Error(std::string(
            "common_speculative_init returned null — does the model contain MTP heads "
            "and does the draft context have ctx_type=:mtp with n_rs_seq>0?"));
    }

    return fine::Ok(fine::make_resource<LlamaSpeculative>(
        spec, std::move(ctx_tgt), std::move(ctx_dft),
        static_cast<uint32_t>(n_draft), needs_ckpt));
}
// Dirty: common_speculative_init probes the contexts (KV clear + setup work)
// and can block well past the normal-scheduler budget.
FINE_NIF(speculative_init, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// Build the live counter snapshot as a flat map { atom => term }. Used by
// speculative_stats (queried from Elixir) and by the streaming NIF when it
// emits {:done, stats} / {:stats, snapshot}. Lock-free reads via std::atomic;
// safe to call from any thread while generate_mtp_tokens is in flight.
static ERL_NIF_TERM build_mtp_stats_map(ErlNifEnv* env, const LlamaSpeculative& s) {
    uint64_t iters   = s.n_iters.load(std::memory_order_relaxed);
    uint64_t dgen    = s.n_drafts_generated.load(std::memory_order_relaxed);
    uint64_t dacc    = s.n_drafts_accepted.load(std::memory_order_relaxed);
    uint64_t emitted = s.n_tokens_emitted.load(std::memory_order_relaxed);
    uint64_t udraft  = s.us_draft.load(std::memory_order_relaxed);
    uint64_t uverify = s.us_verify.load(std::memory_order_relaxed);
    uint64_t usample = s.us_sample.load(std::memory_order_relaxed);
    uint64_t uckpt   = s.us_ckpt.load(std::memory_order_relaxed);
    uint64_t uother  = s.us_other.load(std::memory_order_relaxed);
    uint64_t utotal  = s.us_total.load(std::memory_order_relaxed);

    double acceptance_rate = dgen > 0 ? (double)dacc / (double)dgen : 0.0;
    double tokens_per_sec  = utotal > 0 ? (double)emitted * 1e6 / (double)utotal : 0.0;

    ERL_NIF_TERM tk[6] = {
        enif_make_atom(env, "draft"),
        enif_make_atom(env, "verify"),
        enif_make_atom(env, "sample"),
        enif_make_atom(env, "ckpt"),
        enif_make_atom(env, "other"),
        enif_make_atom(env, "total"),
    };
    ERL_NIF_TERM tv[6] = {
        enif_make_uint64(env, udraft),
        enif_make_uint64(env, uverify),
        enif_make_uint64(env, usample),
        enif_make_uint64(env, uckpt),
        enif_make_uint64(env, uother),
        enif_make_uint64(env, utotal),
    };
    ERL_NIF_TERM timing;
    enif_make_map_from_arrays(env, tk, tv, 6, &timing);

    ERL_NIF_TERM keys[8] = {
        enif_make_atom(env, "iters"),
        enif_make_atom(env, "drafts_generated"),
        enif_make_atom(env, "drafts_accepted"),
        enif_make_atom(env, "tokens_emitted"),
        enif_make_atom(env, "acceptance_rate"),
        enif_make_atom(env, "tokens_per_sec"),
        enif_make_atom(env, "timing_us"),
        enif_make_atom(env, "n_draft"),
    };
    ERL_NIF_TERM vals[8] = {
        enif_make_uint64(env, iters),
        enif_make_uint64(env, dgen),
        enif_make_uint64(env, dacc),
        enif_make_uint64(env, emitted),
        enif_make_double(env, acceptance_rate),
        enif_make_double(env, tokens_per_sec),
        timing,
        enif_make_uint(env, s.n_draft),
    };
    ERL_NIF_TERM map;
    enif_make_map_from_arrays(env, keys, vals, 8, &map);
    return map;
}

fine::Term speculative_stats(ErlNifEnv* env, fine::ResourcePtr<LlamaSpeculative> spec) {
    return fine::Term(build_mtp_stats_map(env, *spec));
}
FINE_NIF(speculative_stats, 0);

fine::Ok<> speculative_print_stats(ErlNifEnv* env, fine::ResourcePtr<LlamaSpeculative> spec) {
    common_speculative_print_stats(spec->spec);
    return fine::Ok();
}
FINE_NIF(speculative_print_stats, 0);

// Streaming MTP generation. Drives a target/draft speculative loop entirely in C++,
// streaming {ref, {:token, id, text}} messages to caller_pid and finally one of:
//   {ref, :eog}                          — model emitted end-of-generation
//   {ref, {:done, stats_map}}            — hit max_tokens (or eog after some output)
//   {ref, {:error, reason_binary}}       — fatal error
// If emit_stats_every > 0, also sends {ref, {:stats, snapshot_map}} every Nth
// emitted token. Stats counters on the LlamaSpeculative resource are updated
// throughout and remain readable lock-free via speculative_stats/1.
fine::Ok<> generate_mtp_tokens(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaSpeculative> spec_res,
    fine::ResourcePtr<LlamaSampler> sampler_res,
    std::vector<int64_t> prompt_token_ids,
    int64_t max_tokens,
    int64_t emit_stats_every,
    ErlNifPid caller_pid,
    fine::Term ref,
    fine::ResourcePtr<CancelFlag> cancel)
{
    auto& sp = *spec_res;
    auto* ctx_tgt = sp.ctx_tgt->ctx;
    auto* ctx_dft = sp.ctx_dft->ctx;
    auto* sampler = sampler_res->sampler;
    const auto* vocab = sp.ctx_tgt->model->vocab();
    const llama_seq_id seq_id = 0;
    const int32_t n_draft = static_cast<int32_t>(sp.n_draft);

    // Cancellation: polled per speculative iteration and installed as the
    // target context's abort callback so the prompt prefill can stop
    // mid-decode (ret == 2 handled by the prefill error path below).
    AbortCallbackScope abort_scope(ctx_tgt, cancel.get());

    MsgEnvGuard msg_env;

    auto send_error = [&](const std::string& msg) {
        enif_clear_env(msg_env);
        ERL_NIF_TERM rc = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM inner = enif_make_tuple2(msg_env,
            enif_make_atom(msg_env, "error"),
            make_binary_term(msg_env, msg.data(), msg.size()));
        ERL_NIF_TERM tup = enif_make_tuple2(msg_env, rc, inner);
        enif_send(env, &caller_pid, msg_env, tup);
    };

    if (prompt_token_ids.empty()) {
        send_error("prompt cannot be empty");
        return fine::Ok();
    }

    std::vector<llama_token> prompt(prompt_token_ids.begin(), prompt_token_ids.end());

    // Wipe any prior KV on seq 0 in both contexts so the spec begins fresh.
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    llama_memory_clear(llama_get_memory(ctx_dft), true);
    sp.ctx_tgt->forget_batch();
    sp.ctx_dft->forget_batch();

    // Prefill the target context with the prompt, then hand each decoded batch
    // to the speculative state.
    //
    // Upstream removed common_speculative_need_embd in f785fc9ea: a draft
    // implementation that needs the target's hidden states now arranges its own
    // extraction -- the MTP impls call llama_set_embeddings_nextn(ctx_tgt, ...)
    // in their constructors and read it back through
    // llama_get_embeddings_nextn, which is a separate path from the per-token
    // logits flag. So the caller no longer has to request logits on every
    // prefill position; upstream's examples/speculative-simple now passes false
    // for the whole prompt. We ask for logits on the final prompt token only,
    // because we sample the first generated token from them below.
    int n_batch = llama_n_batch(ctx_tgt);
    llama_pos n_past = 0;
    for (size_t i = 0; i < prompt.size(); i += n_batch) {
        int n = std::min(static_cast<int>(prompt.size() - i), n_batch);
        bool is_last_chunk = (i + n >= prompt.size());

        llama_batch batch = llama_batch_init(n, 0, 1);
        BatchFreeGuard batch_guard(batch);
        for (int j = 0; j < n; j++) {
            const bool want_logits = is_last_chunk && j == n - 1;
            common_batch_add(batch, prompt[i + j], static_cast<llama_pos>(i + j),
                             { seq_id }, want_logits);
        }

        int ret = decode_tracked(*sp.ctx_tgt, batch);
        if (ret != 0) {
            send_error("prompt decode failed: code=" + std::to_string(ret));
            return fine::Ok();
        }
        bool proc_ok = common_speculative_process(sp.spec, batch);
        if (!proc_ok) {
            fprintf(stderr,
                "MTP prefill: common_speculative_process returned false "
                "at chunk i=%zu n=%d\n", i, n);
        }
    }
    n_past = static_cast<llama_pos>(prompt.size());


    // Prime the speculative state AFTER prefill+process have populated the
    // draft ctx's KV. common_speculative_begin checks ctx_dft.pos_max and
    // warns if prefill hasn't run yet — calling it before prefill leaves
    // the MTP head's pending_h uninitialised and drafts degrade badly.
    common_speculative_begin(sp.spec, seq_id, prompt);

    // Sample the first generated token from the prompt's last logits.
    char piece_buf[1024];
    std::vector<char> large_buf;
    // Hot atom, interned once (immediate term, env-independent).
    const ERL_NIF_TERM atom_token = enif_make_atom(env, "token");

    auto send_token = [&](llama_token tok, bool special) -> bool {
        int n = llama_token_to_piece(vocab, tok, piece_buf, sizeof(piece_buf),
                                     0, special);
        const char* data = piece_buf;
        int len = n;
        if (n < 0) {
            large_buf.resize(-n);
            len = llama_token_to_piece(vocab, tok, large_buf.data(),
                                       large_buf.size(), 0, special);
            data = large_buf.data();
            if (len < 0) len = 0;
        }
        enif_clear_env(msg_env);
        ERL_NIF_TERM rc = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM inner = enif_make_tuple3(msg_env,
            atom_token,
            enif_make_int64(msg_env, tok),
            make_binary_term(msg_env, data, len > 0 ? len : 0));
        ERL_NIF_TERM tup = enif_make_tuple2(msg_env, rc, inner);
        return enif_send(env, &caller_pid, msg_env, tup);
    };

    auto maybe_send_stats = [&]() {
        if (emit_stats_every <= 0) return;
        uint64_t emitted = sp.n_tokens_emitted.load(std::memory_order_relaxed);
        if (emitted == 0 || (emitted % static_cast<uint64_t>(emit_stats_every)) != 0) return;
        enif_clear_env(msg_env);
        ERL_NIF_TERM rc = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM inner = enif_make_tuple2(msg_env,
            enif_make_atom(msg_env, "stats"),
            build_mtp_stats_map(msg_env, sp));
        ERL_NIF_TERM tup = enif_make_tuple2(msg_env, rc, inner);
        enif_send(env, &caller_pid, msg_env, tup);
    };

    auto send_done = [&](const char* tag) {
        enif_clear_env(msg_env);
        ERL_NIF_TERM rc = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM payload;
        if (tag == nullptr) {
            payload = enif_make_tuple2(msg_env,
                enif_make_atom(msg_env, "done"),
                build_mtp_stats_map(msg_env, sp));
        } else {
            payload = enif_make_atom(msg_env, tag);
        }
        ERL_NIF_TERM tup = enif_make_tuple2(msg_env, rc, payload);
        enif_send(env, &caller_pid, msg_env, tup);
    };

    const auto t_session_start = std::chrono::steady_clock::now();

    // Sample the first generated token from the prompt's last position.
    {
        auto t0 = std::chrono::steady_clock::now();
        llama_token tok = llama_sampler_sample(sampler, ctx_tgt, -1);
        sp.us_sample.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count(),
            std::memory_order_relaxed);

        // llama_sampler_sample() already accepts the token internally.

        if (llama_vocab_is_eog(vocab, tok)) {
            sp.us_total.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_session_start).count(),
                std::memory_order_relaxed);
            send_done("eog");
            return fine::Ok();
        }

        if (!send_token(tok, false)) {
            return fine::Ok();
        }
        sp.n_tokens_emitted.fetch_add(1, std::memory_order_relaxed);
        prompt.push_back(tok);
    }

    llama_token sampled = prompt.back();
    int64_t n_emitted = 1;

    // Soft seq_rm helper: trims [from, inf) on a context, ignoring failure
    // (e.g. when there's nothing past `from` to remove). The MTP loop calls
    // it at points where the exact prior position depends on how many drafts
    // were accepted last iteration, so a no-op return is fine.
    auto soft_seq_rm = [](llama_context* c, llama_seq_id sid, llama_pos from) {
        llama_memory_seq_rm(llama_get_memory(c), sid, from, -1);
    };

    // Hybrid models like Qwen 3.6 (GDN + attention) report
    // COMMON_CONTEXT_SEQ_RM_TYPE_FULL, meaning partial seq_rm fails outright.
    // To recover on partial-draft-accept we save the recurrent state of both
    // contexts before each speculative iteration and restore it on rollback,
    // mirroring upstream's `slot.spec_ckpt` mechanism. We use ON_DEVICE +
    // PARTIAL_ONLY so the save stays in GPU buffers (cheap on Metal/CUDA).
    constexpr llama_state_seq_flags ckpt_flags =
        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    std::vector<uint8_t> ckpt_tgt;
    std::vector<uint8_t> ckpt_dft;

    // Main speculative loop.
    while (n_emitted < max_tokens) {
        if (cancel->cancelled.load(std::memory_order_relaxed)) {
            // Caller abandoned the stream — stop quietly.
            return fine::Ok();
        }

        sp.n_iters.fetch_add(1, std::memory_order_relaxed);

        // Anchor for the "other" bucket: time between known timer ends
        // (us_draft, us_verify, us_sample) accumulates into us_other.
        auto t_anchor = std::chrono::steady_clock::now();

        // 0. Ensure the draft ctx is at pos n_past - 1 BEFORE drafting.
        //    After a partial-accept in the previous iteration, ctx_dft may
        //    still hold positions [n_past, n_past + drafts_prev) that need
        //    to be discarded; otherwise common_speculative_draft would try
        //    to decode at pos n_past with pos_max >= n_past and fail the
        //    M-RoPE consistency check.
        soft_seq_rm(ctx_dft, seq_id, n_past);

        // Snapshot both contexts so we can roll back on partial draft accept.
        // Skip entirely on dense models — common_context_can_seq_rm reported
        // PART at init time, so llama_memory_seq_rm handles partial rejection
        // natively and the checkpoint would be pure overhead.
        if (sp.needs_ckpt) {
            auto t_ck0 = std::chrono::steady_clock::now();
            size_t sz_tgt = llama_state_seq_get_size_ext(ctx_tgt, seq_id, ckpt_flags);
            ckpt_tgt.resize(sz_tgt);
            if (sz_tgt > 0) {
                llama_state_seq_get_data_ext(ctx_tgt, ckpt_tgt.data(), sz_tgt, seq_id, ckpt_flags);
            }
            size_t sz_dft = llama_state_seq_get_size_ext(ctx_dft, seq_id, ckpt_flags);
            ckpt_dft.resize(sz_dft);
            if (sz_dft > 0) {
                llama_state_seq_get_data_ext(ctx_dft, ckpt_dft.data(), sz_dft, seq_id, ckpt_flags);
            }
            // Bill this to us_ckpt, then slide the anchor past it so the same
            // microseconds are not also counted as unaccounted "other".
            auto t_ck1 = std::chrono::steady_clock::now();
            sp.us_ckpt.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t_ck1 - t_ck0).count(),
                std::memory_order_relaxed);
            t_anchor += (t_ck1 - t_ck0);
        }

        // 1. Generate drafts from the MTP head's current state.
        std::vector<llama_token> drafts;
        {
            auto t0 = std::chrono::steady_clock::now();
            sp.us_other.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t0 - t_anchor).count(),
                std::memory_order_relaxed);

            auto& dp = common_speculative_get_draft_params(sp.spec, seq_id);
            dp.drafting = true;
            dp.n_max    = n_draft;
            dp.n_past   = n_past;
            dp.id_last  = sampled;
            dp.prompt   = &prompt;
            dp.result   = &drafts;

            common_speculative_draft(sp.spec);
            t_anchor = std::chrono::steady_clock::now();
            sp.us_draft.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t_anchor - t0).count(),
                std::memory_order_relaxed);
        }

        // 2. Build the verification batch: [sampled, drafts...] at consecutive
        //    positions starting at n_past, all with logits.
        const int n_verify = 1 + static_cast<int>(drafts.size());
        llama_batch batch = llama_batch_init(n_verify, 0, 1);
        BatchFreeGuard batch_guard(batch);
        common_batch_add(batch, sampled, n_past, { seq_id }, true);
        for (size_t i = 0; i < drafts.size(); i++) {
            common_batch_add(batch, drafts[i],
                             n_past + 1 + static_cast<llama_pos>(i),
                             { seq_id }, true);
        }

        // 2b. Roll the draft ctx back to n_past so common_speculative_process
        //     can re-decode the verify batch on it. common_speculative_draft
        //     advances ctx_dft to roughly n_past + drafts.size() via internal
        //     AR decoding; without this rollback, the next llama_decode on
        //     ctx_dft would hit an "inconsistent sequence positions" abort
        //     (M-RoPE requires the current pos_max to be < the batch's first
        //     position). Mirrors the upstream server's seq_rm between draft
        //     and process (server-context.cpp:2347–2353).
        soft_seq_rm(ctx_dft, seq_id, n_past);

        // 3. Decode on the target context, then feed back into the spec.
        {
            auto t0 = std::chrono::steady_clock::now();
            sp.us_other.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t0 - t_anchor).count(),
                std::memory_order_relaxed);

            int ret = decode_tracked(*sp.ctx_tgt, batch);
            if (ret != 0) {
                send_error("verify decode failed: code=" + std::to_string(ret));
                return fine::Ok();
            }
            if (!common_speculative_process(sp.spec, batch)) {
                send_error("common_speculative_process failed");
                return fine::Ok();
            }
            t_anchor = std::chrono::steady_clock::now();
            sp.us_verify.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t_anchor - t0).count(),
                std::memory_order_relaxed);
        }

        // 4. Verify: sample at each position, accept the longest prefix of
        //    drafts that matches, then also keep the model's own next-token
        //    from the position after the last accepted draft.
        int n_accepted_drafts = 0;
        int n_accepted_total  = 0;
        bool eog = false;
        bool send_failed = false;

        // `n_emitted < max_tokens` is re-checked here and not just by the outer
        // `while`: this loop emits up to n_verify (= 1 + n_draft) tokens per
        // iteration, so gating only on entry overshoots the caller's budget by
        // up to n_draft - 1. Worse, the overshoot is not even constant --- how
        // many tokens the final iteration emits depends on how many drafts the
        // target accepts, which varies between runs on a reused session. That
        // made `max_tokens: 16` return 16, 17 or 18 tokens for the same prompt
        // under greedy decoding: the token *sequence* was deterministic, the
        // stopping point was not. Breaking mid-iteration leaves
        // n_accepted_total < n_verify, so the rollback below discards the
        // positions that were decoded but never emitted.
        for (int i = 0; i < n_verify && n_emitted < max_tokens; i++) {
            auto t0 = std::chrono::steady_clock::now();
            sp.us_other.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t0 - t_anchor).count(),
                std::memory_order_relaxed);
            llama_token tok = llama_sampler_sample(sampler, ctx_tgt, i);
            t_anchor = std::chrono::steady_clock::now();
            sp.us_sample.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t_anchor - t0).count(),
                std::memory_order_relaxed);
            // llama_sampler_sample() already accepts the token internally.

            if (llama_vocab_is_eog(vocab, tok)) {
                eog = true;
                break;
            }

            if (!send_token(tok, false)) {
                send_failed = true;
                break;
            }
            sp.n_tokens_emitted.fetch_add(1, std::memory_order_relaxed);
            n_emitted += 1;
            n_accepted_total += 1;
            prompt.push_back(tok);
            sampled = tok;

            if (i < (int) drafts.size() && tok == drafts[i]) {
                n_accepted_drafts += 1;
                continue;
            }
            break;  // mismatch (or sampled-from-final-position): stop here
        }


        sp.n_drafts_generated.fetch_add(drafts.size(), std::memory_order_relaxed);
        sp.n_drafts_accepted.fetch_add(n_accepted_drafts, std::memory_order_relaxed);

        // 5. Inform the spec state.
        common_speculative_accept(sp.spec, seq_id, static_cast<uint16_t>(n_accepted_drafts));

        const int n_unaccepted = n_verify - n_accepted_total;
        if (n_unaccepted > 0) {
            if (sp.needs_ckpt) {
                // Hybrid model: partial seq_rm isn't supported, so restore
                // both contexts from the pre-iteration recurrent-state
                // snapshot and re-decode just the accepted prefix.
                auto t_rs0 = std::chrono::steady_clock::now();
                if (!ckpt_tgt.empty()) {
                    llama_state_seq_set_data_ext(ctx_tgt, ckpt_tgt.data(),
                                                  ckpt_tgt.size(), seq_id, ckpt_flags);
                }
                soft_seq_rm(ctx_tgt, seq_id, n_past);

                if (!ckpt_dft.empty()) {
                    llama_state_seq_set_data_ext(ctx_dft, ckpt_dft.data(),
                                                  ckpt_dft.size(), seq_id, ckpt_flags);
                }
                soft_seq_rm(ctx_dft, seq_id, n_past);
                auto t_rs1 = std::chrono::steady_clock::now();
                sp.us_ckpt.fetch_add(
                    std::chrono::duration_cast<std::chrono::microseconds>(t_rs1 - t_rs0).count(),
                    std::memory_order_relaxed);
                // Same anchor slide as the save above: us_other is closed out
                // from t_anchor at the end of the iter, so without this the
                // restore would be billed twice.
                t_anchor += (t_rs1 - t_rs0);

                // Re-decode the accepted tokens on the target so the next
                // iteration's draft starts from a consistent state.
                //
                // The KV must be rebuilt with the token that was `sampled` at
                // the top of this iteration (batch element 0, at pos n_past),
                // followed by all but the LAST emitted token. The last emitted
                // token becomes the next iteration's `sampled` and is decoded
                // then — including it here would duplicate it in the context.
                // `sampled` sits at prompt[size - n_accepted_total - 1], since
                // the accept loop pushed n_accepted_total tokens after it.
                if (n_accepted_total > 0) {
                    llama_batch redo = llama_batch_init(n_accepted_total, 0, 1);
                    BatchFreeGuard redo_guard(redo);
                    for (int i = 0; i < n_accepted_total; i++) {
                        llama_token tok =
                            prompt[prompt.size() - n_accepted_total - 1 + i];
                        common_batch_add(redo, tok,
                                         n_past + static_cast<llama_pos>(i),
                                         { seq_id },
                                         /*logits=*/ i == n_accepted_total - 1);
                    }
                    int ret = decode_tracked(*sp.ctx_tgt, redo);
                    if (ret != 0) {
                        send_error("rollback re-decode failed: code=" + std::to_string(ret));
                        return fine::Ok();
                    }
                }
            } else {
                // Dense model: native partial seq_rm trims the unaccepted
                // tail of the verify batch in-place. Much cheaper.
                soft_seq_rm(ctx_tgt, seq_id, n_past + n_accepted_total);
                soft_seq_rm(ctx_dft, seq_id, n_past + n_accepted_total);
            }
        }

        n_past += n_accepted_total;

        maybe_send_stats();

        // Close out us_other for this iter — captures the post-sample-loop
        // work plus any implicit GPU-sync wait that bleeds into the next
        // iter from llama_decode's async submission on Metal.
        sp.us_other.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_anchor).count(),
            std::memory_order_relaxed);

        if (send_failed) {
            // Caller process is gone; stop quietly.
            return fine::Ok();
        }

        if (eog) {
            sp.us_total.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_session_start).count(),
                std::memory_order_relaxed);
            send_done("eog");
            return fine::Ok();
        }

        if (n_accepted_total == 0) {
            // Should never happen — the first sampled token (position 0) is
            // always taken from the target model itself, so verification
            // emits at least one token per iteration.
            send_error("speculative loop made no progress");
            return fine::Ok();
        }
    }

    sp.us_total.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_session_start).count(),
        std::memory_order_relaxed);
    send_done(nullptr);
    return fine::Ok();
}
FINE_NIF(generate_mtp_tokens, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Streaming generation ---

fine::Ok<> generate_tokens(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx_res,
    fine::ResourcePtr<LlamaSampler> sampler_res,
    std::vector<int64_t> prompt_token_ids,
    int64_t max_tokens,
    ErlNifPid caller_pid,
    fine::Term ref,
    fine::ResourcePtr<CancelFlag> cancel)
{
    auto* ctx = ctx_res->ctx;
    auto* sampler = sampler_res->sampler;
    const auto* vocab = ctx_res->model->vocab();

    // Cancellation: polled per generated token AND installed as the abort
    // callback so a long prefill decode stops mid-flight (ret == 2).
    AbortCallbackScope abort_scope(ctx, cancel.get());

    std::vector<llama_token> prompt_tokens(prompt_token_ids.begin(), prompt_token_ids.end());

    if (prompt_tokens.empty()) {
        // Send error
        MsgEnvGuard msg_env;
        ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy,
            enif_make_tuple2(msg_env,
                enif_make_atom(msg_env, "error"),
                make_binary_term(msg_env, "prompt cannot be empty", 22)));
        enif_send(env, &caller_pid, msg_env, msg);
        return fine::Ok();
    }

    // Process prompt in chunks
    int n_batch = llama_n_batch(ctx);
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
        int n = std::min(static_cast<int>(prompt_tokens.size() - i), n_batch);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, n);
        int ret = decode_tracked(*ctx_res, batch);
        if (ret == 2) {
            // Aborted by the cancel flag mid-prefill — stop quietly.
            return fine::Ok();
        }
        if (ret != 0) {
            MsgEnvGuard msg_env;
            ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy,
                enif_make_tuple2(msg_env,
                    enif_make_atom(msg_env, "error"),
                    make_binary_term(msg_env, "prompt decode failed", 20)));
            enif_send(env, &caller_pid, msg_env, msg);
            return fine::Ok();
        }
    }

    // Allocate reusable message env
    MsgEnvGuard msg_env;

    // Atoms are immediate, environment-independent terms — intern the hot ones
    // once instead of per token, and reuse the detokenize fallback buffer.
    const ERL_NIF_TERM atom_token = enif_make_atom(env, "token");
    const ERL_NIF_TERM atom_eog   = enif_make_atom(env, "eog");
    const ERL_NIF_TERM atom_done  = enif_make_atom(env, "done");
    const ERL_NIF_TERM atom_error = enif_make_atom(env, "error");
    std::vector<char> large_buf;

    // Generation loop
    for (int64_t i = 0; i < max_tokens; i++) {
        if (cancel->cancelled.load(std::memory_order_relaxed)) {
            // Caller abandoned the stream — stop quietly; the consumer is
            // not reading messages anymore.
            return fine::Ok();
        }

        // llama_sampler_sample() already accepts the selected token; calling
        // llama_sampler_accept() again would double-advance grammar state.
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            enif_clear_env(msg_env);
            ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy, atom_eog);
            enif_send(env, &caller_pid, msg_env, msg);
            return fine::Ok();
        }

        // Detokenize (fast path uses the stack buffer; large_buf only on overflow)
        char buf[1024];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        const char* piece_data = buf;
        int piece_len = n;

        if (n < 0) {
            large_buf.resize(-n);
            piece_len = llama_token_to_piece(vocab, new_token,
                large_buf.data(), large_buf.size(), 0, false);
            piece_data = large_buf.data();
            if (piece_len < 0) piece_len = 0;
        }

        // Send {:token, token_id, text}
        enif_clear_env(msg_env);
        ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM inner = enif_make_tuple3(msg_env,
            atom_token,
            enif_make_int64(msg_env, new_token),
            make_binary_term(msg_env, piece_data, piece_len > 0 ? piece_len : 0));
        ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy, inner);

        if (!enif_send(env, &caller_pid, msg_env, msg)) {
            // Caller is dead, stop generating
            return fine::Ok();
        }

        // Decode next token
        llama_batch batch = llama_batch_get_one(&new_token, 1);
        int ret = decode_tracked(*ctx_res, batch);
        if (ret == 2) {
            // Aborted by the cancel flag — stop quietly.
            return fine::Ok();
        }
        if (ret != 0) {
            enif_clear_env(msg_env);
            ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM err_msg = enif_make_tuple2(msg_env, ref_copy,
                enif_make_tuple2(msg_env,
                    atom_error,
                    make_binary_term(msg_env, "decode failed during generation", 30)));
            enif_send(env, &caller_pid, msg_env, err_msg);
            return fine::Ok();
        }
    }

    // Max tokens reached
    enif_clear_env(msg_env);
    ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
    ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy, atom_done);
    enif_send(env, &caller_pid, msg_env, msg);

    return fine::Ok();
}
FINE_NIF(generate_tokens, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- High-level generation ---

std::variant<fine::Ok<std::string>, fine::Error<std::string>>
generate(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaContext> ctx_res,
    fine::ResourcePtr<LlamaSampler> sampler_res,
    std::vector<int64_t> prompt_token_ids,
    int64_t max_tokens,
    fine::ResourcePtr<CancelFlag> cancel)
{
    auto* ctx = ctx_res->ctx;
    auto* sampler = sampler_res->sampler;
    const auto* vocab = ctx_res->model->vocab();

    // Cancellation: polled per token; the abort callback interrupts long
    // prefill decodes (ret == 2). A cancelled call returns the partial text.
    AbortCallbackScope abort_scope(ctx, cancel.get());

    // Convert prompt tokens
    std::vector<llama_token> prompt_tokens(prompt_token_ids.begin(), prompt_token_ids.end());

    if (prompt_tokens.empty()) {
        return fine::Error(std::string("prompt cannot be empty"));
    }

    // Process prompt in chunks of n_batch
    int n_batch = llama_n_batch(ctx);
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
        int n = std::min(static_cast<int>(prompt_tokens.size() - i), n_batch);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, n);
        int ret = decode_tracked(*ctx_res, batch);
        if (ret == 2) {
            return fine::Ok(std::string());
        }
        if (ret != 0) {
            return fine::Error(std::string("prompt decode failed with code: " + std::to_string(ret)));
        }
    }

    // Generation loop
    std::string result;
    for (int64_t i = 0; i < max_tokens; i++) {
        if (cancel->cancelled.load(std::memory_order_relaxed)) {
            break;
        }

        // llama_sampler_sample() applies the sampler chain, selects a token, and
        // already accepts it (advancing grammar state / penalties). Do NOT call
        // llama_sampler_accept() again — a double-accept corrupts grammar state.
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        // Check for end-of-generation
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Detokenize the new token
        char buf[1024];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (n < 0) {
            std::vector<char> large_buf(-n);
            n = llama_token_to_piece(vocab, new_token, large_buf.data(), large_buf.size(), 0, false);
            if (n > 0) result.append(large_buf.data(), n);
        } else if (n > 0) {
            result.append(buf, n);
        }

        // Decode the new token for next iteration
        llama_batch batch = llama_batch_get_one(&new_token, 1);
        int ret = decode_tracked(*ctx_res, batch);
        if (ret == 2) {
            break;
        }
        if (ret != 0) {
            return fine::Error(std::string("generation decode failed with code: " + std::to_string(ret)));
        }
    }

    return fine::Ok(result);
}
FINE_NIF(generate, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- JSON Schema to Grammar ---

// The schema text is untrusted. `nlohmann::json::parse` and
// `json_schema_to_grammar` are both recursive descent, so nesting depth in the
// text is C-stack depth: a deeply nested schema is a stack overflow (SIGSEGV),
// which try/catch cannot recover. Bound size and depth before parsing.
std::variant<fine::Ok<std::string>, fine::Error<std::string>>
json_schema_to_grammar_nif(ErlNifEnv* env, std::string json_str) {
    if (json_str.size() > kMaxSchemaBytes) {
        return fine::Error(std::string("schema too large"));
    }
    if (json_nesting_depth(json_str) > kMaxSchemaDepth) {
        return fine::Error(std::string("schema nested too deeply"));
    }

    try {
        auto schema = nlohmann::ordered_json::parse(json_str);
        std::string grammar = json_schema_to_grammar(schema);
        return fine::Ok(grammar);
    } catch (const std::exception& e) {
        return fine::Error(std::string(e.what()));
    }
}
// Dirty: JSON parsing + grammar construction scale with schema size and can
// run for milliseconds on real-world schemas.
FINE_NIF(json_schema_to_grammar_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Init ---

FINE_INIT("Elixir.LlamaCppEx.NIF");
