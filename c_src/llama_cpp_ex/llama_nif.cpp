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

using namespace llama_cpp_ex;

// --- Resource registrations ---

FINE_RESOURCE(LlamaModel);
FINE_RESOURCE(LlamaContext);
FINE_RESOURCE(LlamaSampler);
FINE_RESOURCE(LlamaSpeculative);

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
           bool use_mlock, bool use_direct_io, bool vocab_only, bool check_tensors) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = static_cast<int32_t>(n_gpu_layers);
    params.use_mmap = use_mmap;
    params.main_gpu = static_cast<int32_t>(main_gpu);
    params.split_mode = static_cast<enum llama_split_mode>(split_mode);
    params.use_mlock = use_mlock;
    params.use_direct_io = use_direct_io;
    params.vocab_only = vocab_only;
    params.check_tensors = check_tensors;

    std::vector<float> ts_float;
    if (!tensor_split.empty()) {
        ts_float.reserve(tensor_split.size());
        for (auto v : tensor_split) ts_float.push_back(static_cast<float>(v));
        params.tensor_split = ts_float.data();
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
FINE_NIF(tokenize, 0);

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
FINE_NIF(detokenize, 0);

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

    return fine::Ok(fine::make_resource<LlamaContext>(ctx, model));
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

fine::ResourcePtr<LlamaSampler>
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
    auto chain_params = llama_sampler_chain_default_params();
    auto* chain = llama_sampler_chain_init(chain_params);

    // Grammar sampler goes first (before penalties/temperature)
    if (!grammar_str.empty()) {
        const auto* vocab = model->vocab();
        auto* grammar = llama_sampler_init_grammar(
            vocab, grammar_str.c_str(), grammar_root.c_str());
        if (grammar) {
            llama_sampler_chain_add(chain, grammar);
        }
    }

    // Add samplers in recommended order: penalties -> top_k -> top_p -> min_p -> temp -> dist/greedy
    if (penalty_repeat != 1.0 || penalty_freq != 0.0 || penalty_present != 0.0) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_penalties(64, static_cast<float>(penalty_repeat),
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

    return fine::make_resource<LlamaSampler>(chain);
}
FINE_NIF(sampler_init, 0);

fine::Ok<> sampler_accept(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler, int64_t token) {
    llama_sampler_accept(sampler->sampler, static_cast<llama_token>(token));
    return fine::Ok();
}
FINE_NIF(sampler_accept, 0);

fine::Ok<> sampler_reset(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler) {
    llama_sampler_reset(sampler->sampler);
    return fine::Ok();
}
FINE_NIF(sampler_reset, 0);

// Dirty: the sampler chain runs a softmax over the full vocab (100k+ entries)
// and grammar samplers can take multiple ms — too slow for a normal scheduler.
int64_t sampler_sample(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler,
                       fine::ResourcePtr<LlamaContext> ctx) {
    return llama_sampler_sample(sampler->sampler, ctx->ctx, -1);
}
FINE_NIF(sampler_sample, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- Decode ---

std::variant<fine::Ok<>, fine::Error<std::string>>
decode(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx, std::vector<int64_t> token_ids) {
    std::vector<llama_token> tokens(token_ids.begin(), token_ids.end());

    // Process in chunks of n_batch
    int n_batch = llama_n_batch(ctx->ctx);
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        int n = std::min(static_cast<int>(tokens.size() - i), n_batch);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n);
        int ret = llama_decode(ctx->ctx, batch);
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
    return fine::Ok();
}
FINE_NIF(memory_clear, 0);

bool memory_seq_rm(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
                   int64_t seq_id, int64_t p0, int64_t p1) {
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

fine::Ok<> memory_seq_cp(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx,
                         int64_t seq_id_src, int64_t seq_id_dst,
                         int64_t p0, int64_t p1) {
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
    llama_memory_seq_keep(
        llama_get_memory(ctx->ctx),
        static_cast<llama_seq_id>(seq_id));
    return fine::Ok();
}
FINE_NIF(memory_seq_keep, 0);

// --- Memory seq_pos_max ---

int64_t memory_seq_pos_max(ErlNifEnv* env, fine::ResourcePtr<LlamaContext> ctx, int64_t seq_id) {
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

    // Clear memory for a fresh decode
    llama_memory_clear(llama_get_memory(ctx->ctx), true);

    // Build batch with explicit seq_id and position tracking
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]      = static_cast<llama_token>(token_ids[i]);
        batch.pos[i]        = static_cast<llama_pos>(i);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = static_cast<llama_seq_id>(seq_id);
        batch.logits[i]     = true; // all tokens get embeddings
    }

    int ret = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        return fine::Error(std::string("embed_decode failed with code: " + std::to_string(ret)));
    }

    return fine::Ok();
}
FINE_NIF(embed_decode, ERL_NIF_DIRTY_JOB_CPU_BOUND);

std::variant<fine::Ok<std::vector<double>>, fine::Error<std::string>>
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

    std::vector<double> out(n_embd);

    if (normalize == 2) {
        // L2 normalization
        double sum = 0.0;
        for (int i = 0; i < n_embd; i++) sum += (double)embd[i] * (double)embd[i];
        double norm = sum > 0.0 ? 1.0 / std::sqrt(sum) : 0.0;
        for (int i = 0; i < n_embd; i++) out[i] = (double)embd[i] * norm;
    } else if (normalize == 0) {
        // Max-abs normalization
        double max_abs = 0.0;
        for (int i = 0; i < n_embd; i++) {
            double a = std::abs((double)embd[i]);
            if (a > max_abs) max_abs = a;
        }
        double norm = max_abs > 0.0 ? 1.0 / max_abs : 0.0;
        for (int i = 0; i < n_embd; i++) out[i] = (double)embd[i] * norm;
    } else {
        // No normalization
        for (int i = 0; i < n_embd; i++) out[i] = (double)embd[i];
    }

    return fine::Ok(out);
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

    // Fresh decode for this batch
    llama_memory_clear(llama_get_memory(ctx->ctx), true);

    llama_batch batch = llama_batch_init(total, 0, 1);
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

    int ret = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

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

        int ret = llama_decode(ctx->ctx, batch);

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
    batch.n_tokens = n;

    for (int i = 0; i < n; i++) {
        auto& [seq_id, token_id, pos] = entries[i];
        batch.token[i]      = static_cast<llama_token>(token_id);
        batch.pos[i]        = static_cast<llama_pos>(pos);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = static_cast<llama_seq_id>(seq_id);
        batch.logits[i]     = true; // need logits for all entries to sample
    }

    int ret = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

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

    int ret = llama_decode(ctx->ctx, batch);

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

    int ret = llama_decode(ctx->ctx, batch);

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
    llama_context* ctx,
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

    int ret = llama_decode(ctx, batch);

    // Purge donatable idle caches one at a time while the KV cache is full.
    while (ret == 1 && purge_idx < purgeable.size()) {
        auto victim = static_cast<llama_seq_id>(purgeable[purge_idx++]);
        llama_memory_seq_rm(llama_get_memory(ctx), victim, -1, -1);
        purged.push_back(victim);
        ret = llama_decode(ctx, batch);
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
                llama_sampler_sample(smpl, ctx, static_cast<int32_t>(i));
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
        int rc = bes_decode_range(ctx, vocab, entries, begin, mid, samplers,
                                  purgeable, purge_idx, purged, n_splits,
                                  failed, results, batch);
        if (rc != 0) return rc;
        return bes_decode_range(ctx, vocab, entries, mid, end, samplers,
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

    llama_batch& batch = ctx->reserve_batch(static_cast<int32_t>(entries.size()));

    int rc = bes_decode_range(ctx->ctx, vocab, entries, 0, entries.size(),
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
int64_t sampler_sample_at(
    ErlNifEnv* env,
    fine::ResourcePtr<LlamaSampler> sampler,
    fine::ResourcePtr<LlamaContext> ctx,
    int64_t idx)
{
    return llama_sampler_sample(sampler->sampler, ctx->ctx, static_cast<int32_t>(idx));
}
FINE_NIF(sampler_sample_at, ERL_NIF_DIRTY_JOB_CPU_BOUND);

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
    uint64_t uother  = s.us_other.load(std::memory_order_relaxed);
    uint64_t utotal  = s.us_total.load(std::memory_order_relaxed);

    double acceptance_rate = dgen > 0 ? (double)dacc / (double)dgen : 0.0;
    double tokens_per_sec  = utotal > 0 ? (double)emitted * 1e6 / (double)utotal : 0.0;

    ERL_NIF_TERM tk[5] = {
        enif_make_atom(env, "draft"),
        enif_make_atom(env, "verify"),
        enif_make_atom(env, "sample"),
        enif_make_atom(env, "other"),
        enif_make_atom(env, "total"),
    };
    ERL_NIF_TERM tv[5] = {
        enif_make_uint64(env, udraft),
        enif_make_uint64(env, uverify),
        enif_make_uint64(env, usample),
        enif_make_uint64(env, uother),
        enif_make_uint64(env, utotal),
    };
    ERL_NIF_TERM timing;
    enif_make_map_from_arrays(env, tk, tv, 5, &timing);

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
    fine::Term ref)
{
    auto& sp = *spec_res;
    auto* ctx_tgt = sp.ctx_tgt->ctx;
    auto* ctx_dft = sp.ctx_dft->ctx;
    auto* sampler = sampler_res->sampler;
    const auto* vocab = sp.ctx_tgt->model->vocab();
    const llama_seq_id seq_id = 0;
    const int32_t n_draft = static_cast<int32_t>(sp.n_draft);

    ErlNifEnv* msg_env = enif_alloc_env();

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
        enif_free_env(msg_env);
        return fine::Ok();
    }

    std::vector<llama_token> prompt(prompt_token_ids.begin(), prompt_token_ids.end());

    // Wipe any prior KV on seq 0 in both contexts so the spec begins fresh.
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    llama_memory_clear(llama_get_memory(ctx_dft), true);

    // For MTP, hidden states are extracted via set_embeddings_pre_norm on
    // ctx_tgt (set up in the MTP impl's constructor). We only need to know
    // that drafts depend on per-position outputs, so logits=true must be
    // requested for every prefill token.
    const bool need_embd = common_speculative_need_embd(sp.spec);

    // Prefill the target context with the prompt. For MTP we request logits
    // at every position so the streaming hook in common_speculative_process
    // can mirror t_h_pre_norm into ctx_dft (see speculative.cpp). The full
    // batch is then fed back to the speculative state.
    int n_batch = llama_n_batch(ctx_tgt);
    llama_pos n_past = 0;
    for (size_t i = 0; i < prompt.size(); i += n_batch) {
        int n = std::min(static_cast<int>(prompt.size() - i), n_batch);
        bool is_last_chunk = (i + n >= prompt.size());

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int j = 0; j < n; j++) {
            const bool want_logits = need_embd
                ? true
                : (is_last_chunk && j == n - 1);
            common_batch_add(batch, prompt[i + j], static_cast<llama_pos>(i + j),
                             { seq_id }, want_logits);
        }

        int ret = llama_decode(ctx_tgt, batch);
        if (ret != 0) {
            llama_batch_free(batch);
            send_error("prompt decode failed: code=" + std::to_string(ret));
            enif_free_env(msg_env);
            return fine::Ok();
        }
        bool proc_ok = common_speculative_process(sp.spec, batch);
        if (!proc_ok) {
            fprintf(stderr,
                "MTP prefill: common_speculative_process returned false "
                "at chunk i=%zu n=%d need_embd=%d logits_on_each=%d\n",
                i, n, (int) need_embd, (int) need_embd);
        }
        llama_batch_free(batch);
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
            enif_free_env(msg_env);
            return fine::Ok();
        }

        if (!send_token(tok, false)) {
            enif_free_env(msg_env);
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

            int ret = llama_decode(ctx_tgt, batch);
            if (ret != 0) {
                llama_batch_free(batch);
                send_error("verify decode failed: code=" + std::to_string(ret));
                enif_free_env(msg_env);
                return fine::Ok();
            }
            if (!common_speculative_process(sp.spec, batch)) {
                llama_batch_free(batch);
                send_error("common_speculative_process failed");
                enif_free_env(msg_env);
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

        for (int i = 0; i < n_verify; i++) {
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

        llama_batch_free(batch);

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

                // Re-decode the accepted tokens on the target so the next
                // iteration's draft starts from a consistent state.
                if (n_accepted_total > 0) {
                    llama_batch redo = llama_batch_init(n_accepted_total, 0, 1);
                    for (int i = 0; i < n_accepted_total; i++) {
                        llama_token tok =
                            prompt[prompt.size() - n_accepted_total + i];
                        common_batch_add(redo, tok,
                                         n_past + static_cast<llama_pos>(i),
                                         { seq_id },
                                         /*logits=*/ i == n_accepted_total - 1);
                    }
                    int ret = llama_decode(ctx_tgt, redo);
                    llama_batch_free(redo);
                    if (ret != 0) {
                        send_error("rollback re-decode failed: code=" + std::to_string(ret));
                        enif_free_env(msg_env);
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
            enif_free_env(msg_env);
            return fine::Ok();
        }

        if (eog) {
            sp.us_total.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_session_start).count(),
                std::memory_order_relaxed);
            send_done("eog");
            enif_free_env(msg_env);
            return fine::Ok();
        }

        if (n_accepted_total == 0) {
            // Should never happen — the first sampled token (position 0) is
            // always taken from the target model itself, so verification
            // emits at least one token per iteration.
            send_error("speculative loop made no progress");
            enif_free_env(msg_env);
            return fine::Ok();
        }
    }

    sp.us_total.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_session_start).count(),
        std::memory_order_relaxed);
    send_done(nullptr);
    enif_free_env(msg_env);
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
    fine::Term ref)
{
    auto* ctx = ctx_res->ctx;
    auto* sampler = sampler_res->sampler;
    const auto* vocab = ctx_res->model->vocab();

    std::vector<llama_token> prompt_tokens(prompt_token_ids.begin(), prompt_token_ids.end());

    if (prompt_tokens.empty()) {
        // Send error
        ErlNifEnv* msg_env = enif_alloc_env();
        ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
        ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy,
            enif_make_tuple2(msg_env,
                enif_make_atom(msg_env, "error"),
                make_binary_term(msg_env, "prompt cannot be empty", 22)));
        enif_send(env, &caller_pid, msg_env, msg);
        enif_free_env(msg_env);
        return fine::Ok();
    }

    // Process prompt in chunks
    int n_batch = llama_n_batch(ctx);
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
        int n = std::min(static_cast<int>(prompt_tokens.size() - i), n_batch);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, n);
        if (llama_decode(ctx, batch) != 0) {
            ErlNifEnv* msg_env = enif_alloc_env();
            ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy,
                enif_make_tuple2(msg_env,
                    enif_make_atom(msg_env, "error"),
                    make_binary_term(msg_env, "prompt decode failed", 20)));
            enif_send(env, &caller_pid, msg_env, msg);
            enif_free_env(msg_env);
            return fine::Ok();
        }
    }

    // Allocate reusable message env
    ErlNifEnv* msg_env = enif_alloc_env();

    // Atoms are immediate, environment-independent terms — intern the hot ones
    // once instead of per token, and reuse the detokenize fallback buffer.
    const ERL_NIF_TERM atom_token = enif_make_atom(env, "token");
    const ERL_NIF_TERM atom_eog   = enif_make_atom(env, "eog");
    const ERL_NIF_TERM atom_done  = enif_make_atom(env, "done");
    const ERL_NIF_TERM atom_error = enif_make_atom(env, "error");
    std::vector<char> large_buf;

    // Generation loop
    for (int64_t i = 0; i < max_tokens; i++) {
        // llama_sampler_sample() already accepts the selected token; calling
        // llama_sampler_accept() again would double-advance grammar state.
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            enif_clear_env(msg_env);
            ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy, atom_eog);
            enif_send(env, &caller_pid, msg_env, msg);
            enif_free_env(msg_env);
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
            enif_free_env(msg_env);
            return fine::Ok();
        }

        // Decode next token
        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            enif_clear_env(msg_env);
            ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM err_msg = enif_make_tuple2(msg_env, ref_copy,
                enif_make_tuple2(msg_env,
                    atom_error,
                    make_binary_term(msg_env, "decode failed during generation", 30)));
            enif_send(env, &caller_pid, msg_env, err_msg);
            enif_free_env(msg_env);
            return fine::Ok();
        }
    }

    // Max tokens reached
    enif_clear_env(msg_env);
    ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
    ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy, atom_done);
    enif_send(env, &caller_pid, msg_env, msg);
    enif_free_env(msg_env);

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
    int64_t max_tokens)
{
    auto* ctx = ctx_res->ctx;
    auto* sampler = sampler_res->sampler;
    const auto* vocab = ctx_res->model->vocab();

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
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            return fine::Error(std::string("prompt decode failed with code: " + std::to_string(ret)));
        }
    }

    // Generation loop
    std::string result;
    for (int64_t i = 0; i < max_tokens; i++) {
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
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            return fine::Error(std::string("generation decode failed with code: " + std::to_string(ret)));
        }
    }

    return fine::Ok(result);
}
FINE_NIF(generate, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// --- JSON Schema to Grammar ---

std::variant<fine::Ok<std::string>, fine::Error<std::string>>
json_schema_to_grammar_nif(ErlNifEnv* env, std::string json_str) {
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
