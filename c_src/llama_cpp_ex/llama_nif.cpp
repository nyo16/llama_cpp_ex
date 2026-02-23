#include "llama_nif.h"
#include <fine.hpp>
#include <llama.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace llama_cpp_ex;

// --- Resource registrations ---

FINE_RESOURCE(LlamaModel);
FINE_RESOURCE(LlamaContext);
FINE_RESOURCE(LlamaSampler);

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

// --- Model ---

std::variant<fine::Ok<fine::ResourcePtr<LlamaModel>>, fine::Error<std::string>>
model_load(ErlNifEnv* env, std::string path, int64_t n_gpu_layers, bool use_mmap) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = static_cast<int32_t>(n_gpu_layers);
    params.use_mmap = use_mmap;

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
    int64_t n_threads_batch)
{
    auto params = llama_context_default_params();
    params.n_ctx         = static_cast<uint32_t>(n_ctx);
    params.n_batch       = static_cast<uint32_t>(n_batch);
    params.n_ubatch      = static_cast<uint32_t>(n_ubatch);
    params.n_threads     = static_cast<int32_t>(n_threads);
    params.n_threads_batch = static_cast<int32_t>(n_threads_batch);

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

// --- Sampler ---

fine::ResourcePtr<LlamaSampler>
sampler_init(
    ErlNifEnv* env,
    int64_t seed,
    double temp,
    int64_t top_k,
    double top_p,
    double min_p,
    double penalty_repeat)
{
    auto chain_params = llama_sampler_chain_default_params();
    auto* chain = llama_sampler_chain_init(chain_params);

    // Add samplers in recommended order: penalties -> top_k -> top_p -> min_p -> temp -> dist/greedy
    if (penalty_repeat != 1.0) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_penalties(64, static_cast<float>(penalty_repeat), 0.0f, 0.0f));
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

int64_t sampler_sample(ErlNifEnv* env, fine::ResourcePtr<LlamaSampler> sampler,
                       fine::ResourcePtr<LlamaContext> ctx) {
    return llama_sampler_sample(sampler->sampler, ctx->ctx, -1);
}
FINE_NIF(sampler_sample, 0);

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

    // Generation loop
    for (int64_t i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

        if (llama_vocab_is_eog(vocab, new_token)) {
            enif_clear_env(msg_env);
            ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
            ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy,
                enif_make_atom(msg_env, "eog"));
            enif_send(env, &caller_pid, msg_env, msg);
            enif_free_env(msg_env);
            return fine::Ok();
        }

        // Detokenize
        char buf[1024];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        const char* piece_data = buf;
        int piece_len = n;
        std::vector<char> large_buf;

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
            enif_make_atom(msg_env, "token"),
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
                    enif_make_atom(msg_env, "error"),
                    make_binary_term(msg_env, "decode failed during generation", 30)));
            enif_send(env, &caller_pid, msg_env, err_msg);
            enif_free_env(msg_env);
            return fine::Ok();
        }
    }

    // Max tokens reached
    enif_clear_env(msg_env);
    ERL_NIF_TERM ref_copy = enif_make_copy(msg_env, ref);
    ERL_NIF_TERM msg = enif_make_tuple2(msg_env, ref_copy,
        enif_make_atom(msg_env, "done"));
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
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

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

// --- Init ---

FINE_INIT("Elixir.LlamaCppEx.NIF");
