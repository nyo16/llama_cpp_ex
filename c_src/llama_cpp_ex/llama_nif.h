#pragma once

#include <fine.hpp>
#include <llama.h>
#include "chat.h"

namespace llama_cpp_ex {

// RAII wrapper for llama_model*
class LlamaModel {
public:
    llama_model* model;
    common_chat_templates_ptr chat_templates;

    explicit LlamaModel(llama_model* m)
        : model(m)
        , chat_templates(common_chat_templates_init(m, ""))
    {}
    ~LlamaModel() {
        // Release chat_templates before freeing the model
        chat_templates.reset();
        if (model) llama_model_free(model);
    }

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    const llama_vocab* vocab() const {
        return llama_model_get_vocab(model);
    }
};

// RAII wrapper for llama_context*
// Holds a ResourcePtr to the model to prevent premature GC.
class LlamaContext {
public:
    llama_context* ctx;
    fine::ResourcePtr<LlamaModel> model;

    LlamaContext(llama_context* c, fine::ResourcePtr<LlamaModel> m)
        : ctx(c), model(std::move(m)) {}
    ~LlamaContext() {
        if (ctx) llama_free(ctx);
    }

    LlamaContext(const LlamaContext&) = delete;
    LlamaContext& operator=(const LlamaContext&) = delete;
};

// RAII wrapper for llama_sampler*
class LlamaSampler {
public:
    llama_sampler* sampler;

    explicit LlamaSampler(llama_sampler* s) : sampler(s) {}
    ~LlamaSampler() {
        if (sampler) llama_sampler_free(sampler);
    }

    LlamaSampler(const LlamaSampler&) = delete;
    LlamaSampler& operator=(const LlamaSampler&) = delete;
};

} // namespace llama_cpp_ex
