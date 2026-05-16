#pragma once

#include <fine.hpp>
#include <llama.h>
#include "chat.h"
#include "speculative.h"
#include <atomic>

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

// RAII wrapper for common_speculative* (MTP draft state).
// Holds ResourcePtrs to both the target (main) and draft (MTP) contexts so
// they stay alive while a speculative session is in flight. Counters are
// updated by the streaming generate_mtp_tokens NIF and read lock-free by
// speculative_stats; relaxed ordering is sufficient because readers tolerate
// a slightly stale snapshot and there is no cross-counter invariant to
// preserve.
class LlamaSpeculative {
public:
    common_speculative* spec;
    fine::ResourcePtr<LlamaContext> ctx_tgt;
    fine::ResourcePtr<LlamaContext> ctx_dft;
    uint32_t n_draft;

    // True when ctx_tgt requires checkpointing for partial draft rollback
    // (e.g. hybrid models like Qwen 3.6 MoE with GDN layers). Captured once
    // at speculative_init time. Dense attention-only models report
    // COMMON_CONTEXT_SEQ_RM_TYPE_PART and skip the checkpoint path entirely.
    bool needs_ckpt;

    std::atomic<uint64_t> n_iters{0};
    std::atomic<uint64_t> n_drafts_generated{0};
    std::atomic<uint64_t> n_drafts_accepted{0};
    std::atomic<uint64_t> n_tokens_emitted{0};
    std::atomic<uint64_t> us_draft{0};
    std::atomic<uint64_t> us_verify{0};
    std::atomic<uint64_t> us_sample{0};
    // Everything in the speculative iter NOT inside the three hot-path
    // timers above. On Metal this is dominated by implicit GPU-sync waits
    // from the previous iter's async verify decode (llama_decode returns
    // before the command buffer completes; the wait lands on the next
    // unrelated allocation in the next iter). Sum of draft+verify+sample
    // +other ≈ us_total.
    std::atomic<uint64_t> us_other{0};
    std::atomic<uint64_t> us_total{0};

    LlamaSpeculative(common_speculative* s,
                     fine::ResourcePtr<LlamaContext> tgt,
                     fine::ResourcePtr<LlamaContext> dft,
                     uint32_t n,
                     bool ckpt)
        : spec(s)
        , ctx_tgt(std::move(tgt))
        , ctx_dft(std::move(dft))
        , n_draft(n)
        , needs_ckpt(ckpt)
    {}
    ~LlamaSpeculative() {
        if (spec) common_speculative_free(spec);
    }

    LlamaSpeculative(const LlamaSpeculative&) = delete;
    LlamaSpeculative& operator=(const LlamaSpeculative&) = delete;
};

} // namespace llama_cpp_ex
