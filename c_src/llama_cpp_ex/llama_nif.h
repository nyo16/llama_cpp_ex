#pragma once

#include <fine.hpp>
#include <llama.h>
#include "chat.h"
#include "speculative.h"
#include <atomic>
#include <vector>

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
//
// INVARIANT: a context is driven by a single process (the Server GenServer or
// one owning caller). The reusable `batch` below relies on that — decode-side
// NIFs must never run concurrently on the same context from multiple
// processes. (A single process can't overlap NIF calls, so no locking.)
class LlamaContext {
public:
    llama_context* ctx;
    fine::ResourcePtr<LlamaModel> model;

    // Reusable explicit batch for the decode-side NIFs (batch_eval,
    // batch_eval_sample, decode_token, prefill), allocated once on first use
    // and grown on demand instead of llama_batch_init/free per call.
    llama_batch batch{};
    int32_t batch_capacity = 0;

    // Whether this context was created with kv_unified: true. Captured because
    // llama_kv_cache::seq_cp aborts (GGML_ASSERT(is_full),
    // llama-kv-cache.cpp:502) on a *partial* cross-sequence copy when the two
    // sequences live in different streams, which is the split (non-unified)
    // case. There is no public getter, and the only upstream probe
    // (common_context_can_seq_rm) clears KV memory and decodes two tokens as a
    // side effect, so it must never be called on a live context.
    bool kv_unified = false;

    // Shape of the last *successful* llama_decode on this context.
    //
    // `sampler_sample_at/3` takes its index straight from Elixir, and
    // llama_sampler_sample -> llama_get_logits_ith aborts on anything upstream's
    // output_resolve_row (llama-context.cpp:841-866) rejects. That function
    // throws in exactly three cases: a negative index past `n_outputs`, a
    // non-negative index past the batch's token count, and an index whose batch
    // token did not request logits. The abort is un-gated GGML_ASSERT ->
    // abort(), i.e. the whole VM, so the boundary has to reproduce all three
    // checks — which needs the batch shape, not just a row count.
    //
    // `logits_last` is per *batch token* (upstream's `output_ids` domain), not
    // per output row. Both reset whenever KV memory is cleared, since the logits
    // go with it.
    int32_t n_outputs_last = 0;
    std::vector<char> logits_last;

    void record_batch(const llama_batch& b) {
        logits_last.assign(static_cast<size_t>(b.n_tokens), 0);
        if (!b.logits) {
            // A null logits array means llama.cpp outputs the last token only
            // (llama-batch.cpp:120-129).
            n_outputs_last = b.n_tokens > 0 ? 1 : 0;
            if (b.n_tokens > 0) logits_last.back() = 1;
            return;
        }
        n_outputs_last = 0;
        for (int32_t i = 0; i < b.n_tokens; i++) {
            if (b.logits[i]) {
                logits_last[static_cast<size_t>(i)] = 1;
                n_outputs_last++;
            }
        }
    }

    void forget_batch() {
        n_outputs_last = 0;
        logits_last.clear();
    }

    // Mirrors output_resolve_row's accept set. -1 means "last output row".
    bool valid_logits_idx(int64_t idx) const {
        if (idx < 0) {
            return idx == -1 && n_outputs_last > 0;
        }
        return idx < static_cast<int64_t>(logits_last.size()) &&
               logits_last[static_cast<size_t>(idx)] != 0;
    }

    LlamaContext(llama_context* c, fine::ResourcePtr<LlamaModel> m)
        : ctx(c), model(std::move(m)) {}

    // Returns the reusable batch with capacity for at least n tokens
    // (per-token seq-id capacity 1 — all decode builders use single-seq
    // entries). Contents are stale; the caller fills 0..n-1 and n_tokens.
    llama_batch& reserve_batch(int32_t n) {
        if (batch_capacity < n) {
            if (batch_capacity > 0) llama_batch_free(batch);
            batch = llama_batch_init(n, 0, 1);
            batch_capacity = n;
        }
        return batch;
    }

    ~LlamaContext() {
        if (batch_capacity > 0) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
    }

    LlamaContext(const LlamaContext&) = delete;
    LlamaContext& operator=(const LlamaContext&) = delete;
};

// Cooperative cancellation flag for the stateless generation loops. The
// owning Elixir process holds the resource and sets it via request_cancel/1;
// the generating NIF polls it per iteration and also installs it as the
// context's abort callback so a long prefill aborts mid-decode.
class CancelFlag {
public:
    std::atomic<bool> cancelled{false};
};

// RAII wrapper for llama_sampler*
// Holds a ResourcePtr to the model for the same reason LlamaContext and
// LlamaSpeculative do: llama_sampler_init_grammar captures the raw
// `const llama_vocab*` returned by model->vocab() and keeps dereferencing it on
// every accept/reset/sample. Without this link, dropping the model term in
// Elixir frees the vocab under a live sampler and the next Sampler.reset/1 or
// accept/2 reads freed heap. `model` is null for samplers built without a
// vocab-dependent stage, which is why it is a plain ResourcePtr and not a
// constructor requirement.
class LlamaSampler {
public:
    llama_sampler* sampler;
    fine::ResourcePtr<LlamaModel> model;

    explicit LlamaSampler(llama_sampler* s) : sampler(s) {}
    LlamaSampler(llama_sampler* s, fine::ResourcePtr<LlamaModel> m)
        : sampler(s), model(std::move(m)) {}
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
    // Recurrent-state save/restore, which only hybrid models pay (needs_ckpt).
    // Broken out of us_other because on a model like Qwen 3.8 — 48 SSM layers
    // beside 16 attention ones — the snapshot is over a hundred MiB and taken
    // every iteration, which is enough on its own to make speculation a net
    // loss. Attributing it to "other" hid that behind a bucket whose documented
    // cause is GPU-sync waits.
    std::atomic<uint64_t> us_ckpt{0};
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
