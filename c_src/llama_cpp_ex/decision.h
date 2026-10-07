#pragma once

// Typed decision models: llama.cpp's /v1/systemone API, without the server.
//
// A decision model answers typed questions (choice, score, noul) about a state
// in one forward pass per prompt; no token is generated. Upstream implements it
// in tools/server/server-decision.cpp and the decision paths of
// server-context.cpp, neither of which this build compiles (LLAMA_BUILD_SERVER
// is OFF). decision.cpp ports both: the request parsing, prompt rendering and
// answer formatting verbatim, and the slot scheduling as a sequential loop over
// one context.
//
// Not ported: image input. It needs libmtmd, which this build does not link; a
// request with images is rejected.
//
// Kept free of Erlang/fine so it stays a plain port that can be diffed against
// upstream on every llama.cpp bump (see docs/release-guide.md).

#include <llama.h>

#include <memory>
#include <string>

namespace llama_cpp_ex::decision {

// The model's "<arch>.decision.type" metadata: "" when it is not a decision
// model, otherwise the type as written in the file, supported or not.
std::string model_type(const llama_model * model);

// True for the types whose scores come from the embeddings output (laya, kev,
// clef). Their context needs embeddings on, pooling NONE, and n_batch equal to
// n_ubatch, which is what upstream's common_init_result forces for them.
bool type_reads_embeddings(const std::string & type);

// True for the types whose prompts share a prefix across the questions of one
// request (openjev, lev, kev, nimble, pplx-decider). The prefix is evaluated
// once when the context has n_seq_max >= 2.
bool type_shares_prompt(const std::string & type);

class Engine {
public:
    // Reads the decision metadata and the "systemone" template of the
    // context's model. Throws std::runtime_error if the model is not a
    // supported decision model or the context is not set up for it.
    explicit Engine(llama_context * ctx);
    ~Engine();

    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    const std::string & type() const;

    // body: a /v1/systemone request ({"state": ..., "questions": {...}}).
    // Returns the response JSON text: {"answers": {...}, "usage": {...}}.
    // Throws std::invalid_argument for a bad request and std::runtime_error
    // for a failure to evaluate it. Leaves the context's memory in an
    // unspecified state.
    std::string decide(const std::string & body);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace llama_cpp_ex::decision
