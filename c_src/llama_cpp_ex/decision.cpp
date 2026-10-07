// Port of llama.cpp's tools/server/server-decision.cpp plus the decision paths
// of tools/server/server-context.cpp (send_decision, the batch layout rules and
// the /v1/systemone handler). See decision.h for scope.
//
// Ported at 88dcc460d. Functions that are upstream's carry its name in a
// trailing comment so a bump can diff them one by one; everything below the
// "evaluation" banner replaces server slots and is ours.

#include "decision.h"

#include "chat.h"
#include "common.h"
#include "json.h"

// Staging API, not part of include/llama.h. Only the enum is used: the order of
// each batch entry is applied by common_batch::get_sub_batch.
#include "../../vendor/llama.cpp/src/llama-ext.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>

using json = common_json;

namespace llama_cpp_ex::decision {

namespace {

// The values double as laya's output column (upstream: task.decision.column =
// question.type), so they must stay 0, 1, 2 in this order.
enum question_type {
    QUESTION_CHOICE,
    QUESTION_SCORE,
    QUESTION_NOUL,
};

struct decision_option {
    std::string key;
    json description; // null if not provided
};

struct decision_question {
    std::string id;
    question_type type;
    json instructions;
    std::vector<decision_option> options; // in the order of the model outputs
};

// One prompt and where to read its outputs. Mirrors server_task::decision plus
// the task's tokens.
struct decision_task {
    llama_tokens tokens;

    std::vector<llama_token> labels;       // logits of these tokens, at the last prompt token
    std::vector<int32_t>     label_groups; // if set, number of labels per output, the output is their max
    std::vector<int32_t>     markers;      // embeddings[column] at these prompt positions
    int32_t                  column  = 0;
    // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
    int32_t                  pointer = -1;

    // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
    // the scores are the first n_scores rows of the embeddings
    std::vector<int32_t> order;
    int32_t              n_scores = 0;

    // first prompt position that is read, -1 if none
    int32_t pos_first() const {
        int32_t pos = pointer;
        for (const int32_t marker : markers) {
            pos = pos < 0 ? marker : std::min(pos, marker);
        }
        return pos;
    }
};

const char * question_type_name(question_type type) { // decision_question_type_name
    switch (type) {
        case QUESTION_CHOICE: return "choice";
        case QUESTION_SCORE:  return "score";
        case QUESTION_NOUL:   return "noul";
    }
    return "";
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
const size_t DECISION_LEV_N_RATINGS = 9;

std::string meta_str(const llama_model * model, const std::string & key) { // decision_meta_str
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

// server-common's json_value(body, key, std::string()): the default when the
// key is missing, null, or not a string.
std::string json_string(const json & body, const std::string & key) {
    if (body.contains(key) && body.at(key).is_string()) {
        return body.at(key).get<std::string>();
    }
    return std::string();
}

// replace text in all strings of a JSON value
json replace_text(const json & val, const std::string & search, const std::string & replace) { // decision_replace_text
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(replace_text(item, search, replace));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = replace_text(item, search, replace);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
json sort_keys(const json & val) { // decision_sort_keys
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(sort_keys(item));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = sort_keys(item);
        }
        json out = json::object();
        for (const auto & [key, item] : sorted) {
            out[key] = item;
        }
        return out;
    }
    return val;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels (kev/api.py: render)
std::string kev_render(const json & val, int indent = 0) { // decision_kev_render
    const std::string pad(2 * indent, ' ');
    if (val.is_null()) {
        return "";
    }
    if (val.is_string()) {
        return val.get<std::string>();
    }
    if (val.is_boolean()) {
        return val.get<bool>() ? "True" : "False";
    }
    if (val.is_array()) {
        std::string out;
        for (const auto & item : val) {
            const std::string text = kev_render(item, indent + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + kev_render(item, is_nested ? indent + 1 : 0);
        }
        return out;
    }
    return val.dump();
}

// kev text input: special tokens written in the text must not be parsed as such
std::string kev_text(const json & val) { // decision_kev_text
    static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(kev_render(val), re_special, "<\xC2\xA6$1\xC2\xA6>");
}

// confidence formulas are the ones published by TypeSafe

double confidence_choice(const std::vector<double> & probs) { // decision_confidence_choice
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

double confidence_score(const std::vector<double> & probs) { // decision_confidence_score
    if (probs.size() < 2) {
        return 1.0;
    }
    const size_t n    = probs.size();
    const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

    // mean distance to the mode, relative to the one of a uniform distribution around its center
    double dist         = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; i++) {
        dist         += probs[i] * std::fabs((double) i - (double) mode);
        dist_uniform += std::fabs((double) i - (n - 1) / 2.0) / n;
    }
    return std::max(0.0, 1.0 - dist / dist_uniform);
}

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
const std::string CLEF_MARKER        = "<<clef:";
const std::string CLEF_SEP           = "<<clef:sep>>";
const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
const std::string CLEF_MARK_OPTION   = "<<clef:option>>";

} // namespace

std::string model_type(const llama_model * model) {
    const std::string arch = meta_str(model, "general.architecture");
    if (arch.empty()) {
        return "";
    }
    return meta_str(model, arch + ".decision.type");
}

// Mirrors the decision branch of common_init_result::common_init_result.
bool type_reads_embeddings(const std::string & type) {
    return type == "laya" || type == "kev" || type == "clef";
}

// Mirrors server_decision_context::can_share_prompt.
bool type_shares_prompt(const std::string & type) {
    return type == "openjev" || type == "lev" || type == "kev" || type == "nimble" || type == "pplx-decider" || type == "lfm2-d1";
}

struct Engine::Impl {
    llama_context     * ctx   = nullptr;
    const llama_model * model = nullptr;

    common_decision_type type = COMMON_DECISION_TYPE_NONE;
    std::string          type_name;

    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   choice_sorted   = false; // choice options are in the order of their keys

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    common_batch batch;

    explicit Impl(llama_context * c) : ctx(c), model(llama_get_model(c)), batch(c) {
        init();
    }

    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_LFM2_D1:
                return true;
            default:
                return false;
        }
    }

    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    bool reads_embeddings() const {
        return type == COMMON_DECISION_TYPE_LAYA || type == COMMON_DECISION_TYPE_KEV || type == COMMON_DECISION_TYPE_CLEF;
    }

    void init() { // server_decision_context::init
        const common_decision_type model_type = common_get_decision_type(model);
        if (model_type == COMMON_DECISION_TYPE_NONE) {
            throw std::runtime_error("the model is not a decision model");
        }

        const std::string prefix = meta_str(model, "general.architecture") + ".decision.";
        type_name = meta_str(model, prefix + "type");

        vocab = llama_model_get_vocab(model);

        const char * tmpl_src = llama_model_chat_template(model, "systemone");
        if (tmpl_src == nullptr) {
            throw std::runtime_error("decision model has no \"systemone\" template");
        }
        tmpl = std::make_shared<const common_chat_template>(tmpl_src, "", "");

        const std::string prefix_temp = prefix + "temperature.";
        for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
            char key[256];
            char val[64];
            if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {
                continue;
            }
            if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
                continue;
            }
            const float temp = std::strtof(val, nullptr);
            if (temp <= 0.0f) {
                throw std::runtime_error(string_format("invalid decision temperature: %s = %s", key, val));
            }
            temperatures[key + prefix_temp.size()] = temp;
        }

        if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
            // one letter per option, each must be a single token
            const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            for (const char c : letters) {
                const auto toks = common_tokenize(vocab, std::string(1, c), false, false);
                if (toks.size() != 1) {
                    throw std::runtime_error(string_format("decision label '%c' is not a single token", c));
                }
                labels.push_back(toks[0]);
            }
            n_options_max   = labels.size();
            noul_true_first = true;
        } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_NIMBLE || model_type == COMMON_DECISION_TYPE_PPLX_DECIDER) {
            // label codes are A..Z then AA..ZZ, only the ones that are a single token are used
            std::vector<std::string> codes;
            for (char a = 'A'; a <= 'Z'; a++) {
                codes.push_back(std::string(1, a));
            }
            for (char a = 'A'; a <= 'Z'; a++) {
                for (char b = 'A'; b <= 'Z'; b++) {
                    codes.push_back(std::string{a, b});
                }
            }
            for (const auto & code : codes) {
                const auto toks = common_tokenize(vocab, code, false, false);
                if (toks.size() == 1 && labels.size() < 255) {
                    labels.push_back(toks[0]);
                    label_texts.push_back(code);
                }
            }
            n_options_max = labels.size();
        } else if (model_type == COMMON_DECISION_TYPE_KEV) {
            // the hidden state of an option is read at the token that ends it
            const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
            if (toks.size() != 1) {
                throw std::runtime_error("decision model has no <|box_end|> token");
            }
            token_marker  = toks[0];
            n_options_max = 255;
        } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
            token_marker = llama_vocab_mask(vocab);
            token_sep    = llama_vocab_sep(vocab);
            if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
                throw std::runtime_error("decision model has no mask or sep token");
            }
            text_marker = common_token_to_piece(vocab, token_marker, true);

            const std::string val = meta_str(model, prefix + "max_head_tokens");
            max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
            if (max_head_tokens == 0) {
                throw std::runtime_error("decision model has no valid max_head_tokens");
            }
            n_options_max = 255;
        } else if (model_type == COMMON_DECISION_TYPE_CLEF) {
            n_options_max   = 255;
            noul_true_first = true;
            choice_sorted   = true;
        } else if (model_type == COMMON_DECISION_TYPE_LFM2_D1) {
            n_options_max   = 255;
            noul_true_first = true;
        } else {
            throw std::runtime_error("unsupported decision model type: " + type_name);
        }
        type = model_type;

        // Upstream's common_init_result forces this context shape; here the
        // context comes from Elixir, so check it rather than trust it.
        if (reads_embeddings() && llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
            throw std::runtime_error("a " + type_name + " decision model needs a context with embeddings: true and pooling_type: :none");
        }
    }

    //
    // request parsing
    //

    std::vector<decision_question> parse_questions(const json & body) const { // server_decision_context::parse_questions
        // d1 accepts a null state (images only upstream; here it just asks about nothing but the questions)
        if (!body.contains("state") || (body.at("state").is_null() && type != COMMON_DECISION_TYPE_LFM2_D1)) {
            throw std::invalid_argument("\"state\" must be provided");
        }
        if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
            throw std::invalid_argument("\"questions\" must be a non-empty object");
        }

        std::vector<decision_question> questions;
        for (const auto & [id, q] : body.at("questions").items()) {
            auto err = [&id = id](const std::string & msg) {
                return std::invalid_argument("questions." + id + ": " + msg);
            };
            if (!q.is_object()) {
                throw err("must be an object");
            }
            if (!q.contains("instructions") || q.at("instructions").is_null()) {
                throw err("\"instructions\" must be provided");
            }

            decision_question question;
            question.id           = id;
            question.instructions = q.at("instructions");

            const std::string type_name = json_string(q, "type");
            const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

            if (type_name == "choice") {
                question.type = QUESTION_CHOICE;
                if (!criteria.is_object() || criteria.empty()) {
                    throw err("\"criteria\" must be a non-empty object");
                }
                for (const auto & [key, description] : criteria.items()) {
                    question.options.push_back({key, description});
                }
                if (choice_sorted) {
                    std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                        return a.key < b.key;
                    });
                }
            } else if (type_name == "score") {
                question.type = QUESTION_SCORE;
                if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                    throw err("\"criteria\" must be an array of 2 to 10 levels");
                }
                for (size_t i = 0; i < criteria.size(); i++) {
                    question.options.push_back({std::to_string(i), criteria.at(i)});
                }
            } else if (type_name == "noul") {
                question.type = QUESTION_NOUL;
                if (!criteria.is_null() && !criteria.is_object()) {
                    throw err("\"criteria\" must be an object");
                }
                for (const char * key : {"false", "true"}) {
                    question.options.push_back({key, criteria.is_object() && criteria.contains(key) ? criteria.at(key) : json()});
                }
                if (noul_true_first) {
                    std::swap(question.options[0], question.options[1]);
                }
            } else {
                throw err("\"type\" must be one of: choice, score, noul");
            }

            if (question.options.size() > n_options_max) {
                throw err(string_format("too many options (%zu), this model supports at most %zu", question.options.size(), n_options_max));
            }

            questions.push_back(std::move(question));
        }
        return questions;
    }

    // server_decision_context::parse_state, without image loading: this build
    // has no libmtmd, so any image is an error instead of being taken out of
    // the state. A state without images comes back unchanged, as upstream.
    json parse_state(const json & body) const {
        auto reject = []() {
            return std::invalid_argument("image input is not supported: LlamaCppEx is built without libmtmd");
        };
        if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
            throw std::invalid_argument("\"videos\" is not supported");
        }
        if (body.contains("images") && !body.at("images").is_null()) {
            if (!body.at("images").is_array()) {
                throw std::invalid_argument("\"images\" must be an array");
            }
            if (!body.at("images").empty()) {
                throw reject();
            }
        }

        const json & state = body.at("state");
        const bool is_wrapped = state.is_object() && state.contains("messages");
        const json & messages = is_wrapped ? state.at("messages") : state;
        if (!messages.is_array()) {
            return state;
        }
        for (const auto & msg : messages) {
            if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
                continue;
            }
            for (const auto & part : msg.at("content")) {
                if (part.is_object() && json_string(part, "type") == "image_url" && part.contains("image_url")) {
                    throw reject();
                }
            }
        }
        return state;
    }

    //
    // prompt
    //

    size_t n_variants(const decision_question & question) const { // server_decision_context::n_variants
        // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
        if (type == COMMON_DECISION_TYPE_LEV && question.type == QUESTION_CHOICE && question.options.size() > 1) {
            return 2;
        }
        return 1;
    }

    size_t n_outputs(const decision_question & question) const { // server_decision_context::n_outputs
        if (type == COMMON_DECISION_TYPE_LEV && question.type == QUESTION_NOUL) {
            return DECISION_LEV_N_RATINGS;
        }
        return question.options.size();
    }

    // label codes follow prompt.py of the model repo
    void d1_labels(const decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const { // server_decision_context::d1_labels
        const size_t n_options = question.options.size();

        auto get_single_tokens = [&](const std::vector<std::string> & forms) {
            llama_tokens out;
            for (const auto & form : forms) {
                const auto toks = common_tokenize(vocab, form, false, false);
                if (toks.size() == 1 && std::find(out.begin(), out.end(), toks[0]) == out.end()) {
                    out.push_back(toks[0]);
                }
            }
            return out;
        };

        if (question.type != QUESTION_CHOICE) {
            for (const auto & opt : question.options) {
                llama_tokens group;
                if (question.type == QUESTION_SCORE) {
                    group = get_single_tokens({opt.key});
                } else if (opt.key == "true") {
                    group = get_single_tokens({"yes", "Yes", "YES"});
                } else {
                    group = get_single_tokens({"no", "No", "NO"});
                }
                if (group.empty()) {
                    throw std::runtime_error("decision label is not a single token: " + opt.key);
                }
                texts.push_back(opt.key);
                groups.push_back(group);
            }
            return;
        }

        bool is_letters = true;
        for (const auto & opt : question.options) {
            is_letters = is_letters && opt.key.size() == 1 && std::isalpha((unsigned char) opt.key[0]);
        }

        std::vector<std::string> codes;
        for (size_t i = 0; i < n_options; i++) {
            if (is_letters) {
                codes.push_back(question.options[i].key);
            } else if (n_options <= 26) {
                codes.push_back(std::string(1, 'A' + i));
            } else {
                codes.push_back(string_format("%02zu", i));
            }
        }

        std::vector<std::string> pool;
        for (char c = 'A'; c <= 'Z'; c++) {
            pool.push_back(std::string(1, c));
        }
        for (int i = 0; i < 100; i++) {
            pool.push_back(string_format("%02d", i));
        }
        for (char c = 'a'; c <= 'z'; c++) {
            pool.push_back(std::string(1, c));
        }
        for (int i = 0; i < 200; i++) {
            pool.push_back(string_format("#%d", i));
        }
        for (char a = 'A'; a <= 'Z'; a++) {
            for (char b = 'A'; b <= 'Z'; b++) {
                pool.push_back(std::string{a, b});
            }
        }

        llama_tokens used;
        auto take = [&](const std::string & code) {
            const auto toks = common_tokenize(vocab, code, false, false);
            if (toks.size() != 1 || std::find(used.begin(), used.end(), toks[0]) != used.end()) {
                return false;
            }
            used.push_back(toks[0]);
            llama_tokens group = {toks[0]};
            for (const llama_token tok : get_single_tokens({" " + code})) {
                if (tok != toks[0]) {
                    group.push_back(tok);
                }
            }
            texts.push_back(code);
            groups.push_back(group);
            return true;
        };
        for (const auto & code : codes) {
            bool is_taken = take(code);
            for (size_t i = 0; !is_taken && i < pool.size(); i++) {
                is_taken = take(pool[i]);
            }
            if (!is_taken) {
                throw std::invalid_argument(string_format("no single-token label left for %zu options", n_options));
            }
        }
    }

    json render_options(const decision_question & question, size_t variant) const { // server_decision_context::render_options
        const size_t n_options = question.options.size();

        std::vector<std::string>  d1_texts;
        std::vector<llama_tokens> d1_groups;
        if (type == COMMON_DECISION_TYPE_LFM2_D1) {
            d1_labels(question, d1_texts, d1_groups);
        }
        // the second variant shows the options in the reverse order
        json options = json::array();
        for (size_t i = 0; i < n_options; i++) {
            const auto & opt = question.options[variant == 0 ? i : n_options - 1 - i];
            json option = json{
                {"key",         opt.key},
                {"description", opt.description},
            };
            if (type == COMMON_DECISION_TYPE_KEV) {
                option["key"] = kev_text(opt.key);
                if (!opt.description.is_null()) {
                    option["description"] = kev_text(opt.description);
                }
            }
            if (!label_texts.empty()) {
                option["label"] = label_texts[i];
            }
            if (!d1_texts.empty()) {
                option["label"] = d1_texts[i];
            }
            options.push_back(option);
        }
        return options;
    }

    std::string run_template(const json & inp) const {
        jinja::context jctx(tmpl->source());
        jinja::global_from_json(jctx, inp, false);
        jinja::runtime runtime(jctx);
        const jinja::value results = runtime.execute(tmpl->prog);
        return jinja::runtime::gather_string_parts(results)->as_string().str();
    }

    // server_decision_context::render with n_images == 0
    std::string render(
            const json & state,
            const std::vector<decision_question> & questions,
            const decision_question & question,
            size_t variant) const {
        // the template is given raw JSON values, it serializes the ones that are not strings
        json inp = json{
            {"id",           question.id},
            {"type",         question_type_name(question.type)},
            {"instructions", question.instructions},
            {"state",        state},
            {"options",      render_options(question, variant)},
        };

        // the nimble prompt lists all the questions of the request
        if (type == COMMON_DECISION_TYPE_NIMBLE) {
            inp["questions"] = json::array();
            for (const auto & q : questions) {
                inp["questions"].push_back(json{
                    {"id",           q.id},
                    {"type",         question_type_name(q.type)},
                    {"instructions", q.instructions},
                    {"options",      render_options(q, 0)},
                });
            }
        }

        // lev was trained with sorted keys
        if (type == COMMON_DECISION_TYPE_LEV) {
            inp = sort_keys(inp);
        }

        // the kev template only takes text
        if (type == COMMON_DECISION_TYPE_KEV) {
            inp["state"]        = kev_text(state);
            inp["instructions"] = kev_text(question.instructions);
        }

        // the input must not contain the marker of the options
        if (!text_marker.empty()) {
            inp = replace_text(inp, text_marker, " ");
        }

        inp["images"] = json::array();

        return run_template(inp);
    }

    // server_decision_context::fill_task without files
    decision_task fill_task(
            const json & state,
            const std::vector<decision_question> & questions,
            const decision_question & question,
            size_t variant) const {
        decision_task task;
        const std::string prompt = render(state, questions, question, variant);

        if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER) {
            // lev reads the ratings of a noul question at its first labels, not at the digits
            task.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
        }
        if (type == COMMON_DECISION_TYPE_LFM2_D1) {
            std::vector<std::string>  texts;
            std::vector<llama_tokens> groups;
            d1_labels(question, texts, groups);
            for (const auto & group : groups) {
                task.labels.insert(task.labels.end(), group.begin(), group.end());
                task.label_groups.push_back(group.size());
            }
        }

        llama_tokens tokens = common_tokenize(vocab, prompt, false, true);
        if (type == COMMON_DECISION_TYPE_LAYA) {
            fill_task_laya(tokens, question, task);
        }
        if (type == COMMON_DECISION_TYPE_KEV) {
            // an option is read at its end token, the question at the last token
            for (size_t i = 0; i < tokens.size(); i++) {
                if (tokens[i] == token_marker) {
                    task.markers.push_back(i);
                }
            }
            if (task.markers.size() != question.options.size()) {
                throw std::runtime_error("unexpected layout of the decision prompt");
            }
            task.pointer = tokens.size() - 1;
        }
        task.tokens = std::move(tokens);
        return task;
    }

    // the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
    // options and question are cut to fit max_head_tokens, the same way the model was trained
    void fill_task_laya(llama_tokens & tokens, const decision_question & question, decision_task & task) const { // server_decision_context::fill_task_laya
        const size_t n_options = question.options.size();

        std::vector<size_t> markers;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == token_marker) {
                markers.push_back(i);
            }
        }
        const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
        if (markers.size() != n_options || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) {
            throw invalid;
        }
        const size_t head_end = markers[0] - 1;
        const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();
        if (opts_end + 1 >= tokens.size()) {
            throw invalid;
        }

        // marker + text of each option
        std::vector<llama_tokens> options;
        size_t n_options_tokens = 0;
        auto set_max = [&](size_t n_max) {
            n_options_tokens = 0;
            for (auto & opt : options) {
                opt.resize(std::min(opt.size(), n_max));
                n_options_tokens += opt.size();
            }
        };
        for (size_t i = 0; i < n_options; i++) {
            const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
            options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
        }
        set_max(max_option_tokens + 1);
        if (n_options_tokens + 16 > max_head_tokens) {
            // too many or too long options, shrink them evenly
            set_max(std::max((size_t) 4, (max_head_tokens - std::min(max_head_tokens, (size_t) 16)) / n_options));
        }
        const size_t n_question_max = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));

        llama_tokens out;
        out.push_back(tokens[0]);
        out.insert(out.end(), tokens.begin() + 1, tokens.begin() + std::min(head_end, 1 + n_question_max));
        out.push_back(token_sep);
        for (const auto & opt : options) {
            task.markers.push_back(out.size());
            out.insert(out.end(), opt.begin(), opt.end());
        }
        out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
        tokens = std::move(out);

        // the output has one score per question type
        task.column = question.type;
    }

    // server_decision_context::fill_task_joint without files
    decision_task fill_task_joint(const json & state, const std::vector<decision_question> & questions) const {
        decision_task task;

        json inp_questions = json::array();
        for (const auto & question : questions) {
            json options = json::array();
            for (const auto & opt : question.options) {
                options.push_back(json{
                    {"key",         opt.key},
                    {"description", opt.description},
                });
            }
            inp_questions.push_back(json{
                {"id",           question.id},
                {"type",         question_type_name(question.type)},
                {"instructions", question.instructions},
                {"options",      options},
            });
        }

        // the template is given raw JSON values with sorted keys, and no marker in the input
        json inp = json{
            {"state",     state},
            {"questions", inp_questions},
        };
        inp = replace_text(sort_keys(inp), CLEF_MARKER, "<<clef ");

        inp["images"]        = json::array();
        inp["sep"]           = CLEF_SEP;
        inp["mark_question"] = CLEF_MARK_QUESTION;
        inp["mark_option"]   = CLEF_MARK_OPTION;

        const std::string prompt = run_template(inp);

        const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

        // the model was trained with the pieces tokenized one by one
        const std::vector<std::string> pieces = string_split(prompt, CLEF_SEP);

        size_t i_question = 0;
        for (size_t i_piece = 0; i_piece < pieces.size(); i_piece++) {
            std::string piece = pieces[i_piece];
            int32_t order = LLAMA_DECISION_ORDER_NONE;
            if (string_starts_with(piece, CLEF_MARK_QUESTION)) {
                piece = piece.substr(CLEF_MARK_QUESTION.size());
                if (i_question >= questions.size()) {
                    throw invalid;
                }
                switch (questions[i_question++].type) {
                    case QUESTION_NOUL:   order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
                    case QUESTION_CHOICE: order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
                    case QUESTION_SCORE:  order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
                }
            } else if (string_starts_with(piece, CLEF_MARK_OPTION)) {
                piece = piece.substr(CLEF_MARK_OPTION.size());
                order = LLAMA_DECISION_ORDER_OPTION;
                task.n_scores++;
            }

            const llama_tokens piece_tokens = common_tokenize(vocab, piece, false, true);
            if (order != LLAMA_DECISION_ORDER_NONE && piece_tokens.empty()) {
                throw std::invalid_argument("the instructions and the options of a question must not be empty");
            }
            task.tokens.insert(task.tokens.end(), piece_tokens.begin(), piece_tokens.end());
            task.order.resize(task.tokens.size(), order);
        }

        size_t n_options = 0;
        for (const auto & question : questions) {
            n_options += question.options.size();
        }
        if (i_question != questions.size() || (size_t) task.n_scores != n_options) {
            throw invalid;
        }
        return task;
    }

    //
    // answer
    //

    float get_temperature(const decision_question & question) const { // server_decision_context::get_temperature
        const size_t n = question.options.size();
        const std::string type_name = question_type_name(question.type);

        // the temperature can depend on the number of options, the buckets are the ones used to fit it
        std::string bucket;
        if (type == COMMON_DECISION_TYPE_LEV) {
            bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
        } else {
            bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
        }

        for (const auto & name : {type_name + "." + bucket, type_name}) {
            const auto it = temperatures.find(name);
            if (it != temperatures.end()) {
                return it->second;
            }
        }
        return 1.0f;
    }

    json format_answer(const decision_question & question, const std::vector<std::vector<float>> & scores) const { // server_decision_context::format_answer
        const size_t n = n_outputs(question);
        if (scores.size() != n_variants(question)) {
            throw std::runtime_error("decision result does not match the number of variants");
        }

        // softmax over the outputs of each variant, then the average of the variants
        const float temperature = get_temperature(question);
        std::vector<double> probs(n, 0.0);
        for (size_t v = 0; v < scores.size(); v++) {
            const auto & s = scores[v];
            if (s.size() != n) {
                throw std::runtime_error("decision result does not match the number of options");
            }
            // a joint head returns NaN if it could not use the decision order
            if (std::any_of(s.begin(), s.end(), [](float v) { return std::isnan(v); })) {
                throw std::runtime_error("the model could not evaluate the decision");
            }
            const float score_max = *std::max_element(s.begin(), s.end());
            std::vector<double> p(n);
            double sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                p[i] = std::exp((double) (s[i] - score_max) / temperature);
                sum += p[i];
            }
            for (size_t i = 0; i < n; i++) {
                // the second variant is in the reverse order
                probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
            }
        }

        json answer = json{{"type", question_type_name(question.type)}};

        if (question.type == QUESTION_NOUL) {
            if (type == COMMON_DECISION_TYPE_LEV) {
                double expected = 0.0;
                for (size_t i = 0; i < n; i++) {
                    expected += probs[i] * i / (n - 1);
                }
                answer["noul"] = expected;
                return answer;
            }
            for (size_t i = 0; i < n; i++) {
                if (question.options[i].key == "true") {
                    answer["noul"] = probs[i];
                }
            }
            return answer;
        }

        json probabilities = json::object();
        for (size_t i = 0; i < n; i++) {
            probabilities[question.options[i].key] = probs[i];
        }

        if (question.type == QUESTION_CHOICE) {
            const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
            answer["choice"]        = question.options[best].key;
            answer["probabilities"] = probabilities;
            answer["confidence"]    = confidence_choice(probs);
        } else {
            double expected = 0.0;
            json legend = json::object();
            for (size_t i = 0; i < n; i++) {
                expected += i * probs[i];
                legend[question.options[i].key] = question.options[i].description;
            }
            answer["score"]         = expected;
            answer["legend"]        = legend;
            answer["probabilities"] = probabilities;
            answer["confidence"]    = confidence_score(probs);
        }
        return answer;
    }

    //
    // evaluation
    //
    // Replaces upstream's slots. Upstream runs the tasks of a request on
    // n_parallel slots and shares a prompt prefix by evaluating it on one slot
    // and seq_cp'ing it to the others (server_decision_group_tasks,
    // server_slot::copy_prompt_to). This runs them one after another on one
    // context: the shared prefix is decoded once on seq 0 and copied to seq 1
    // for each task. A whole-sequence copy works on recurrent and hybrid
    // memory too, where a partial seq_rm back to the prefix would not.
    //

    // The positions that must come out of the last batch of a task, as
    // upstream's batch layout rules (server-context.cpp, "the outputs of a
    // decision are read from one batch"): the last token for label readouts,
    // everything from the first read position for embedding readouts, and the
    // whole prompt for the models that see it at once (laya is a non-causal
    // encoder, clef's joint head reads the whole batch).
    int32_t first_output(const decision_task & task) const {
        if (type == COMMON_DECISION_TYPE_LAYA || type == COMMON_DECISION_TYPE_CLEF) {
            return 0;
        }
        if (!task.labels.empty()) {
            return (int32_t) task.tokens.size() - 1;
        }
        return task.pos_first();
    }

    void check_fits(const decision_task & task) const {
        const int32_t n_tokens = (int32_t) task.tokens.size();
        const int32_t n_ctx    = (int32_t) llama_n_ctx(ctx);
        const int32_t n_batch  = (int32_t) llama_n_batch(ctx);
        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx);

        if (n_tokens == 0) {
            throw std::runtime_error("the decision prompt is empty");
        }
        if (n_tokens > n_ctx) {
            throw std::invalid_argument(string_format(
                "the prompt (%d tokens) does not fit in the context, increase n_ctx (current: %d)", n_tokens, n_ctx));
        }
        if (first_output(task) == 0) {
            const int32_t n_max = std::min(n_batch, n_ubatch);
            if (n_tokens > n_max) {
                throw std::invalid_argument(string_format(
                    "the prompt (%d tokens) must be evaluated in one batch, increase n_batch and n_ubatch (current: %d)", n_tokens, n_max));
            }
            return;
        }
        const int32_t n_read = n_tokens - first_output(task);
        if (n_read > n_batch) {
            throw std::invalid_argument(string_format(
                "the question and its options (%d tokens) are too large to process. "
                "increase the batch size (current batch size: %d)", n_read, n_batch));
        }
    }

    // Decodes tokens [from, to) of `task` on `seq`, at positions equal to their
    // index. Every position from `out_from` on is an output, and all of them
    // land in the last batch (check_fits guarantees they fit in one). Returns
    // the position of the first entry of that last batch.
    int32_t decode(const decision_task & task, int32_t from, int32_t to, llama_seq_id seq, int32_t out_from) {
        const int32_t n_batch = (int32_t) llama_n_batch(ctx);
        const int32_t last    = std::max(from, std::min(out_from, to - n_batch));

        auto run = [&](int32_t off, int32_t end) {
            batch.clear();
            for (int32_t i = off; i < end; i++) {
                batch.add(task.tokens[i], i, seq, i >= out_from);
                if (!task.order.empty()) {
                    batch.tokens.back().decision_order = task.order[i];
                }
            }
            const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                throw std::runtime_error(string_format("decision decode failed with code: %d", ret));
            }
        };

        for (int32_t off = from; off < last; off += n_batch) {
            run(off, std::min(last, off + n_batch));
        }
        run(last, to);
        return last;
    }

    // server_context::send_decision: the raw scores of one evaluated task
    std::vector<float> read_scores(const decision_task & task, int32_t batch_start) const {
        const int32_t n_tokens = (int32_t) task.tokens.size();
        std::vector<float> scores;

        if (!task.labels.empty()) {
            const float * logits = llama_get_logits_ith(ctx, n_tokens - 1 - batch_start);
            if (logits == nullptr) {
                throw std::runtime_error("failed to get logits");
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : task.labels) {
                if (label < 0 || label >= n_vocab) {
                    throw std::runtime_error("decision label is outside the vocabulary");
                }
                scores.push_back(logits[label]);
            }
            if (!task.label_groups.empty()) {
                std::vector<float> grouped;
                size_t i = 0;
                for (const int32_t n : task.label_groups) {
                    if (n <= 0 || i + n > scores.size()) {
                        throw std::runtime_error("decision label groups do not match the labels");
                    }
                    grouped.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
                    i += n;
                }
                scores = std::move(grouped);
            }
            return scores;
        }

        const int32_t out_from = first_output(task);
        auto get_embd = [&](int32_t pos) -> const float * {
            return pos >= out_from && pos < n_tokens ? llama_get_embeddings_ith(ctx, pos - batch_start) : nullptr;
        };

        // joint head (decision model): the scores are the first rows
        for (int32_t i = 0; i < task.n_scores; i++) {
            const float * embd = get_embd(i);
            if (embd == nullptr) {
                throw std::runtime_error("failed to get embeddings");
            }
            scores.push_back(embd[0]);
        }

        const int32_t n_embd_out = llama_model_n_embd_out(model);
        const int32_t n_pointer  = n_embd_out / 2;
        const float * embd_q = task.pointer >= 0 ? get_embd(task.pointer) : nullptr;
        if (task.column < 0 || task.column >= n_embd_out) {
            throw std::runtime_error("decision output column is outside the embeddings");
        }

        for (const int32_t marker : task.markers) {
            const float * embd = get_embd(marker);
            if (embd == nullptr || (task.pointer >= 0 && embd_q == nullptr)) {
                throw std::runtime_error("failed to get embeddings, the question and its options must fit in one batch");
            }
            if (task.pointer < 0) {
                scores.push_back(embd[task.column]);
                continue;
            }
            float dot = 0.0f;
            for (int32_t i = 0; i < n_pointer; i++) {
                dot += embd_q[i] * embd[n_pointer + i];
            }
            scores.push_back(dot / sqrtf((float) n_pointer));
        }
        return scores;
    }

    // Tokens every task starts with, capped so each task keeps at least one
    // token of its own (server_decision_group_tasks) and so no read position
    // falls inside the prefix, where it would not be an output.
    int32_t shared_prefix(const std::vector<decision_task> & tasks) const {
        if (!can_share_prompt() || tasks.size() < 2 || llama_n_seq_max(ctx) < 2) {
            return 0;
        }
        const llama_tokens & first = tasks[0].tokens;
        size_t n_shared = first.size() - 1;
        for (const auto & task : tasks) {
            size_t n_common = 0;
            while (n_common < n_shared && n_common < task.tokens.size() && task.tokens[n_common] == first[n_common]) {
                n_common++;
            }
            n_shared = std::min({n_shared, n_common, task.tokens.size() - 1});
            n_shared = std::min(n_shared, (size_t) first_output(task));
        }
        return (int32_t) n_shared;
    }

    std::vector<std::vector<float>> evaluate(const std::vector<decision_task> & tasks) {
        for (const auto & task : tasks) {
            check_fits(task);
        }

        llama_set_embeddings(ctx, reads_embeddings());
        llama_memory_t mem = llama_get_memory(ctx);
        auto clear = [&]() {
            if (mem != nullptr) {
                llama_memory_clear(mem, true);
            }
        };

        std::vector<std::vector<float>> results;
        results.reserve(tasks.size());

        const int32_t n_shared = mem != nullptr ? shared_prefix(tasks) : 0;
        clear();
        if (n_shared > 0) {
            decode(tasks[0], 0, n_shared, 0, n_shared);
        }

        for (const auto & task : tasks) {
            llama_seq_id seq = 0;
            if (n_shared > 0) {
                seq = 1;
                llama_memory_seq_rm(mem, seq, -1, -1);
                llama_memory_seq_cp(mem, 0, seq, -1, -1);
            } else {
                clear();
            }
            const int32_t start = decode(task, n_shared, (int32_t) task.tokens.size(), seq, first_output(task));
            results.push_back(read_scores(task, start));
        }
        clear();
        return results;
    }

    // The /v1/systemone handler (server-context.cpp, post_systemone), minus
    // HTTP and slots.
    std::string decide(const std::string & text) {
        json body;
        try {
            body = json::parse(text);
        } catch (const std::exception & e) {
            throw std::invalid_argument(std::string("invalid request JSON: ") + e.what());
        }
        if (!body.is_object()) {
            throw std::invalid_argument("the request must be a JSON object");
        }

        const auto questions = parse_questions(body);
        const json state     = parse_state(body);

        std::vector<decision_task> tasks;
        if (is_joint()) {
            tasks.push_back(fill_task_joint(state, questions));
        } else {
            for (const auto & question : questions) {
                for (size_t variant = 0; variant < n_variants(question); variant++) {
                    tasks.push_back(fill_task(state, questions, question, variant));
                }
            }
        }

        const auto results = evaluate(tasks);

        size_t n_tokens = 0;
        for (const auto & task : tasks) {
            n_tokens += task.tokens.size();
        }

        json answers = json::object();
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (is_joint()) {
                // one result with the scores of all the questions, in order
                const auto & all = results[0];
                if (i_score + question.options.size() > all.size()) {
                    throw std::runtime_error("decision result does not match the number of options");
                }
                scores.emplace_back(all.begin() + i_score, all.begin() + i_score + question.options.size());
                i_score += question.options.size();
            }
            for (size_t variant = 0; !is_joint() && variant < n_variants(question); variant++) {
                scores.push_back(results[i_result++]);
            }
            answers[question.id] = format_answer(question, scores);
        }

        return json{
            {"answers", answers},
            {"usage",   json{
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        }.dump();
    }
};

Engine::Engine(llama_context * ctx) : impl(std::make_unique<Impl>(ctx)) {}

Engine::~Engine() = default;

const std::string & Engine::type() const {
    return impl->type_name;
}

std::string Engine::decide(const std::string & body) {
    return impl->decide(body);
}

} // namespace llama_cpp_ex::decision
