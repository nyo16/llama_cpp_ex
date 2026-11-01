// Port of llama.cpp's tools/server/server-decision.cpp plus the decision paths
// of tools/server/server-context.cpp (send_decision, the batch layout rules and
// the /v1/systemone handler). See decision.h for scope.
//
// Ported at 785322d17. Functions that are upstream's carry its name in a
// trailing comment so a bump can diff them one by one; everything below the
// "evaluation" banner replaces server slots and is ours.

#include "decision.h"
#include "input_tokenize.h"

#include "chat.h"
#include "common.h"
#include "json.h"

// Staging API, not part of include/llama.h. Only the enum is used: the order of
// each batch entry is applied by common_batch::get_sub_batch.
#include "../../vendor/llama.cpp/src/llama-ext.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
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
        for (size_t i = 0; i < order.size(); i++) {
            if (order[i] != 0) {
                return pos < 0 ? (int32_t) i : std::min(pos, (int32_t) i);
            }
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

// h2o-lightning follows serve_config.json and h2o_lightning_shim.py of the model repo
// a longer state keeps its head and its tail
const size_t DECISION_H2O_MAX_STATE_TOKENS = 32000;

// the recursive functions of this file walk request JSON, a deeper value would overflow the stack
// (unreachable through the NIF, which refuses a request nested deeper than 64 levels before parsing it)
const int RECURSIVE_LIMIT = 128;

std::invalid_argument too_deep() {
    return std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
}

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
json replace_text(const json & val, const std::string & search, const std::string & replace, int depth = 0) { // decision_replace_text
    if (depth >= RECURSIVE_LIMIT) {
        throw too_deep();
    }
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(replace_text(item, search, replace, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = replace_text(item, search, replace, depth + 1);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
json sort_keys(const json & val, int depth = 0) { // decision_sort_keys
    if (depth >= RECURSIVE_LIMIT) {
        throw too_deep();
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(sort_keys(item, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = sort_keys(item, depth + 1);
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
std::string kev_render(const json & val, int indent = 0, int depth = 0) { // decision_kev_render
    if (depth >= RECURSIVE_LIMIT) {
        throw too_deep();
    }
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
            const std::string text = kev_render(item, indent + 1, depth + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + kev_render(item, is_nested ? indent + 1 : 0, depth + 1);
        }
        return out;
    }
    return val.dump();
}

// same result as std::regex_replace with <\|([A-Za-z0-9_]+)\|>
// std::regex is not used: libstdc++ recurses for each char, a long input overflows the stack
std::string escape_special_tokens(const std::string & text) { // server_decision_escape_special_tokens
    const auto is_name = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    };
    std::string out;
    size_t n_done = 0; // the chars of text that are in out
    for (size_t i = text.find("<|"); i != std::string::npos; i = text.find("<|", std::max(i + 1, n_done))) {
        size_t j = i + 2;
        while (j < text.size() && is_name(text[j])) {
            j++;
        }
        if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
            out.append(text, n_done, i - n_done);
            out += "<\xC2\xA6";
            out.append(text, i + 2, j - i - 2);
            out += "\xC2\xA6>";
            n_done = j + 2;
        }
    }
    if (n_done == 0) {
        return text;
    }
    out.append(text, n_done, std::string::npos);
    return out;
}

// kev text input: special tokens written in the text must not be parsed as such
std::string kev_text(const json & val) { // decision_kev_text
    return escape_special_tokens(kev_render(val));
}

// lfm2-d1-omni, lfm3-d1: special tokens written in the input must not be parsed as such, in keys too
template <typename F>
json escape_json(const json & val, const F & escape, int depth = 0) { // decision_escape
    if (depth >= RECURSIVE_LIMIT) {
        throw too_deep();
    }
    if (val.is_string()) {
        return escape(val.get<std::string>());
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(escape_json(item, escape, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[escape(key)] = escape_json(item, escape, depth + 1);
        }
        return out;
    }
    return val;
}

// lfm3-d1-omni: a spoken question, its instructions are an input_audio part
bool is_spoken(const json & instructions) { // decision_is_spoken
    return instructions.is_object() && json_string(instructions, "type") == "input_audio";
}

//
// h2o-lightning: the texts are rendered the way the reference shim renders them with Python (json.dumps, str.lstrip)
//

// one code point of UTF-8 text at s[i], i moves past it; an invalid byte is read as is
uint32_t utf8_next(const std::string & s, size_t & i) { // decision_utf8_next
    const unsigned char c = s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
    if (i + n > s.size()) {
        n = 1;
    }
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
    for (size_t k = 1; k < n; k++) {
        cp = (cp << 6) | (s[i + k] & 0x3f);
    }
    i += n;
    return cp;
}

// str.isspace() of Python
bool py_isspace(uint32_t cp) { // decision_py_isspace
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

std::string py_rstrip(const std::string & s) { // decision_py_rstrip
    size_t end = 0;
    for (size_t i = 0; i < s.size();) {
        if (!py_isspace(utf8_next(s, i))) {
            end = i;
        }
    }
    return s.substr(0, end);
}

std::string py_lstrip(const std::string & s) { // decision_py_lstrip
    size_t i = 0;
    while (i < s.size()) {
        size_t next = i;
        if (!py_isspace(utf8_next(s, next))) {
            break;
        }
        i = next;
    }
    return s.substr(i);
}

// a code point as JSON escapes, a surrogate pair above the BMP
std::string py_escape_cp(uint32_t cp) { // decision_py_escape_cp
    if (cp < 0x10000) {
        return string_format("\\u%04x", cp);
    }
    cp -= 0x10000;
    return string_format("\\u%04x\\u%04x", 0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
}

// repr() of a Python float: the shortest digits that read back the same, in exponent form below 1e-4 and from 1e16
std::string py_float(double v) { // decision_py_float
    char buf[64];
    // the fewest digits that read back the same
    for (int prec = 0; prec < 17; prec++) {
        snprintf(buf, sizeof(buf), "%.*e", prec, v);
        if (std::strtod(buf, nullptr) == v) {
            break;
        }
    }
    const std::string sci(buf); // for example "-1.25e+03"
    const size_t pos_e = sci.find('e');
    const int    exp   = std::atoi(sci.c_str() + pos_e + 1);
    std::string sign;
    std::string digits;
    for (size_t i = 0; i < pos_e; i++) {
        if (sci[i] == '-') {
            sign = "-";
        } else if (sci[i] != '.') {
            digits += sci[i];
        }
    }
    const int n = (int) digits.size();
    if (exp < -4 || exp >= 16) {
        return sign + digits.substr(0, 1) + (n > 1 ? "." + digits.substr(1) : "") + string_format("e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    }
    if (exp >= n - 1) {
        return sign + digits + std::string(exp - (n - 1), '0') + ".0";
    }
    if (exp >= 0) {
        return sign + digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
    }
    return sign + "0." + std::string(-exp - 1, '0') + digits;
}

void py_str(std::string & out, const std::string & s, bool ensure_ascii) { // decision_py_str
    out += '"';
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; i++; continue;
            case '\\': out += "\\\\"; i++; continue;
            case '\b': out += "\\b";  i++; continue;
            case '\f': out += "\\f";  i++; continue;
            case '\n': out += "\\n";  i++; continue;
            case '\r': out += "\\r";  i++; continue;
            case '\t': out += "\\t";  i++; continue;
            default:   break;
        }
        if (c < 0x20 || (ensure_ascii && c == 0x7f)) {
            out += string_format("\\u%04x", c);
            i++;
        } else if (c < 0x80 || !ensure_ascii) {
            out += (char) c;
            i++;
        } else {
            out += py_escape_cp(utf8_next(s, i));
        }
    }
    out += '"';
}

// json.dumps() of Python: indent < 0 is no indent, keys stay in their order
void py_dump(std::string & out, const json & val, int indent, int level, const std::string & item_sep, const std::string & key_sep, bool ensure_ascii) { // decision_py_dump
    if (level >= RECURSIVE_LIMIT) {
        throw too_deep();
    }
    const auto newline = [&](int lvl) {
        return indent < 0 ? std::string() : "\n" + std::string(indent * lvl, ' ');
    };
    if (val.is_null()) {
        out += "null";
    } else if (val.is_boolean()) {
        out += val.get<bool>() ? "true" : "false";
    } else if (val.is_number_float()) {
        out += py_float(val.get<double>());
    } else if (val.is_number()) {
        out += val.dump();
    } else if (val.is_string()) {
        py_str(out, val.get<std::string>(), ensure_ascii);
    } else if (val.is_array()) {
        if (val.empty()) {
            out += "[]";
            return;
        }
        out += "[" + newline(level + 1);
        for (size_t i = 0; i < val.size(); i++) {
            if (i > 0) {
                out += item_sep + newline(level + 1);
            }
            py_dump(out, val.at(i), indent, level + 1, item_sep, key_sep, ensure_ascii);
        }
        out += newline(level) + "]";
    } else if (val.is_object()) {
        if (val.empty()) {
            out += "{}";
            return;
        }
        out += "{" + newline(level + 1);
        bool first = true;
        for (const auto & [key, item] : val.items()) {
            if (!first) {
                out += item_sep + newline(level + 1);
            }
            first = false;
            py_str(out, key, ensure_ascii);
            out += key_sep;
            py_dump(out, item, indent, level + 1, item_sep, key_sep, ensure_ascii);
        }
        out += newline(level) + "}";
    }
}

std::string py_dumps(const json & val, int indent, const std::string & item_sep, const std::string & key_sep, bool ensure_ascii) { // decision_py_dumps
    std::string out;
    py_dump(out, val, indent, 0, item_sep, key_sep, ensure_ascii);
    return out;
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

// VLLM_DECISION2, VLLM_D3: normalized entropy, as in the reference implementation
double confidence_entropy(const std::vector<double> & probs) { // decision_confidence_entropy
    if (probs.size() < 2) {
        return 1.0;
    }
    double entropy = 0.0;
    for (const double p : probs) {
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }
    return std::max(0.0, std::min(1.0, 1.0 - entropy / std::log((double) probs.size())));
}

bool is_payload(const json & val) { // decision_is_payload
    return val.is_string() || val.is_object() || val.is_array();
}

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
const std::string CLEF_MARKER        = "<<clef:";
const std::string CLEF_SEP           = "<<clef:sep>>";
const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
const std::string CLEF_MARK_OPTION   = "<<clef:option>>";

// given to the lfm2-d1-omni template: text between the pieces of the prompt, and at the start of the pieces that are cut to a token budget
const std::string D1OMNI_MARKER        = "<<d1omni:";
const std::string D1OMNI_SEP           = "<<d1omni:sep>>";
const std::string D1OMNI_MARK_STATE    = "<<d1omni:state>>";
const std::string D1OMNI_MARK_QUESTION = "<<d1omni:question>>";
const std::string D1OMNI_MARK_OPTION   = "<<d1omni:option>>";

// max_length of the model config, the converter checks it. Upstream also has
// the image and audio budgets, which only apply to media.
const size_t D1OMNI_MAX_TOKENS = 16384;

//
// caller text (ours, not upstream's: see docs/release-guide.md before porting over it)
//
// A prompt is rendered from the "systemone" template with the caller's state,
// instructions and options in it, and upstream tokenizes the result with
// special parsing: a state holding "<|im_start|>" or "<think>" produced those
// tokens instead of text, so a guard model could not see the markers it is
// asked about. Upstream breaks such text on four paths (kev, lfm2-d1-omni,
// lfm3-d1, h2o-lightning) and only for some tokens.
//
// Here the caller's text goes through llama_cpp_ex::caller_text_guard (strict:
// user-defined tokens too) before rendering, the rendered prompt gets it back as
// caller spans, and tokenize_pieces keeps every special token out of those
// spans. A request with no special-token text renders and tokenizes exactly as
// upstream.

// A rendered prompt and the spans of it that came from the caller.
struct rendered_prompt {
    std::string text;
    std::vector<std::pair<size_t, size_t>> inputs; // [begin, end), in order

    static rendered_prompt from_pieces(const std::vector<llama_cpp_ex::prompt_piece> & pieces) {
        rendered_prompt out;
        for (const auto & piece : pieces) {
            if (piece.is_input) {
                out.inputs.emplace_back(out.text.size(), out.text.size() + piece.text.size());
            }
            out.text += piece.text;
        }
        return out;
    }

    bool overlaps_input(size_t begin, size_t end) const {
        for (const auto & [a, b] : inputs) {
            if (a < end && begin < b) {
                return true;
            }
        }
        return false;
    }

    // text[begin, end) as template and caller pieces
    std::vector<llama_cpp_ex::prompt_piece> pieces(size_t begin, size_t end) const {
        std::vector<llama_cpp_ex::prompt_piece> out;
        size_t pos = begin;
        for (const auto & [a, b] : inputs) {
            if (b <= begin || a >= end) {
                continue;
            }
            const size_t s = std::max(a, begin);
            const size_t e = std::min(b, end);
            if (s > pos) {
                out.push_back({text.substr(pos, s - pos), false});
            }
            out.push_back({text.substr(s, e - s), true});
            pos = e;
        }
        if (end > pos) {
            out.push_back({text.substr(pos, end - pos), false});
        }
        return out;
    }

    // the [begin, end) ranges string_split(text, sep) gives; caller text holds no separator
    std::vector<std::pair<size_t, size_t>> split(const std::string & sep) const {
        std::vector<std::pair<size_t, size_t>> out;
        size_t start = 0;
        size_t from  = 0;
        for (size_t pos = text.find(sep); pos != std::string::npos; pos = text.find(sep, from)) {
            if (overlaps_input(pos, pos + sep.size())) {
                from = pos + 1;
                continue;
            }
            out.emplace_back(start, pos);
            start = from = pos + sep.size();
        }
        out.emplace_back(start, text.size());
        return out;
    }

    // text[begin, end) starts with prefix, as string_starts_with on that piece
    bool starts_with(size_t begin, size_t end, const std::string & prefix) const {
        return end - begin >= prefix.size() && text.compare(begin, prefix.size(), prefix) == 0;
    }
};

// Every string of a JSON value, keys included, through guard.protect.
json protect_all(const llama_cpp_ex::caller_text_guard & guard, const json & val, const std::string & nonce,
                 std::vector<std::string> & originals, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw too_deep();
    }
    if (val.is_string()) {
        return guard.protect(val.get<std::string>(), nonce, originals);
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(protect_all(guard, item, nonce, originals, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[guard.protect(key, nonce, originals)] = protect_all(guard, item, nonce, originals, depth + 1);
        }
        return out;
    }
    return val;
}

// An option's label is the model's own, the rest of it the caller's.
json protect_options(const llama_cpp_ex::caller_text_guard & guard, const json & options, const std::string & nonce,
                     std::vector<std::string> & originals) {
    if (!options.is_array()) {
        return protect_all(guard, options, nonce, originals);
    }
    json out = json::array();
    for (const auto & opt : options) {
        if (!opt.is_object()) {
            out.push_back(protect_all(guard, opt, nonce, originals));
            continue;
        }
        json po = json::object();
        for (const auto & [key, val] : opt.items()) {
            po[key] = key == "label" ? val : protect_all(guard, val, nonce, originals);
        }
        out.push_back(po);
    }
    return out;
}

// The template input with the caller's text guarded. The keys this port adds
// itself are left as they are.
json protect_request(const llama_cpp_ex::caller_text_guard & guard, const json & inp, const std::string & nonce,
                     std::vector<std::string> & originals) {
    json out = json::object();
    for (const auto & [key, val] : inp.items()) {
        if (key == "type" || key == "audio" || key == "images" || key == "sep" ||
            key == "mark_state" || key == "mark_question" || key == "mark_option") {
            out[key] = val;
        } else if (key == "options") {
            out[key] = protect_options(guard, val, nonce, originals);
        } else if (key == "questions" && val.is_array()) {
            json questions = json::array();
            for (const auto & q : val) {
                json pq = json::object();
                for (const auto & [qk, qv] : q.items()) {
                    if (qk == "type") {
                        pq[qk] = qv;
                    } else if (qk == "options") {
                        pq[qk] = protect_options(guard, qv, nonce, originals);
                    } else {
                        pq[qk] = protect_all(guard, qv, nonce, originals);
                    }
                }
                questions.push_back(pq);
            }
            out[key] = questions;
        } else {
            out[key] = protect_all(guard, val, nonce, originals);
        }
    }
    return out;
}

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
    return type == "laya" || type == "kev" || type == "clef" || type == "lfm2-d1-omni" || type == "vllm-decision2";
}

// Mirrors server_decision_context::can_share_prompt.
bool type_shares_prompt(const std::string & type) {
    return type == "openjev" || type == "granite-switch" || type == "lev" || type == "kev" || type == "nimble" ||
           type == "pplx-decider" || type == "vllm-decision2" || type == "lfm2-d1" || type == "h2o-lightning";
}

struct Engine::Impl {
    llama_context     * ctx   = nullptr;
    const llama_model * model = nullptr;

    common_decision_type type = COMMON_DECISION_TYPE_NONE;
    std::string          type_name;

    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    std::map<size_t, std::vector<float>> score_bias; // added to the scores of a score question, by number of levels
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   noul_criteria_order = false; // noul options are in the order of the criteria keys, if given
    bool   choice_sorted   = false; // choice options are in the order of their keys
    bool   strict_payload  = false; // state, instructions and descriptions must be text, an object or an array
    bool   canonical_legend = false; // non-string score descriptions are returned as canonical JSON text

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER, VLLM_D3, H2O_LIGHTNING
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    // H2O_LIGHTNING, LFM3_D1, LFM3_D1_OMNI
    std::vector<std::string> quoted_spellings; // texts of the special tokens, quoted in the input, longest first
    std::string              quoted_first;     // the first bytes of these texts

    common_batch batch;

    // ours: the caller's text may not produce special tokens (see caller_text_guard)
    std::unique_ptr<llama_cpp_ex::special_tokens> specials;
    std::unique_ptr<llama_cpp_ex::caller_text_guard> guard;

    explicit Impl(llama_context * c) : ctx(c), model(llama_get_model(c)), batch(c) {
        init();
    }

    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_DECISION2:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
                return true;
            default:
                return false;
        }
    }

    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    bool is_lfm3_d1() const {
        return type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI;
    }

    bool reads_embeddings() const {
        return type == COMMON_DECISION_TYPE_LAYA || type == COMMON_DECISION_TYPE_KEV || type == COMMON_DECISION_TYPE_CLEF ||
               type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || type == COMMON_DECISION_TYPE_VLLM_DECISION2;
    }

    void init() { // server_decision_context::init
        const common_decision_type model_type = common_get_decision_type(model);
        if (model_type == COMMON_DECISION_TYPE_NONE) {
            throw std::runtime_error("the model is not a decision model");
        }

        const std::string prefix = meta_str(model, "general.architecture") + ".decision.";
        type_name = meta_str(model, prefix + "type");

        vocab = llama_model_get_vocab(model);
        specials = std::make_unique<llama_cpp_ex::special_tokens>(vocab);
        guard    = std::make_unique<llama_cpp_ex::caller_text_guard>(*specials, llama_cpp_ex::input_policy::strict);

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

        // "<n_levels>.<level>" -> offset
        const std::string prefix_bias = prefix + "score_bias.";
        for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
            char key[256];
            char val[64];
            if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_bias)) {
                continue;
            }
            if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
                continue;
            }
            const auto parts = string_split<std::string>(key + prefix_bias.size(), '.');
            const size_t n     = parts.size() == 2 ? std::strtoul(parts[0].c_str(), nullptr, 10) : 0;
            const size_t level = parts.size() == 2 ? std::strtoul(parts[1].c_str(), nullptr, 10) : 0;
            if (n < 2 || n > 255 || level >= n) {
                throw std::runtime_error(string_format("invalid decision score bias: %s", key));
            }
            auto & bias = score_bias[n];
            bias.resize(n, 0.0f);
            bias[level] = std::strtof(val, nullptr);
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
        } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_GRANITE_SWITCH || model_type == COMMON_DECISION_TYPE_NIMBLE || model_type == COMMON_DECISION_TYPE_PPLX_DECIDER || model_type == COMMON_DECISION_TYPE_VLLM_D3) {
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
            if (model_type == COMMON_DECISION_TYPE_VLLM_D3) {
                canonical_legend = true;
            }
            if (model_type == COMMON_DECISION_TYPE_GRANITE_SWITCH) {
                noul_true_first = true;
            }
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
        } else if (model_type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
            n_options_max       = 255;
            noul_criteria_order = true;
            strict_payload      = true;
            canonical_legend    = true;
        } else if (model_type == COMMON_DECISION_TYPE_LFM2_D1 || model_type == COMMON_DECISION_TYPE_LFM3_D1 || model_type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
            n_options_max   = 255;
            noul_true_first = true;
            if (model_type != COMMON_DECISION_TYPE_LFM2_D1) {
                // the texts of the special tokens are broken in the input
                init_quoted_spellings(LLAMA_TOKEN_ATTR_CONTROL, 2);
            }
        } else if (model_type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
            token_marker = llama_vocab_mask(vocab);
            if (token_marker == LLAMA_TOKEN_NULL) {
                throw std::runtime_error("decision model has no mask token");
            }
            n_options_max = 255;
        } else if (model_type == COMMON_DECISION_TYPE_H2O_LIGHTNING) {
            // the answer is read at the token of a label, its text is given to the template
            for (uint32_t i = 0; llama_model_decision_label(model, i) != nullptr; i++) {
                const std::string text = llama_model_decision_label(model, i);
                const auto toks = common_tokenize(vocab, text, false, false);
                if (toks.size() != 1 || std::find(labels.begin(), labels.end(), toks[0]) != labels.end()) {
                    throw std::runtime_error(string_format("decision label \"%s\" is not a single distinct token", text.c_str()));
                }
                labels.push_back(toks[0]);
                label_texts.push_back(text);
            }
            if (labels.empty()) {
                throw std::runtime_error("decision model has no labels");
            }
            // the texts of the control and user-defined tokens are quoted in the input, so that they stay text
            init_quoted_spellings(LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED, 1);
            n_options_max   = labels.size();
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

    // the texts of the tokens with these attributes, quoted_spellings and quoted_first
    void init_quoted_spellings(uint32_t attrs, size_t min_size) { // server_decision_context::init_quoted_spellings
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
            if (llama_vocab_get_attr(vocab, id) & attrs) {
                const std::string text = llama_vocab_get_text(vocab, id);
                if (text.size() >= min_size) {
                    quoted_spellings.push_back(text);
                }
            }
        }
        std::stable_sort(quoted_spellings.begin(), quoted_spellings.end(), [](const std::string & a, const std::string & b) {
            return a.size() > b.size();
        });
        for (const auto & spelling : quoted_spellings) {
            if (quoted_first.find(spelling[0]) == std::string::npos) {
                quoted_first += spelling[0];
            }
        }
    }

    //
    // request parsing
    //

    std::vector<decision_question> parse_questions(const json & body) const { // server_decision_context::parse_questions
        // lfm2-d1, lfm2-d1-omni, lfm3-d1 and vllm-d3 accept a null state (for images only upstream; here it just asks about nothing but the questions)
        const bool null_state = type == COMMON_DECISION_TYPE_LFM2_D1 || type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || type == COMMON_DECISION_TYPE_VLLM_D3 || type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI;
        if (!body.contains("state") || (body.at("state").is_null() && !null_state)) {
            throw std::invalid_argument("\"state\" must be provided");
        }
        if (strict_payload && !is_payload(body.at("state"))) {
            throw std::invalid_argument("\"state\" must be text, an object or an array");
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
            // the vllm-d3 prompt has a default question
            // h2o-lightning: the instructions can be left out, the template has a default, and the type defaults to choice
            const bool is_h2o = type == COMMON_DECISION_TYPE_H2O_LIGHTNING;
            const json instructions = q.contains("instructions") ? q.at("instructions") : (is_h2o ? json(std::string()) : json());
            if (instructions.is_null() && type != COMMON_DECISION_TYPE_VLLM_D3 && !is_h2o) {
                throw err("\"instructions\" must be provided");
            }
            if (strict_payload && (!is_payload(instructions) || instructions == "")) {
                throw err("\"instructions\" must be non-empty text, an object or an array");
            }
            if (type == COMMON_DECISION_TYPE_LFM3_D1 && is_spoken(instructions)) {
                throw err("spoken instructions need a model with audio input");
            }
            // upstream's fill_task loads the clip of a spoken question for every type, through libmtmd
            if (is_spoken(instructions)) {
                throw err("audio input is not supported: LlamaCppEx is built without libmtmd");
            }

            decision_question question;
            question.id           = id;
            question.instructions = instructions;

            // server-common's json_value(q, "type", default): the default when the key is missing, null, or not a string
            const std::string type_name = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : std::string(is_h2o ? "choice" : "");
            const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

            if (type_name == "choice") {
                question.type = QUESTION_CHOICE;
                if (!criteria.is_object() || criteria.empty()) {
                    throw err("\"criteria\" must be a non-empty object");
                }
                for (const auto & [key, description] : criteria.items()) {
                    if (strict_payload && (key.empty() || !(description.is_null() || is_payload(description)))) {
                        throw err("option keys must be non-empty, descriptions must be null, text, an object or an array");
                    }
                    question.options.push_back({key, description});
                }
                if (strict_payload && question.options.size() < 2) {
                    throw err("\"criteria\" must have at least 2 options");
                }
                if (choice_sorted) {
                    std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                        return a.key < b.key;
                    });
                }
            } else if (type_name == "score") {
                question.type = QUESTION_SCORE;
                // h2o-lightning: from 1 level, the maximum is n_options_max
                if (is_h2o && (!criteria.is_array() || criteria.empty())) {
                    throw err("\"criteria\" must be a non-empty array");
                }
                if (!is_h2o && (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)) {
                    throw err("\"criteria\" must be an array of 2 to 10 levels");
                }
                for (size_t i = 0; i < criteria.size(); i++) {
                    if (strict_payload && !is_payload(criteria.at(i))) {
                        throw err("level descriptions must be text, an object or an array");
                    }
                    question.options.push_back({std::to_string(i), criteria.at(i)});
                }
            } else if (type_name == "noul") {
                question.type = QUESTION_NOUL;
                if (!criteria.is_null() && !criteria.is_object()) {
                    throw err("\"criteria\" must be an object");
                }
                if (strict_payload && criteria.is_object()) {
                    for (const auto & [key, description] : criteria.items()) {
                        if ((key != "false" && key != "true") || !is_payload(description)) {
                            throw err("\"criteria\" may only describe \"false\" and \"true\" with text, an object or an array");
                        }
                    }
                }
                for (const char * key : {"false", "true"}) {
                    json description;
                    if (criteria.is_object() && criteria.contains(key)) {
                        description = criteria.at(key);
                    } else if (criteria.is_object() && (type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || is_h2o)) {
                        // lfm2-d1-omni and h2o-lightning also read the descriptions under "no" and "yes"
                        const char * alias = std::string(key) == "true" ? "yes" : "no";
                        description = criteria.contains(alias) ? criteria.at(alias) : json();
                    }
                    question.options.push_back({key, description});
                }
                if (noul_true_first || (noul_criteria_order && criteria.is_object() && criteria.size() == 2 && criteria.begin().key() == "true")) {
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

    // server_decision_context::parse_state, without media loading: this build
    // has no libmtmd, so any image or audio clip is an error instead of being
    // taken out of the state. A state without media comes back unchanged, as
    // upstream.
    json parse_state(const json & body) const {
        auto reject = [](const char * kind) {
            return std::invalid_argument(std::string(kind) + " input is not supported: LlamaCppEx is built without libmtmd");
        };
        if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
            throw std::invalid_argument("\"videos\" is not supported");
        }
        // "images" is an alias of "files"
        for (const char * key : {"files", "images"}) {
            if (!body.contains(key) || body.at(key).is_null()) {
                continue;
            }
            if (!body.at(key).is_array()) {
                throw std::invalid_argument(string_format("\"%s\" must be an array", key));
            }
            if (!body.at(key).empty()) {
                throw reject("image");
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
                    throw reject("image");
                }
                if (part.is_object() && json_string(part, "type") == "input_audio" && part.contains("input_audio")) {
                    throw reject("audio");
                }
                // lfm3-d1 and lfm3-d1-omni also take videos
                if (is_lfm3_d1() && part.is_object() && json_string(part, "type") == "video_url" && part.contains("video_url")) {
                    throw reject("video");
                }
            }
        }
        return state;
    }

    // a state longer than max_state_tokens keeps its head and its tail around a "..." line
    std::string h2o_cut_state(const std::string & text) const { // server_decision_context::h2o_cut_state
        // a byte-level BPE never has more tokens than bytes
        const size_t max_state_tokens = DECISION_H2O_MAX_STATE_TOKENS;
        if (text.size() <= max_state_tokens) {
            return text;
        }
        const llama_tokens tokens = tokenize_caller(text);
        if (tokens.size() <= max_state_tokens) {
            return text;
        }
        // each half is the text of its token pieces, a cut inside a character gives U+FFFD as in a JSON response of the server
        const size_t half = max_state_tokens / 2;
        auto text_of = [&](llama_tokens::const_iterator begin, llama_tokens::const_iterator end) {
            std::string out;
            for (auto it = begin; it != end; ++it) {
                out += common_token_to_piece(vocab, *it, true);
            }
            return json::parse(json::make(out).dump_safe()).get<std::string>();
        };
        return text_of(tokens.begin(), tokens.begin() + half) + "\n...\n" + text_of(tokens.end() - half, tokens.end());
    }

    // the special token texts written in the input are quoted with JSON escapes, so they stay text
    std::string h2o_quote(const std::string & text) const { // server_decision_context::h2o_quote
        std::string out;
        for (size_t i = 0; i < text.size();) {
            bool is_quoted = false;
            if (quoted_first.find(text[i]) == std::string::npos) {
                out += text[i++];
                continue;
            }
            for (const auto & spelling : quoted_spellings) {
                if (text.compare(i, spelling.size(), spelling) == 0) {
                    std::string quoted = spelling;
                    string_replace_all(quoted, "<", "\\u003c");
                    string_replace_all(quoted, ">", "\\u003e");
                    out += quoted;
                    i += spelling.size();
                    is_quoted = true;
                    break;
                }
            }
            if (!is_quoted) {
                out += text[i++];
            }
        }
        return out;
    }

    // LFM3_D1: a zero-width space after the first character of each special token text (d1 prompt.py: plain)
    std::string d1_break(const std::string & text) const { // server_decision_context::d1_break
        static const std::string zwsp = "\xE2\x80\x8B";
        std::string cur = text;
        // again for a match that overlapped the one broken before it
        for (bool changed = true; changed;) {
            changed = false;
            std::string out;
            for (size_t i = 0; i < cur.size();) {
                const std::string * hit = nullptr;
                if (quoted_first.find(cur[i]) != std::string::npos) {
                    for (const auto & spelling : quoted_spellings) {
                        if (cur.compare(i, spelling.size(), spelling) == 0) {
                            hit = &spelling;
                            break;
                        }
                    }
                }
                if (hit == nullptr) {
                    out += cur[i++];
                    continue;
                }
                // after the first character, its UTF-8 bytes
                const unsigned char c = (*hit)[0];
                const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
                out += hit->substr(0, n) + zwsp + hit->substr(n);
                i += hit->size();
                changed = true;
            }
            cur = std::move(out);
        }
        return cur;
    }

    // the state as the model reads it, done once per request; returns the state for most models
    json prepare_state(const json & state) const { // server_decision_context::prepare_state
        if (type != COMMON_DECISION_TYPE_H2O_LIGHTNING) {
            return state;
        }
        // done once per request: compact JSON, the cut, the quoting
        return h2o_quote(h2o_cut_state(state.is_string() ? state.get<std::string>() : py_dumps(state, -1, ",", ":", false)));
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
            } else if (is_lfm3_d1()) {
                // A..Z, then AA, AB..
                codes.push_back(i < 26 ? std::string(1, 'A' + i) : std::string{(char) ('A' + (i - 26) / 26), (char) ('A' + (i - 26) % 26)});
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
        if (type == COMMON_DECISION_TYPE_LFM2_D1 || is_lfm3_d1()) {
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

    // H2O_LIGHTNING: the input of the template, its texts are rendered as the reference shim does
    json render_h2o(const json & state, const decision_question & question) const { // server_decision_context::render_h2o
        // the instructions are JSON with an indent of 1, the descriptions are default JSON, the state is from prepare_state()
        const std::string instructions = py_lstrip(question.instructions.is_string() ?
                question.instructions.get<std::string>() : py_dumps(question.instructions, 1, ",", ": ", true));

        json options = json::array();
        for (size_t i = 0; i < question.options.size(); i++) {
            const auto & opt = question.options[i];
            json description; // null if not provided, the template has a default, except a score level that is shown as JSON null
            if (question.type == QUESTION_SCORE && opt.description.is_null()) {
                description = "null";
            } else if (!opt.description.is_null()) {
                description = h2o_quote(opt.description.is_string() ? opt.description.get<std::string>() : py_dumps(opt.description, -1, ", ", ": ", false));
            }
            const std::string key = h2o_quote(opt.key);

            // the user turn is stripped like Python's str.strip(), the template only trims ASCII whitespace
            // so the text that ends the turn is stripped here, a space keeps a description that is only whitespace
            if (i + 1 == question.options.size()) {
                const bool is_key = question.type == QUESTION_CHOICE && description.is_null();
                if (is_key || description.is_string()) {
                    const std::string text = is_key ? key : description.get<std::string>();
                    const std::string stripped = py_rstrip(text);
                    if (stripped != text) {
                        description = stripped.empty() ? " " : stripped;
                    }
                }
            }

            json option = json::object();
            option["label"]       = label_texts[i];
            option["key"]         = key;
            option["description"] = description;
            options.push_back(option);
        }

        json inp = json::object();
        inp["type"]         = question_type_name(question.type);
        inp["state"]        = state;
        inp["instructions"] = h2o_quote(instructions);
        inp["options"]      = options;
        return inp;
    }

    // ours: renders with the caller's text guarded, see caller_text_guard
    rendered_prompt run_template(const json & inp) const {
        std::vector<std::string> originals;
        const std::string nonce = llama_cpp_ex::random_nonce();
        const json guarded = protect_request(*guard, inp, nonce, originals);

        jinja::context jctx(tmpl->source());
        jinja::global_from_json(jctx, guarded, false);
        jinja::runtime runtime(jctx);
        const jinja::value results = runtime.execute(tmpl->prog);
        std::string text = jinja::runtime::gather_string_parts(results)->as_string().str();
        if (originals.empty()) {
            return {std::move(text), {}};
        }
        return rendered_prompt::from_pieces(llama_cpp_ex::restore_pieces({{std::move(text), false}}, nonce, originals));
    }

    // ours: as common_tokenize(vocab, text, false, true), except that no
    // special token comes from the caller's spans
    llama_tokens tokenize_prompt(const rendered_prompt & prompt, size_t begin, size_t end) const {
        return llama_cpp_ex::tokenize_pieces(vocab, *specials, prompt.pieces(begin, end), false,
                                             llama_cpp_ex::input_policy::strict);
    }

    // ours: caller text alone, tokenized so that it produces no special token
    llama_tokens tokenize_caller(const std::string & text) const {
        return llama_cpp_ex::tokenize_pieces(vocab, *specials, {{text, true}}, false,
                                             llama_cpp_ex::input_policy::strict);
    }

    // server_decision_context::render with n_images == 0 (so is_audio == false)
    rendered_prompt render(
            const json & state,
            const std::vector<decision_question> & questions,
            const decision_question & question,
            size_t variant) const {
        // the template is given raw JSON values, it serializes the ones that are not strings
        // h2o-lightning: the template is given texts
        json inp = type == COMMON_DECISION_TYPE_H2O_LIGHTNING ? render_h2o(state, question) : json{
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

        // spoken questions are refused in parse_questions, so upstream's is_spoken branches never apply here
        if (is_lfm3_d1()) {
            inp = escape_json(inp, [this](const std::string & text) { return d1_break(text); });
        }

        if (type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
            inp = replace_text(escape_json(inp, escape_special_tokens), D1OMNI_MARKER, "<<d1omni ");
            inp["audio"]         = false;
            inp["sep"]           = D1OMNI_SEP;
            inp["mark_state"]    = D1OMNI_MARK_STATE;
            inp["mark_question"] = D1OMNI_MARK_QUESTION;
            inp["mark_option"]   = D1OMNI_MARK_OPTION;
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
        // each question has its own prompt, with the spans that the head reads
        if (type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
            return fill_task_joint(state, { question });
        }
        if (type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
            return fill_task_d1omni(state, questions, question, variant);
        }

        decision_task task;
        const rendered_prompt prompt = render(state, questions, question, variant);

        if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_GRANITE_SWITCH || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER || type == COMMON_DECISION_TYPE_VLLM_D3 ||
            type == COMMON_DECISION_TYPE_H2O_LIGHTNING) {
            // lev reads the ratings of a noul question at its first labels, not at the digits
            task.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
        }
        if (type == COMMON_DECISION_TYPE_LFM2_D1 || is_lfm3_d1()) {
            std::vector<std::string>  texts;
            std::vector<llama_tokens> groups;
            d1_labels(question, texts, groups);
            for (const auto & group : groups) {
                task.labels.insert(task.labels.end(), group.begin(), group.end());
                task.label_groups.push_back(group.size());
            }
        }

        llama_tokens tokens = tokenize_prompt(prompt, 0, prompt.text.size());
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

    // server_decision_context::fill_task_d1omni without files: no media
    // prefix, so the text gets the whole D1OMNI_MAX_TOKENS budget.
    // each piece is cut to its budget as the model was trained (d1-omni prompt.py: encode), the state gets the room that is left
    decision_task fill_task_d1omni(
            const json & state,
            const std::vector<decision_question> & questions,
            const decision_question & question,
            size_t variant) const {
        decision_task task;
        const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

        // with no files the media markers before the first separator are empty
        const rendered_prompt prompt = render(state, questions, question, variant);
        const auto pieces = prompt.split(D1OMNI_SEP);
        if (pieces.empty() || pieces[0].first != pieces[0].second) {
            throw invalid;
        }

        const size_t n_max = D1OMNI_MAX_TOKENS;

        // the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
        const int64_t n_options      = question.options.size();
        const int64_t n_budget       = std::max<int64_t>(96, std::min<int64_t>(24 * n_options + 32, n_max / 2));
        const int64_t n_option_max   = std::max<int64_t>(2, (n_budget - 3 * n_options) / n_options);
        const int64_t n_question_max = std::max<int64_t>(16, n_budget);

        llama_tokens head;  // before the state
        llama_tokens body;  // the state
        llama_tokens tail;  // after the state
        bool has_state = false;
        for (size_t i_piece = 1; i_piece < pieces.size(); i_piece++) {
            auto [begin, end] = pieces[i_piece];
            int64_t n_piece_max = -1;
            bool is_state = false;
            if (prompt.starts_with(begin, end, D1OMNI_MARK_STATE)) {
                begin   += D1OMNI_MARK_STATE.size();
                is_state = true;
            } else if (prompt.starts_with(begin, end, D1OMNI_MARK_QUESTION)) {
                begin += D1OMNI_MARK_QUESTION.size();
                n_piece_max = n_question_max;
            } else if (prompt.starts_with(begin, end, D1OMNI_MARK_OPTION)) {
                begin += D1OMNI_MARK_OPTION.size();
                n_piece_max = n_option_max;
            }

            llama_tokens tokens = tokenize_prompt(prompt, begin, end);
            if (n_piece_max >= 0 && (int64_t) tokens.size() > n_piece_max) {
                tokens.resize(n_piece_max);
            }

            if (is_state) {
                if (has_state) {
                    throw invalid;
                }
                body      = std::move(tokens);
                has_state = true;
            } else {
                llama_tokens & dst = has_state ? tail : head;
                dst.insert(dst.end(), tokens.begin(), tokens.end());
            }
        }
        if (!has_state) {
            throw invalid;
        }

        const size_t n_room = n_max - std::min(n_max, head.size() + tail.size());
        body.resize(std::min(body.size(), n_room));

        llama_tokens tokens = std::move(head);
        tokens.insert(tokens.end(), body.begin(), body.end());
        tokens.insert(tokens.end(), tail.begin(), tail.end());
        tokens.resize(std::min(tokens.size(), n_max));

        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == token_marker) {
                task.markers.push_back(i);
            }
        }
        if ((int64_t) task.markers.size() != n_options) {
            throw std::invalid_argument("the options do not fit in the context");
        }

        // the output has one score per question type
        task.column = question.type;
        task.tokens = std::move(tokens);
        return task;
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

        const rendered_prompt prompt = run_template(inp);

        const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

        // the model was trained with the pieces tokenized one by one
        const auto pieces = prompt.split(CLEF_SEP);

        size_t i_question = 0;
        for (size_t i_piece = 0; i_piece < pieces.size(); i_piece++) {
            auto [begin, end] = pieces[i_piece];
            int32_t order = LLAMA_DECISION_ORDER_NONE;
            if (prompt.starts_with(begin, end, CLEF_MARK_QUESTION)) {
                begin += CLEF_MARK_QUESTION.size();
                if (i_question >= questions.size()) {
                    throw invalid;
                }
                switch (questions[i_question++].type) {
                    case QUESTION_NOUL:   order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
                    case QUESTION_CHOICE: order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
                    case QUESTION_SCORE:  order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
                }
            } else if (prompt.starts_with(begin, end, CLEF_MARK_OPTION)) {
                begin += CLEF_MARK_OPTION.size();
                order = LLAMA_DECISION_ORDER_OPTION;
                task.n_scores++;
            }

            const llama_tokens piece_tokens = tokenize_prompt(prompt, begin, end);
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

    // server_decision_context::format_answer with has_media == false, so
    // lfm2-d1-omni keeps its calibrated temperatures (upstream uses 1.0 for
    // media) and a model's "image" temperature is never used
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
            // a joint head returns NaN if it could not use the decision order, +inf would make the softmax NaN
            if (std::any_of(s.begin(), s.end(), [](float v) { return std::isnan(v) || v == INFINITY; })) {
                throw std::runtime_error("the model could not evaluate the decision");
            }
            std::vector<double> logits(s.begin(), s.end());
            if (question.type == QUESTION_SCORE && score_bias.count(n) > 0) {
                for (size_t i = 0; i < n; i++) {
                    logits[i] += score_bias.at(n)[i];
                }
            }
            const double logit_max = *std::max_element(logits.begin(), logits.end());
            std::vector<double> p(n);
            double sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                p[i] = std::exp((logits[i] - logit_max) / temperature);
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
            answer["confidence"]    = get_confidence(question.type, probs);
        } else {
            double expected = 0.0;
            json legend = json::object();
            for (size_t i = 0; i < n; i++) {
                expected += i * probs[i];
                const json & description = question.options[i].description;
                legend[question.options[i].key] = canonical_legend && !description.is_string() ? json(sort_keys(description).dump()) : description;
            }
            // lfm3-d1: the median level, the first whose cumulative probability reaches one half (d1 api.py)
            if (is_lfm3_d1()) {
                double mass = 0.0;
                size_t level = n - 1;
                for (size_t i = 0; i < n && mass < 0.5; i++) {
                    mass += probs[i];
                    level = i;
                }
                expected = (double) level;
            }
            answer["score"]         = expected;
            answer["legend"]        = legend;
            answer["probabilities"] = probabilities;
            answer["confidence"]    = get_confidence(question.type, probs);
        }
        return answer;
    }

    double get_confidence(question_type qtype, const std::vector<double> & probs) const { // server_decision_context::get_decision_confidence
        if (type == COMMON_DECISION_TYPE_VLLM_DECISION2 || type == COMMON_DECISION_TYPE_VLLM_D3) {
            return confidence_entropy(probs);
        }
        // lfm3-d1: the probability of the answer (d1 api.py)
        if (is_lfm3_d1()) {
            return *std::max_element(probs.begin(), probs.end());
        }
        switch (qtype) {
            case QUESTION_CHOICE: return confidence_choice(probs);
            case QUESTION_SCORE:  return confidence_score(probs);
            case QUESTION_NOUL:   break;
        }
        throw std::runtime_error("no confidence for this question type");
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
    // whole prompt for the models that read every position (laya and
    // lfm2-d1-omni score markers anywhere in it; clef's joint head reads the
    // whole batch).
    int32_t first_output(const decision_task & task) const {
        if (type == COMMON_DECISION_TYPE_LAYA || type == COMMON_DECISION_TYPE_CLEF || type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
            return 0;
        }
        if (!task.labels.empty()) {
            return (int32_t) task.tokens.size() - 1;
        }
        return task.pos_first();
    }

    // The position from which the prompt must be in the last batch: the first
    // output, or the whole prompt over a non-causal context (laya,
    // lfm2-d1-omni, vllm-d3, lfm3-d1), whose tokens only attend within their batch
    // (server_slot::can_split).
    int32_t first_batched(const decision_task & task) const {
        return llama_get_causal_attn(ctx) ? first_output(task) : 0;
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
        if (first_batched(task) == 0) {
            const int32_t n_max = std::min(n_batch, n_ubatch);
            if (n_tokens > n_max) {
                throw std::invalid_argument(string_format(
                    "the prompt (%d tokens) must be evaluated in one batch, increase n_batch and n_ubatch (current: %d)", n_tokens, n_max));
            }
            return;
        }
        const int32_t n_read = n_tokens - first_batched(task);
        if (n_read > n_batch) {
            throw std::invalid_argument(string_format(
                "the question and its options (%d tokens) are too large to process. "
                "increase the batch size (current batch size: %d)", n_read, n_batch));
        }
    }

    // Decodes tokens [from, to) of `task` on `seq`, at positions equal to their
    // index. Every position from `batch_from` on lands in the last batch
    // (check_fits guarantees they fit in one). Every position from `out_from`
    // (>= batch_from) on is an output, and every position is for the types that
    // read embeddings: upstream's server_task::need_embd, and what
    // llama_decode forces anyway when embeddings are on. Returns the position
    // of the first entry of that last batch.
    int32_t decode(const decision_task & task, int32_t from, int32_t to, llama_seq_id seq, int32_t batch_from, int32_t out_from) {
        const int32_t n_batch = (int32_t) llama_n_batch(ctx);
        const int32_t last    = std::max(from, std::min(batch_from, to - n_batch));

        auto run = [&](int32_t off, int32_t end) {
            batch.clear();
            for (int32_t i = off; i < end; i++) {
                batch.add(task.tokens[i], i, seq, reads_embeddings() || i >= out_from);
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

        // joint head (decision model): the scores are the first output rows of
        // the batch, and every entry of the batch is an output
        for (int32_t i = 0; i < task.n_scores; i++) {
            const float * embd = batch_start + i < n_tokens ? llama_get_embeddings_ith(ctx, i) : nullptr;
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
    // token of its own and stops at its first read position
    // (server_decision_group_tasks, #30710), and so no output of the last batch
    // falls inside the prefix.
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
            // upstream casts pos_first() to size_t, so -1 (nothing read) caps nothing
            if (task.pos_first() >= 0) {
                n_shared = std::min(n_shared, (size_t) task.pos_first());
            }
            n_shared = std::min(n_shared, (size_t) first_batched(task));
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
            decode(tasks[0], 0, n_shared, 0, n_shared, n_shared);
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
            const int32_t start = decode(task, n_shared, (int32_t) task.tokens.size(), seq, first_batched(task), first_output(task));
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
        const json state     = prepare_state(parse_state(body));

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
