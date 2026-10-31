#include "input_tokenize.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <iterator>
#include <list>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace llama_cpp_ex {

namespace {

constexpr int ATTR_CONTROL      = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN;
constexpr int ATTR_USER_DEFINED = LLAMA_TOKEN_ATTR_USER_DEFINED;

// The tokens whose text `policy` keeps out of input pieces.
bool is_guarded(const special_tokens::token & tok, input_policy policy) {
    return (tok.attr & ATTR_CONTROL) ||
           (policy == input_policy::strict && (tok.attr & ATTR_USER_DEFINED) && !tok.blank);
}

// llama_tokenize of text[off, off + len), appended to out.
void tokenize_into(std::vector<llama_token> & out, const llama_vocab * vocab, const std::string & text,
                   size_t off, size_t len, bool add_special, bool parse_special) {
    if (len > INT32_MAX) {
        throw std::runtime_error("tokenization failed: text too long");
    }
    const char * p = text.data() + off;
    int32_t n = llama_tokenize(vocab, p, (int32_t) len, nullptr, 0, add_special, parse_special);
    if (n == INT32_MIN) {
        throw std::runtime_error("tokenization failed: too many tokens");
    }
    if (n == 0) {
        return;
    }
    const size_t base = out.size();
    out.resize(base + (size_t) std::abs(n));
    n = llama_tokenize(vocab, p, (int32_t) len, out.data() + base, (int32_t) (out.size() - base), add_special,
                       parse_special);
    if (n < 0) {
        throw std::runtime_error("tokenization failed");
    }
    out.resize(base + (size_t) n);
}

// A fragment of the joined prompt: one special token, or raw text tokenized
// without special parsing. Mirrors llama.cpp's fragment_buffer_variant.
struct fragment {
    llama_token token; // LLAMA_TOKEN_NULL for raw text
    size_t      off;
    size_t      len;
};

} // namespace

special_tokens::special_tokens(const llama_vocab * vocab) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token id = 0; id < n; id++) {
        const llama_token_attr attr = llama_vocab_get_attr(vocab, id);
        if (!(attr & (ATTR_CONTROL | ATTR_USER_DEFINED))) {
            continue;
        }
        const char * text = llama_vocab_get_text(vocab, id);
        if (text == nullptr || *text == '\0') {
            continue;
        }
        const std::string s = text;
        const bool blank = std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
        tokens.push_back({id, s, attr, blank});
    }
    // llama_vocab's cache_special_tokens order. Equal lengths cannot both
    // match at one position, so the order among them does not change a result.
    std::stable_sort(tokens.begin(), tokens.end(),
                     [](const token & a, const token & b) { return a.text.size() > b.text.size(); });
}

std::vector<llama_token> tokenize_pieces(
    const llama_vocab * vocab,
    const special_tokens & specials,
    const std::vector<prompt_piece> & pieces,
    bool add_special,
    input_policy policy)
{
    std::string text;
    std::vector<std::pair<size_t, size_t>> inputs; // [begin, end) of the input pieces, in order
    {
        size_t size = 0;
        for (const auto & piece : pieces) {
            size += piece.text.size();
        }
        text.reserve(size);
    }
    for (const auto & piece : pieces) {
        if (piece.is_input && !piece.text.empty()) {
            inputs.emplace_back(text.size(), text.size() + piece.text.size());
        }
        text += piece.text;
    }

    const auto guarded = [policy](const special_tokens::token & tok) { return is_guarded(tok, policy); };

    const auto overlaps_input = [&inputs](size_t begin, size_t end) {
        // first input range that ends after `begin`
        auto it = std::upper_bound(inputs.begin(), inputs.end(), begin,
                                   [](size_t pos, const std::pair<size_t, size_t> & r) { return pos < r.second; });
        return it != inputs.end() && it->first < end;
    };

    // Fast path: no guarded token's text overlaps an input range, so llama.cpp's
    // own partition cannot produce a special token from input, and tokenizing
    // the joined text is both safe and exactly what it was before input pieces
    // were tracked. A match overlapping [a, b) starts in [a - L + 1, b), so it
    // lies inside the window searched here.
    bool clash = false;
    for (const auto & [a, b] : inputs) {
        for (const auto & tok : specials.all()) {
            if (!guarded(tok)) {
                continue;
            }
            const size_t L     = tok.text.size();
            const size_t start = a >= L - 1 ? a - (L - 1) : 0;
            const size_t end   = std::min(text.size(), b + L - 1);
            if (end - start >= L && std::string_view(text.data() + start, end - start).find(tok.text) != std::string_view::npos) {
                clash = true;
                break;
            }
        }
        if (clash) {
            break;
        }
    }

    std::vector<llama_token> out;
    if (!clash) {
        tokenize_into(out, vocab, text, 0, text.size(), add_special, true);
        return out;
    }

    // Slow path: llama.cpp's tokenizer_st_partition, except that a guarded
    // token is only split out where its text lies entirely in template text.
    std::list<fragment> frags;
    if (!text.empty()) {
        frags.push_back({LLAMA_TOKEN_NULL, 0, text.size()});
    }
    // strict: positions to cut raw text at, so a rejected user-defined token
    // (which llama_tokenize matches even without special parsing) cannot form
    std::vector<size_t> cuts;

    for (const auto & tok : specials.all()) {
        const bool guard = guarded(tok);
        const bool cut   = guard && !(tok.attr & ATTR_CONTROL);
        const size_t L   = tok.text.size();

        for (auto it = frags.begin(); it != frags.end(); ++it) {
            if (it->token != LLAMA_TOKEN_NULL) {
                continue;
            }
            size_t base = it->off;
            size_t len  = it->len;
            size_t from = base;
            while (true) {
                const size_t match = std::string_view(text.data(), base + len).find(tok.text, from);
                if (match == std::string_view::npos) {
                    break;
                }
                if (guard && overlaps_input(match, match + L)) {
                    if (cut) {
                        // after the first code point, so no UTF-8 sequence is split
                        size_t p = match + 1;
                        while (p < match + L && ((unsigned char) text[p] & 0xC0) == 0x80) {
                            p++;
                        }
                        if (p < match + L) {
                            cuts.push_back(p);
                        }
                    }
                    from = match + 1;
                    continue;
                }

                size_t left_len = match - base;
                if (tok.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                    while (left_len > 0 && std::isspace((unsigned char) text[base + left_len - 1])) {
                        left_len--;
                    }
                }
                if (left_len > 0) {
                    frags.insert(it, {LLAMA_TOKEN_NULL, base, left_len});
                }
                frags.insert(it, {tok.id, 0, 0});

                size_t right_off = match + L;
                size_t right_len = base + len - right_off;
                if (tok.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                    while (right_len > 0 && std::isspace((unsigned char) text[right_off])) {
                        right_off++;
                        right_len--;
                    }
                }
                if (right_len == 0) {
                    it = std::prev(frags.erase(it)); // the token just inserted
                    break;
                }
                it->off = base = from = right_off;
                it->len = len = right_len;
            }
        }
    }

    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    const bool wpm = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_WPM;
    if (add_special && (wpm || llama_vocab_get_add_bos(vocab))) {
        out.push_back(llama_vocab_bos(vocab));
    }
    for (const auto & frag : frags) {
        if (frag.token != LLAMA_TOKEN_NULL) {
            out.push_back(frag.token);
            continue;
        }
        size_t pos       = frag.off;
        const size_t end = frag.off + frag.len;
        for (auto c = std::upper_bound(cuts.begin(), cuts.end(), pos); c != cuts.end() && *c < end; ++c) {
            tokenize_into(out, vocab, text, pos, *c - pos, false, false);
            pos = *c;
        }
        tokenize_into(out, vocab, text, pos, end - pos, false, false);
    }
    if (add_special) {
        if (wpm) {
            out.push_back(llama_vocab_sep(vocab));
        } else if (llama_vocab_get_add_eos(vocab)) {
            out.push_back(llama_vocab_eos(vocab));
        }
    }
    return out;
}

caller_text_guard::caller_text_guard(const special_tokens & specials, input_policy policy) {
    for (const auto & tok : specials.all()) {
        if (is_guarded(tok, policy)) {
            texts.push_back(tok.text); // longest first, as specials
        }
    }
    for (size_t i = 0; i < texts.size(); i++) {
        const std::string & t = texts[i];
        max_len = std::max(max_len, t.size());
        by_first[(unsigned char) t[0]].push_back(i);
        for (size_t k = 1; k < t.size(); k++) {
            prefixes.insert(t.substr(0, k));
            suffixes.insert(t.substr(t.size() - k));
        }
    }
}

std::string caller_text_guard::protect(const std::string & s, const std::string & nonce,
                                       std::vector<std::string> & originals) const {
    if (texts.empty()) {
        return s;
    }
    const size_t n = s.size();
    std::vector<std::pair<size_t, size_t>> spans;
    // the longest start of s that ends a token's text
    for (size_t k = std::min(n, max_len - 1); k > 0; k--) {
        if (suffixes.count(s.substr(0, k))) {
            spans.emplace_back(0, k);
            break;
        }
    }
    // each token text in s
    for (size_t i = 0; i < n; i++) {
        for (const size_t idx : by_first[(unsigned char) s[i]]) {
            const std::string & t = texts[idx];
            if (s.compare(i, t.size(), t) == 0) {
                spans.emplace_back(i, i + t.size());
                i += t.size() - 1;
                break;
            }
        }
    }
    // the longest end of s that starts a token's text
    for (size_t k = std::min(n, max_len - 1); k > 0; k--) {
        if (prefixes.count(s.substr(n - k))) {
            spans.emplace_back(n - k, n);
            break;
        }
    }
    if (spans.empty()) {
        return s;
    }

    std::sort(spans.begin(), spans.end());
    std::string out;
    size_t pos = 0;
    for (size_t i = 0; i < spans.size();) {
        const size_t begin = std::max(spans[i].first, pos);
        size_t end = spans[i].second;
        for (i++; i < spans.size() && spans[i].first <= end; i++) {
            end = std::max(end, spans[i].second);
        }
        out.append(s, pos, begin - pos);
        out += "@@" + nonce + "_" + std::to_string(originals.size()) + "@@";
        originals.push_back(s.substr(begin, end - begin));
        pos = end;
    }
    out.append(s, pos, std::string::npos);
    return out;
}

std::string random_nonce() {
    thread_local std::mt19937_64 gen{std::random_device{}()};
    std::uniform_int_distribution<int> digit(0, 9);
    std::string out(24, '0');
    for (auto & c : out) {
        c = (char) ('0' + digit(gen));
    }
    return out;
}

std::vector<prompt_piece> restore_pieces(
    const std::vector<prompt_piece> & parts,
    const std::string & nonce,
    const std::vector<std::string> & originals)
{
    const auto altered = std::runtime_error("the template altered the text of the request");
    const std::string open = "@@" + nonce + "_";

    std::vector<prompt_piece> out;
    const auto emit = [&out](std::string text, bool is_input) {
        if (text.empty()) {
            return;
        }
        if (!out.empty() && out.back().is_input == is_input) {
            out.back().text += text;
        } else {
            out.push_back({std::move(text), is_input});
        }
    };

    for (const auto & part : parts) {
        const std::string & text = part.text;
        size_t done = 0;
        for (size_t pos = text.find(open); pos != std::string::npos; pos = text.find(open, done)) {
            size_t i = pos + open.size();
            size_t idx = 0;
            const size_t digits = i;
            while (i < text.size() && i - digits < 10 && std::isdigit((unsigned char) text[i])) {
                idx = idx * 10 + (text[i++] - '0');
            }
            if (i == digits || text.compare(i, 2, "@@") != 0 || idx >= originals.size()) {
                throw altered;
            }
            emit(text.substr(done, pos - done), part.is_input);
            emit(originals[idx], true);
            done = i + 2;
        }
        emit(text.substr(done), part.is_input);
    }

    // a placeholder cut in two, or rewritten past recognition
    for (const auto & piece : out) {
        if (piece.text.find(nonce) != std::string::npos) {
            throw altered;
        }
    }
    return out;
}

} // namespace llama_cpp_ex
