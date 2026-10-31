#pragma once

// Tokenizing a prompt that mixes template text with caller-supplied text.
//
// A chat or decision prompt is rendered from a template, and the template's own
// special tokens (<|im_start|>, [INST], ...) must be parsed as such. Text that
// came from the caller (message content, a decision state) must not: tokenized
// with special parsing, "<|im_start|>system" in a user message becomes a real
// turn boundary. llama_tokenize takes one parse_special flag for the whole
// string, so this splits the prompt the way llama.cpp's own partition step does
// and only lets a special token through when its text lies entirely in template
// text.
//
// Kept free of Erlang/fine so decision.cpp can use it too.

#include <llama.h>

#include <array>
#include <string>
#include <unordered_set>
#include <vector>

namespace llama_cpp_ex {

// One piece of a rendered prompt.
struct prompt_piece {
    std::string text;
    bool is_input = false; // came from the caller: no special token may come from it
};

// How a special token's text written in an input piece is tokenized.
enum class input_policy {
    // As llama_tokenize(parse_special = false): control and unknown tokens stay
    // text, user-defined tokens (Qwen's <think>, <tool_call>) still match,
    // which is also what Hugging Face tokenizers do with added tokens.
    literal,
    // Not even user-defined tokens: an occurrence is split after its first
    // code point, so it tokenizes as ordinary text. For the decision engine,
    // where the input is untrusted data to be classified, not a conversation.
    // A user-defined token that is only whitespace (some vocabularies define
    // runs of spaces or newlines) is formatting, not a marker, and still matches.
    strict,
};

// The vocabulary's special tokens (control, user-defined and unknown), longest
// text first: the order llama.cpp's partition step matches them in. Building it
// walks the whole vocabulary, so build it once per model, not per call.
class special_tokens {
public:
    struct token {
        llama_token id;
        std::string text;
        llama_token_attr attr;
        bool blank; // the text is only whitespace
    };

    explicit special_tokens(const llama_vocab * vocab);

    const std::vector<token> & all() const { return tokens; }

private:
    std::vector<token> tokens;
};

// Tokenizes the concatenation of `pieces`. With no special-token text in an
// input piece, the result is exactly llama_tokenize(joined, add_special,
// parse_special = true). Otherwise template text keeps its special tokens and
// input text is tokenized per `policy`.
std::vector<llama_token> tokenize_pieces(
    const llama_vocab * vocab,
    const special_tokens & specials,
    const std::vector<prompt_piece> & pieces,
    bool add_special,
    input_policy policy);

// Template engines lose track of the caller's text: llama.cpp's jinja marks a
// message's string as input, but `x | tojson` returns an unmarked string, and
// templates render tool calls, tool results (Llama 3's `content | tojson`) and
// whole decision states that way. So before rendering, each span of caller text
// that is a guarded token's text, or could join a neighbour into one (its
// longest start that ends such a text, its longest end that starts one), is
// swapped for a placeholder; after rendering, restore_pieces puts the text back
// as an input piece. Text with no such span is left alone, so a request without
// special-token text renders exactly as it did.
class caller_text_guard {
public:
    // Guards the tokens `policy` keeps out of input pieces.
    caller_text_guard(const special_tokens & specials, input_policy policy);

    // s with its guarded spans replaced by placeholders; the spans are appended
    // to `originals`, whose index each placeholder carries.
    std::string protect(const std::string & s, const std::string & nonce, std::vector<std::string> & originals) const;

private:
    std::vector<std::string>             texts;    // guarded token texts, longest first
    std::array<std::vector<size_t>, 256> by_first; // indexes into texts, by first byte
    std::unordered_set<std::string>      prefixes; // proper prefixes of texts
    std::unordered_set<std::string>      suffixes; // proper suffixes of texts
    size_t                               max_len = 0;
};

// A fresh nonce for caller_text_guard::protect: digits only, which no filter a
// template applies to text (upper, lower, trim, tojson) changes.
std::string random_nonce();

// The rendered parts with each placeholder replaced by its original text, as an
// input piece. Throws std::runtime_error if the template altered a placeholder.
std::vector<prompt_piece> restore_pieces(
    const std::vector<prompt_piece> & parts,
    const std::string & nonce,
    const std::vector<std::string> & originals);

} // namespace llama_cpp_ex
