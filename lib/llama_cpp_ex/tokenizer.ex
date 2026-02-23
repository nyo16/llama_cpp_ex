defmodule LlamaCppEx.Tokenizer do
  @moduledoc """
  Text tokenization and detokenization.
  """

  @doc """
  Encodes text into a list of token IDs.

  ## Options

    * `:add_special` - Add special tokens (BOS/EOS). Defaults to `true`.
    * `:parse_special` - Parse special token text (e.g., `<|im_start|>`). Defaults to `true`.

  """
  @spec encode(LlamaCppEx.Model.t(), String.t(), keyword()) :: {:ok, [integer()]}
  def encode(%LlamaCppEx.Model{ref: ref}, text, opts \\ []) do
    add_special = Keyword.get(opts, :add_special, true)
    parse_special = Keyword.get(opts, :parse_special, true)
    {:ok, LlamaCppEx.NIF.tokenize(ref, text, add_special, parse_special)}
  end

  @doc """
  Decodes a list of token IDs back into text.
  """
  @spec decode(LlamaCppEx.Model.t(), [integer()]) :: {:ok, String.t()}
  def decode(%LlamaCppEx.Model{ref: ref}, tokens) when is_list(tokens) do
    {:ok, LlamaCppEx.NIF.detokenize(ref, tokens)}
  end

  @doc """
  Converts a single token ID to its text representation.
  """
  @spec token_to_piece(LlamaCppEx.Model.t(), integer()) :: String.t()
  def token_to_piece(%LlamaCppEx.Model{ref: ref}, token) do
    LlamaCppEx.NIF.token_to_piece(ref, token)
  end

  @doc "Returns the vocabulary size."
  @spec vocab_size(LlamaCppEx.Model.t()) :: integer()
  def vocab_size(%LlamaCppEx.Model{ref: ref}), do: LlamaCppEx.NIF.vocab_n_tokens(ref)

  @doc "Returns the BOS (beginning of sentence) token ID."
  @spec bos_token(LlamaCppEx.Model.t()) :: integer()
  def bos_token(%LlamaCppEx.Model{ref: ref}), do: LlamaCppEx.NIF.vocab_bos(ref)

  @doc "Returns the EOS (end of sentence) token ID."
  @spec eos_token(LlamaCppEx.Model.t()) :: integer()
  def eos_token(%LlamaCppEx.Model{ref: ref}), do: LlamaCppEx.NIF.vocab_eos(ref)

  @doc "Returns whether a token is an end-of-generation token."
  @spec eog?(LlamaCppEx.Model.t(), integer()) :: boolean()
  def eog?(%LlamaCppEx.Model{ref: ref}, token), do: LlamaCppEx.NIF.vocab_is_eog(ref, token)
end
