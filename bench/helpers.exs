defmodule Bench.Helpers do
  @moduledoc false

  # The short suite's filler, kept as the first element so `prompts/0` below is
  # byte-identical to what produced the numbers in bench/results/.
  @filler_sentences [
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. "
  ]

  @filler hd(@filler_sentences)

  @long_token_sizes [1024, 8192, 32768]

  def setup(opts \\ []) do
    model_path = System.get_env("LLAMA_MODEL_PATH") || raise "Set LLAMA_MODEL_PATH"
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, -1)

    :ok = LlamaCppEx.init()
    {:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: n_gpu_layers)
    model
  end

  def start_server(opts \\ []) do
    model_path = System.get_env("LLAMA_MODEL_PATH") || raise "Set LLAMA_MODEL_PATH"
    n_parallel = Keyword.get(opts, :n_parallel, 4)

    server_opts = [
      model_path: model_path,
      n_gpu_layers: Keyword.get(opts, :n_gpu_layers, -1),
      n_parallel: n_parallel,
      n_ctx: Keyword.get(opts, :n_ctx, 4096),
      temp: 0.0
    ]

    # Pass through new options
    server_opts =
      server_opts
      |> maybe_add(opts, :cache_prompt)
      |> maybe_add(opts, :batch_strategy)
      |> maybe_add(opts, :kv_unified)
      |> maybe_add(opts, :prompt_cache_ram_mb)

    {:ok, server} = LlamaCppEx.Server.start_link(server_opts)
    server
  end

  @doc """
  The short-prompt suite: roughly 6, 110 and 220 tokens.

  Deliberately unchanged — it is the only common axis between these runs and
  the numbers already recorded under `bench/results/`, so a regression at the
  old sizes stays visible. Everything above 220 tokens lives in
  `long_prompts/1`.
  """
  def prompts do
    %{
      "short" => "The capital of France is",
      "medium" => String.duplicate(@filler, 10),
      "long" => String.duplicate(@filler, 20)
    }
  end

  @doc """
  Token counts for the long-prompt suite: 1k / 8k / 32k.
  """
  def long_token_sizes, do: @long_token_sizes

  @doc """
  `%{"1k" => text, "8k" => text, "32k" => text}`, each tokenizing to exactly
  that many tokens under `model`'s vocab.

  Every per-token cost in the server — slot selection, prefix matching, prompt
  retention — is invisible below a few hundred tokens, which is where the short
  suite stops. Sizes are made exact rather than estimated for the same reason:
  the costs these probe are linear in the token count, so an "approximately 8k"
  prompt makes the numbers unusable for comparison across runs.

  Needs a context at least `32768 + max_tokens` wide to run the 32k case.
  """
  def long_prompts(model) do
    Map.new(@long_token_sizes, fn n -> {size_label(n), prompt_of_tokens(model, n)} end)
  end

  @doc """
  Text that tokenizes to exactly `n_tokens` under `model`'s vocab, using the
  same tokenizer options the server uses (so the count includes BOS).
  """
  def prompt_of_tokens(model, n_tokens) do
    # Detokenize/re-tokenize is not a round trip in general — decoded text can
    # re-merge into a different number of tokens — so converge instead of
    # assuming: cut the pool to the target, re-encode, correct by the error.
    # Plain-ASCII filler settles in one or two passes.
    {:ok, pool} = LlamaCppEx.Tokenizer.encode(model, filler_text(n_tokens))
    converge(model, pool, n_tokens, n_tokens, 8)
  end

  @doc """
  A deterministic synthetic token-id list of length `n`.

  For benchmarking the pure-Elixir scheduling code (`LlamaCppEx.Server.Slots`),
  which only ever compares token ids and never consults a vocab. No model, no
  NIF, no GPU — which is what makes those measurements reproducible.
  """
  def token_ids(n, opts \\ []) do
    seed = Keyword.get(opts, :seed, 0)
    Enum.map(1..n//1, &rem(&1 * 2_654_435_761 + seed, 32_000))
  end

  @doc """
  `"1k"`, `"8k"`, `"32k"` — the benchee input label for a token count.
  """
  def size_label(n) when rem(n, 1024) == 0, do: "#{div(n, 1024)}k"
  def size_label(n), do: "#{n}tok"

  defp converge(_model, _pool, _take, target, 0) do
    raise "could not build a prompt of exactly #{target} tokens"
  end

  defp converge(model, pool, take, target, budget) do
    {:ok, text} = LlamaCppEx.Tokenizer.decode(model, Enum.take(pool, take))
    {:ok, actual} = LlamaCppEx.Tokenizer.encode(model, text)

    case length(actual) do
      ^target -> text
      got -> converge(model, pool, take + (target - got), target, budget - 1)
    end
  end

  # Enough filler to tokenize past `n_tokens` with room for the converge loop to
  # ask for more. The sentences are cycled rather than one repeated line so the
  # text is not degenerate input for the tokenizer.
  defp filler_text(n_tokens) do
    @filler_sentences
    |> Stream.cycle()
    |> Enum.take(div(n_tokens, 4) + 16)
    |> IO.iodata_to_binary()
  end

  defp maybe_add(server_opts, source_opts, key) do
    case Keyword.fetch(source_opts, key) do
      {:ok, val} -> Keyword.put(server_opts, key, val)
      :error -> server_opts
    end
  end
end
