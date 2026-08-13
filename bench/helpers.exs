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

    defaults = [
      model_path: model_path,
      n_gpu_layers: -1,
      n_parallel: 4,
      n_ctx: 4096,
      temp: 0.0
    ]

    # Anything the server declares is forwardable. The list used to be four
    # hand-maintained `maybe_add` calls, which meant every new tuning knob was
    # silently dropped here until someone noticed — and a benchmark that ignores
    # the option it is measuring is worse than no benchmark.
    forwardable = LlamaCppEx.Server.start_option_keys()
    {passthrough, unknown} = Keyword.split(opts, forwardable)

    unknown = Keyword.drop(unknown, [:model_path])

    if unknown != [] do
      raise ArgumentError,
            "Bench.Helpers.start_server/1 got options the server does not accept: " <>
              "#{inspect(Keyword.keys(unknown))}"
    end

    {:ok, server} = LlamaCppEx.Server.start_link(Keyword.merge(defaults, passthrough))
    server
  end

  @doc """
  Blocks until a server has finished loading, and returns the model.

  `Server.start_link/1` returns *before* the load: `init/1` stays cheap and
  `handle_continue/2` does the work. `Server.get_model/1` is not a wait — it
  raises `ArgumentError` as soon as one blocking timeout elapses, which a 63 GB
  model on a cold page cache comfortably outlives. A benchmark that used it
  instead lost the caller mid-load and took the VM down with
  `CUDA error: driver shutting down`.

  Timing a load therefore means timing this call, not `start_link/1`.
  """
  def await_model(server, timeout_ms \\ 900_000) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_await_model(server, deadline)
  end

  defp do_await_model(server, deadline) do
    case LlamaCppEx.Server.fetch_model(server) do
      {:ok, model} ->
        model

      {:error, :not_ready} ->
        if System.monotonic_time(:millisecond) < deadline do
          Process.sleep(200)
          do_await_model(server, deadline)
        else
          raise "model still loading after the deadline"
        end

      {:error, reason} ->
        raise "server failed to load its model: #{inspect(reason)}"
    end
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
end
