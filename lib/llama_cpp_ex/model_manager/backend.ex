defmodule LlamaCppEx.ModelManager.Backend do
  @moduledoc """
  Behaviour for the model I/O the manager performs on the write path.

  The default implementation is `LlamaCppEx.ModelManager.ModelIO`, which delegates
  to `LlamaCppEx.Hub`, `LlamaCppEx.Model`, and `LlamaCppEx.Server`. Tests inject a
  fake via the `:io` start option to exercise load/unload lifecycle without real
  GGUF files.

  Inference dispatch (`generate`/`stream`/`chat`/`embed`) does NOT go through this
  behaviour — it reads the ETS table directly from the caller and calls the
  relevant module, keeping the manager process off the hot path.
  """

  alias LlamaCppEx.Model
  alias LlamaCppEx.ModelManager.Entry

  @doc """
  Resolves a source to a local file path and its byte size, downloading from the
  Hub if needed.
  """
  @callback resolve_source(Entry.source(), keyword()) ::
              {:ok, String.t(), non_neg_integer()} | {:error, term()}

  @doc "Loads a model directly (for `:direct` mode)."
  @callback load_model(String.t(), keyword()) :: {:ok, Model.t()} | {:error, term()}

  @doc "Starts a backing `LlamaCppEx.Server` for `id` (for `:server` mode)."
  @callback start_server(id :: term(), path :: String.t(), keyword()) ::
              {:ok, pid()} | {:error, term()}

  @doc "Stops a backing server, dropping its context and model refs."
  @callback stop_server(pid()) :: :ok
end
