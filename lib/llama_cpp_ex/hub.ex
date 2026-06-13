defmodule LlamaCppEx.Hub do
  @moduledoc """
  Download GGUF models from HuggingFace Hub.

  Requires the optional `req` dependency. Add it to your `mix.exs`:

      {:req, "~> 0.5"}

  ## Examples

      # Search for GGUF models
      {:ok, results} = LlamaCppEx.Hub.search("qwen3 gguf", limit: 5)

      # List GGUF files in a repository
      {:ok, files} = LlamaCppEx.Hub.list_gguf_files("Qwen/Qwen3-4B-GGUF")

      # Download a model (cached locally)
      {:ok, path} = LlamaCppEx.Hub.download(
        "Qwen/Qwen3-4B-GGUF",
        "qwen3-4b-q4_k_m.gguf"
      )

  ## Authentication

  For private or gated repositories, set the `HF_TOKEN` environment variable
  or pass the `:token` option:

      LlamaCppEx.Hub.download("org/private-model", "model.gguf", token: "hf_...")

  ## Caching

  Downloaded files are cached in `~/.cache/llama_cpp_ex/models/` by default.
  Override with the `:cache_dir` option or `LLAMA_CACHE_DIR` environment variable.
  ETag headers are stored alongside cached files to detect upstream changes.

  ## Offline Mode

  Set `LLAMA_OFFLINE=1` to use only cached files without network access.

  ## Proxies

  Requests honor the standard proxy environment variables automatically:
  `HTTPS_PROXY`/`HTTP_PROXY` (and their lowercase forms), falling back to
  `ALL_PROXY`, with `NO_PROXY` respected for host bypass. Because HuggingFace is
  served over HTTPS, the `HTTPS_PROXY` value is the one that applies; an HTTP
  proxy tunnels HTTPS via the `CONNECT` method.

      # honored automatically
      export HTTPS_PROXY=http://127.0.0.1:8118

  Override or disable proxying per call with the `:proxy` (a URL string, a Mint
  `{scheme, address, port, opts}` tuple, or `false`) and `:no_proxy` options:

      LlamaCppEx.Hub.search("qwen3 gguf", proxy: "http://user:pass@127.0.0.1:8118")
      LlamaCppEx.Hub.download("org/model", "model.gguf", proxy: false)

  ### SOCKS is not supported

  The underlying HTTP client (Req → Finch → Mint) supports HTTP/1 proxies only —
  plain forwarding and HTTPS-over-`CONNECT` tunneling. It has **no SOCKS
  support**, so a `socks5://` value (e.g. from `ALL_PROXY`) is ignored with a
  warning. To use a SOCKS upstream, run a local HTTP-to-SOCKS bridge such as
  [Privoxy](https://www.privoxy.org) or [gost](https://github.com/go-gost/gost)
  and point `HTTPS_PROXY` at the bridge's HTTP port.
  """

  require Logger

  @hf_base_url "https://huggingface.co"
  @hf_api_url "https://huggingface.co/api/models"
  @default_cache_dir Path.expand("~/.cache/llama_cpp_ex/models")

  # --- Search ---

  @doc """
  Search HuggingFace Hub for GGUF models.

  Returns a list of model info maps with `:id`, `:downloads`, `:likes`,
  `:last_modified`, and `:tags`.

  ## Options

    * `:limit` - Maximum results. Defaults to `10`.
    * `:sort` - Sort by `"downloads"`, `"likes"`, or `"lastModified"`. Defaults to `"downloads"`.
    * `:direction` - Sort direction, `-1` for descending. Defaults to `-1`.
    * `:token` - HuggingFace API token.
    * `:proxy`, `:no_proxy` - Proxy overrides. See the "Proxies" section above.

  ## Examples

      {:ok, models} = LlamaCppEx.Hub.search("llama gguf q4")
      Enum.each(models, fn m -> IO.puts("\#{m.id} (\#{m.downloads} downloads)") end)

  """
  @spec search(String.t(), keyword()) :: {:ok, [map()]} | {:error, String.t()}
  def search(query, opts \\ []) do
    with :ok <- ensure_req() do
      limit = Keyword.get(opts, :limit, 10)
      sort = Keyword.get(opts, :sort, "downloads")
      direction = Keyword.get(opts, :direction, -1)
      headers = auth_headers(opts)

      params = [
        search: query,
        filter: "gguf",
        sort: sort,
        direction: direction,
        limit: limit
      ]

      req_opts = [headers: headers, params: params] ++ proxy_request_options(@hf_api_url, opts)

      case Req.get(@hf_api_url, req_opts) do
        {:ok, %{status: 200, body: body}} when is_list(body) ->
          models =
            Enum.map(body, fn m ->
              %{
                id: m["id"] || m["modelId"],
                downloads: m["downloads"] || 0,
                likes: m["likes"] || 0,
                last_modified: m["lastModified"],
                tags: m["tags"] || [],
                private: m["private"] || false,
                gated: m["gated"] || false
              }
            end)

          {:ok, models}

        {:ok, %{status: status}} ->
          {:error, "HuggingFace search returned status #{status}"}

        {:error, exception} ->
          {:error, "network error: #{Exception.message(exception)}"}
      end
    end
  end

  # --- Download ---

  @doc """
  Download a GGUF file from HuggingFace Hub, returning the local path.

  Uses ETag-based caching — if the file exists locally and the ETag matches,
  the cached version is returned without re-downloading.

  ## Options

    * `:cache_dir` - Local cache directory. Defaults to `~/.cache/llama_cpp_ex/models/`
      or the `LLAMA_CACHE_DIR` environment variable.
    * `:token` - HuggingFace API token. Defaults to `HF_TOKEN` environment variable.
    * `:revision` - Git revision (branch, tag, or commit). Defaults to `"main"`.
    * `:force` - Force re-download even if cached. Defaults to `false`.
    * `:proxy`, `:no_proxy` - Proxy overrides. See the "Proxies" section above.

  """
  @spec download(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def download(repo_id, filename, opts \\ []) do
    with :ok <- ensure_req() do
      dest = cache_path(repo_id, filename, opts)
      force = Keyword.get(opts, :force, false)

      cond do
        offline?() and File.exists?(dest) ->
          Logger.debug("Offline mode: using cached #{dest}")
          {:ok, dest}

        offline?() ->
          {:error, "offline mode enabled but file not cached: #{repo_id}/#{filename}"}

        File.exists?(dest) and not force ->
          Logger.debug("Using cached model: #{dest}")
          {:ok, dest}

        true ->
          url = build_download_url(repo_id, filename, opts)
          headers = auth_headers(opts)
          do_download_to(url, dest, headers, proxy_request_options(url, opts))
      end
    end
  end

  # --- Listing ---

  @doc """
  List GGUF files available in a HuggingFace repository.

  Returns a list of maps with `:filename` and `:size` (bytes).

  ## Options

    * `:token` - HuggingFace API token.

  ## Examples

      {:ok, files} = LlamaCppEx.Hub.list_gguf_files("Qwen/Qwen3-4B-GGUF")
      Enum.each(files, fn f ->
        size_mb = Float.round(f.size / 1_000_000, 1)
        IO.puts("\#{f.filename} (\#{size_mb} MB)")
      end)

  """
  @spec list_gguf_files(String.t(), keyword()) ::
          {:ok, [%{filename: String.t(), size: integer()}]} | {:error, String.t()}
  def list_gguf_files(repo_id, opts \\ []) do
    with :ok <- ensure_req() do
      revision = Keyword.get(opts, :revision, "main")
      url = "#{@hf_api_url}/#{repo_id}/tree/#{revision}"
      headers = auth_headers(opts)

      case Req.get(url, [headers: headers] ++ proxy_request_options(url, opts)) do
        {:ok, %{status: 200, body: body}} when is_list(body) ->
          files =
            body
            |> Enum.filter(
              &(&1["type"] == "file" and String.ends_with?(&1["path"] || "", ".gguf"))
            )
            |> Enum.map(fn f -> %{filename: f["path"], size: f["size"] || 0} end)
            |> Enum.sort_by(& &1.size)

          {:ok, files}

        {:ok, %{status: 401}} ->
          {:error, "authentication required — set HF_TOKEN or pass :token option"}

        {:ok, %{status: 403}} ->
          {:error,
           "access denied — this may be a gated model requiring access approval at #{@hf_base_url}/#{repo_id}"}

        {:ok, %{status: 404}} ->
          {:error, "repository not found: #{repo_id}"}

        {:ok, %{status: status}} ->
          {:error, "HuggingFace API returned status #{status}"}

        {:error, exception} ->
          {:error, "network error: #{Exception.message(exception)}"}
      end
    end
  end

  # --- Model Info ---

  @doc """
  Get model repository metadata from HuggingFace Hub API.

  ## Options

    * `:token` - HuggingFace API token.

  """
  @spec get_model_info(String.t(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def get_model_info(repo_id, opts \\ []) do
    with :ok <- ensure_req() do
      url = "#{@hf_api_url}/#{repo_id}"
      headers = auth_headers(opts)

      case Req.get(url, [headers: headers] ++ proxy_request_options(url, opts)) do
        {:ok, %{status: 200, body: body}} ->
          {:ok, body}

        {:ok, %{status: 401}} ->
          {:error, "authentication required — set HF_TOKEN or pass :token option"}

        {:ok, %{status: 403}} ->
          {:error,
           "access denied — this may be a gated model requiring access approval at #{@hf_base_url}/#{repo_id}"}

        {:ok, %{status: 404}} ->
          {:error, "repository not found: #{repo_id}"}

        {:ok, %{status: status}} ->
          {:error, "HuggingFace API returned status #{status}"}

        {:error, exception} ->
          {:error, "network error: #{Exception.message(exception)}"}
      end
    end
  end

  # --- Public Helpers ---

  @doc """
  Filter a list of HuggingFace siblings entries to only GGUF files.

  Returns maps with `:filename` and `:size`.
  """
  @spec filter_gguf_files([map()]) :: [%{filename: String.t(), size: integer()}]
  def filter_gguf_files(siblings) do
    siblings
    |> Enum.filter(&String.ends_with?(&1["rfilename"] || "", ".gguf"))
    |> Enum.map(fn s ->
      %{filename: s["rfilename"], size: s["size"] || 0}
    end)
    |> Enum.sort_by(& &1.size)
  end

  @doc """
  Build the download URL for a file in a HuggingFace repository.
  """
  @spec build_download_url(String.t(), String.t(), keyword()) :: String.t()
  def build_download_url(repo_id, filename, opts \\ []) do
    revision = Keyword.get(opts, :revision, "main")
    "#{@hf_base_url}/#{repo_id}/resolve/#{revision}/#{filename}"
  end

  @doc """
  Build authentication headers from options or environment.

  Checks for tokens in order: `:token` option, `HF_TOKEN` env var,
  `HUGGING_FACE_HUB_TOKEN` env var (legacy).
  """
  @spec auth_headers(keyword()) :: [{String.t(), String.t()}]
  def auth_headers(opts) do
    token =
      Keyword.get(opts, :token) ||
        System.get_env("HF_TOKEN") ||
        System.get_env("HUGGING_FACE_HUB_TOKEN")

    if token do
      [{"authorization", "Bearer #{token}"}]
    else
      []
    end
  end

  @doc false
  # Builds the Req `connect_options` needed to route a request through a proxy.
  #
  # The proxy is resolved from the `:proxy` option (a URL string, a Mint
  # `{scheme, address, port, opts}` tuple, or `false` to disable) and otherwise
  # from the standard proxy environment variables. Returns `[]` when no usable
  # proxy applies. SOCKS proxies are detected and skipped — see the "Proxies"
  # section in the module doc for the reasoning and the workaround.
  @spec proxy_request_options(String.t(), keyword()) :: keyword()
  def proxy_request_options(url, opts \\ []) do
    case Keyword.get(opts, :proxy, :auto) do
      {scheme, host, port, proxy_opts}
      when is_atom(scheme) and is_binary(host) and is_integer(port) and is_list(proxy_opts) ->
        [connect_options: [proxy: {scheme, host, port, proxy_opts}]]

      proxy_setting ->
        target = URI.parse(url)

        case resolve_proxy_url(proxy_setting, target.scheme) do
          nil -> []
          proxy_url -> build_proxy_options(proxy_url, target.host, opts)
        end
    end
  end

  @doc """
  Build the local cache path for a model file.
  """
  @spec cache_path(String.t(), String.t(), keyword()) :: String.t()
  def cache_path(repo_id, filename, opts \\ []) do
    cache_dir =
      Keyword.get(opts, :cache_dir) ||
        System.get_env("LLAMA_CACHE_DIR") ||
        @default_cache_dir

    Path.join([cache_dir, repo_id, filename])
  end

  # --- Private ---

  defp offline? do
    System.get_env("LLAMA_OFFLINE") in ["1", "true"]
  end

  defp ensure_req do
    if Code.ensure_loaded?(Req) do
      :ok
    else
      {:error,
       "the :req dependency is required for HuggingFace Hub downloads. " <>
         "Add {:req, \"~> 0.5\"} to your mix.exs deps."}
    end
  end

  # --- Proxy resolution ---

  defp resolve_proxy_url(false, _scheme), do: nil
  defp resolve_proxy_url(url, _scheme) when is_binary(url), do: url
  defp resolve_proxy_url(:auto, scheme), do: env_proxy_url(scheme)

  # Scheme-specific vars take precedence over the catch-all ALL_PROXY, so a
  # usable HTTP proxy always wins over an (unusable) SOCKS ALL_PROXY.
  defp env_proxy_url("https"), do: env_any(["HTTPS_PROXY", "https_proxy"]) || env_all_proxy()
  defp env_proxy_url("http"), do: env_any(["HTTP_PROXY", "http_proxy"]) || env_all_proxy()
  defp env_proxy_url(_other), do: env_all_proxy()

  defp env_all_proxy, do: env_any(["ALL_PROXY", "all_proxy"])

  defp env_any(keys) do
    Enum.find_value(keys, fn key ->
      case System.get_env(key) do
        nil -> nil
        "" -> nil
        value -> value
      end
    end)
  end

  defp build_proxy_options(proxy_url, host, opts) do
    no_proxy = Keyword.get(opts, :no_proxy) || env_any(["NO_PROXY", "no_proxy"]) || ""

    if bypass_proxy?(host, no_proxy) do
      []
    else
      case parse_proxy(proxy_url) do
        {:ok, proxy, []} ->
          [connect_options: [proxy: proxy]]

        {:ok, proxy, proxy_headers} ->
          [connect_options: [proxy: proxy, proxy_headers: proxy_headers]]

        {:error, {:socks, scheme}} ->
          Logger.warning(
            "ignoring #{scheme} proxy #{redact_proxy(proxy_url)}: the HTTP client (Req/Finch/Mint) " <>
              "supports HTTP/HTTPS CONNECT proxies only, not SOCKS. Run a local HTTP-to-SOCKS " <>
              "bridge (e.g. Privoxy or gost) and point HTTPS_PROXY at it instead."
          )

          []

        {:error, :invalid} ->
          Logger.warning("ignoring malformed proxy URL #{redact_proxy(proxy_url)}")
          []
      end
    end
  end

  defp bypass_proxy?(nil, _no_proxy), do: false

  defp bypass_proxy?(host, no_proxy) do
    no_proxy
    |> String.split(",", trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.any?(&host_matches_no_proxy?(host, &1))
  end

  defp host_matches_no_proxy?(_host, ""), do: false
  defp host_matches_no_proxy?(_host, "*"), do: true

  defp host_matches_no_proxy?(host, entry) do
    entry = String.trim_leading(entry, ".")
    host == entry or String.ends_with?(host, "." <> entry)
  end

  defp parse_proxy(proxy_url) do
    uri = proxy_url |> normalize_proxy_url() |> URI.parse()

    case uri.scheme do
      scheme when scheme in ["http", "https"] ->
        proxy = {proxy_scheme_atom(scheme), uri.host, uri.port || default_proxy_port(scheme), []}
        {:ok, proxy, proxy_auth_headers(uri.userinfo)}

      "socks" <> _ ->
        {:error, {:socks, String.upcase(uri.scheme)}}

      _other ->
        {:error, :invalid}
    end
  end

  # Proxy URLs from `ALL_PROXY` or a bare `host:port` option may omit the scheme.
  defp normalize_proxy_url(url) do
    if Regex.match?(~r{^[a-zA-Z][a-zA-Z0-9+.\-]*://}, url), do: url, else: "http://" <> url
  end

  defp proxy_scheme_atom("http"), do: :http
  defp proxy_scheme_atom("https"), do: :https

  defp default_proxy_port("http"), do: 80
  defp default_proxy_port("https"), do: 443

  defp proxy_auth_headers(nil), do: []

  defp proxy_auth_headers(userinfo),
    do: [{"proxy-authorization", "Basic " <> Base.encode64(userinfo)}]

  # Strips credentials before a proxy URL is written to the log.
  defp redact_proxy(proxy_url) do
    case URI.parse(normalize_proxy_url(proxy_url)) do
      %URI{userinfo: nil} = uri -> URI.to_string(uri)
      %URI{} = uri -> URI.to_string(%{uri | userinfo: "***"})
    end
  end

  defp do_download_to(url, dest, headers, proxy_opts) do
    Logger.info("Downloading to #{dest}")
    File.mkdir_p!(Path.dirname(dest))

    tmp_dest = dest <> ".download"

    try do
      case do_stream_download(url, tmp_dest, headers, proxy_opts) do
        {:ok, etag} ->
          File.rename!(tmp_dest, dest)

          if etag do
            File.write!(dest <> ".etag", etag)
          end

          Logger.info("Download complete: #{dest}")
          {:ok, dest}

        {:error, reason} ->
          File.rm(tmp_dest)
          {:error, reason}
      end
    rescue
      e ->
        File.rm(tmp_dest)
        {:error, "download failed: #{Exception.message(e)}"}
    end
  end

  defp do_stream_download(url, dest, headers, proxy_opts) do
    # Use Req with output to file — handles redirects correctly
    req_opts = [headers: headers, max_redirects: 10, into: File.stream!(dest)] ++ proxy_opts

    case Req.get(url, req_opts) do
      {:ok, %{status: 200} = resp} ->
        etag = get_header(resp, "etag")
        {:ok, etag}

      {:ok, %{status: 401}} ->
        {:error, "authentication required — set HF_TOKEN or pass :token option"}

      {:ok, %{status: 403}} ->
        {:error, "access denied — this may be a gated model requiring access approval"}

      {:ok, %{status: 404}} ->
        {:error, "file not found: #{url}"}

      {:ok, %{status: status}} ->
        {:error, "download failed with status #{status}"}

      {:error, exception} ->
        {:error, "network error: #{Exception.message(exception)}"}
    end
  end

  defp get_header(%{headers: headers}, key) do
    case Map.get(headers, key) do
      [value | _] -> value
      _ -> nil
    end
  end
end
