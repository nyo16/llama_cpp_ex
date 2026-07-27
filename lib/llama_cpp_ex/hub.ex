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
  The cache directory is created with mode `0o700` and cached files with `0o600`
  — gated-repository content is credential-adjacent.

  A cached file is returned as-is: `download/3` makes **no** request to the Hub
  when the file is already present. The upstream ETag is recorded in a
  `<file>.etag` sidecar, but nothing reads it back yet, so a file that changed
  upstream is *not* detected — pass `force: true` to refresh it.

  ## Integrity

  A fresh download is verified against the SHA-256 that HuggingFace publishes for
  the file, and a file whose digest does not match is deleted rather than cached.
  See `download/3` for the limits of that guarantee.

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
      params = [
        search: query,
        filter: "gguf",
        sort: Keyword.get(opts, :sort, "downloads"),
        direction: Keyword.get(opts, :direction, -1),
        limit: Keyword.get(opts, :limit, 10)
      ]

      case hf_get(@hf_api_url, opts, params: params) do
        {:ok, %{status: 200, body: body}} when is_list(body) ->
          {:ok, parse_search_results(body)}

        {:ok, %{status: status}} ->
          {:error, "HuggingFace search returned status #{status}"}

        {:error, exception} ->
          {:error, "network error: #{Exception.message(exception)}"}
      end
    end
  end

  defp parse_search_results(body) do
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
  end

  # --- Download ---

  @doc """
  Download a GGUF file from HuggingFace Hub, returning the local path.

  An already-cached file is returned immediately, without contacting the Hub.
  There is no upstream revalidation — the ETag written to the `<file>.etag`
  sidecar is never read back — so pass `force: true` to refresh a cached file.
  The cache key includes `:revision`, so two revisions of the same file cache
  separately.

  A fresh download is streamed into a randomly named temporary file opened with
  `O_EXCL`, verified against the SHA-256 HuggingFace publishes for the file, and
  only then renamed into place with mode `0o600`. A missing published digest is a
  failure, not a warning — see `:verify_checksum`.

  > #### Integrity is not authenticity {: .warning}
  >
  > The digest comes from the same origin as the bytes, so it detects corruption
  > or tampering *between* HuggingFace and you — not a malicious file published by
  > the repository owner. GGUF parsing happens in C++, so for repositories you do
  > not trust, also pass `check_tensors: true` to `LlamaCppEx.Model.load/2`.

  ## Options

    * `:cache_dir` - Local cache directory. Defaults to `~/.cache/llama_cpp_ex/models/`
      or the `LLAMA_CACHE_DIR` environment variable.
    * `:token` - HuggingFace API token. Defaults to `HF_TOKEN` environment variable.
    * `:revision` - Git revision (branch, tag, or commit). Defaults to `"main"`.
      Part of the cache key.
    * `:force` - Force re-download even if cached. Defaults to `false`.
    * `:verify_checksum` - Integrity policy. Defaults to `true`.
      * `true` — fail closed. The download is checked against the SHA-256
        HuggingFace publishes, and a file the Hub lists *without* one is refused.
        Verification used to be downgradable by the metadata response itself: it
        fell back to a warning when `siblings[].lfs.sha256` was absent, so
        stripping one JSON key was enough to have the bytes cached unverified.
      * `:best_effort` — warn and proceed when the Hub publishes no digest. For
        the rare non-LFS blob that is genuinely small enough to have none.
      * `false` — skip the check and the metadata request entirely. Logged as a
        warning.
    * `:proxy`, `:no_proxy` - Proxy overrides. See the "Proxies" section above.
  """
  @spec download(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def download(repo_id, filename, opts \\ []) do
    with :ok <- ensure_req(),
         {:ok, dest} <- safe_cache_path(repo_id, filename, opts) do
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
          fetch_and_cache(repo_id, filename, dest, opts)
      end
    end
  end

  defp fetch_and_cache(repo_id, filename, dest, opts) do
    with {:ok, expected_sha256} <- expected_sha256(repo_id, filename, opts) do
      url = build_download_url(repo_id, filename, opts)
      do_download_to(url, dest, "#{repo_id}/#{filename}", expected_sha256, opts)
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

      case hf_get(url, opts) do
        {:ok, %{status: 200, body: body}} when is_list(body) -> {:ok, parse_gguf_tree(body)}
        other -> hf_api_error(other, repo_id)
      end
    end
  end

  defp parse_gguf_tree(body) do
    body
    |> Enum.filter(&(&1["type"] == "file" and String.ends_with?(&1["path"] || "", ".gguf")))
    |> Enum.map(fn f -> %{filename: f["path"], size: f["size"] || 0} end)
    |> Enum.sort_by(& &1.size)
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

      case hf_get(url, opts) do
        {:ok, %{status: 200, body: body}} -> {:ok, body}
        other -> hf_api_error(other, repo_id)
      end
    end
  end

  # --- Shared HF API request helpers ---

  # Issues a GET with auth headers and proxy options applied.
  defp hf_get(url, opts, extra_req_opts \\ []) do
    req_opts = [headers: auth_headers(opts)] ++ extra_req_opts ++ proxy_request_options(url, opts)
    http_client(opts).(url, req_opts)
  end

  # Test seam. The HTTP client is a `(url, req_opts) -> {:ok, resp} | {:error, e}`
  # function, so the test suite can drive the download path with no network
  # access. `Req.Test` would be the idiomatic stub, but it requires the optional
  # :plug dependency, which this project does not carry.
  defp http_client(opts), do: Keyword.get(opts, :http_client) || (&Req.get/2)

  # Maps a non-200 HF API response or transport error to an error tuple.
  defp hf_api_error({:ok, %{status: 401}}, _repo_id),
    do: {:error, "authentication required — set HF_TOKEN or pass :token option"}

  defp hf_api_error({:ok, %{status: 403}}, repo_id),
    do:
      {:error,
       "access denied — this may be a gated model requiring access approval at #{@hf_base_url}/#{repo_id}"}

  defp hf_api_error({:ok, %{status: 404}}, repo_id),
    do: {:error, "repository not found: #{repo_id}"}

  defp hf_api_error({:ok, %{status: 200, body: body}}, repo_id),
    do:
      {:error,
       "unexpected HuggingFace API response for #{repo_id}: #{inspect(body, limit: 5, printable_limit: 120)}"}

  defp hf_api_error({:ok, %{status: status}}, _repo_id),
    do: {:error, "HuggingFace API returned status #{status}"}

  defp hf_api_error({:error, exception}, _repo_id),
    do: {:error, "network error: #{Exception.message(exception)}"}

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

  The path is `<cache_dir>/<repo_id>/<revision>/<filename>`. `revision` is part of
  the key because otherwise pinning `revision: "<sha>"` returned whatever had been
  cached for `main` — the pin bought nothing, silently. It defaults to `"main"`.

  All three components are caller-supplied and become path components, so each is
  validated first: one that is absolute, empty, `"."`, `".."`, `~`-prefixed, or
  contains a null byte would escape the cache and is rejected with an
  `ArgumentError`. `download/3` performs the same validation but surfaces it as
  `{:error, reason}`.
  """
  @spec cache_path(String.t(), String.t(), keyword()) :: String.t()
  def cache_path(repo_id, filename, opts \\ []) do
    case safe_cache_path(repo_id, filename, opts) do
      {:ok, path} -> path
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  # --- Cache path validation ---

  # `cache_path/3` without the raise, so `download/3` keeps its
  # `{:error, String.t()}` contract.
  defp safe_cache_path(repo_id, filename, opts) do
    cache_dir =
      Keyword.get(opts, :cache_dir) ||
        System.get_env("LLAMA_CACHE_DIR") ||
        @default_cache_dir

    revision = Keyword.get(opts, :revision, "main")

    with :ok <- validate_path_fragment(repo_id, "repository id"),
         :ok <- validate_path_fragment(revision, "revision"),
         :ok <- validate_path_fragment(filename, "filename") do
      {:ok, Path.join([cache_dir, repo_id, revision, filename])}
    end
  end

  # Both fragments are caller-supplied and end up in a `Path.join/1`, so anything
  # that resolves outside the cache directory has to be refused rather than
  # sanitized: a rewritten repo id would silently cache one repo's file under
  # another's name. `\` counts as a separator too — harmless on Unix, a real
  # escape on Windows.
  defp validate_path_fragment(value, label) when is_binary(value) do
    cond do
      value == "" ->
        {:error, "invalid #{label}: must not be empty"}

      String.contains?(value, <<0>>) ->
        {:error, "invalid #{label} #{inspect(value)}: must not contain a null byte"}

      absolute_fragment?(value) ->
        {:error, "invalid #{label} #{inspect(value)}: must be a relative path"}

      Enum.any?(String.split(value, ["/", "\\"]), &unsafe_component?/1) ->
        {:error,
         "invalid #{label} #{inspect(value)}: path components must not be empty, " <>
           "\".\", \"..\", or start with \"~\""}

      true ->
        :ok
    end
  end

  defp validate_path_fragment(value, label),
    do: {:error, "invalid #{label}: expected a string, got #{inspect(value)}"}

  defp absolute_fragment?(value),
    do: String.starts_with?(value, ["/", "\\"]) or Regex.match?(~r/\A[a-zA-Z]:/, value)

  defp unsafe_component?(""), do: true
  defp unsafe_component?("."), do: true
  defp unsafe_component?(".."), do: true
  defp unsafe_component?("~" <> _rest), do: true
  defp unsafe_component?(_component), do: false

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

  # --- Integrity ---

  # Resolves the SHA-256 HuggingFace publishes for the file, or `:unverified` when
  # the caller has deliberately opted out.
  #
  # There is no longer a "the metadata did not mention a digest, so proceed"
  # outcome. `sibling_sha256/5` fell through to `{:ok, nil}` and
  # `verify_integrity/3` turned `nil` into a warning and `:ok`, which made
  # verification downgradable *by the metadata response*: a party who can shape it
  # — a MITM with a trusted cert, or the TLS-terminating corporate proxy this
  # module explicitly supports — strips one JSON key and the bytes are cached
  # unverified. Every GGUF large enough to matter is LFS-backed, so a missing
  # digest is now an error naming its own escape hatch.
  defp expected_sha256(repo_id, filename, opts) do
    case Keyword.get(opts, :verify_checksum, true) do
      false ->
        Logger.warning(
          "integrity verification disabled for #{repo_id}/#{filename} — the downloaded " <>
            "bytes will not be checked against HuggingFace's SHA-256"
        )

        {:ok, :unverified}

      mode when mode in [true, :best_effort] ->
        fetch_expected_sha256(repo_id, filename, opts, mode)

      other ->
        {:error,
         "invalid :verify_checksum #{inspect(other)}: expected true (fail closed), " <>
           ":best_effort (warn when the Hub publishes no digest), or false (skip)"}
    end
  end

  # `siblings[].lfs.sha256` from the revision API is the file's real SHA-256 (the
  # same value the resolve URL returns as `x-linked-etag`); the plain `etag` on
  # the download response is the CDN's, which is a different hash entirely.
  defp fetch_expected_sha256(repo_id, filename, opts, mode) do
    revision = Keyword.get(opts, :revision, "main")

    case hf_get("#{@hf_api_url}/#{repo_id}/revision/#{revision}", opts, params: [blobs: true]) do
      {:ok, %{status: 200, body: %{"siblings" => siblings}}} when is_list(siblings) ->
        sibling_sha256(siblings, repo_id, filename, revision, mode)

      other ->
        {:error, reason} = hf_api_error(other, repo_id)

        {:error,
         "cannot verify #{repo_id}/#{filename}: #{reason} — pass " <>
           "`verify_checksum: false` to download without integrity checking"}
    end
  end

  defp sibling_sha256(siblings, repo_id, filename, revision, mode) do
    case Enum.find(siblings, &(&1["rfilename"] == filename)) do
      nil ->
        {:error, "file not found in #{repo_id}@#{revision}: #{filename}"}

      %{"lfs" => %{"sha256" => sha}} when is_binary(sha) ->
        {:ok, String.downcase(sha)}

      %{} when mode == :best_effort ->
        Logger.warning(
          "#{repo_id}/#{filename}@#{revision} is not LFS-backed and HuggingFace publishes " <>
            "no SHA-256 for it; proceeding unverified because verify_checksum: :best_effort"
        )

        {:ok, :unverified}

      %{} ->
        {:error,
         "#{repo_id}/#{filename}@#{revision} is listed but HuggingFace publishes no " <>
           "SHA-256 for it, so the download cannot be integrity-checked. Every GGUF " <>
           "large enough to matter is LFS-backed, so this usually means the metadata " <>
           "response was not what it claimed. Pass `verify_checksum: :best_effort` to " <>
           "warn and proceed, or `verify_checksum: false` to skip the check entirely"}
    end
  end

  # No `nil` clause: a missing digest is refused in sibling_sha256/5 rather than
  # downgraded here. That clause was the downgrade path.
  defp verify_integrity(_path, :unverified, _label), do: :ok

  defp verify_integrity(path, expected, label) do
    actual = sha256_file(path)

    if actual == expected do
      :ok
    else
      {:error,
       "checksum mismatch for #{label}: expected SHA-256 #{expected}, got #{actual} — " <>
         "the downloaded file was discarded"}
    end
  end

  # Hashes what actually landed on disk rather than the byte stream, so a
  # truncated or partially written file fails too.
  defp sha256_file(path) do
    path
    |> File.stream!(1024 * 1024)
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  # --- Download ---

  defp do_download_to(url, dest, label, expected_sha256, opts) do
    Logger.info("Downloading to #{dest}")
    tmp_dest = tmp_download_path(dest)

    try do
      stage_download(url, dest, tmp_dest, label, expected_sha256, opts)
    rescue
      # File.Error from mkdir/chmod/rename/the File.stream! sink, ErlangError from
      # lower-level IO — programming errors (KeyError, MatchError, …) propagate.
      e in [File.Error, ErlangError] ->
        {:error, abort_download(e, tmp_dest)}
    end
  end

  defp stage_download(url, dest, tmp_dest, label, expected_sha256, opts) do
    mkdir_private!(Path.dirname(dest))

    with {:ok, etag} <- do_stream_download(url, tmp_dest, opts),
         :ok <- verify_integrity(tmp_dest, expected_sha256, label) do
      # chmod before the rename, so the file is never visible at the final path
      # with a umask-derived mode.
      File.chmod!(tmp_dest, 0o600)
      File.rename!(tmp_dest, dest)
      write_etag(dest, etag)
      Logger.info("Download complete: #{dest}")
      {:ok, dest}
    else
      {:error, reason} ->
        File.rm(tmp_dest)
        {:error, reason}
    end
  end

  # Unpredictable name: a fixed `<dest>.part` can be pre-created as a symlink to
  # redirect the write, and two concurrent downloads of the same file would share
  # it and corrupt each other.
  defp tmp_download_path(dest) do
    suffix = 12 |> :crypto.strong_rand_bytes() |> Base.url_encode64(padding: false)
    dest <> "." <> suffix <> ".part"
  end

  # Cleans up after a failed download and returns the user-facing reason.
  # `:eexist` means the exclusive create lost the temporary name to something
  # already there — most likely a planted symlink — so that entry is left alone
  # rather than deleted.
  defp abort_download(%File.Error{reason: :eexist, path: path}, _tmp_dest) do
    "download aborted: #{path} already exists — refusing to write through it " <>
      "(possible symlink attack or concurrent download)"
  end

  defp abort_download(exception, tmp_dest) do
    File.rm(tmp_dest)
    "download failed: #{Exception.message(exception)}"
  end

  # Creates the directory and any missing parents with mode 0o700. Only
  # directories this call creates are chmodded; an existing cache directory keeps
  # whatever the user chose.
  defp mkdir_private!(dir) do
    [first | rest] = Path.split(dir)

    Enum.reduce(rest, mkdir_private_component!(first), fn component, parent ->
      mkdir_private_component!(Path.join(parent, component))
    end)
    |> then(fn _deepest -> :ok end)
  end

  defp mkdir_private_component!(path) do
    if File.dir?(path) do
      path
    else
      File.mkdir_p!(path)
      File.chmod!(path, 0o700)
      path
    end
  end

  defp write_etag(_dest, nil), do: :ok

  defp write_etag(dest, etag) do
    path = dest <> ".etag"
    File.write!(path, etag)
    File.chmod!(path, 0o600)
  end

  defp do_stream_download(url, tmp_dest, opts) do
    # O_EXCL: an entry already at `tmp_dest` — a symlink planted by another user
    # on a shared cache dir, say — fails the open instead of being written
    # through. `File.stream!/2` cannot express this: its mode argument is
    # `stream_mode()`, which has no `:write`, `:binary` or `:exclusive`, so the
    # device is opened here and fed by Req's `into:` callback instead. (Dialyzer
    # caught the earlier `File.stream!(path, [:write, :binary, :exclusive])`: the
    # call could never succeed, so the protection was not actually in place.)
    case File.open(tmp_dest, [:write, :binary, :exclusive]) do
      {:ok, device} ->
        try do
          stream_into_device(url, device, opts)
        after
          File.close(device)
        end

      {:error, :eexist} ->
        {:error, "temp file already exists: #{tmp_dest}"}

      {:error, posix} ->
        {:error, "cannot open #{tmp_dest}: #{:file.format_error(posix)}"}
    end
  end

  defp stream_into_device(url, device, opts) do
    req_opts =
      [
        headers: auth_headers(opts),
        max_redirects: 10,
        # Req's retry re-runs the request with this same `into:` closure, which
        # still holds `device` positioned at the end of whatever the failed attempt
        # already wrote (`deps/req/lib/req/steps.ex:2315` calls `run_request/1` on
        # the unhalted request). A transient 503 or a dropped connection therefore
        # *appended a second response body* instead of restarting the file — a
        # corrupt GGUF with a plausible size and a valid ETag. There is no hook
        # that runs before a retry with access to the device, so retry is off here
        # and a failed download is the caller's to repeat: `download/3` deletes the
        # temp file and returns an error, so a repeat starts clean.
        retry: false,
        # Unlike the Collectable form, a function `into:` is invoked for every
        # status, so the body of a 404 or a gated-model 403 would otherwise land
        # in the temp file. Only 200 bodies are written; the caller deletes the
        # temp file on any error either way.
        into: fn {:data, data}, {req, resp} ->
          if resp.status == 200, do: write_chunk!(device, data)
          {:cont, {req, resp}}
        end
      ] ++ proxy_request_options(url, opts)

    case http_client(opts).(url, req_opts) do
      {:ok, %{status: 200} = resp} ->
        {:ok, get_header(resp, "etag")}

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
  catch
    :throw, {:write_failed, posix} ->
      {:error, "cannot write download: #{:file.format_error(posix)}"}
  end

  # A failed write must become an error tuple, not an exception.
  #
  # The result of the old `IO.binwrite/2` was discarded, but not for the reason it
  # looked like: `IO.binwrite/2` does not return `{:error, reason}` at all — it
  # calls `:erlang.error(reason)` (`elixir/lib/io.ex:305-310`). So a full disk
  # raised `** (ErlangError) :enospc` from inside Req's streaming callback, which
  # unwound past `download/3` into the caller's crash report — with the request
  # options, including `:token`, in the stacktrace. `:file.write/2` is the variant
  # that reports, so the failure is handled here and the temp file is deleted.
  #
  # Thrown rather than returned because this runs inside Req's `into:` callback,
  # where the only way out is to unwind: `{:halt, _}` ends the stream but still
  # reports `{:ok, resp}`, which is the truncated-file-with-a-valid-etag outcome.
  defp write_chunk!(device, data) do
    case :file.write(device, data) do
      :ok -> :ok
      {:error, posix} -> throw({:write_failed, posix})
    end
  end

  defp get_header(%{headers: headers}, key) do
    case Map.get(headers, key) do
      [value | _rest] -> value
      _missing -> nil
    end
  end
end
