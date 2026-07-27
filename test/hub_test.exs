defmodule LlamaCppEx.HubTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias LlamaCppEx.Hub

  describe "cache_path/3" do
    @tag :tmp_dir
    test "builds correct directory structure", %{tmp_dir: tmp_dir} do
      path = Hub.cache_path("Qwen/Qwen3-4B-GGUF", "model.gguf", cache_dir: tmp_dir)
      assert path == Path.join([tmp_dir, "Qwen", "Qwen3-4B-GGUF", "main", "model.gguf"])
    end

    @tag :tmp_dir
    test "uses LLAMA_CACHE_DIR env var", %{tmp_dir: tmp_dir} do
      System.put_env("LLAMA_CACHE_DIR", tmp_dir)
      path = Hub.cache_path("org/model", "file.gguf")
      assert path == Path.join([tmp_dir, "org", "model", "main", "file.gguf"])
    after
      System.delete_env("LLAMA_CACHE_DIR")
    end

    # W-17: `revision` was absent from the cache key, so pinning
    # `revision: "<sha>"` returned whatever had been cached for `main` and the pin
    # bought nothing — silently, which is the whole problem with a pin that does
    # not pin.
    @tag :tmp_dir
    test "two revisions of the same file cache separately", %{tmp_dir: tmp_dir} do
      main = Hub.cache_path("org/model", "model.gguf", cache_dir: tmp_dir)
      pinned = Hub.cache_path("org/model", "model.gguf", cache_dir: tmp_dir, revision: "abc123")

      assert main != pinned
      assert main == Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      assert pinned == Path.join([tmp_dir, "org", "model", "abc123", "model.gguf"])
    end

    @tag :tmp_dir
    test "an explicit revision: \"main\" is the same path as the default", %{tmp_dir: tmp_dir} do
      assert Hub.cache_path("org/model", "f.gguf", cache_dir: tmp_dir) ==
               Hub.cache_path("org/model", "f.gguf", cache_dir: tmp_dir, revision: "main")
    end

    # The revision is a caller-supplied path component now, so it gets the same
    # traversal validation as the repo id and the filename.
    @tag :tmp_dir
    test "a traversing revision is refused", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid revision.*path components/s, fn ->
        Hub.cache_path("org/model", "model.gguf", cache_dir: tmp_dir, revision: "../../../etc")
      end

      assert_raise ArgumentError, ~r/invalid revision.*relative path/s, fn ->
        Hub.cache_path("org/model", "model.gguf", cache_dir: tmp_dir, revision: "/etc")
      end
    end
  end

  describe "cache_path/3 rejects components that escape the cache directory" do
    @tag :tmp_dir
    test "a repo id containing ../", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid repository id.*path components/s, fn ->
        Hub.cache_path("../../../../tmp", "model.gguf", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "a filename containing ../", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid filename.*path components/s, fn ->
        Hub.cache_path("org/model", "../../../../tmp/evil.gguf", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "an absolute filename", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r{invalid filename "/etc/cron\.d/evil".*relative path}s, fn ->
        Hub.cache_path("org/model", "/etc/cron.d/evil", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "an absolute repo id", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid repository id.*relative path/s, fn ->
        Hub.cache_path("/etc", "passwd", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "a Windows drive-qualified filename", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid filename.*relative path/s, fn ->
        Hub.cache_path("org/model", "C:\\Windows\\evil.gguf", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "a backslash-separated .. component", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid filename.*path components/s, fn ->
        Hub.cache_path("org/model", "sub\\..\\..\\evil.gguf", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "a ~-prefixed component", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid repository id.*path components/s, fn ->
        Hub.cache_path("~", ".ssh/authorized_keys", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "an empty or dot-only component", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid filename.*must not be empty/s, fn ->
        Hub.cache_path("org/model", "", cache_dir: tmp_dir)
      end

      assert_raise ArgumentError, ~r/invalid repository id.*path components/s, fn ->
        Hub.cache_path("org//model", "file.gguf", cache_dir: tmp_dir)
      end

      assert_raise ArgumentError, ~r/invalid filename.*path components/s, fn ->
        Hub.cache_path("org/model", "./file.gguf", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "a null byte", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid filename.*null byte/s, fn ->
        Hub.cache_path("org/model", "model.gguf\0.txt", cache_dir: tmp_dir)
      end
    end

    @tag :tmp_dir
    test "a non-string fragment", %{tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/invalid repository id.*expected a string/s, fn ->
        Hub.cache_path(:org_model, "file.gguf", cache_dir: tmp_dir)
      end
    end
  end

  describe "build_download_url/3" do
    test "constructs correct default URL" do
      url = Hub.build_download_url("Qwen/Qwen3-4B-GGUF", "qwen3-4b-q4_k_m.gguf")

      assert url ==
               "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/qwen3-4b-q4_k_m.gguf"
    end

    test "handles revision parameter" do
      url = Hub.build_download_url("org/model", "file.gguf", revision: "v1.0")
      assert url == "https://huggingface.co/org/model/resolve/v1.0/file.gguf"
    end

    test "handles commit sha revision" do
      url = Hub.build_download_url("org/model", "file.gguf", revision: "abc123def")
      assert url == "https://huggingface.co/org/model/resolve/abc123def/file.gguf"
    end
  end

  # Three of these used to assert nothing about what they were named for: the
  # "HF_TOKEN env var" test set no env var (it was a copy of the option test),
  # "option takes precedence" passed a single token so there was nothing to take
  # precedence over, and "empty when no token" asserted only `is_list/1`, which
  # holds for every possible return value.
  #
  # Testing the documented fallback chain means touching the OS environment.
  # Nothing else in the suite reads HF_TOKEN or HUGGING_FACE_HUB_TOKEN, and
  # ExUnit runs a single module's tests sequentially, so the restore below is
  # enough to keep `async: true`.
  describe "auth_headers/1" do
    @token_vars ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]

    setup do
      saved = Map.new(@token_vars, &{&1, System.get_env(&1)})
      Enum.each(@token_vars, &System.delete_env/1)

      on_exit(fn ->
        Enum.each(saved, fn
          {var, nil} -> System.delete_env(var)
          {var, value} -> System.put_env(var, value)
        end)
      end)

      :ok
    end

    test "includes token from option" do
      assert Hub.auth_headers(token: "hf_test123") == [
               {"authorization", "Bearer hf_test123"}
             ]
    end

    test "empty when no token is available anywhere" do
      assert Hub.auth_headers(token: nil) == []
      assert Hub.auth_headers([]) == []
    end

    test "falls back to HF_TOKEN when no option is given" do
      System.put_env("HF_TOKEN", "hf_env_token")

      assert Hub.auth_headers([]) == [{"authorization", "Bearer hf_env_token"}]
      assert Hub.auth_headers(token: nil) == [{"authorization", "Bearer hf_env_token"}]
    end

    test "falls back to the legacy HUGGING_FACE_HUB_TOKEN last" do
      System.put_env("HUGGING_FACE_HUB_TOKEN", "hf_legacy")

      assert Hub.auth_headers([]) == [{"authorization", "Bearer hf_legacy"}]

      # HF_TOKEN wins over the legacy name.
      System.put_env("HF_TOKEN", "hf_current")
      assert Hub.auth_headers([]) == [{"authorization", "Bearer hf_current"}]
    end

    test "the :token option takes precedence over both env vars" do
      System.put_env("HF_TOKEN", "hf_env")
      System.put_env("HUGGING_FACE_HUB_TOKEN", "hf_legacy")

      assert Hub.auth_headers(token: "hf_option") == [{"authorization", "Bearer hf_option"}]
    end
  end

  describe "filter_gguf_files/1" do
    test "filters to only .gguf files" do
      siblings = [
        %{"rfilename" => "model-Q4_K_M.gguf", "size" => 1000},
        %{"rfilename" => "model-Q8_0.gguf", "size" => 2000},
        %{"rfilename" => "README.md", "size" => 100},
        %{"rfilename" => "config.json", "size" => 50}
      ]

      result = Hub.filter_gguf_files(siblings)
      assert length(result) == 2
      filenames = Enum.map(result, & &1.filename)
      assert "model-Q4_K_M.gguf" in filenames
      assert "model-Q8_0.gguf" in filenames
    end

    test "returns maps with filename and size" do
      siblings = [%{"rfilename" => "model.gguf", "size" => 42}]
      [file] = Hub.filter_gguf_files(siblings)
      assert file.filename == "model.gguf"
      assert file.size == 42
    end

    test "handles missing size" do
      siblings = [%{"rfilename" => "model.gguf"}]
      [file] = Hub.filter_gguf_files(siblings)
      assert file.size == 0
    end

    test "returns empty for no GGUF files" do
      siblings = [%{"rfilename" => "README.md"}, %{"rfilename" => "config.json"}]
      assert Hub.filter_gguf_files(siblings) == []
    end

    test "sorted by size ascending" do
      siblings = [
        %{"rfilename" => "big.gguf", "size" => 9000},
        %{"rfilename" => "small.gguf", "size" => 100},
        %{"rfilename" => "mid.gguf", "size" => 5000}
      ]

      result = Hub.filter_gguf_files(siblings)
      sizes = Enum.map(result, & &1.size)
      assert sizes == [100, 5000, 9000]
    end
  end

  describe "download/3 rejects traversal before touching the network" do
    @tag :tmp_dir
    test "a traversing filename writes nothing outside the cache", %{tmp_dir: tmp_dir} do
      cache_dir = Path.join([tmp_dir, "cache", "a"])
      outside = Path.join(tmp_dir, "outside.gguf")

      assert {:error, message} =
               Hub.download("org/model", "../../../../outside.gguf",
                 cache_dir: cache_dir,
                 http_client: no_network()
               )

      assert message =~ "invalid filename"
      refute File.exists?(outside)
    end

    @tag :tmp_dir
    test "a traversing repo id writes nothing outside the cache", %{tmp_dir: tmp_dir} do
      cache_dir = Path.join([tmp_dir, "cache", "a"])
      outside = Path.join(tmp_dir, "evil.gguf")

      assert {:error, message} =
               Hub.download("../../..", "evil.gguf",
                 cache_dir: cache_dir,
                 http_client: no_network()
               )

      assert message =~ "invalid repository id"
      refute File.exists?(outside)
    end
  end

  describe "download/3 integrity verification" do
    @tag :tmp_dir
    @tag capture_log: true
    test "caches a file whose SHA-256 matches the published digest", %{tmp_dir: tmp_dir} do
      body = "gguf bytes"

      assert {:ok, path} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: hub_stub(sha256(body), body)
               )

      assert path == Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      assert File.read!(path) == body
      assert File.read!(path <> ".etag") == ~s("stub-etag")
      assert leftover_parts(path) == []
    end

    @tag :tmp_dir
    @tag capture_log: true
    test "a checksum mismatch discards the download", %{tmp_dir: tmp_dir} do
      body = "tampered bytes"
      expected = sha256("what the repository published")

      assert {:error, message} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: hub_stub(expected, body)
               )

      assert message =~ "checksum mismatch for org/model/model.gguf"
      assert message =~ expected
      assert message =~ sha256(body)

      dest = Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      refute File.exists?(dest)
      assert leftover_parts(dest) == []
    end

    @tag :tmp_dir
    test "an unreachable metadata endpoint fails the download", %{tmp_dir: tmp_dir} do
      client = fn _url, _req_opts -> {:ok, Req.Response.new(status: 500)} end

      assert {:error, message} =
               Hub.download("org/model", "model.gguf", cache_dir: tmp_dir, http_client: client)

      assert message =~ "cannot verify org/model/model.gguf"
      assert message =~ "verify_checksum: false"
      refute File.exists?(Path.join([tmp_dir, "org", "model", "main", "model.gguf"]))
    end

    @tag :tmp_dir
    test "a file the revision does not list fails the download", %{tmp_dir: tmp_dir} do
      client = hub_stub(sha256("x"), "x", filename: "other.gguf")

      assert {:error, message} =
               Hub.download("org/model", "model.gguf", cache_dir: tmp_dir, http_client: client)

      assert message =~ "file not found in org/model@main: model.gguf"
    end

    # The downgrade this closes: `sibling_sha256/5` fell through to `{:ok, nil}`
    # and `verify_integrity/3` turned `nil` into a warning and `:ok`, so
    # verification was downgradable by the *metadata* response — strip one JSON
    # key and the bytes are cached unverified. A MITM with a trusted cert, or the
    # TLS-terminating corporate proxy this module explicitly supports, can do that.
    @tag :tmp_dir
    test "a file with no published digest is refused by default", %{tmp_dir: tmp_dir} do
      body = "small gguf"

      assert {:error, message} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: hub_stub(nil, body)
               )

      assert message =~ "publishes no SHA-256"
      # And it names both escape hatches, so the error is actionable.
      assert message =~ "verify_checksum: :best_effort"
      assert message =~ "verify_checksum: false"

      # Nothing was cached, and no temp file survived.
      dest = Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      refute File.exists?(dest)
      assert leftover_parts(dest) == []
    end

    @tag :tmp_dir
    test "verify_checksum: :best_effort warns and proceeds", %{tmp_dir: tmp_dir} do
      body = "small gguf"

      log =
        capture_log(fn ->
          assert {:ok, path} =
                   Hub.download("org/model", "model.gguf",
                     cache_dir: tmp_dir,
                     verify_checksum: :best_effort,
                     http_client: hub_stub(nil, body)
                   )

          assert File.read!(path) == body
        end)

      assert log =~ "publishes no SHA-256"
      assert log =~ ":best_effort"
    end

    @tag :tmp_dir
    test ":best_effort still verifies when a digest IS published", %{tmp_dir: tmp_dir} do
      body = "gguf bytes"

      assert {:error, message} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 verify_checksum: :best_effort,
                 http_client: hub_stub(String.duplicate("a", 64), body)
               )

      assert message =~ "checksum mismatch"
    end

    @tag :tmp_dir
    test "an unrecognised :verify_checksum value is rejected, not treated as truthy", %{
      tmp_dir: tmp_dir
    } do
      # `if Keyword.get(opts, :verify_checksum, true)` used to accept anything
      # truthy, so a typo like `verify_checksum: :yes` silently meant "verify" and
      # `verify_checksum: nil` silently meant "do not".
      for value <- [:yes, :required, "true", 1, nil] do
        assert {:error, message} =
                 Hub.download("org/model", "model.gguf",
                   cache_dir: tmp_dir,
                   verify_checksum: value,
                   http_client: no_network()
                 ),
               "verify_checksum: #{inspect(value)} should be rejected"

        assert message =~ "invalid :verify_checksum"
      end
    end

    @tag :tmp_dir
    test "verify_checksum: false skips the metadata request and warns", %{tmp_dir: tmp_dir} do
      body = "unverified gguf"
      client = hub_stub(:no_metadata_request, body)

      log =
        capture_log(fn ->
          assert {:ok, path} =
                   Hub.download("org/model", "model.gguf",
                     cache_dir: tmp_dir,
                     verify_checksum: false,
                     http_client: client
                   )

          assert File.read!(path) == body
        end)

      assert log =~ "integrity verification disabled for org/model/model.gguf"
    end
  end

  describe "download/3 temporary file handling" do
    @tag :tmp_dir
    @tag capture_log: true
    test "the temp file is created exclusively before any body arrives", %{tmp_dir: tmp_dir} do
      body = "gguf bytes"
      dest = Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      test_pid = self()

      # Runs while the request is in flight, i.e. after `File.open/2` and before
      # the first byte is written — the exact window a planted symlink used.
      observe = fn ->
        [tmp] = leftover_parts(dest)

        send(
          test_pid,
          {:tmp, tmp, File.lstat!(tmp).type, File.open(tmp, [:write, :binary, :exclusive])}
        )
      end

      assert {:ok, ^dest} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: hub_stub(sha256(body), body, before_body: observe)
               )

      assert_received {:tmp, _tmp, type, open_result}

      # It already exists as a regular file. The lazy `File.stream!` sink this
      # replaced did not create it until the first write, so there was a window
      # in which the name could still be claimed.
      assert type == :regular

      # O_EXCL on the very path the library chose: a second creator loses. This
      # is the call `do_stream_download/3` itself makes, so an entry planted at
      # that name ahead of the download fails the open rather than being written
      # through.
      assert open_result == {:error, :eexist}

      assert File.read!(dest) == body
      assert leftover_parts(dest) == []
    end

    @tag :tmp_dir
    @tag capture_log: true
    test "a symlink swapped in mid-download cannot redirect the write", %{tmp_dir: tmp_dir} do
      victim = Path.join(tmp_dir, "victim")
      File.write!(victim, "untouched")
      body = "gguf bytes"
      dest = Path.join([tmp_dir, "org", "model", "main", "model.gguf"])

      # Even an attacker who wins the race *after* the exclusive open cannot get
      # bytes into the victim: the device is already bound to the original
      # inode, so the body goes to the now-unlinked file, and the checksum then
      # fails because the symlink reads the victim's contents back instead.
      swap = fn ->
        [tmp] = leftover_parts(dest)
        File.rm!(tmp)
        File.ln_s!(victim, tmp)
      end

      assert {:error, message} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: hub_stub(sha256(body), body, before_body: swap)
               )

      assert message =~ "checksum mismatch"
      assert File.read!(victim) == "untouched"
      refute File.exists?(dest)
      assert leftover_parts(dest) == []
    end

    @tag :tmp_dir
    @tag capture_log: true
    test "the temp target is unpredictable and lands beside the destination", %{tmp_dir: tmp_dir} do
      body = "gguf bytes"
      dest = Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      test_pid = self()

      record = fn -> send(test_pid, {:tmp, hd(leftover_parts(dest))}) end

      download = fn ->
        File.rm_rf!(Path.dirname(dest))

        {:ok, ^dest} =
          Hub.download("org/model", "model.gguf",
            cache_dir: tmp_dir,
            http_client: hub_stub(sha256(body), body, before_body: record)
          )

        receive do
          {:tmp, tmp} -> tmp
        after
          0 -> flunk("the stub never observed a temp file")
        end
      end

      first = download.()
      second = download.()

      # A fixed `<dest>.part` can be pre-created as a symlink, and two concurrent
      # downloads of the same file would share it.
      refute first == second
      assert Path.dirname(first) == Path.dirname(dest)
      refute first in [dest <> ".part", dest <> ".download"]

      # 12 random bytes, base64url-encoded, between the filename and ".part".
      entropy = first |> Path.basename(".part") |> String.replace_prefix("model.gguf.", "")
      assert String.length(entropy) >= 16
    end

    @tag :tmp_dir
    @tag capture_log: true
    test "a non-200 body is never written to the temp file", %{tmp_dir: tmp_dir} do
      dest = Path.join([tmp_dir, "org", "model", "main", "model.gguf"])
      test_pid = self()

      # A function `into:` is driven for every status, so without hub.ex's
      # explicit `resp.status == 200` guard an error page would land in the
      # cache. Measured after the body is fed, before the temp file is removed.
      measure = fn -> send(test_pid, {:written, File.stat!(hd(leftover_parts(dest))).size}) end

      assert {:error, message} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 verify_checksum: false,
                 http_client:
                   hub_stub(:no_metadata_request, "<html>404 not found</html>",
                     status: 404,
                     after_body: measure
                   )
               )

      assert message =~ "file not found"
      assert_received {:written, 0}
      refute File.exists?(dest)
      assert leftover_parts(dest) == []
    end
  end

  describe "download/3 cache permissions" do
    @tag :tmp_dir
    @tag capture_log: true
    test "cached files are 0o600 and the directories they land in are 0o700", %{tmp_dir: tmp_dir} do
      body = "gguf bytes"
      cache_dir = Path.join(tmp_dir, "cache")

      assert {:ok, path} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: cache_dir,
                 http_client: hub_stub(sha256(body), body)
               )

      assert mode(path) == 0o600
      assert mode(path <> ".etag") == 0o600
      assert mode(cache_dir) == 0o700
      assert mode(Path.join(cache_dir, "org")) == 0o700
      assert mode(Path.join([cache_dir, "org", "model"])) == 0o700
      assert mode(Path.join([cache_dir, "org", "model", "main"])) == 0o700
    end
  end

  describe "download/3 caching" do
    @tag :tmp_dir
    test "a cached file is returned with no HTTP request at all", %{tmp_dir: tmp_dir} do
      repo_dir = Path.join([tmp_dir, "test-org", "test-model", "main"])
      File.mkdir_p!(repo_dir)
      cached_path = Path.join(repo_dir, "model.gguf")
      File.write!(cached_path, "fake model data")

      assert {:ok, ^cached_path} =
               Hub.download("test-org/test-model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: no_network()
               )
    end

    # KNOWN GAP: `Hub` writes a `<file>.etag` sidecar but never reads it back, so
    # a file that changed upstream is served from the cache unrevalidated. This
    # pins the behaviour the code actually has, which is also what the moduledoc
    # and `download/3`'s @doc now describe — neither claims ETag revalidation any
    # more. Implementing revalidation means a metadata round-trip on every cached
    # load (including `ModelManager`'s) plus an offline fallback; when that lands,
    # this test flips to asserting the refresh.
    @tag :tmp_dir
    test "a stale cached file is not revalidated against the Hub", %{tmp_dir: tmp_dir} do
      repo_dir = Path.join([tmp_dir, "org", "model", "main"])
      File.mkdir_p!(repo_dir)
      cached = Path.join(repo_dir, "model.gguf")
      File.write!(cached, "stale bytes")
      File.write!(cached <> ".etag", ~s("old-etag"))

      fresh = "fresh bytes"

      assert {:ok, ^cached} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 http_client: hub_stub(sha256(fresh), fresh)
               )

      assert File.read!(cached) == "stale bytes"
      assert File.read!(cached <> ".etag") == ~s("old-etag")
    end

    @tag :tmp_dir
    @tag capture_log: true
    test "force: true re-downloads over a cached file", %{tmp_dir: tmp_dir} do
      repo_dir = Path.join([tmp_dir, "org", "model", "main"])
      File.mkdir_p!(repo_dir)
      cached = Path.join(repo_dir, "model.gguf")
      File.write!(cached, "stale bytes")

      fresh = "fresh bytes"

      assert {:ok, ^cached} =
               Hub.download("org/model", "model.gguf",
                 cache_dir: tmp_dir,
                 force: true,
                 http_client: hub_stub(sha256(fresh), fresh)
               )

      assert File.read!(cached) == fresh
      assert mode(cached) == 0o600
    end

    @tag :tmp_dir
    test "offline mode returns cached file", %{tmp_dir: tmp_dir} do
      System.put_env("LLAMA_OFFLINE", "1")
      repo_dir = Path.join([tmp_dir, "org", "model", "main"])
      File.mkdir_p!(repo_dir)
      path = Path.join(repo_dir, "cached.gguf")
      File.write!(path, "cached")

      assert {:ok, ^path} =
               Hub.download("org/model", "cached.gguf",
                 cache_dir: tmp_dir,
                 http_client: no_network()
               )
    after
      System.delete_env("LLAMA_OFFLINE")
    end

    @tag :tmp_dir
    test "offline mode errors when not cached", %{tmp_dir: tmp_dir} do
      System.put_env("LLAMA_OFFLINE", "1")

      assert {:error, msg} =
               Hub.download("org/model", "missing.gguf",
                 cache_dir: tmp_dir,
                 http_client: no_network()
               )

      assert msg =~ "offline"
    after
      System.delete_env("LLAMA_OFFLINE")
    end
  end

  # --- Network isolation ---
  #
  # `hub.ex` takes its HTTP client as the `:http_client` option (a
  # `(url, req_opts)` function defaulting to `&Req.get/2`) so these tests never
  # touch the network. `Req.Test` would be the idiomatic stub, but it needs the
  # optional :plug dependency, which this project does not carry.

  # Fails the test if `download/3` issues any request at all.
  defp no_network do
    fn url, _req_opts -> flunk("download/3 unexpectedly issued a request to #{url}") end
  end

  # Covers both requests `download/3` makes: the revision-metadata lookup that
  # supplies the expected SHA-256, and the file fetch itself. Pass
  # `:no_metadata_request` as the digest to assert the metadata lookup is
  # skipped. `:status` sets the file response's status; `:before_body` and
  # `:after_body` are zero-arity hooks run either side of the body being fed,
  # which is when the library's temp file is open.
  defp hub_stub(digest, body, opts \\ []) do
    filename = Keyword.get(opts, :filename, "model.gguf")
    status = Keyword.get(opts, :status, 200)
    before_body = Keyword.get(opts, :before_body, fn -> :ok end)
    after_body = Keyword.get(opts, :after_body, fn -> :ok end)

    fn url, req_opts ->
      if String.contains?(url, "/revision/") do
        stub_metadata(url, filename, digest)
      else
        stub_file(req_opts, body, status, {before_body, after_body})
      end
    end
  end

  defp stub_metadata(url, _filename, :no_metadata_request) do
    flunk("download/3 requested #{url} even though verify_checksum was false")
  end

  defp stub_metadata(_url, filename, digest) do
    {:ok, Req.Response.new(status: 200, body: %{"siblings" => [sibling(filename, digest)]})}
  end

  defp stub_file(req_opts, body, status, {before_body, after_body}) do
    into = Keyword.fetch!(req_opts, :into)
    before_body.()
    resp = Req.Response.new(status: status, headers: [{"etag", ~s("stub-etag")}])
    {_req, resp} = feed(into, [body], resp)
    after_body.()
    {:ok, resp}
  end

  # Mirrors Req's function-`into:` contract: the callback is invoked once per
  # data chunk with `{:data, chunk}` and the `{req, resp}` accumulator, and its
  # `{:cont, acc}` / `{:halt, acc}` return decides whether streaming continues.
  # Unlike the Collectable form, Req drives it for *every* status, which is why
  # `hub.ex` checks `resp.status` before writing — feeding a non-200 body here is
  # what gives that check teeth.
  defp feed(into, chunks, resp) when is_function(into, 2) do
    Enum.reduce_while(chunks, {Req.new(), resp}, fn chunk, acc ->
      case into.({:data, chunk}, acc) do
        {:cont, acc} -> {:cont, acc}
        {:halt, acc} -> {:halt, acc}
      end
    end)
  end

  defp sibling(filename, nil), do: %{"rfilename" => filename}

  defp sibling(filename, digest),
    do: %{"rfilename" => filename, "lfs" => %{"sha256" => digest}}

  defp sha256(data), do: :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)

  defp mode(path), do: Bitwise.band(File.stat!(path).mode, 0o777)

  defp leftover_parts(dest), do: Path.wildcard(Path.dirname(dest) <> "/*.part")
end
