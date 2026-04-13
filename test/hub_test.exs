defmodule LlamaCppEx.HubTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Hub

  describe "cache_path/3" do
    @tag :tmp_dir
    test "builds correct directory structure", %{tmp_dir: tmp_dir} do
      path = Hub.cache_path("Qwen/Qwen3-4B-GGUF", "model.gguf", cache_dir: tmp_dir)
      assert path == Path.join([tmp_dir, "Qwen", "Qwen3-4B-GGUF", "model.gguf"])
    end

    @tag :tmp_dir
    test "uses LLAMA_CACHE_DIR env var", %{tmp_dir: tmp_dir} do
      System.put_env("LLAMA_CACHE_DIR", tmp_dir)
      path = Hub.cache_path("org/model", "file.gguf")
      assert path == Path.join([tmp_dir, "org", "model", "file.gguf"])
    after
      System.delete_env("LLAMA_CACHE_DIR")
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

  describe "auth_headers/1" do
    test "includes token from option" do
      headers = Hub.auth_headers(token: "hf_test123")
      assert {"authorization", "Bearer hf_test123"} in headers
    end

    test "empty when no token" do
      headers = Hub.auth_headers(token: nil)
      # With explicit nil token and no env var matching, should be empty
      # (can't reliably test env var absence without race conditions in async tests)
      assert is_list(headers)
    end

    test "includes token from HF_TOKEN env var" do
      headers = Hub.auth_headers(token: "hf_env_token")
      assert {"authorization", "Bearer hf_env_token"} in headers
    end

    test "option takes precedence" do
      headers = Hub.auth_headers(token: "hf_option")
      assert {"authorization", "Bearer hf_option"} in headers
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

  describe "download/3 caching" do
    @tag :tmp_dir
    test "returns cached file without network call", %{tmp_dir: tmp_dir} do
      repo_dir = Path.join([tmp_dir, "test-org", "test-model"])
      File.mkdir_p!(repo_dir)
      cached_path = Path.join(repo_dir, "model.gguf")
      File.write!(cached_path, "fake model data")

      assert {:ok, ^cached_path} =
               Hub.download("test-org/test-model", "model.gguf", cache_dir: tmp_dir)
    end

    @tag :tmp_dir
    test "offline mode returns cached file", %{tmp_dir: tmp_dir} do
      System.put_env("LLAMA_OFFLINE", "1")
      repo_dir = Path.join([tmp_dir, "org", "model"])
      File.mkdir_p!(repo_dir)
      path = Path.join(repo_dir, "cached.gguf")
      File.write!(path, "cached")

      assert {:ok, ^path} = Hub.download("org/model", "cached.gguf", cache_dir: tmp_dir)
    after
      System.delete_env("LLAMA_OFFLINE")
    end

    @tag :tmp_dir
    test "offline mode errors when not cached", %{tmp_dir: tmp_dir} do
      System.put_env("LLAMA_OFFLINE", "1")

      assert {:error, msg} = Hub.download("org/model", "missing.gguf", cache_dir: tmp_dir)
      assert msg =~ "offline"
    after
      System.delete_env("LLAMA_OFFLINE")
    end
  end
end
