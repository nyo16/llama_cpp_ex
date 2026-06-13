defmodule LlamaCppEx.HubProxyTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias LlamaCppEx.Hub

  @https "https://huggingface.co/api/models"
  @http "http://example.com/models"

  describe "proxy_request_options/2 with explicit :proxy option" do
    test "http proxy on an https target tunnels via CONNECT (proxy tuple keeps the proxy's own scheme)" do
      assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}]] =
               Hub.proxy_request_options(@https, proxy: "http://127.0.0.1:8118", no_proxy: "")
    end

    test "http proxy on an http target" do
      assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}]] =
               Hub.proxy_request_options(@http, proxy: "http://127.0.0.1:8118", no_proxy: "")
    end

    test "an https proxy scheme is preserved" do
      assert [connect_options: [proxy: {:https, "proxy.local", 8443, []}]] =
               Hub.proxy_request_options(@https, proxy: "https://proxy.local:8443", no_proxy: "")
    end

    test "a scheme-less proxy URL defaults to http" do
      assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}]] =
               Hub.proxy_request_options(@https, proxy: "127.0.0.1:8118", no_proxy: "")
    end

    test "defaults to port 80 for an http proxy without an explicit port" do
      assert [connect_options: [proxy: {:http, "proxy.local", 80, []}]] =
               Hub.proxy_request_options(@https, proxy: "http://proxy.local", no_proxy: "")
    end

    test "defaults to port 443 for an https proxy without an explicit port" do
      assert [connect_options: [proxy: {:https, "proxy.local", 443, []}]] =
               Hub.proxy_request_options(@https, proxy: "https://proxy.local", no_proxy: "")
    end

    test "userinfo becomes a basic proxy-authorization header and is stripped from the host" do
      assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}, proxy_headers: headers]] =
               Hub.proxy_request_options(@https,
                 proxy: "http://user:pass@127.0.0.1:8118",
                 no_proxy: ""
               )

      assert {"proxy-authorization", "Basic " <> encoded} =
               List.keyfind(headers, "proxy-authorization", 0)

      assert Base.decode64!(encoded) == "user:pass"
    end

    test "proxy: false disables proxying entirely" do
      assert [] = Hub.proxy_request_options(@https, proxy: false)
    end

    test "a Mint proxy tuple is passed through unchanged" do
      assert [connect_options: [proxy: {:http, "h", 3128, []}]] =
               Hub.proxy_request_options(@https, proxy: {:http, "h", 3128, []})
    end
  end

  describe "proxy_request_options/2 SOCKS handling (Mint has no SOCKS support)" do
    test "a socks5 proxy is ignored with an actionable warning" do
      log =
        capture_log(fn ->
          assert [] =
                   Hub.proxy_request_options(@https,
                     proxy: "socks5://127.0.0.1:1080",
                     no_proxy: ""
                   )
        end)

      assert log =~ "SOCKS"
    end

    test "a socks5h proxy is also ignored" do
      assert [] =
               Hub.proxy_request_options(@https, proxy: "socks5h://127.0.0.1:1080", no_proxy: "")
    end
  end

  describe "proxy_request_options/2 NO_PROXY handling" do
    test "an exact host match bypasses the proxy" do
      assert [] =
               Hub.proxy_request_options("https://internal.corp/x",
                 proxy: "http://127.0.0.1:8118",
                 no_proxy: "internal.corp"
               )
    end

    test "a domain suffix match bypasses the proxy" do
      assert [] =
               Hub.proxy_request_options("https://api.internal.corp/x",
                 proxy: "http://127.0.0.1:8118",
                 no_proxy: ".internal.corp"
               )
    end

    test "a wildcard bypasses all hosts" do
      assert [] =
               Hub.proxy_request_options(@https, proxy: "http://127.0.0.1:8118", no_proxy: "*")
    end

    test "a non-matching no_proxy entry still uses the proxy" do
      assert [connect_options: [proxy: {:http, _, _, _}]] =
               Hub.proxy_request_options(@https,
                 proxy: "http://127.0.0.1:8118",
                 no_proxy: "internal.corp"
               )
    end
  end

  describe "proxy_request_options/2 environment auto-detection" do
    test "HTTPS_PROXY is used for https targets" do
      with_env(%{"HTTPS_PROXY" => "http://127.0.0.1:8118"}, fn ->
        assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}]] =
                 Hub.proxy_request_options(@https)
      end)
    end

    test "HTTP_PROXY is used for http targets" do
      with_env(%{"HTTP_PROXY" => "http://127.0.0.1:8118"}, fn ->
        assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}]] =
                 Hub.proxy_request_options(@http)
      end)
    end

    test "the scheme-specific HTTPS_PROXY wins over a (SOCKS) ALL_PROXY" do
      with_env(
        %{"HTTPS_PROXY" => "http://127.0.0.1:8118", "ALL_PROXY" => "socks5://127.0.0.1:1080"},
        fn ->
          assert [connect_options: [proxy: {:http, "127.0.0.1", 8118, []}]] =
                   Hub.proxy_request_options(@https)
        end
      )
    end

    test "ALL_PROXY is used as a fallback when no scheme-specific proxy is set" do
      with_env(%{"ALL_PROXY" => "http://127.0.0.1:3128"}, fn ->
        assert [connect_options: [proxy: {:http, "127.0.0.1", 3128, []}]] =
                 Hub.proxy_request_options(@https)
      end)
    end

    test "no proxy configured returns an empty option list" do
      with_env(%{}, fn ->
        assert [] = Hub.proxy_request_options(@https)
      end)
    end
  end

  # Clears every proxy-related env var, applies the given overrides, runs `fun`,
  # then restores the original environment. Keeps these tests deterministic
  # regardless of the developer's ambient shell (which may itself set a proxy).
  defp with_env(env, fun) do
    keys =
      ~w(HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy)

    saved = Map.new(keys, fn k -> {k, System.get_env(k)} end)

    try do
      Enum.each(keys, &System.delete_env/1)
      Enum.each(env, fn {k, v} -> System.put_env(k, v) end)
      fun.()
    after
      Enum.each(keys, fn k ->
        case saved[k] do
          nil -> System.delete_env(k)
          v -> System.put_env(k, v)
        end
      end)
    end
  end
end
