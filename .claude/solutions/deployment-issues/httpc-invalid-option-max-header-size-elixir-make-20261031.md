---
module: "LlamaCppEx.MixProject"
date: "2026-10-31"
problem_type: deployment_issue
component: precompiled_nif_download
symptoms:
  - "[notice] Invalid option {max_header_size,65536} ignored — logged once while downloading the precompiled NIF on OTP 29.0.3"
  - "upstream report for other libraries: {:error, {:header_too_long, {:max, 10240}}} when fetching a NIF from GitHub releases"
root_cause: "OTP 29.0.6 / 28.5.0.6 / 27.3.4.17 added an httpc max_header_size request option defaulting to 10,240 bytes and GitHub release redirects can exceed it; elixir_make 0.10.1 passes max_header_size: 65_536, which older OTP patch releases reject as an unknown option (httpc request_options_defaults) and log at info"
severity: medium
tags: [elixir-make, httpc, otp, precompiled-nif, github-releases, dependencies]
---

# "Invalid option {max_header_size,65536} ignored" when installing the NIF

## Symptoms

Notice during `mix compile` of a project depending on `llama_cpp_ex` 0.8.63,
OTP 29.0.3, elixir_make 0.10.1. Download and checksum succeeded.

## Investigation

1. **Assumed elixir_make bug** — `httpc:request_options_defaults/0` in
   inets 9.3.1 (OTP 27.2) and 9.7.1 (OTP 29.0.3) lists only sync, stream,
   body_format, full_result, headers_as_is, receiver, socket_opts,
   ipv6_host_with_brackets → `max_header_size` is unknown there.
2. **Upstream history** — elixir-lang/elixir_make#98 added the option because
   newer OTP patch releases enforce a 10 KB header cap; vix failed with
   `header_too_long`. So the option is required on new OTP and harmless on old.
3. **Our exposure** — `{:elixir_make, "~> 0.8"}` let users resolve 0.9/0.10.0,
   which would fail on current OTP patches if GitHub's redirect headers exceed
   10 KB. Measured our `v0.8.63` redirect: 5,595 bytes (302) + 902 (200) —
   below the cap today, but it depends on the signed Location URL.

## Root Cause

An OTP patch release introduced a new enforced default; the library fix uses a
new option name older OTP rejects with a notice. Our dependency floor allowed
the pre-fix downloader.

## Solution

```elixir
# mix.exs
{:elixir_make, "~> 0.10.1", runtime: false},
```

README "Logging" documents the notice as harmless.

## Prevention

- When a CI/user notice comes from a dependency, check the dependency's
  latest release and changelog before patching around it; here the "bug" was
  the fix.
- Keep build-time downloader deps' floors at the version that handles current
  OTP behaviour.
- Related annotation in the same session: `"xref: [exclude: ...]" ... is
  deprecated` came from old elixir_make/mint/req releases, fixed by updating
  them (PR #113), not by editing our mix.exs.
