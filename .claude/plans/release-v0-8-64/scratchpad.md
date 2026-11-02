# Scratchpad: release v0.8.64

## DECISION

- No llama.cpp bump (user, 2026-10-31). Upstream master was 22 commits
  ahead of `418f905a4` (head `ddf48f85c`); do that as its own release with
  the full release-guide diff (header, decision port, tokenizer partition).
- Merge #115 (docs only) before tagging (user).
- No research agents spawned: the process is fixed by docs/release-guide.md
  and was exercised end to end for v0.8.63 the same day.
- Skip decision_compare.exs: no decision or tokenizer code changed since
  v0.8.63, which was ALL MATCH on 9 models.

## Known, not regressions

- Llama-3.2-1B-Instruct `:tools` round trip: calls the tool again (10/11).
- OTP 29.0.3 locally: `Invalid option {max_header_size,65536} ignored` during
  the NIF download is expected (httpc predates the option).
- "Found packages with security advisories" should be gone after #114
  (decimal 3.2.0); if it reappears, check `mix deps.get` output.

## DEAD-END

(none yet)

## Flake found on #115 CI (2026-10-31)

`ServerSmokeTest` "halting a stream early cancels generation" got another
server's `:done` (prompt 39, `max_tokens`) — the preceding chat_completion
test's server, still alive after its test exited (ExUnit doesn't wait for
linked processes), emitting telemetry after `GenServer.reply`. Fixed by
pinning `metadata.server` in the collectors. Not reproducible locally on Metal
with seed 489492 (timing); evidence is the CI log ordering (handler attach at
54.311, stale done logged 54.314).
