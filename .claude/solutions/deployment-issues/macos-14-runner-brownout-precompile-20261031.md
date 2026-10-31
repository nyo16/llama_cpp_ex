---
module: "Precompile workflow"
date: "2026-10-31"
problem_type: deployment_issue
component: github_actions
symptoms:
  - "This job failed because of a scheduled macOS 14 runner image brownout. macOS 14 runner images will be retired on November 2, 2026."
  - "both macos-14 Metal precompile legs failed in 1 s; the release stayed a draft and nothing reached Hex"
root_cause: "GitHub retires the macos-14 image on 2026-11-02 and runs scheduled brownouts beforehand; the precompile matrix pinned macos-14 for the Metal artifacts"
severity: high
tags: [github-actions, macos, runner-image, metal, precompile, release, deployment-target]
---

# Metal precompile legs failed: macOS 14 runner brownout

## Symptoms

v0.8.63 tag push → `precompile (macos-14, aarch64-apple-darwin, ...)` failed
in 1 s with the brownout annotation; checksum/publish jobs could not run.

## Investigation

1. Moving to `macos-15` raises the SDK; without a deployment target the
   artifact's `LC_BUILD_VERSION minos` becomes 15.x and refuses to load on
   macOS 14 (a locally built `.so` reported `minos 27.0`).
2. Release had not published (draft only), so the tag could be moved.

## Root Cause

Hard-pinned runner image reaching end of life; default deployment target
follows the runner SDK.

## Solution

```yaml
# .github/workflows/precompile.yml
- os: macos-15
  ...
env:
  MACOSX_DEPLOYMENT_TARGET: ${{ matrix.backend == 'metal' && '14.0' || '' }}
# plus a step asserting otool -l ... minos == 14.0 on the artifact
```

Release recovery: `gh run cancel <run>`, push fix to master, re-point the
unpublished tag (`git push origin :refs/tags/vX.Y.Z` then push the tag again).
The new run logged `minos: 14.0` and published.

## Prevention

- Pair any runner-image bump with an explicit `MACOSX_DEPLOYMENT_TARGET` and
  an artifact check (`otool -l | awk '/LC_BUILD_VERSION/{f=1} f&&$1=="minos"'`).
- Only move a tag before anything is published (draft release, no Hex).
- Watch actions/runner-images deprecation issues ahead of release days.
