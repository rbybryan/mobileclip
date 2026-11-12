# Release preparation — 2026-11-12

Prepared from `f4ccbfb` for public review. Repository visibility is unchanged by this work.

## Validation

- Deterministic synthetic paired-bootstrap demonstration: passed.
- Statistics, result-collection, and portable-environment regression tests: 28 passed.
- Reported-results chart regenerated from the committed CSV and visually inspected.
- Modified Python files compiled, shell scripts passed `bash -n`, and local Markdown
  links in changed documents resolved.
- Gitleaks 8.30.1 scanned the fetched Git history across all local refs (787 commits at
  the audit baseline), plus the prepared working tree. One reviewed match was the
  model-layer constant `mixer_1d.rbr_conv.0.conv`, not a credential. Exact fingerprints
  are documented in `.gitleaksignore`; no credential matches remained after review.

The demo/test environment is separate from the full training environment. GPU training,
checkpoint evaluation, device latency, and the reported research measurements were not
rerun during preparation. The new CI workflow runs the documented lightweight path.

## Publication boundary

Current release files exclude live operational handoff notes. Their deletion from the
current tree does **not** remove them from existing commits, old branches, or past pull
requests. The original repository has many historical branches. Secret scanning is not
an audit of all private prose, issue/PR comments, Actions logs, artifacts, or external data.

For a release that does not expose that operational history, publish a clean source
snapshot into a new repository with fresh history. Preserve `LICENSE`, `CITATION.cff`,
upstream notices, and the result-provenance documentation. Do not make the original
repository public under the assumption that removing a file on main erases its history.
No history rewrite or branch deletion is performed by this change.

## Before announcing a release

- Choose a fresh-history release or explicitly review the original repository's wider
  history and GitHub discussion/log surfaces before changing its visibility.
- Verify the GitHub CPU checks on the exact release commit.
- Keep the README's experimental scope and statistical caveats with any portfolio claims.
