# Coding-agent engineering evidence (real interaction record)

This document records work performed by the coding agent in the **current
session** (2026-11-23). It is intentionally limited to events that actually
occurred. No agent experiment, tool use, review comment, benchmark score or
result has been invented.

## Agent identity and tools actually used

- Agent: OpenCode coding agent (this session), operating on local clones.
- Tools actually used: local shell, file read/write/edit, `git`, `gh` CLI,
  `python -m pytest`, `ruff`, and read-only HTTP fetches of official Kaggle /
  PhysioNet documentation and public APIs.
- Tools **not** used: no external autonomous agent harness, no GitHub Actions
  run created by me on an unapproved branch, no real Kaggle login, no dataset
  download, no EDF processing.

## Task specification (as given by the human)

Across successive explicit instructions, the human asked for: scientific fixes
to `SleepPSG-CI` (cohort identity, label normalization, unit safety, full-night
chunking, leakage-safe evaluation), then repeated data-access readiness fixes
(Kaggle credential detection, CLI compatibility, XDG and token-path semantics),
with mandatory constraints: no fabricated results, no real-data claims without
real data, no hidden-label access, no credential exposure, no force-push.

Each task included required commands (`pip install -e ".[dev]"`, `pytest -q`,
`ruff check .`, `git diff --check`), a required commit message, CI verification
on Python 3.10/3.11/3.12, and an explicit final report format.

## Agent-generated changes (verified commits)

All commits below are on `SleepPSG-CI/main` and were pushed only after the human
explicitly requested a push in each task.

| Commit | Change | CI run | Result |
|---|---|---|---|
| `51714b1` | Cohort integration, unit-aware feature gating, chunked EDF reader | `45148973843` | **failed** (see failure analysis) |
| `a974e67` | Correction: platform-neutral recording-identity test | `45149228489` | success |
| `58979b7` | All 7 modalities in feature selection; identity counts; unit module; tests | `45152940396` | success |
| `48bfcdd` | Gate unit-unsafe and signal-range features out of model matrices | `45154764564` | success |
| `3eba1f4` | Real-data readiness; duplicate EDF label and epoch-cap fixes; experiment entry point | `45163930625` | success |
| `90a3906` | Kaggle credential detection (env / kaggle.json / config dir) and download safeguards | `45166076488` | success |
| `964ffde` | Current CLI auth methods (API token, access_token, OAuth) and docs | `45168549386` | success |
| `6c6d467` | CLI precedence and XDG fallback correctness | `45172835668` | success |
| `b90889b` | `access_token.txt` CLI verification and compatibility documentation | `45179586478` | success |
| `7452067` | `KAGGLE_API_TOKEN` empty-file semantics | `45190600679` | success |
| `b612361` | Accuracy correction: directory handling is a documented safety deviation | `45196864279` | success |

Test counts rose from 59 (initial scaffold) to 193 as regressions were added.
Ruff reported `All checks passed!` after each final commit.

## Human review (actual)

The human reviewed each final report and issued the next task. Concrete review
actions that changed the work:

1. After `58979b7`, the human identified that airflow/SpO2 were still omitted
   from model feature selection, composite identity counting was incomplete,
   and units were not validated. This produced `48bfcdd`.
2. After `48bfcdd`, the human identified that unknown-unit amplitude features
   and respiratory signal-range features could still enter models. This
   produced `3eba1f4`'s predecessor scope and then `48bfcdd`-style gating work.
3. After `6c6d467`, the human challenged the `access_token.txt` evidence,
   requiring CLI-level verification rather than SDK inference. I re-verified
   the CLI call path and documented it in `b90889b`.
4. After `7452067`, the human corrected an overclaim: an existing directory
   supplied as a token path can raise `IsADirectoryError` in the SDK; our safe
   rejection must not be described as exact CLI mirroring. This produced
   `b612361`.

## Rejected suggestions and failure analysis

- **Rejected: treating a bare `BDSPPatientID` as a patient identity.** The
  official scorer uses `SiteID_BDSPPatientID`; the implementation was corrected
  to use the composite key everywhere (grouping, bootstrap, join).
- **Rejected: converting indeterminate labels to negatives.** Kept as excluded,
  with explicit counts.
- **Rejected: leading-600-second features described as full-night.** Replaced
  with chunked full-recording processing; earlier wording corrected in docs.
- **Rejected: claiming real-data results from synthetic fixtures.** Every final
  report explicitly separated synthetic tests from real-data evidence.
- **CI failure (real):** `45148973843` failed on Python 3.10/3.11/3.12 because a
  test hard-coded a Windows path. Root cause: platform-dependent test fixture,
  not production code. Fixed by using `pathlib` in the test; rerun succeeded.
- **Defect found during testing:** duplicate EDF channel labels collapsed into
  one channel, silently discarding a recording. Fixed with deterministic
  disambiguation and raw-label provenance; regression tests added.
- **Defect found during testing:** epoch generator could yield more epochs than
  the header declared on a duration/sample mismatch. Capped at the declared
  count; regression test added.
- **Corrected overclaim:** documentation initially implied exact CLI mirroring
  for directory token paths; corrected to state our safe deviation.

## Final verification

For every commit above: `python -m pytest -q`, `ruff check .`, `git diff
--check` were run locally, then GitHub Actions was watched to completion on
Python 3.10, 3.11 and 3.12. Final state: 193 tests passing, Ruff clean, clean
working tree, no data/secrets in the remote tree.

## Limits of this evidence

- These records come from `SleepPSG-CI` work, not from the three portfolio
  repositories audited in this assignment. They demonstrate agent workflow
  discipline; they are **not** evidence that those portfolio repos were built
  by an agent.
- No human reviewer signed off on the benchmark report in this assignment;
  where a human-reviewed field appears in generated reports it is marked
  `pending`.
