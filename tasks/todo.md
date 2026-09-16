# Task Checklist

## Task 1: Project configuration and input validation

**Acceptance criteria:**
- [x] Dependencies, Git ignores, model config, and frozen dataset validation are present.
- [x] Dataset validation enforces required columns, labels, unique IDs, eight demos, and sixty tests.

**Verification:**
- [x] Focused dataset/config tests pass.

**Dependencies:** None

## Task 2: Prompt rendering and isolation

**Acceptance criteria:**
- [x] Zero-shot and few-shot templates are external files.
- [x] Test-row label, reason, source type, and source reference cannot enter prompt rendering.
- [x] Few-shot contains exactly eight demo examples.

**Verification:**
- [x] Focused prompt tests pass.

**Dependencies:** Task 1

## Task 3: Resumable prediction

**Acceptance criteria:**
- [x] OpenAI-compatible calls use configured model, endpoint, timeout, retry count, and temperature.
- [x] Raw output, normalized prediction, API status/error, and latency are saved.
- [x] Existing finalized rows are skipped and never overwritten.

**Verification:**
- [x] Focused prediction tests pass without network access.

**Dependencies:** Tasks 1-2

## Task 4: Evaluation

**Acceptance criteria:**
- [x] Ground truth is joined by sample ID only inside evaluation code.
- [x] Required global and per-class metrics are programmatically generated.
- [x] Incorrect, invalid, and API-failure rows are written to `error_cases.csv` and remain distinguishable.

**Verification:**
- [x] Focused evaluation tests pass.

**Dependencies:** Tasks 1 and 3

## Task 5: Unified orchestration

**Acceptance criteria:**
- [x] One CLI runs prediction then evaluation.
- [x] `--limit` uses the identical pipeline and isolated output directory.
- [x] A local fake-client integration test produces all three artifacts.

**Verification:**
- [x] Integration test and CLI help pass.

**Dependencies:** Tasks 3-4

## Task 6: Documentation

**Acceptance criteria:**
- [x] README contains setup and execution commands.
- [x] `docs/implementation_report.md` contains every requested topic without background padding.

**Verification:**
- [x] Commands and paths in documentation match the code.

**Dependencies:** Task 5

## Task 7: Delivery

**Acceptance criteria:**
- [x] Full test, lint, format, and compile checks pass.
- [x] No secret, `.env`, result artifact, or dataset modification is committed.
- [x] Commits are pushed to the target repository.

**Verification:**
- [x] Remote branch contains the final commit hash.

**Dependencies:** Tasks 1-6
