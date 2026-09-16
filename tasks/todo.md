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
- [ ] Zero-shot and few-shot templates are external files.
- [ ] Test-row label, reason, source type, and source reference cannot enter prompt rendering.
- [ ] Few-shot contains exactly eight demo examples.

**Verification:**
- [ ] Focused prompt tests pass.

**Dependencies:** Task 1

## Task 3: Resumable prediction

**Acceptance criteria:**
- [ ] OpenAI-compatible calls use configured model, endpoint, timeout, retry count, and temperature.
- [ ] Raw output, normalized prediction, API status/error, and latency are saved.
- [ ] Existing finalized rows are skipped and never overwritten.

**Verification:**
- [ ] Focused prediction tests pass without network access.

**Dependencies:** Tasks 1-2

## Task 4: Evaluation

**Acceptance criteria:**
- [ ] Ground truth is joined by sample ID only inside evaluation code.
- [ ] Required global and per-class metrics are programmatically generated.
- [ ] Incorrect, invalid, and API-failure rows are written to `error_cases.csv` and remain distinguishable.

**Verification:**
- [ ] Focused evaluation tests pass.

**Dependencies:** Tasks 1 and 3

## Task 5: Unified orchestration

**Acceptance criteria:**
- [ ] One CLI runs prediction then evaluation.
- [ ] `--limit` uses the identical pipeline and isolated output directory.
- [ ] A local fake-client integration test produces all three artifacts.

**Verification:**
- [ ] Integration test and CLI help pass.

**Dependencies:** Tasks 3-4

## Task 6: Documentation

**Acceptance criteria:**
- [ ] README contains setup and execution commands.
- [ ] `docs/implementation_report.md` contains every requested topic without background padding.

**Verification:**
- [ ] Commands and paths in documentation match the code.

**Dependencies:** Task 5

## Task 7: Delivery

**Acceptance criteria:**
- [ ] Full test, lint, format, and compile checks pass.
- [ ] No secret, `.env`, result artifact, or dataset modification is committed.
- [ ] Commits are pushed to the target repository.

**Verification:**
- [ ] Remote branch contains the final commit hash.

**Dependencies:** Tasks 1-6
