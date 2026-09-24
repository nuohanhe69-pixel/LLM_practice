#!/usr/bin/env bash
set -euo pipefail

audit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
handoff_dir="$audit_dir/handoff_v3"
artifact_dir="$audit_dir/artifacts"
work_dir="$(mktemp -d /tmp/challenge-audit-v3.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

cp -R "$handoff_dir" "$work_dir/handoff_v3"
uv venv "$work_dir/venv" --python 3.12
uv pip install --python "$work_dir/venv/bin/python" -r "$handoff_dir/audit_requirements.txt"
audit_python="$work_dir/venv/bin/python"

cd "$work_dir/handoff_v3"
{
  "$audit_python" revise_candidates.py --input baseline/candidates_full.jsonl --out .
  "$audit_python" - baseline/candidates_full.jsonl baseline/model_input_only.jsonl <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
target.write_text(
    "".join(
        json.dumps({"id": row["id"], "error_text": row["error_text"]}, ensure_ascii=False)
        + "\n"
        for row in rows
    ),
    encoding="utf-8",
)
print(f"Generated baseline model input: {len(rows)} rows")
PY
  "$audit_python" challenge_shortcut_audit.py \
    --dataset baseline/candidates_full.jsonl \
    --model-input baseline/model_input_only.jsonl \
    --out baseline --seeds 10
  "$audit_python" challenge_shortcut_audit.py \
    --dataset candidates_full_v3.jsonl \
    --model-input model_input_only_v3.jsonl \
    --out . --seeds 10
  "$audit_python" build_handoff.py
} 2>&1 | tee "$work_dir/run_log.txt"

"$audit_python" -m unittest discover -s . -p 'test*py' -v \
  2>&1 | tee "$work_dir/combined_test_results.txt"

"$audit_python" - "$handoff_dir" "$work_dir/handoff_v3" "$work_dir/integrity_report.json" <<'PY'
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

source, generated, report_path = map(Path, sys.argv[1:])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


matched = [
    "candidates_full_v3.jsonl",
    "model_input_only_v3.jsonl",
    "change_log.json",
    "baseline/audit.json",
    "audit.json",
    "audit.md",
    "三问题_专项处理与验收报告.md",
    "内部_六条盲审ID映射.csv",
]
for relative in matched:
    assert (source / relative).read_bytes() == (generated / relative).read_bytes(), relative

with zipfile.ZipFile(source / "仅发审阅者_匿名容量镜像盲审包.zip") as original:
    with zipfile.ZipFile(generated / "仅发审阅者_匿名容量镜像盲审包.zip") as rebuilt:
        assert set(original.namelist()) == set(rebuilt.namelist())
        for name in original.namelist():
            assert original.read(name) == rebuilt.read(name), name

full = rows(generated / "candidates_full_v3.jsonl")
model = rows(generated / "model_input_only_v3.jsonl")
old = {row["id"]: row for row in rows(generated / "baseline/candidates_full.jsonl")}
new = {row["id"]: row for row in full}
assert len(full) == len(model) == len(new) == 60
assert set(new) == {f"T{number:03d}" for number in range(1, 61)}
duplicate_text_groups = sum(
    count > 1 for count in collections.Counter(row["error_text"] for row in full).values()
)
assert duplicate_text_groups == 0
assert all(item == {"id": row["id"], "error_text": row["error_text"]}
           for item, row in zip(model, full, strict=True))
assert collections.Counter(row["label"] for row in full) == {
    label: 15 for label in ("NETWORK_API", "CONTEXT_LIMIT", "ENV_DEPENDENCY", "CODE_RUNTIME")
}
assert set(collections.Counter((row["label"], row["primary_distractor"])
                               for row in full).values()) == {5}
assert len({(row["label"], row["primary_distractor"]) for row in full}) == 12
assert old["T014"] == new["T014"]
assert "last_forwarded" in new["T014"]["error_text"]
assert new["T049"]["evidence_pattern"] == "remote_service_queue_scope_vs_model_per_request_window"
assert "tenant_slot_group=sg_7 active=8 allowed=8" in new["T049"]["error_text"]
assert "processing_started=<none>" in new["T049"]["error_text"]
assert "body_sha256=4f27" in new["T049"]["error_text"]
assert new["T049"]["error_text"].count("body_sha256=4f27") == 2
assert "model response@12:16:15" in new["T049"]["error_text"]
assert "no payload compaction" in new["T049"]["error_text"]
assert 30210 + 1024 < 32768
assert any(change["id"] == "T049" and change["type"] == "mechanism_substitution"
           for change in json.loads((generated / "change_log.json").read_text())["changes"])
pending = sorted(row["id"] for row in full if row.get("review_flags"))
assert pending == ["T011", "T042", "T044"]
assert all(new[ident]["review_status"] == "REQUIRES_UNINVOLVED_REVIEWERS_FOR_CHALLENGE_STRENGTH"
           for ident in pending)

audit = json.loads((generated / "audit.json").read_text(encoding="utf-8"))
baseline = json.loads((generated / "baseline/audit.json").read_text(encoding="utf-8"))
assert audit["status"] == baseline["status"] == "SCREENING_COMPLETED_NOT_ACCEPTANCE"
assert audit["structural_errors"] == baseline["structural_errors"] == []
similarity = next(item["cosine"] for item in audit["top_near_neighbors"]
                  if {item["id_a"], item["id_b"]} == {"T014", "T049"})
report = {
    "status": "PROGRAMMATIC_SCREENING_PASSED_NOT_ACCEPTANCE",
    "source_dataset_sha256": digest(source / "baseline/candidates_full.jsonl"),
    "v3_dataset_sha256": digest(source / "candidates_full_v3.jsonl"),
    "source_artifacts_reproduced": matched,
    "blind_packet_entries_reproduced": True,
    "samples": len(full),
    "unique_ids": len(new),
    "exact_duplicate_text_groups": duplicate_text_groups,
    "label_counts": audit["label_counts"],
    "directed_boundary_counts": audit["directed_pair_counts"],
    "model_input_fields": ["id", "error_text"],
    "t014_t049_similarity": similarity,
    "high_similarity_pairs_reported": len(audit["top_near_neighbors"]),
    "t049_mechanism": new["T049"]["evidence_pattern"],
    "independent_difficulty_review_pending": pending,
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Integrity and source reproduction checks passed")
PY

mkdir -p "$artifact_dir/baseline" "$artifact_dir/v3"
cp baseline/audit.json baseline/audit.md "$artifact_dir/baseline/"
cp audit.json audit.md "$artifact_dir/v3/"
cp 三问题_专项处理与验收报告.md "$artifact_dir/v3/"
cp "$work_dir/run_log.txt" "$work_dir/combined_test_results.txt" \
  "$work_dir/integrity_report.json" "$artifact_dir/"
printf 'Challenge audit complete: %s\n' "$artifact_dir"
