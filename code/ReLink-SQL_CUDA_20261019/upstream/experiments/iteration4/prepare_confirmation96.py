#!/usr/bin/env python3
"""Freeze 48+48 public questions before reference execution, without replacement.

Historical artifacts are projected only to public identities/questions/groups.
No model calls. Reference SQL is accessed only for the persisted selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from recoverylab.sql import connect, execute

SEED = "20260913-confirmation96-v1-db-disjoint-frozen-selection"
OUTPUT = ROOT / "iteration4/data/confirmation96_v1"
TRAIN_DEV = ROOT / "iteration4/data/cascade_train377_dev64"
SKIP_VALUES = {"sql", "query", "reference", "answer", "observation", "text", "raw_text", "messages",
               "prompt", "response", "feedback", "correct", "labels", "logprobs", "gold_sql", "gold_result"}
IDENTIFIER = re.compile(r"(?:bird[A-Za-z0-9_]*:\d+|spider[A-Za-z0-9_]*:[^\s/:|]+:[A-Za-z0-9_-]+)")
BIRD_NUMBER = re.compile(r"bird[A-Za-z0-9_]*:(\d+)")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def family(dataset):
    value = str(dataset).casefold()
    return "bird" if value.startswith("bird") else "spider" if value.startswith("spider") else value


def question_key(question):
    return sha(norm(question).encode())  # Conservative across datasets AND databases.


def group_key(dataset, group):
    return family(dataset), norm(group)


def dump(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    default = lambda x: x.hex() if isinstance(x, bytes) else str(x)
    text = ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=default) + "\n" for row in value)
            if lines else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=default) + "\n")
    path.write_text(text)
    return sha(text.encode())


def project_identity(value, ids, questions, groups, bird_ids, inherited_dataset=None):
    """Recursively inspect identity metadata, never gold/text/observation fields."""
    if isinstance(value, dict):
        dataset = value.get("dataset", inherited_dataset)
        group = value.get("group", value.get("db_id"))
        if dataset is not None and group is not None and not isinstance(group, (list, dict)):
            groups.add(group_key(dataset, group))
        question = value.get("question")
        if isinstance(question, str):
            questions.add(question_key(question))
        for key, child in value.items():
            key_lower = str(key).casefold()
            if key_lower in SKIP_VALUES or key_lower.startswith("gold") or key_lower.startswith("private") or key_lower.endswith("_sql"):
                continue
            if key in ("id", "task_id", "source_task_id") and isinstance(child, (str, int)):
                ids.add(str(child))
            # Mapping keys can themselves be original task IDs.
            for match in IDENTIFIER.finditer(str(key)):
                ids.add(match.group())
            project_identity(child, ids, questions, groups, bird_ids, dataset)
    elif isinstance(value, list):
        for child in value:
            project_identity(child, ids, questions, groups, bird_ids, inherited_dataset)
    elif isinstance(value, str):
        for match in IDENTIFIER.finditer(value):
            ids.add(match.group())
        for match in BIRD_NUMBER.finditer(value):
            bird_ids.add(int(match.group(1)))


def history_skip(path, output):
    if path.is_relative_to(output):
        return "current_new_cohort"
    rel = path.relative_to(PROJECT)
    if any(part in {".git", ".venv", "__pycache__", "models", "vendor", "source_materials", "research_failure_routing"} for part in rel.parts):
        return "dependencies_models_vendor_or_literature"
    if "private" in rel.parts or any(part.startswith("private_") for part in rel.parts):
        return "private_labels_not_read"
    if path.is_relative_to(ROOT / "data") and not path.is_relative_to(ROOT / "data/pilot"):
        return "official_source_or_external_data_not_a_project_cohort"
    if re.search(r"(?:^|_)(?:scored|scores|gold|labels|paired|summary)(?:_|\.|$)", path.name) or path.name == "path_records.jsonl" or path.name.startswith("evaluation_"):
        return "evaluation_outputs_not_read"
    return None


def scan_history(output):
    command = ["rg", "--files", "--hidden", "--no-ignore", "-g", "*.json", "-g", "*.jsonl",
        "-g", "!**/.venv/**", "-g", "!**/.git/**", "-g", "!**/__pycache__/**", str(PROJECT)]
    paths = sorted({Path(line).resolve() for line in subprocess.check_output(command, text=True).splitlines()})
    ids, questions, groups, bird_ids = set(), set(), set(), set()
    files, skipped = [], Counter()
    for path in paths:
        reason = history_skip(path, output)
        if reason:
            skipped[reason] += 1; continue
        content = path.read_bytes()
        ignored_tail = False
        if path.suffix == ".jsonl":
            lines = content.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b"\n"):
                try:
                    json.loads(lines[-1])
                except json.JSONDecodeError:
                    lines = lines[:-1]; ignored_tail = True
            records = [json.loads(line) for line in lines if line.strip()]
        else:
            try:
                records = [json.loads(content)]
            except json.JSONDecodeError:
                # A partial mutable metadata file is not safe to overlook.
                raise ValueError("Historical JSON is incomplete; retry after writer finishes: " + str(path))
        for record in records:
            project_identity(record, ids, questions, groups, bird_ids)
        files.append(dict(path=str(path.relative_to(PROJECT)), bytes=len(content), sha256=sha(content),
                          projected_top_level_records=len(records), ignored_incomplete_tail=ignored_tail))
    return dict(ids=ids, questions=questions, groups=groups, bird_ids=bird_ids, files=files, skipped=dict(skipped))


def blocked_train_dev():
    groups, files, counts = set(), [], {}
    for split, expected in (("train", 377), ("development", 64)):
        path = TRAIN_DEV / (split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != expected or len({row["id"] for row in rows}) != expected:
            raise ValueError("Current frozen train377/dev64 cohort changed")
        groups.update(group_key(row["dataset"], row["group"]) for row in rows)
        counts[split] = len(rows)
        files.append(dict(path=str(path), sha256=file_sha(path), tasks=len(rows)))
    return groups, files, counts


def select_pool(sources, databases, history, blocked, per_family=48):
    """Public fields only; source SQL is neither indexed nor inspected here."""
    pools, counters, seen = defaultdict(list), Counter(), set()
    for dataset, path, rows in sources:
        fam = family(dataset)
        for index, row in enumerate(rows):
            db, question = row["db_id"], row["question"]
            key = question_key(question)
            counters[fam + "/source_rows"] += 1
            if key in history["questions"]:
                counters[fam + "/prior_question"] += 1; continue
            if group_key(dataset, db) in blocked:
                counters[fam + "/train_development_db"] += 1; continue
            old_qkey = sha((fam + "\0" + norm(question)).encode())
            ident = ("bird4_confirmation:" + str(row["question_id"]) if fam == "bird" else
                     "spider4_confirmation:" + db + ":" + old_qkey[:12])
            aliases = {ident}
            if fam == "bird":
                numeric = int(row["question_id"])
                if numeric in history["bird_ids"]:
                    counters[fam + "/prior_numeric_id"] += 1; continue
                aliases.update("bird" + prefix + ":" + str(numeric) for prefix in ("", "2", "3_holdout"))
            else:
                aliases.update(("spider2:" + db + ":" + sha(("20260913-iteration2" + question).encode())[:12],
                                "spider3_holdout:" + db + ":" + old_qkey[:12]))
            if aliases & history["ids"]:
                counters[fam + "/prior_id"] += 1; continue
            if db not in databases[fam]:
                counters[fam + "/missing_database"] += 1; continue
            if key in seen:
                counters[fam + "/duplicate_source_question"] += 1; continue
            seen.add(key)
            rank = sha((SEED + "\0" + fam + "\0" + db + "\0" + norm(question)).encode())
            task = dict(id=ident, dataset=dataset, group=db, split="confirmation", kind="sql", question=question,
                evidence=row.get("evidence", ""), db_path=str(databases[fam][db].resolve()))
            pools[fam].append(dict(rank=rank, task=task, source=str(path), source_row=index,
                                  question_key=key, source_id_aliases=sorted(aliases)))
    eligible = {fam: len(pools[fam]) for fam in ("spider", "bird")}
    if any(count < per_family for count in eligible.values()):
        return [], eligible, dict(counters)
    selected = [candidate for fam in ("spider", "bird")
                for candidate in sorted(pools[fam], key=lambda value: (value["rank"], value["task"]["id"]))[:per_family]]
    return selected, eligible, dict(counters)


def source_data():
    spider = ROOT / "data/spider/spider_data"
    paths = [("spider", spider / "train_spider.json"), ("spider", spider / "dev.json"),
             ("bird_dev_20251106", ROOT / "data/bird_full_dev_20251106/dev.json")]
    sources = [(dataset, path, json.loads(path.read_text())) for dataset, path in paths]
    if len(sources[-1][2]) != 1534:
        raise ValueError("Require the pinned official full BIRD 1534 source")
    dbs = {"spider": {path.stem: path for path in (spider / "database").glob("*/*.sqlite")},
           "bird": {path.stem: path for path in (ROOT / "data/bird_databases/minidev/MINIDEV/dev_databases").glob("*/*.sqlite")}}
    metadata = [dict(dataset=dataset, path=str(path), rows=len(rows), sha256=file_sha(path)) for dataset, path, rows in sources]
    return sources, dbs, metadata


def references_after_freeze(selected, sources, public_path, expected_hash, reference_executor=execute):
    if file_sha(public_path) != expected_hash:
        raise ValueError("Public selection must be persisted unchanged before reference access")
    by_source = {str(path): rows for _, path, rows in sources}
    private, audit = [], []
    for candidate in selected:
        task = candidate["task"]
        row = by_source[candidate["source"]][candidate["source_row"]]
        sql = row["query"] if family(task["dataset"]) == "spider" else row["SQL"]
        result = reference_executor(task["db_path"], sql, timeout=5, max_rows=50000)
        # Keep all selected questions, including failed references. Never advance
        # to another candidate or silently rewrite a reference SQL.
        result["seconds"] = 0.0
        private.append(dict(id=task["id"], split="confirmation", gold_sql=sql, gold_result=result))
        audit.append(dict(id=task["id"], family=family(task["dataset"]), status=result["status"],
                          reason=None if result["status"] == "ok" else "reference_execution_failed_no_replacement"))
        if len(private) % 16 == 0:
            print(json.dumps(dict(reference_preparation=len(private), selected=len(selected),
                                  failures=sum(row["status"] != "ok" for row in audit))), flush=True)
    return private, audit


def prepare(output=OUTPUT):
    output = output.resolve()
    if output.exists():
        raise ValueError("New confirmation directory required; never overwrite a selection")
    history = scan_history(output)
    blocked, blocked_files, train_dev_counts = blocked_train_dev()
    sources, dbs, source_meta = source_data()
    selected, eligible, counters = select_pool(sources, dbs, history, blocked)
    base = dict(protocol="confirmation96-fixed-public-before-reference-v1", split="confirmation", seed=SEED,
        requested_per_family=48, eligible_public_pool=eligible, exclusion_counters=counters,
        source_files=source_meta, training_development_files=blocked_files, training_development_counts=train_dev_counts,
        blocked_train_development_groups=sorted(blocked), historical_files=history["files"],
        history_skipped_categories=history["skipped"], historical_id_count=len(history["ids"]),
        historical_normalized_question_count=len(history["questions"]), historical_bird_numeric_id_count=len(history["bird_ids"]),
        historical_ids_sha256=sha("\n".join(sorted(history["ids"])).encode()),
        historical_questions_sha256=sha("\n".join(sorted(history["questions"])).encode()),
        question_normalization="Unicode NFKC + casefold + whitespace collapse; exclude exact normalized wording globally across dataset/database",
        selection_rule="Fixed new seed hash of family/database/normalized question; exactly48+48 BEFORE opening selected references; no outcome/difficulty/structure filtering",
        script_sha256=file_sha(Path(__file__)), metric_sha256=file_sha(ROOT / "recoverylab/sql.py"),
        prior_gold_fields_consulted=False, new_reference_sql_stdout=False, gpu_used=False)
    if not selected:
        output.mkdir(parents=True)
        dump(output / "insufficient_pool_audit.json", base | {"status": "insufficient_pool_no_selection_no_references"})
        return dict(status="insufficient_pool", eligible=eligible, requested={"spider": 48, "bird": 48}, references_accessed=False)
    # Re-scan before freezing to include concurrent completed/publicly planned
    # tasks. Existing-output append-only changes with the same IDs are harmless.
    latest = scan_history(output)
    used_questions = history["questions"] | latest["questions"]
    used_ids = history["ids"] | latest["ids"]
    used_bird_ids = history["bird_ids"] | latest["bird_ids"]
    if any(candidate["question_key"] in used_questions or set(candidate["source_id_aliases"]) & used_ids
           or (family(candidate["task"]["dataset"]) == "bird" and
               int(candidate["task"]["id"].rsplit(":", 1)[1]) in used_bird_ids)
           for candidate in selected):
        raise ValueError("Historical cohort expanded during selection; no references read, rerun public audit")
    schemas, snapshots = {}, {}
    for candidate in selected:
        task = candidate["task"]; path = task["db_path"]
        if path not in schemas:
            with connect(path) as con:
                ddl = [row[0] for row in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name") if row[0]]
                pragmas = dict(page_count=con.execute("PRAGMA page_count").fetchone()[0], page_size=con.execute("PRAGMA page_size").fetchone()[0],
                               user_version=con.execute("PRAGMA user_version").fetchone()[0])
            schemas[path] = ";\n".join(ddl) + ";"
            snapshots[path] = dict(path=path, bytes=Path(path).stat().st_size, sha256=file_sha(path), schema_sha256=sha(schemas[path].encode()), **pragmas)
        task["schema"] = schemas[path]
    public = [candidate["task"] for candidate in selected]
    assert len(public) == 96 and len({task["id"] for task in public}) == 96
    assert not any(group_key(task["dataset"], task["group"]) in blocked for task in public)
    public_hash = dump(output / "confirmation.jsonl", public, lines=True)
    selections = [{key: value for key, value in candidate.items() if key != "task"} | dict(id=candidate["task"]["id"],
        family=family(candidate["task"]["dataset"]), group=candidate["task"]["group"],
        overlaps_other_historical_database=group_key(candidate["task"]["dataset"], candidate["task"]["group"]) in (history["groups"] | latest["groups"])) for candidate in selected]
    freeze = base | dict(status="public_selection_frozen_references_not_yet_accessed", selected_count=96,
        counts=dict(Counter(task["dataset"] for task in public)), public_sha256=public_hash,
        historical_question_overlap=0, historical_id_overlap=0, train_development_database_overlap=0,
        all_reference_failures_retained=True, replacements_allowed=False, selection=selections,
        database_snapshots=list(snapshots.values()), final_history_files=latest["files"],
        relative_to_current_train_dev_db_disjoint=True, entirely_unseen_historical_databases_claimed=False,
        other_historical_database_overlap_questions={fam: sum(row["family"] == fam and row["overlaps_other_historical_database"] for row in selections) for fam in ("bird", "spider")},
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    frozen_hash = dump(output / "selection_frozen.json", freeze)
    # Source metadata is public provenance; no reference SQL is printed here.
    private, reference_audit = references_after_freeze(selected, sources, output / "confirmation.jsonl", public_hash)
    private_hash = dump(output / "private/confirmation.jsonl", private, lines=True)
    (output / "private/confirmation.jsonl").chmod(0o600)
    audit_hash = dump(output / "reference_audit.json", dict(fixed_public_sha256=public_hash, selected_count=96,
        successful_references=sum(row["status"] == "ok" for row in reference_audit),
        failed_references=sum(row["status"] != "ok" for row in reference_audit),
        replacements=0, per_task=reference_audit))
    bird_source_meta = ROOT / "data/bird_full_dev_20251106/source.json"
    final = freeze | dict(status="prepared_fixed96", selection_frozen_sha256=frozen_hash,
        private_sha256=private_hash, reference_audit_sha256=audit_hash,
        reference_successes=sum(row["status"] == "ok" for row in reference_audit),
        reference_failures=sum(row["status"] != "ok" for row in reference_audit),
        public_selection_precedes_reference_execution=True, bird_source=json.loads(bird_source_meta.read_text()),
        limitation="BIRD official1534 question source uses existing Mini-Dev physical DB snapshots. Disjoint only from the declared current train377/dev64 DBs, not all historical tests or foundation-model pretraining.")
    dump(output / "manifest.json", final)
    return dict(status="prepared_fixed96", public_path=str(output / "confirmation.jsonl"), counts=final["counts"],
        public_sha256=public_hash, manifest=str(output / "manifest.json"), reference_failures=final["reference_failures"],
        historical_question_overlap=0, historical_id_overlap=0, train_development_database_overlap=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
