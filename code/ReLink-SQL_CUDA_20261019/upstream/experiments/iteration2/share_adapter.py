#!/usr/bin/env python3
"""Gold-free MLX port of the published SHARE inference pipeline.

Only the MLXEngine imports model dependencies. Prompt/data-flow tests run with
the Python standard library. Evaluation is deliberately outside this module.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/share"
SOURCE_REVISION = "a3c76f4bde46c8820fd79809213e3822a52aa7c3"
CHECKPOINTS = {
    "bam": ("birdsql/share-bam", "68ad411d3c37d3a58a817ab2d76b60f1eca0c6bd"),
    "sam": ("birdsql/share-sam", "926179d5207eeb4fd693bf2ea52315e97847fe07"),
    "lom": ("birdsql/share-lom", "f207b0f47d0bb952630baf237b9ae6db7139f36b"),
}
STAGES = ("bam_sql2sr", "bam_mask", "sam_augment", "lom_repair", "generator_sql")
ROLE = dict(zip(STAGES, ("bam", "bam", "sam", "lom", "generator")))
PUBLIC_FIELDS = {"id", "task_id", "dataset", "group", "split", "kind", "question",
                 "evidence", "db_path", "db_id", "schema"}
PREDICTION_FIELDS = {"id", "task_id", "sql", "text", "model", "revision",
                     "prompt_tokens", "generation_tokens", "service_seconds",
                     "action", "finish_reason"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def model_metadata(model_path: str) -> dict:
    """Fingerprint local config/provenance without loading weights or the model."""
    path = Path(model_path).expanduser()
    result = {"path": model_path, "local_directory_exists": path.is_dir()}
    if path.is_dir():
        for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json",
                     "share_provenance.json"):
            if (path / name).is_file():
                result[name + "_sha256"] = file_digest(path / name)
        if (path / "config.json").is_file():
            config = json.loads((path / "config.json").read_text())
            result["quantization"] = config.get("quantization", config.get("quantization_config"))
            result["dtype"] = config.get("torch_dtype", config.get("dtype"))
    return result


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_templates(path: Path | None = None) -> dict[str, str]:
    """Read literal strings from the author's file without importing its code."""
    path = path or VENDOR / "src/prompts/for_infer.py"
    names = {"system_prompt", "sql2sr_user_prompt", "mask_schema_user_prompt",
             "fill_in_schema_user_prompt", "sr2sr_user_prompt", "ft_sr_to_sql"}
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != names or not all(isinstance(v, str) for v in result.values()):
        raise ValueError("Incomplete/nonliteral official SHARE prompt file")
    return result


def public_task(row: dict) -> dict:
    """Reject accidental label-bearing inputs rather than silently strip labels."""
    extra = set(row) - PUBLIC_FIELDS
    if extra:
        raise ValueError(f"Tasks must be public-only; unexpected fields: {sorted(extra)}")
    task = dict(row)
    task["id"] = str(task.get("id", task.get("task_id", "")))
    if not task["id"] or not isinstance(task.get("question"), str):
        raise ValueError("Every public task needs id and question")
    if task.get("kind", "sql") != "sql":
        raise ValueError("SHARE supports SQL tasks only")
    task["db_id"] = task.get("db_id") or task.get("group")
    if not task["db_id"]:
        raise ValueError("Every public task needs db_id or group")
    task["evidence"] = task.get("evidence", "")
    return task


def public_predictions(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        extra = set(row) - PREDICTION_FIELDS
        if extra:
            raise ValueError(f"Initial predictions must be public-only: {sorted(extra)}")
        task_id = str(row.get("task_id", row.get("id", "")))
        sql = row.get("sql", row.get("text"))
        if not task_id or not isinstance(sql, str):
            raise ValueError("Initial prediction requires task_id and sql (or text)")
        if task_id in result:
            raise ValueError(f"Duplicate initial prediction {task_id}")
        result[task_id] = dict(row, sql=sql)
    return result


def table_info_from_sqlite(task: dict) -> dict:
    """Fallback public metadata; no row values or target query are read."""
    path = Path(task["db_path"]).resolve()
    db = sqlite3.connect("file:" + quote(str(path), safe="/") + "?mode=ro", uri=True)
    try:
        db.execute("PRAGMA query_only=ON")
        tables = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        columns = [[-1, "*"]]
        pk = []
        fk = []
        index = {}
        for t_idx, table in enumerate(tables):
            qtable = '"' + table.replace('"', '""') + '"'
            for row in db.execute(f"PRAGMA table_info({qtable})"):
                index[(table.lower(), row[1].lower())] = len(columns)
                if row[5]:
                    pk.append(len(columns))
                columns.append([t_idx, row[1]])
        for table in tables:
            qtable = '"' + table.replace('"', '""') + '"'
            for row in db.execute(f"PRAGMA foreign_key_list({qtable})"):
                src = index.get((table.lower(), str(row[3]).lower()))
                dst = index.get((str(row[2]).lower(), str(row[4]).lower()))
                if src is not None and dst is not None:
                    fk.append([src, dst])
        return dict(db_id=task["db_id"], table_names_original=tables,
                    column_names_original=columns, primary_keys=pk, foreign_keys=fk)
    finally:
        db.close()


def referenced_columns(info: dict, text: str) -> list[list[str]]:
    """Author's case-sensitive substring matcher, with stable set ordering.

    Do not replace this with a SQL parser: SAM output is SR, not SQL. Stable
    lexical sorting replaces the author's Python set iteration only.
    """
    tables = info["table_names_original"]
    columns = [[tables[int(t)], c] for t, c in info["column_names_original"][1:]]
    chosen_tables = {t for t in tables if t in text}
    chosen_columns = sorted({c for _, c in columns if c in text})
    return [[t, c] for c in chosen_columns for t, name in columns
            if name == c and t in chosen_tables]


def full_schema(info: dict) -> str:
    tables = info["table_names_original"]
    by_table = {i: [] for i in range(len(tables))}
    for t_idx, column in info["column_names_original"][1:]:
        table = tables[t_idx]
        table = f"`{table}`" if " " in table else table
        column = f"`{column}`" if " " in column else column
        by_table[t_idx].append(f"{table}.{column}")
    return "\n\n".join(f"### Table: {tables[i]}\n{cols}" for i, cols in by_table.items())


def fenced_sr(text: str) -> str:
    match = re.search(r"```SR(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def augmented_columns(info: dict, initial_sql: str, sam_response: str) -> list[str]:
    before = referenced_columns(info, initial_sql)
    after = referenced_columns(info, sam_response)
    return sorted({f"{t}.{c}" for t, c in before + after})


def column_descriptions(task: dict, columns: list[str], meanings: dict) -> str:
    output = ""
    for entry in columns:
        # Use the same table.column convention as the original implementation.
        table, column = entry.split(".")
        table, column = table.replace("`", ""), column.replace("`", "")
        meaning = meanings.get(f"{task['db_id']}|{table}|{column}")
        if not isinstance(meaning, str) or not meaning:
            continue
        meaning = meaning[1:] if meaning[0] == "#" else meaning
        table = f"`{table}`" if " " in table else table
        column = f"`{column}`" if " " in column else column
        output += f"# {table}.{column}: {meaning}\n"
    return output


def foreign_keys(info: dict) -> dict:
    tables, columns = info["table_names_original"], info["column_names_original"]
    def name(i):
        return f"{tables[columns[i][0]]}.{columns[i][-1]}"
    return {name(src): name(dst) for src, dst in info.get("foreign_keys", [])}


def stage_messages(stage: str, task: dict, initial_sql: str, info: dict,
                   previous: dict[str, str], meanings: dict,
                   templates: dict[str, str]) -> tuple[list[dict], dict]:
    task = public_task(task)
    extra = {}
    if stage == "bam_sql2sr":
        sql = initial_sql.split("\t")[0].replace("\n", " ")
        columns = [f"{t}.{c}" for t, c in referenced_columns(info, sql)]
        prompt = templates["sql2sr_user_prompt"].format(schema=columns, sql=sql)
    elif stage == "bam_mask":
        sr = fenced_sr(previous["bam_sql2sr"])
        extra["input_sr_missing"] = not bool(sr)
        prompt = templates["mask_schema_user_prompt"].format(sr=sr)
    elif stage == "sam_augment":
        sr = fenced_sr(previous["bam_mask"])
        extra["input_sr_missing"] = not bool(sr)
        prompt = templates["fill_in_schema_user_prompt"].format(
            schema=full_schema(info), highlighted_schema=referenced_columns(info, initial_sql),
            question=task["question"], evidence=task["evidence"], masked_sr=sr)
    elif stage in ("lom_repair", "generator_sql"):
        columns = augmented_columns(info, initial_sql, previous["sam_augment"])
        extra["augmented_schema"] = columns
        sr = fenced_sr(previous["sam_augment"] if stage == "lom_repair"
                       else previous["lom_repair"])
        extra["input_sr_missing"] = not bool(sr)
        desc = column_descriptions(task, columns, meanings)
        extra["column_description_characters"] = len(desc)
        if stage == "lom_repair":
            schema = [x.replace("`", "") for x in columns]
            template = templates["sr2sr_user_prompt"]
        else:
            schema = []
            for entry in columns:
                table, column = entry.split(".")
                table = f"`{table}`" if " " in table else table
                column = f"`{column}`" if " " in column else column
                schema.append(f"{table}.{column}")
            template = templates["ft_sr_to_sql"]
        prompt = template.format(schema=schema, fk_dic=foreign_keys(info),
                                 column_description=desc, question=task["question"],
                                 evidence=task["evidence"], sr=sr)
    else:
        raise ValueError(stage)
    messages = [{"role": "user", "content": prompt}]
    if stage != "generator_sql":
        messages.insert(0, {"role": "system", "content": templates["system_prompt"]})
    return messages, extra


def extract_final_sql(response: str) -> str:
    for language in ("sqlite", "sql", ""):
        match = re.search(r"```" + language + r"(.*?)```", response, re.DOTALL)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return response.split("\n\nsqlite\n")[-1]


class MLXEngine:
    """Sequential model residency, greedy sampling, no cross-call KV reuse."""
    def __init__(self, max_context_tokens=8192):
        self.path = None
        self.model = self.tokenizer = None
        self.max_context_tokens = max_context_tokens
        self.load_events = []

    def generate(self, messages, model_path, max_tokens, stop_strings):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        if self.path != model_path:
            self.model = self.tokenizer = None
            gc.collect()
            mx.clear_cache()
            started = time.perf_counter()
            self.model, self.tokenizer = load(model_path,
                tokenizer_config={"trust_remote_code": False})
            mx.eval(self.model.parameters())
            self.load_events.append(dict(model_path=model_path,
                                         seconds=time.perf_counter() - started))
            self.path = model_path
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) + max_tokens > self.max_context_tokens:
            raise ValueError(f"Context overflow: {len(token_ids)} + {max_tokens} > "
                             f"{self.max_context_tokens}; no silent truncation")
        started = time.perf_counter()
        parts = []
        last = None
        stopped = False
        stream = stream_generate(self.model, self.tokenizer, token_ids,
                                 max_tokens=max_tokens, sampler=make_sampler(temp=0))
        try:
            for response in stream:
                last = response
                parts.append(response.text)
                if any(stop in "".join(parts) for stop in stop_strings):
                    stopped = True
                    break
        finally:
            stream.close()
        mx.synchronize()
        seconds = time.perf_counter() - started
        text = "".join(parts)
        for stop in stop_strings:
            text = text.split(stop, 1)[0]
        if last is None:
            raise RuntimeError("Model generated no response object")
        return dict(text=text, prompt_tokens=last.prompt_tokens,
                    generation_tokens=last.generation_tokens, service_seconds=seconds,
                    finish_reason="stop_string" if stopped else last.finish_reason,
                    rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def pipeline(tasks, predictions, tables, meanings, models, output: Path, engine,
             max_tokens=3000, generator_max_tokens=1024, stages=STAGES,
             save_prompts=False, templates=None):
    templates = templates or read_templates()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stages.jsonl"
    existing = {}
    for row in read_jsonl(path) if path.exists() else []:
        key = (row["task_id"], row["stage"])
        if key in existing:
            raise ValueError(f"Duplicate checkpoint row: {key}")
        existing[key] = row
    final = []
    for stage in STAGES:
        if stage not in stages:
            continue
        for task in tasks:
            task = public_task(task)
            task_id = task["id"]
            initial = predictions[task_id]["sql"]
            previous = {s: existing[(task_id, s)]["text"] for s in STAGES
                        if (task_id, s) in existing}
            started = time.perf_counter()
            messages, extra = stage_messages(stage, task, initial, tables[task["db_id"]],
                                              previous, meanings, templates)
            preparation_seconds = time.perf_counter() - started
            cap = generator_max_tokens if stage == "generator_sql" else max_tokens
            stops = [] if stage == "generator_sql" else ["</FINAL_ANSWER>", "<|EOT|>"]
            signature = digest(dict(messages=messages, model=models[ROLE[stage]],
                                     max_tokens=cap, temperature=0, stops=stops))
            key = (task_id, stage)
            if key in existing:
                if existing[key]["call_signature"] != signature:
                    raise ValueError(f"Refusing stale resume for {key}; use a new output dir")
                continue
            attempts = []
            # The author's invalid-text retry permits 11 total attempts.
            for _ in range(1 if stage == "generator_sql" else 11):
                result = engine.generate(messages, models[ROLE[stage]], cap, stops)
                attempts.append(result)
                if len(result["text"].strip()) > 1:
                    break
            text = attempts[-1]["text"]
            if stage != "generator_sql" and len(text.strip()) <= 1:
                text = "[INVALID]"
            row = dict(task_id=task_id, stage=stage, model=models[ROLE[stage]],
                       source_revision=SOURCE_REVISION, call_signature=signature,
                       prompt_sha256=digest(messages), text=text, extra=extra,
                       preparation_seconds=preparation_seconds,
                       attempts=attempts, finish_reason=attempts[-1]["finish_reason"])
            for field in ("prompt_tokens", "generation_tokens", "service_seconds"):
                row[field] = sum(a[field] for a in attempts)
            if save_prompts:
                row["messages"] = messages
            with path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            existing[key] = row
            print(json.dumps(dict(task_id=task_id, stage=stage,
                                  generation_tokens=row["generation_tokens"],
                                  seconds=round(row["service_seconds"], 3))), flush=True)
    for task in tasks:
        task_id = task["id"]
        if not all((task_id, stage) in existing for stage in STAGES):
            continue
        rows = [existing[(task_id, stage)] for stage in STAGES]
        row = dict(task_id=task_id, sql=extract_final_sql(rows[-1]["text"]),
                   action="share_full", initial_prediction=predictions[task_id],
                   model=models["generator"])
        for field in ("prompt_tokens", "generation_tokens", "service_seconds",
                      "preparation_seconds"):
            row["recovery_" + field] = sum(r[field] for r in rows)
        row["incomplete_sr_stage_count"] = sum(
            bool(r["extra"].get("input_sr_missing")) for r in rows)
        final.append(row)
    with (output / "final_sql.jsonl").open("w") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--initial-predictions", type=Path, required=True)
    p.add_argument("--tables", type=Path, action="append", default=[])
    p.add_argument("--column-meanings", type=Path, action="append", default=[])
    for role in ("bam", "sam", "lom", "generator"):
        p.add_argument(f"--{role}-model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--generator-max-tokens", type=int, default=1024)
    p.add_argument("--max-context-tokens", type=int, default=8192)
    p.add_argument("--stage", choices=STAGES, action="append")
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument("--validate-only", action="store_true", help="No MLX imports or model calls")
    args = p.parse_args()
    tasks = [public_task(t) for t in read_jsonl(args.tasks)]
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    if len({t.get("dataset", "unspecified") for t in tasks}) > 1:
        raise ValueError("Run one dataset at a time: Spider/BIRD reuse some db_id values "
                         "but have different schema/column descriptions")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        tasks = tasks[:args.limit]
    predictions = public_predictions(read_jsonl(args.initial_predictions))
    if any(t["id"] not in predictions for t in tasks):
        raise ValueError("Missing initial predictions")
    tables, meanings = {}, {}
    for path in args.tables:
        for info in json.loads(path.read_text()):
            tables[info["db_id"]] = info
    fallback = []
    for task in tasks:
        if task["db_id"] not in tables:
            tables[task["db_id"]] = table_info_from_sqlite(task)
            fallback.append(task["db_id"])
    for path in args.column_meanings:
        for key, value in json.loads(path.read_text()).items():
            if key in meanings and meanings[key] != value:
                raise ValueError(f"Conflicting column descriptions for {key}")
            meanings[key] = value
    models = {role: getattr(args, f"{role}_model") for role in ("bam", "sam", "lom", "generator")}
    inputs = [args.tasks, args.initial_predictions] + args.tables + args.column_meanings
    manifest = dict(protocol="SHARE-MLX-v1", source_revision=SOURCE_REVISION,
                    checkpoints={role: dict(repo=repo, revision=revision)
                                 for role, (repo, revision) in CHECKPOINTS.items()},
                    models=models, tasks=len(tasks),
                    model_metadata={role: model_metadata(path) for role, path in models.items()},
                    selected_task_ids=[t["id"] for t in tasks],
                    files={str(path.resolve()): file_digest(path) for path in inputs},
                    templates_sha256=file_digest(VENDOR / "src/prompts/for_infer.py"),
                    sqlite_metadata_fallback=fallback, meanings_count=len(meanings),
                    max_tokens=args.max_tokens, generator_max_tokens=args.generator_max_tokens,
                    max_context_tokens=args.max_context_tokens,
                    quantization="See converted model config; this runner does not quantize",
                    evaluation="None; labels and SQL execution are outside this runner")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run manifest changed; choose another --output")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if args.validate_only:
        print(json.dumps(dict(validated_tasks=len(tasks), models_loaded=False,
                              metadata_fallback=fallback, meanings_count=len(meanings))))
        return
    engine = MLXEngine(args.max_context_tokens)
    try:
        pipeline(tasks, predictions, tables, meanings, models, args.output, engine,
                 args.max_tokens, args.generator_max_tokens, args.stage or STAGES,
                 args.save_prompts)
    finally:
        with (args.output / "model_loads.jsonl").open("a") as f:
            for event in engine.load_events:
                f.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    main()
