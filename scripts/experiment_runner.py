"""Manifest-driven command runner that resumes after sleep or interruption.

Plan schema::

  {"name": "family", "jobs": [{"id": "seed0", "command": ["python", "..."],
                                  "completion": "results/family/seed0.json"}]}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path, event: dict) -> None:
    event = {"time_unix": time.time(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except subprocess.SubprocessError:
        return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="JSON plan, relative to repository root")
    ap.add_argument("--run-dir", default="", help="default: results/runs/<plan-name>")
    ap.add_argument("--resume", action="store_true", help="skip only jobs with completion artifacts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan_path = (ROOT / args.plan).resolve()
    plan_text = plan_path.read_text(encoding="utf-8")
    plan = json.loads(plan_text)
    name = plan.get("name", plan_path.stem)
    run_dir = (ROOT / args.run_dir) if args.run_dir else ROOT / "results" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.json"
    events = run_dir / "events.jsonl"
    plan_hash = hashlib.sha256(plan_text.encode()).hexdigest()
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['plan_sha256'] != plan_hash or saved['git_commit'] != _git_head():
            raise ValueError('run identity changed: use a new run directory for changed plan/code')
    if not manifest.exists() and not args.dry_run:
        manifest.write_text(json.dumps({
            "name": name, "plan": str(plan_path.relative_to(ROOT)),
            "plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest(),
            "git_commit": _git_head(), "python": sys.executable,
            "created_unix": time.time(), "status": "running",
        }, indent=2), encoding="utf-8")
        _jsonl(events, {"event": "run_started", "git_commit": _git_head()})

    failures = 0
    for job in plan["jobs"]:
        completion = ROOT / job["completion"]
        if args.resume and completion.exists():
            if args.dry_run:
                print("DRY RUN: skip completed", job["id"])
                continue
            _jsonl(events, {"event": "job_skipped_completed", "job": job["id"],
                            "completion": job["completion"]})
            continue
        command = [sys.executable if token == "{python}" else token for token in job["command"]]
        if args.dry_run:
            print("DRY RUN:", command)
            continue
        _jsonl(events, {"event": "job_started", "job": job["id"], "command": command})
        with (run_dir / (job['id']+'.log')).open('a',encoding='utf-8') as log:
            try:
                proc = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                rc = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                _jsonl(events, {'event':'job_interrupted','job':job['id']})
                data=json.loads(manifest.read_text(encoding='utf-8'))
                data['status']='interrupted'
                manifest.write_text(json.dumps(data,indent=2),encoding='utf-8')
                raise
        if rc == 0 and completion.exists():
            _jsonl(events, {"event": "job_completed", "job": job["id"], "returncode": rc,
                            "completion": job["completion"]})
        else:
            failures += 1
            _jsonl(events, {"event": "job_failed", "job": job["id"], "returncode": rc,
                            "completion_exists": completion.exists()})
            break
    if args.dry_run:
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["status"] = "failed" if failures else "completed"
    data["finished_unix"] = time.time()
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _jsonl(events, {"event": "run_" + data["status"]})
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
