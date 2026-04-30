from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from subprocess import Popen
import re
import subprocess
import threading
import uuid
from typing import Any

from ..app.config import PROJECT_ROOT, load_config
from .workflows import build_command, predict_output_dir

_JOB_HISTORY_PATH = PROJECT_ROOT / "results" / "gui_job_history.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Job:
    id: str
    workflow_id: str
    title: str
    payload: dict[str, Any]
    command: list[str]
    status: str = "queued"
    progress: float | None = None
    progress_label: str = "Queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    output_dir: str | None = None
    log_path: str | None = None
    error: str | None = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=3000))
    phase: str | None = None
    process: Popen[str] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "title": self.title,
            "payload": dict(self.payload),
            "command": list(self.command),
            "status": self.status,
            "progress": self.progress,
            "progress_label": self.progress_label,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "error": self.error,
            "log_lines": list(self.log_lines),
            "phase": self.phase,
            "live_progress_path": self._find_live_progress_path(),
            "report_path": self._find_report_path(),
            "figures": self._find_figure_paths(),
        }

    def append_log(self, line: str) -> None:
        self.log_lines.append(line.rstrip("\n"))

    def _find_live_progress_path(self) -> str | None:
        if not self.output_dir:
            return None
        path = Path(self.output_dir) / "live_progress.png"
        return str(path) if path.exists() else None

    def _find_report_path(self) -> str | None:
        if not self.output_dir:
            return None
        candidates = [
            Path(self.output_dir) / "report.html",
            Path(self.output_dir) / "assessment_report.html",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _find_figure_paths(self) -> list[str]:
        if not self.output_dir:
            return []
        output_dir = Path(self.output_dir)
        candidates = [
            output_dir / "network_metrics.png",
            output_dir / "severity_distribution.png",
            output_dir / "speed_histograms.png",
            output_dir / "accident_heatmap.png",
            output_dir / "batch_ai_distribution.png",
        ]
        figures_dir = output_dir / "figures"
        if figures_dir.exists():
            candidates.extend(sorted(figures_dir.glob("*.png")))
        return [str(path) for path in candidates if path.exists()]


class JobManager:
    """Background subprocess job runner for GUI-triggered workflows."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load_history()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [job.to_dict() for job in self._jobs.values()]
        return sorted(jobs, key=lambda item: item["created_at"], reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def create_job(self, workflow_id: str, title: str, payload: dict[str, Any]) -> dict[str, Any]:
        command = build_command(workflow_id, payload)
        output_dir = predict_output_dir(workflow_id, payload)
        job = Job(
            id=uuid.uuid4().hex,
            workflow_id=workflow_id,
            title=title,
            payload=payload,
            command=command,
            output_dir=output_dir,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._save_history_locked()

        worker = threading.Thread(target=self._run_job, args=(job.id,), daemon=True)
        worker.start()
        return job.to_dict()

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            process = job.process if job else None
        if not job or not process:
            return False
        process.terminate()
        job.status = "cancelled"
        job.progress_label = "Cancelled"
        job.finished_at = _utc_now()
        self._save_history()
        return True

    def forget_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status == "running":
                return False
            del self._jobs[job_id]
            self._save_history_locked()
        return True

    def clear_finished(self) -> int:
        with self._lock:
            removable = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status not in {"running", "queued"}
            ]
            for job_id in removable:
                del self._jobs[job_id]
            self._save_history_locked()
        return len(removable)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utc_now()
            job.progress = 0.02
            job.progress_label = "Starting"
            self._save_history_locked()

        env = os.environ.copy()
        src_dir = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src_dir if not env.get("PYTHONPATH") else f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
        env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        if job.output_dir:
            Path(job.output_dir).mkdir(parents=True, exist_ok=True)
            job.log_path = str(Path(job.output_dir) / "gui_job.log")

        process = subprocess.Popen(
            job.command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.process = process

        log_file = open(job.log_path, "a", encoding="utf-8") if job.log_path else None
        try:
            assert process.stdout is not None
            for raw_line in iter(process.stdout.readline, ""):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                if log_file:
                    log_file.write(raw_line)
                    log_file.flush()
                self._ingest_log_line(job, line)
            return_code = process.wait()
        finally:
            if log_file:
                log_file.close()

        with self._lock:
            if job.status != "cancelled":
                job.return_code = return_code
                job.finished_at = _utc_now()
                job.process = None
                if return_code == 0:
                    job.status = "succeeded"
                    job.progress = 1.0
                    job.progress_label = "Completed"
                else:
                    job.status = "failed"
                    job.progress_label = "Failed"
                    job.error = f"Process exited with code {return_code}"
                self._save_history_locked()

    def _ingest_log_line(self, job: Job, line: str) -> None:
        job.append_log(line)
        self._update_progress(job, line)
        self._save_history()

    def _update_progress(self, job: Job, line: str) -> None:
        if job.workflow_id == "simulation.run":
            config_path = str(job.payload.get("config", "configs/thessaloniki/default.yaml"))
            try:
                config = load_config(config_path)
                total_steps = float(config["sumo"]["total_steps"])
            except Exception:
                total_steps = None
            match = re.search(r"\[\s*(\d+)\s+min\]", line)
            if match and total_steps:
                current_seconds = float(match.group(1)) * 60.0
                ratio = min(0.95, max(0.05, current_seconds / total_steps))
                job.progress = ratio
                job.progress_label = f"Simulation minute {match.group(1)}"
                return
            step_match = re.search(r"step=(\d+)", line)
            if step_match and total_steps:
                current_seconds = float(step_match.group(1))
                ratio = min(0.95, max(0.05, current_seconds / total_steps))
                job.progress = ratio
                job.progress_label = f"Simulation step {int(current_seconds):,} / {int(total_steps):,}"
                return
            run_match = re.search(r"── Run (\d+) / (\d+)", line)
            if run_match:
                current = int(run_match.group(1))
                total = int(run_match.group(2))
                job.progress = min(0.95, max(0.05, current / max(total, 1)))
                job.progress_label = f"Run {current} of {total}"
                return
            if "Run complete" in line:
                job.progress = 0.98
                job.progress_label = "Generating outputs"
                return

        if job.workflow_id == "assessment.run":
            phase_match = re.search(r"Phase\s+(\d)(?:b)?/4", line)
            if phase_match:
                current = int(phase_match.group(1))
                job.progress = min(0.95, current / 4.0)
                job.progress_label = line.strip()
                return

        four_phase_match = re.search(r"\[\s*(\d+)/4\]", line)
        if four_phase_match:
            current = int(four_phase_match.group(1))
            job.progress = min(0.95, current / 4.0)
            job.progress_label = line.strip()
            return

        if "saved" in line.lower() or "ready!" in line.lower() or "complete" in line.lower():
            job.progress = max(job.progress or 0.2, 0.9)
            job.progress_label = line.strip()

    def _save_history(self) -> None:
        with self._lock:
            self._save_history_locked()

    def _save_history_locked(self) -> None:
        _JOB_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for job in self._jobs.values():
            item = job.to_dict()
            item["log_lines"] = item["log_lines"][-500:]
            payload.append(item)
        _JOB_HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_history(self) -> None:
        if not _JOB_HISTORY_PATH.exists():
            return
        try:
            payload = json.loads(_JOB_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, list):
            return
        for item in payload:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "unknown"))
            if status in {"running", "queued"}:
                status = "unknown"
            job = Job(
                id=str(item.get("id") or uuid.uuid4().hex),
                workflow_id=str(item.get("workflow_id") or "unknown"),
                title=str(item.get("title") or "Recovered Job"),
                payload=item.get("payload") if isinstance(item.get("payload"), dict) else {},
                command=list(item.get("command") or []),
                status=status,
                progress=item.get("progress") if isinstance(item.get("progress"), (int, float)) else None,
                progress_label=str(item.get("progress_label") or "Recovered from history"),
                created_at=str(item.get("created_at") or _utc_now()),
                started_at=item.get("started_at"),
                finished_at=item.get("finished_at"),
                return_code=item.get("return_code") if isinstance(item.get("return_code"), int) else None,
                output_dir=item.get("output_dir"),
                log_path=item.get("log_path"),
                error=item.get("error"),
                log_lines=deque((str(line) for line in item.get("log_lines", [])), maxlen=3000),
                phase=item.get("phase"),
            )
            self._jobs[job.id] = job


job_manager = JobManager()
