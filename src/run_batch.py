"""Run ``run.py`` across every dataset folder with a live terminal dashboard.

Usage examples:
    python src/run_batch.py
    python src/run_batch.py --jobs 2 --pages 5 --eval
    python src/run_batch.py id_02 id_49 --group-rows

The run.py options accepted by this script are forwarded to every job.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RUN_SCRIPT = REPO_ROOT / "run.py"
SPINNER_FRAMES = ("|", "/", "-", "\\")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(slots=True)
class JobState:
    name: str
    folder: Path
    pdf_path: Path
    status: str = "queued"
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_line: str = "Waiting to start"
    key_preview: str | None = None
    line_count: int = 0
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)

    def elapsed_seconds(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end_time = self.finished_at if self.finished_at is not None else (now or time.time())
        return max(0.0, end_time - self.started_at)


class TerminalDashboard:
    def __init__(self, jobs: Sequence[JobState]) -> None:
        self.jobs = jobs
        self.enabled = sys.stdout.isatty() and os.environ.get("TERM") not in {None, "", "dumb"}
        self.start_time = time.time()
        self.frame = 0
        self._entered = False

    def __enter__(self) -> "TerminalDashboard":
        if self.enabled:
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            sys.stdout.flush()
            self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._entered:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            self._entered = False

    def render(self) -> None:
        if not self.enabled:
            return
        self.frame += 1
        sys.stdout.write("\x1b[H\x1b[2J")
        sys.stdout.write(self._build_screen())
        sys.stdout.flush()

    def _build_screen(self) -> str:
        width, height = shutil.get_terminal_size(fallback=(120, 30))
        now = time.time()

        queued = sum(job.status == "queued" for job in self.jobs)
        running = sum(job.status == "running" for job in self.jobs)
        succeeded = sum(job.status == "succeeded" for job in self.jobs)
        failed = sum(job.status == "failed" for job in self.jobs)
        cancelled = sum(job.status == "cancelled" for job in self.jobs)

        name_width = min(24, max(12, max((len(job.name) for job in self.jobs), default=12)))
        status_width = 11
        elapsed_width = 8
        code_width = 6
        padding = 8
        message_width = max(20, width - name_width - status_width - elapsed_width - code_width - padding)

        lines = [
            "run.py batch dashboard",
            self._key_preview_line(),
            (
                f"Queued: {queued}  Running: {running}  Succeeded: {succeeded}  "
                f"Failed: {failed}  Cancelled: {cancelled}  "
                f"Elapsed: {format_duration(time.time() - self.start_time)}"
            ),
            "Press Ctrl-C to stop all running jobs.",
            "",
            (
                f"{fit_text('Document', name_width)}  "
                f"{fit_text('Status', status_width)}  "
                f"{fit_text('Elapsed', elapsed_width)}  "
                f"{fit_text('Code', code_width)}  "
                f"{fit_text('Latest output', message_width)}"
            ),
            (
                f"{'-' * name_width}  {'-' * status_width}  {'-' * elapsed_width}  "
                f"{'-' * code_width}  {'-' * message_width}"
            ),
        ]

        for job in self.jobs:
            status_label = self._status_label(job)
            code = "" if job.returncode is None else str(job.returncode)
            lines.append(
                f"{fit_text(job.name, name_width)}  "
                f"{fit_text(status_label, status_width)}  "
                f"{fit_text(format_duration(job.elapsed_seconds(now)), elapsed_width)}  "
                f"{fit_text(code, code_width)}  "
                f"{fit_text(clean_line(job.last_line), message_width)}"
            )

        if len(lines) > height:
            visible_rows = max(0, height - 1)
            hidden = len(lines) - visible_rows
            lines = lines[:visible_rows]
            if lines:
                lines[-1] = fit_text(f"... {hidden} more jobs not shown", width)

        return "\n".join(lines) + "\n"

    def _key_preview_line(self) -> str:
        previews = list(dict.fromkeys(
            job.key_preview for job in self.jobs if job.key_preview is not None
        ))
        if not previews:
            return "using kwy: <waiting>"
        if len(previews) == 1:
            return f"using kwy: {previews[0]}"
        return "using kwy: " + ", ".join(previews)

    def _status_label(self, job: JobState) -> str:
        if job.status == "running":
            spinner = SPINNER_FRAMES[self.frame % len(SPINNER_FRAMES)]
            return f"{spinner} RUNNING"
        if job.status == "queued":
            return "QUEUED"
        if job.status == "succeeded":
            return "OK"
        if job.status == "failed":
            return "FAILED"
        if job.status == "cancelled":
            return "CANCELLED"
        return job.status.upper()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fit_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 3:
        return text[:width]
    return f"{text[:width - 3]}..."


def clean_line(text: str) -> str:
    text = ANSI_RE.sub("", text).replace("\t", " ")
    return " ".join(text.split())


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run run.py in parallel across all folders in data/ with a live TUI.",
        epilog="Known run.py flags are forwarded to every dataset job.",
    )
    parser.add_argument(
        "documents",
        nargs="*",
        help="Optional dataset folder names under data/. Defaults to every valid folder.",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=4,
        help="Maximum number of concurrent run.py processes (default: 4).",
    )
    parser.add_argument(
        "--refresh-rate",
        type=positive_float,
        default=0.2,
        metavar="SECONDS",
        help="Dashboard refresh interval in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--run-script",
        type=Path,
        default=DEFAULT_RUN_SCRIPT,
        help=f"Path to the run.py entrypoint (default: {DEFAULT_RUN_SCRIPT}).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch run.py (default: current interpreter).",
    )
    run_options = parser.add_argument_group("run.py options")
    run_options.add_argument(
        "--pages",
        type=positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Forwarded to run.py: minimum training pages; cluster samples are added.",
    )
    run_options.add_argument(
        "--start-page",
        type=positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Forwarded to run.py: fixed physical record start page.",
    )
    evaluation = run_options.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--eval",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: run train/test evaluation (run.py default).",
    )
    evaluation.add_argument(
        "--no-eval",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: disable the independent test evaluation.",
    )
    run_options.add_argument(
        "--eval-pages",
        type=positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Forwarded to run.py: maximum disjoint evaluation pages to sample.",
    )
    run_options.add_argument(
        "--group-rows",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: group OCR words into row-level entries.",
    )
    run_options.add_argument(
        "--reuse-ocr",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: explicitly reuse each document's existing OCR CSV.",
    )
    run_options.add_argument(
        "--train-threshold",
        type=probability,
        default=argparse.SUPPRESS,
        metavar="FLOAT",
        help="Forwarded to run.py: train accuracy threshold for repair.",
    )
    run_options.add_argument(
        "--sonar-drop-percent",
        type=float,
        default=argparse.SUPPRESS,
        metavar="PERCENT",
        help="Forwarded to run.py: relative coverage drop flagged by sonar.",
    )
    run_options.add_argument(
        "--train-max-retries",
        type=nonnegative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Forwarded to run.py: maximum number of repair attempts.",
    )
    run_options.add_argument(
        "--gt-test-window",
        type=positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Forwarded to run.py: pages per LLM call for test ground truth (default: 20).",
    )
    run_options.add_argument(
        "--manual-ground-truth", action="store_true", default=argparse.SUPPRESS,
        help="Forwarded to run.py: defer final test ground truth for UI submission.",
    )
    run_options.add_argument(
        "--holdout-pages",
        type=nonnegative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Forwarded to run.py: train pages hidden from script generation (0 disables).",
    )
    run_options.add_argument(
        "--easy-model",
        default=argparse.SUPPRESS,
        metavar="MODEL",
        help="Forwarded to run.py: OpenRouter model for easy extraction blocks.",
    )
    run_options.add_argument(
        "--hard-model",
        default=argparse.SUPPRESS,
        metavar="MODEL",
        help=(
            "Forwarded to run.py: model for block separation, difficulty routing, "
            "hard blocks, and repair."
        ),
    )
    run_options.add_argument(
        "--reference-model",
        default=argparse.SUPPRESS,
        metavar="MODEL",
        help="Forwarded to run.py: model for train/test ground truth.",
    )
    run_options.add_argument(
        "--template-model",
        default=argparse.SUPPRESS,
        metavar="MODEL",
        help="Forwarded to run.py: model for template inference.",
    )
    run_options.add_argument(
        "--caveman",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: prepend /caveman to every OpenCode prompt.",
    )
    run_options.add_argument(
        "--no-routing",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: use the easy model for every non-reference phase.",
    )
    run_options.add_argument(
        "--save-log",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Forwarded to run.py: save raw OpenCode session output to <doc>/debug/.",
    )
    return parser.parse_known_args(argv)


def build_forwarded_args(args: argparse.Namespace, unknown_args: Sequence[str]) -> list[str]:
    forwarded: list[str] = []

    for flag, attr in (
        ("--pages", "pages"),
        ("--start-page", "start_page"),
        ("--eval-pages", "eval_pages"),
        ("--train-threshold", "train_threshold"),
        ("--sonar-drop-percent", "sonar_drop_percent"),
        ("--train-max-retries", "train_max_retries"),
        ("--gt-test-window", "gt_test_window"),
        ("--holdout-pages", "holdout_pages"),
        ("--easy-model", "easy_model"),
        ("--hard-model", "hard_model"),
        ("--reference-model", "reference_model"),
        ("--template-model", "template_model"),
    ):
        if hasattr(args, attr):
            forwarded.extend([flag, str(getattr(args, attr))])

    for flag, attr in (
        ("--eval", "eval"),
        ("--no-eval", "no_eval"),
        ("--group-rows", "group_rows"),
        ("--reuse-ocr", "reuse_ocr"),
        ("--manual-ground-truth", "manual_ground_truth"),
        ("--no-routing", "no_routing"),
        ("--caveman", "caveman"),
        ("--save-log", "save_log"),
    ):
        if getattr(args, attr, False):
            forwarded.append(flag)

    forwarded.extend(unknown_args)
    return forwarded


def discover_jobs(data_dir: Path, requested_docs: Sequence[str]) -> list[JobState]:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    if requested_docs:
        names = list(dict.fromkeys(requested_docs))
    else:
        names = sorted(entry.name for entry in data_dir.iterdir() if entry.is_dir())

    jobs: list[JobState] = []
    missing: list[str] = []
    invalid: list[str] = []

    for name in names:
        folder = data_dir / name
        pdf_path = folder / f"{name}.pdf"
        if not folder.is_dir():
            missing.append(name)
            continue
        if not pdf_path.is_file():
            invalid.append(name)
            continue
        jobs.append(JobState(name=name, folder=folder, pdf_path=pdf_path))

    if missing:
        raise SystemExit(f"Unknown dataset folders: {', '.join(sorted(missing))}")
    if invalid:
        raise SystemExit(
            "Dataset folders missing their expected PDF: "
            + ", ".join(f"{name} ({data_dir / name / f'{name}.pdf'})" for name in sorted(invalid))
        )
    if not jobs:
        raise SystemExit(f"No runnable dataset folders found in {data_dir}")

    return jobs


def build_command(args: argparse.Namespace, forwarded_args: Sequence[str], job: JobState) -> list[str]:
    return [
        args.python,
        "-u",
        str(args.run_script.resolve()),
        str(job.pdf_path),
        *forwarded_args,
        "--out-dir",
        str(job.folder),
    ]


async def stream_output(stream: asyncio.StreamReader, job: JobState) -> None:
    # Tee the job's full output to data/<doc>/batch.log (append, line-buffered)
    log_path = job.folder / "batch.log"
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # a single line exceeded the reader limit — drain a chunk instead
                line = await stream.read(2 ** 20)
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            log.write(decoded if decoded.endswith("\n") else decoded + "\n")
            stripped = decoded.strip()
            if stripped:
                if stripped.startswith("using kwy: "):
                    job.key_preview = stripped.removeprefix("using kwy: ")
                job.last_line = stripped
                job.line_count += 1


async def terminate_process(job: JobState) -> None:
    process = job.process
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def run_job(
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    forwarded_args: Sequence[str],
    job: JobState,
) -> int:
    async with semaphore:
        command = build_command(args, forwarded_args, job)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        job.status = "running"
        job.started_at = time.time()
        job.last_line = "Launching run.py"

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                limit=2 ** 24,  # 16 MB — repair loop can emit very long lines
            )
        except Exception as exc:
            job.status = "failed"
            job.finished_at = time.time()
            job.last_line = f"Failed to start: {exc}"
            job.returncode = -1
            return job.returncode

        job.process = process
        assert process.stdout is not None
        reader_task = asyncio.create_task(stream_output(process.stdout, job))

        try:
            returncode = await process.wait()
            await reader_task
        except asyncio.CancelledError:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
            await terminate_process(job)
            job.status = "cancelled"
            job.returncode = process.returncode
            job.finished_at = time.time()
            if process.returncode is None:
                job.last_line = "Interrupted"
            raise
        finally:
            job.process = None

        job.returncode = returncode
        job.finished_at = time.time()
        if returncode == 0:
            job.status = "succeeded"
            if clean_line(job.last_line) == "Launching run.py":
                job.last_line = "Completed successfully"
        else:
            job.status = "failed"
            if clean_line(job.last_line) == "Launching run.py":
                job.last_line = f"Failed with exit code {returncode}"
        return returncode


async def render_loop(dashboard: TerminalDashboard, jobs: Sequence[JobState], refresh_rate: float) -> None:
    try:
        while True:
            dashboard.render()
            if all(job.status in {"succeeded", "failed", "cancelled"} for job in jobs):
                break
            await asyncio.sleep(refresh_rate)
    except asyncio.CancelledError:
        dashboard.render()
        raise


def print_final_summary(jobs: Sequence[JobState], interrupted: bool) -> None:
    succeeded = [job for job in jobs if job.status == "succeeded"]
    failed = [job for job in jobs if job.status == "failed"]
    cancelled = [job for job in jobs if job.status == "cancelled"]

    print("Batch summary")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Cancelled: {len(cancelled)}")
    if interrupted:
        print("  Run interrupted by signal.")

    for job in failed + cancelled:
        code = "n/a" if job.returncode is None else str(job.returncode)
        print(
            f"  {job.name}: status={job.status} code={code} "
            f"elapsed={format_duration(job.elapsed_seconds())} last='{clean_line(job.last_line)}'"
        )


async def run_batch(args: argparse.Namespace, forwarded_args: Sequence[str]) -> int:
    jobs = discover_jobs(args.data_dir, args.documents)

    if not args.run_script.is_file():
        raise SystemExit(f"run.py entrypoint not found: {args.run_script}")

    semaphore = asyncio.Semaphore(min(args.jobs, len(jobs)))
    dashboard = TerminalDashboard(jobs)
    interrupt_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, interrupt_event.set)

    job_tasks = [
        asyncio.create_task(run_job(semaphore, args, forwarded_args, job), name=f"run:{job.name}")
        for job in jobs
    ]
    render_task = asyncio.create_task(render_loop(dashboard, jobs, args.refresh_rate))
    interrupt_task = asyncio.create_task(interrupt_event.wait(), name="interrupt-wait")

    interrupted = False

    with dashboard:
        try:
            gather_future = asyncio.gather(*job_tasks)
            done, _ = await asyncio.wait(
                {gather_future, interrupt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if interrupt_task in done:
                interrupted = True
                for task in job_tasks:
                    task.cancel()
                await asyncio.gather(*job_tasks, return_exceptions=True)
                gather_future.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await gather_future
            else:
                await gather_future
        finally:
            interrupt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt_task
            render_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await render_task

    print_final_summary(jobs, interrupted)

    if interrupted:
        return 130
    if any(job.status == "failed" for job in jobs):
        return 1
    if any(job.status == "cancelled" for job in jobs):
        return 130
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args, unknown_args = parse_args(argv or sys.argv[1:])
    forwarded_args = build_forwarded_args(args, unknown_args)
    return asyncio.run(run_batch(args, forwarded_args))


if __name__ == "__main__":
    raise SystemExit(main())
