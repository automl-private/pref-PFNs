"""Record which code and environment produced a checkpoint.

A checkpoint already stores the `MainConfig` that produced it, which fixes the *training
distribution*. It does not fix the *code*: the same config run against a modified prior or a
changed training loop yields a different model. This module supplies the missing half, so a
checkpoint found later can be traced to the commit that produced it without relying on file
timestamps or on someone's memory.

Two repositories are reported, because they are usually not the same one:

- ``code``   -- the repository containing this package.
- ``inputs`` -- the repositories containing files registered with `record_input_file`,
  typically the config passed to `run_training_cli.py`. Configs commonly live in a separate
  project or workspace repository, whose commit is otherwise unrecorded.

The ``dirty`` flag matters as much as the commit. A checkpoint produced from a modified
working tree is not reproducible from any commit and should be treated as provisional.
``dirty=None`` means the state could not be determined, which is deliberately distinct from
``False``: an unknown state is not a clean one.

Nothing here raises. Provenance must never be the reason a training run fails, so every
lookup degrades to ``None``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_input_files: list[Path] = []
_cached: dict | None = None


def record_input_file(path: str | os.PathLike) -> None:
    """Register a file whose repository should appear in the provenance record.

    Intended for configs and other inputs read from outside this package. Call before
    training starts; duplicates and unresolvable paths are ignored.
    """
    try:
        resolved = Path(path).resolve()
    except OSError:
        return
    if resolved not in _input_files:
        _input_files.append(resolved)


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def repo_state(path: str | os.PathLike) -> dict | None:
    """Commit, branch and dirty flag of the repository containing `path`.

    Returns None when `path` is not inside a git repository, which is a normal situation
    (an installed package, a config kept outside version control) and not an error.
    """
    start = Path(path)
    start = start if start.is_dir() else start.parent
    root = _git(start, "rev-parse", "--show-toplevel")
    if not root:
        return None
    status = _git(start, "status", "--porcelain")
    return {
        "root": root,
        "commit": _git(start, "rev-parse", "HEAD"),
        "branch": _git(start, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def provenance() -> dict:
    """A record of the code, inputs, environment and invocation behind a run.

    Cached after the first call: the git subprocesses are not free and `save_checkpoint`
    runs every epoch. Caching also guarantees every checkpoint from a single run reports
    the same code state, which is the honest answer -- the code that ran is the code as it
    was when the process started.
    """
    global _cached
    if _cached is not None:
        return _cached

    try:
        import torch

        # str(): torch.__version__ is a TorchVersion instance, not a plain str. Embedding
        # the object makes any checkpoint carrying this block unloadable under the
        # torch>=2.6 default of weights_only=True. See `.claude/common-pitfalls.md`.
        torch_version = str(torch.__version__)
        cuda_version = torch.version.cuda
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        torch_version = cuda_version = device_name = None

    _cached = {
        "code": repo_state(Path(__file__).resolve().parent),
        "inputs": {str(p): repo_state(p) for p in _input_files},
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch_version,
        "cuda": cuda_version,
        "gpu": device_name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return _cached


def format_provenance(p: dict | None = None) -> str:
    """A compact human-readable block, for training logs."""
    p = provenance() if p is None else p

    def line(label: str, r: dict | None) -> str:
        if r is None:
            return f"  {label:<10} not under version control"
        commit = (r["commit"] or "unknown")[:10]
        if r["dirty"]:
            state = "  DIRTY - not reproducible from any commit"
        elif r["dirty"] is None:
            state = "  (dirty state unknown)"
        else:
            state = ""
        return f"  {label:<10} {commit} on {r['branch'] or '?'}{state}"

    lines = ["[provenance]", line("code", p["code"])]
    for path, state in p["inputs"].items():
        lines.append(line(Path(path).name, state))
    job = p["slurm_job_id"]
    lines.append(f"  {'host':<10} {p['hostname']}{'' if job is None else f' (slurm job {job})'}")
    lines.append(f"  {'torch':<10} {p['torch']} (cuda {p['cuda']})  python {p['python']}")
    if p["gpu"]:
        lines.append(f"  {'gpu':<10} {p['gpu']}")
    lines.append(f"  {'started':<10} {p['started_utc']}")
    lines.append(f"  {'argv':<10} {' '.join(p['argv'])}")
    return "\n".join(lines)


def _main() -> None:
    """Print the provenance stored in a checkpoint: `python -m pfns.provenance CKPT ...`.

    With no argument, prints the provenance of the current process instead.
    """
    if len(sys.argv) == 1:
        print(format_provenance())
        return

    import torch

    for path in sys.argv[1:]:
        print(f"=== {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("provenance") if isinstance(checkpoint, dict) else None
        if stored is None:
            # Every checkpoint trained before this module existed. The config is still
            # embedded; only the code version is missing.
            print("  no provenance recorded (trained before provenance was added)")
        else:
            print(format_provenance(stored))
        print()


if __name__ == "__main__":
    _main()
