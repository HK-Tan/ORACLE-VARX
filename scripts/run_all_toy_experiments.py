#!/usr/bin/env python3
"""Orchestrate toy benchmark runs across phases using tmux.

Parallelizes Phase 1 (DML) by learner — each learner gets its own tmux pane
with BLAS thread limits. Each pane is fully independent (per-experiment edge
trajectories + incremental metrics_summary.csv).

Phases:
    Phase 0: All OLS baselines (sequential, no tmux)
    Phase 1: DML × 4 learners × 3 obs levels (4 parallel tmux panes)

Usage:
    # Run all phases
    python scripts/run_all_toy_experiments.py --phase all

    # Run Phase 1 only
    python scripts/run_all_toy_experiments.py --phase 1

    # Dry run to inspect commands
    python scripts/run_all_toy_experiments.py --phase all --dry-run

    # Override n_jobs per pane
    python scripts/run_all_toy_experiments.py --phase 1 --n-jobs 3

Note:
    Phase 2 (TabPFN, GPU) is not included — run separately on a GPU machine:
        python scripts/run_toy_benchmark.py --phase 2 --device cuda --no-show
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


LEARNERS = ["lgbm", "xgboost", "rf", "extra_trees"]
SCRIPT = "python scripts/run_toy_benchmark.py"


def _get_physical_cores() -> int:
    """Get physical core count (falls back to os.cpu_count)."""
    try:
        return int(subprocess.check_output(
            ["python", "-c", "import psutil; print(psutil.cpu_count(logical=False))"],
            text=True,
        ).strip())
    except Exception:
        return os.cpu_count() or 4


def _compute_threads_per_pane(n_panes: int) -> int:
    """Compute per-pane thread/job count for BLAS and joblib."""
    cores = _get_physical_cores()
    return max(1, (cores - 1) // n_panes)


def get_phase_commands(phase: int, n_jobs: int = None, verbose: bool = True,
                       no_show: bool = True) -> list[str]:
    """Get the shell commands for a given phase.

    Phase 0 returns a single sequential command.
    Phase 1 returns 4 parallel commands (one per learner) with BLAS thread limits.
    """
    verbose_arg = " --verbose" if verbose else ""
    no_show_arg = " --no-show" if no_show else ""

    if phase == 0:
        return [
            f"{SCRIPT} --phase 0{no_show_arg}{verbose_arg}",
        ]
    elif phase == 1:
        n_panes = len(LEARNERS)
        threads = n_jobs if n_jobs else _compute_threads_per_pane(n_panes)
        thread_env = (
            f"OMP_NUM_THREADS={threads} "
            f"MKL_NUM_THREADS={threads} "
            f"OPENBLAS_NUM_THREADS={threads}"
        )
        return [
            f"{thread_env} {SCRIPT} --phase 1 --learner {learner}{no_show_arg}{verbose_arg} --n-jobs {threads}"
            for learner in LEARNERS
        ]
    else:
        raise ValueError(f"Unknown phase: {phase}. Valid: 0, 1, all")


def run_phase_tmux(phase: int, commands: list[str], dry_run: bool = False,
                   sequential: bool = False):
    """Launch commands in tmux panes for parallel execution.

    Args:
        phase: Phase number (for display and tmux session naming).
        commands: Shell commands to run.
        dry_run: If True, print commands without executing.
        sequential: If True, run commands sequentially (no tmux).
    """
    session_name = f"toy_phase{phase}"

    if sequential:
        print(f"\n  Phase {phase}: Running {len(commands)} commands sequentially:")
        for i, cmd in enumerate(commands, 1):
            print(f"    [{i}/{len(commands)}] {cmd}")
        if dry_run:
            return

        for i, cmd in enumerate(commands, 1):
            print(f"\n  [{i}/{len(commands)}] Running: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"  WARNING: Command {i} exited with code {result.returncode}")
        return

    if len(commands) == 1:
        cmd = commands[0]
        print(f"\n  Phase {phase}: Running single command:")
        print(f"    {cmd}")
        if dry_run:
            return

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  WARNING: Phase {phase} exited with code {result.returncode}")
        return

    # Multiple commands — create tmux session with 2x2 grid
    print(f"\n  Phase {phase}: Launching {len(commands)} panes in tmux session '{session_name}'")
    for i, cmd in enumerate(commands):
        print(f"    Pane {i}: {cmd}")

    if dry_run:
        return

    # Kill existing session if it exists
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )

    # Create new session and capture the first pane's unique ID
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-P", "-F", "#{pane_id}"],
        capture_output=True, text=True, check=True,
    )
    pane_ids = [result.stdout.strip()]
    subprocess.run(
        ["tmux", "send-keys", "-t", pane_ids[0], commands[0], "Enter"],
        check=True,
    )

    # Build 2x2 layout using stable pane IDs
    for i, cmd in enumerate(commands[1:], start=1):
        if i == 1:
            split_target = pane_ids[0]
            split_dir = ["-h"]
        elif i == 2:
            split_target = pane_ids[0]
            split_dir = ["-v"]
        elif i == 3:
            split_target = pane_ids[1]
            split_dir = ["-v"]

        result = subprocess.run(
            ["tmux", "split-window"] + split_dir + ["-t", split_target, "-P", "-F", "#{pane_id}"],
            capture_output=True, text=True, check=True,
        )
        pane_ids.append(result.stdout.strip())

        subprocess.run(
            ["tmux", "send-keys", "-t", pane_ids[i], cmd, "Enter"],
            check=True,
        )

    print(f"\n  tmux session '{session_name}' created with {len(commands)} panes.")
    print(f"  Attach with: tmux attach -t {session_name}")
    print(f"  Waiting for all panes to finish...")

    # Poll until all panes finish
    while True:
        time.sleep(30)
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_current_command}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            break

        active_commands = result.stdout.strip().split("\n")
        python_panes = [c for c in active_commands if "python" in c.lower()]
        if not python_panes:
            print(f"  All panes in phase {phase} have finished.")
            break

        n_active = len(python_panes)
        n_done = len(commands) - n_active
        print(f"  Phase {phase}: {n_done}/{len(commands)} panes done, {n_active} still running...")

    # Clean up tmux session
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )


def _phase_description(phase: int) -> str:
    descs = {
        0: "OLS baselines (VAR, ACLE-VAR, VARX x3, ACLE-VARX x3)",
        1: "DML x 4 learners x 3 obs levels (parallel tmux)",
    }
    return descs.get(phase, f"Unknown phase {phase}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate toy benchmark runs using tmux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  0: All OLS baselines — sequential (fast)
  1: DML x 4 learners x 3 obs levels — 4 parallel tmux panes
  all: Phase 0 sequential, then Phase 1 parallel

Example:
  python scripts/run_all_toy_experiments.py --phase all --verbose
        """
    )
    parser.add_argument("--phase", type=str, required=True,
                        help="Phase to run: 0, 1, or 'all'")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="CPU cores per pane (default: auto-computed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Pass --verbose to experiment scripts")
    parser.add_argument("--no-show", action="store_true", default=True,
                        help="Pass --no-show to experiment scripts (default: True)")

    args = parser.parse_args()

    print("=" * 80)
    print("ORACLE-VARX: Toy Benchmark Orchestrator")
    print("=" * 80)

    if args.phase == "all":
        phases = [0, 1]
    else:
        phases = [int(args.phase)]

    total_start = time.perf_counter()

    for phase in phases:
        phase_start = time.perf_counter()
        print(f"\n{'=' * 60}")
        print(f"  PHASE {phase}: {_phase_description(phase)}")
        print(f"{'=' * 60}")

        commands = get_phase_commands(phase, n_jobs=args.n_jobs,
                                     verbose=args.verbose, no_show=args.no_show)
        run_phase_tmux(phase, commands, dry_run=args.dry_run,
                       sequential=(phase == 0))

        phase_elapsed = time.perf_counter() - phase_start
        print(f"  Phase {phase} complete: {phase_elapsed:.0f}s ({phase_elapsed/60:.1f} min)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 80}")
    print(f"ALL PHASES COMPLETE — Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
