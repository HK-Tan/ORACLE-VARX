#!/usr/bin/env python3
"""Orchestrate all experiment runs across phases using tmux.

This script automates the tmux parallel launch pattern for the full
experiment matrix (16 runs -> 35 unique model outputs).

Phases:
    Phase 0: All OLS baselines — VAR, ACLE-VAR, VARX x3, ACLE-VARX x3 (sequential)
    Phase 1: VIX x 4 learners, DML only (4 parallel tmux panes)
    Phase 2: macro5 x 4 learners, DML only (4 parallel tmux panes)
    Phase 3: all10 x 4 learners, DML only (4 parallel tmux panes)

Usage:
    # Run all phases sequentially
    python scripts/run_all_experiments.py --phase all

    # Run a specific phase
    python scripts/run_all_experiments.py --phase 1

    # Dry run to see commands without executing
    python scripts/run_all_experiments.py --phase all --dry-run

    # Override n_jobs per pane
    python scripts/run_all_experiments.py --phase 1 --n-jobs 3

Note:
    TabPFN runs (GPU) are not included in this orchestrator.
    Run them separately on a GPU machine:
        python scripts/run_combined_experiment.py --confounders vix --learner lgbm --device cuda --no-show
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
CONFOUNDER_CONFIGS = {
    1: "vix",
    2: "macro5",
    3: "all10",
}

SCRIPT = "python scripts/run_combined_experiment.py"


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


def get_phase_commands(phase: int, n_jobs: int = None, verbose: bool = True) -> list[str]:
    """Get the shell commands for a given phase.

    Phase 0 returns multiple sequential OLS commands (no tmux needed).
    Phases 1-3 return parallel DML-only commands with BLAS thread limits.
    """
    verbose_arg = " --verbose" if verbose else ""

    if phase == 0:
        # All OLS baselines — run sequentially with full CPU access
        return [
            f"{SCRIPT} --no-confounders --no-show{verbose_arg}",
            f"{SCRIPT} --confounders vix --ols-only --no-show{verbose_arg}",
            f"{SCRIPT} --confounders macro5 --ols-only --no-show{verbose_arg}",
            f"{SCRIPT} --confounders all10 --ols-only --no-show{verbose_arg}",
        ]
    elif phase in CONFOUNDER_CONFIGS:
        conf = CONFOUNDER_CONFIGS[phase]
        n_panes = len(LEARNERS)
        threads = n_jobs if n_jobs else _compute_threads_per_pane(n_panes)
        thread_env = (
            f"OMP_NUM_THREADS={threads} "
            f"MKL_NUM_THREADS={threads} "
            f"OPENBLAS_NUM_THREADS={threads}"
        )
        return [
            f"{thread_env} {SCRIPT} --confounders {conf} --learner {learner} --no-show{verbose_arg} --n-jobs {threads}"
            for learner in LEARNERS
        ]
    else:
        raise ValueError(f"Unknown phase: {phase}. Valid: 0, 1, 2, 3, all")


def run_phase_tmux(phase: int, commands: list[str], dry_run: bool = False,
                   sequential: bool = False):
    """Launch commands in tmux panes for parallel execution.

    Args:
        phase: Phase number (for display and tmux session naming).
        commands: Shell commands to run.
        dry_run: If True, print commands without executing.
        sequential: If True, run all commands sequentially via subprocess.run()
            instead of tmux. Used for Phase 0 (OLS baselines).
    """
    session_name = f"phase{phase}"

    if sequential:
        # Run all commands one at a time — no tmux needed
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
        # Single command — just run directly
        cmd = commands[0]
        print(f"\n  Phase {phase}: Running single command:")
        print(f"    {cmd}")
        if dry_run:
            return

        # Run in foreground and wait
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  WARNING: Phase {phase} exited with code {result.returncode}")
        return

    # Multiple commands — create tmux session with panes
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
    pane_ids = [result.stdout.strip()]  # e.g. ["%0"]
    subprocess.run(
        ["tmux", "send-keys", "-t", pane_ids[0], commands[0], "Enter"],
        check=True,
    )

    # Build 2x2 layout using stable pane IDs:
    #   pane_ids[0] (top-left)  | pane_ids[1] (top-right)
    #   pane_ids[2] (bot-left)  | pane_ids[3] (bot-right)
    for i, cmd in enumerate(commands[1:], start=1):
        if i == 1:
            # Horizontal split from pane 0 -> creates right pane
            split_target = pane_ids[0]
            split_dir = ["-h"]
        elif i == 2:
            # Vertical split from pane 0 (top-left) -> creates bottom-left
            split_target = pane_ids[0]
            split_dir = ["-v"]
        elif i == 3:
            # Vertical split from pane 1 (top-right) -> creates bottom-right
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

    # Wait for all panes to finish by polling
    while True:
        time.sleep(30)
        # Check if any pane is still running a python process
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_current_command}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Session no longer exists
            break

        active_commands = result.stdout.strip().split("\n")
        python_panes = [c for c in active_commands if "python" in c.lower()]
        if not python_panes:
            print(f"  All panes in phase {phase} have finished.")
            break

        # Show progress
        n_active = len(python_panes)
        n_done = len(commands) - n_active
        print(f"  Phase {phase}: {n_done}/{len(commands)} panes done, {n_active} still running...")

    # Clean up tmux session
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate all experiment runs across phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  0: All OLS baselines — VAR, ACLE-VAR, VARX x3, ACLE-VARX x3 (sequential)
  1: VIX x 4 tree learners, DML only (parallel)
  2: macro5 x 4 tree learners, DML only (parallel)
  3: all10 x 4 tree learners, DML only (parallel)
  all: Run phases 0-3 sequentially

Example workflow on c7i.4xlarge:
  python scripts/run_all_experiments.py --phase all --verbose
        """
    )
    parser.add_argument("--phase", type=str, required=True,
                        help="Phase to run: 0, 1, 2, 3, or 'all'")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="CPU cores per pane (default: auto-computed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Pass --verbose to experiment scripts")

    args = parser.parse_args()

    print("=" * 80)
    print("ORACLE-VARX: Full Experiment Suite Orchestrator")
    print("=" * 80)

    if args.phase == "all":
        phases = [0, 1, 2, 3]
    else:
        phases = [int(args.phase)]

    total_start = time.perf_counter()

    for phase in phases:
        phase_start = time.perf_counter()
        print(f"\n{'=' * 60}")
        print(f"  PHASE {phase}: {_phase_description(phase)}")
        print(f"{'=' * 60}")

        commands = get_phase_commands(phase, n_jobs=args.n_jobs, verbose=args.verbose)
        run_phase_tmux(phase, commands, dry_run=args.dry_run,
                       sequential=(phase == 0))

        phase_elapsed = time.perf_counter() - phase_start
        print(f"  Phase {phase} complete: {phase_elapsed:.0f}s ({phase_elapsed/60:.1f} min)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 80}")
    print(f"ALL PHASES COMPLETE — Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 80}")


def _phase_description(phase: int) -> str:
    descs = {
        0: "All OLS baselines (VAR, ACLE-VAR, VARX x3, ACLE-VARX x3)",
        1: "VIX x 4 learners (DML only)",
        2: "macro5 x 4 learners (DML only)",
        3: "all10 x 4 learners (DML only)",
    }
    return descs.get(phase, f"Unknown phase {phase}")


if __name__ == "__main__":
    main()
