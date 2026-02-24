"""One-time migration: transpose coefficients/SE in stored DML result files.

The DML coefficient code had a transpose bug where the last two dimensions
of coefficients (VARXResult) and SE_all (ORACLEVARXResult) were swapped.
This script retroactively fixes previously-saved result.pt / results.pt files.

Usage:
    # Dry run (default) — shows what would be changed
    python scripts/fix_stored_coefficients.py

    # Apply with backups (.bak files created)
    python scripts/fix_stored_coefficients.py --apply

    # Apply without backups
    python scripts/fix_stored_coefficients.py --apply --no-backup
"""

import argparse
import shutil
from pathlib import Path

import torch


def is_dml_varx_path(path: Path) -> bool:
    """Check if a VARXResult file came from a DML model (OR-VARX) based on path.

    Real results:  results/<model_type>/<run_name>/results.pt
        DML if model_type is 'orvarx' or 'orvarx_tabpfn'
    Toy results:   results-toy/<experiment_name>/result.pt
        DML if experiment_name starts with 'OR-VARX'
    """
    parts = path.parts

    # Real results: check grandparent directory name
    for i, part in enumerate(parts):
        if part in ("orvarx", "orvarx_tabpfn"):
            return True
        if part in ("var", "varx", "aclevarx", "oraclevarx", "oraclevarx_tabpfn"):
            return False

    # Toy results: check parent directory name
    parent_name = path.parent.name
    if parent_name.startswith("OR-VARX"):
        return True
    if parent_name.startswith(("VAR_", "VARX_", "ACLE-VAR", "ORACLE-VARX")):
        return False

    return False


def fix_file(path: Path, apply: bool, backup: bool) -> str:
    """Process a single result file. Returns a status string."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result_type = checkpoint.get("result_type", "unknown")

    # ACLEVARXResult: OLS-based, always correct
    if result_type == "ACLEVARXResult":
        return f"SKIP (ACLEVARXResult, OLS)  {path}"

    # VARXResult: need to distinguish DML vs OLS by path
    if result_type == "VARXResult":
        if not is_dml_varx_path(path):
            return f"SKIP (VARXResult, OLS)     {path}"

        coef = checkpoint.get("coefficients")
        if coef is None:
            return f"SKIP (VARXResult, no coef) {path}"

        old_shape = tuple(coef.shape)
        new_coef = coef.transpose(-2, -1)
        new_shape = tuple(new_coef.shape)

        if apply:
            if backup:
                shutil.copy2(path, str(path) + ".bak")
            checkpoint["coefficients"] = new_coef
            torch.save(checkpoint, path)

        action = "FIXED" if apply else "WOULD FIX"
        return f"{action} (VARXResult, DML coef)  {path}  shape: {old_shape} -> {new_shape}"

    # ORACLEVARXResult: always DML-based
    if result_type == "ORACLEVARXResult":
        se = checkpoint.get("SE_all")
        if se is None:
            return f"SKIP (ORACLEVARXResult, SE_all=None) {path}"

        old_shape = tuple(se.shape)
        new_se = se.transpose(-2, -1)
        new_shape = tuple(new_se.shape)

        if apply:
            if backup:
                shutil.copy2(path, str(path) + ".bak")
            checkpoint["SE_all"] = new_se
            torch.save(checkpoint, path)

        action = "FIXED" if apply else "WOULD FIX"
        return f"{action} (ORACLEVARXResult, SE_all) {path}  shape: {old_shape} -> {new_shape}"

    return f"SKIP (unknown type: {result_type}) {path}"


def main():
    parser = argparse.ArgumentParser(
        description="Fix transposed coefficients in stored DML result files."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write fixes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backup files when applying fixes.",
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results", "results-toy"],
        help="Directories to search for result files (default: results results-toy).",
    )
    args = parser.parse_args()

    backup = not args.no_backup

    if not args.apply:
        print("=== DRY RUN (use --apply to write changes) ===\n")
    else:
        print(f"=== APPLYING FIXES (backup={'yes' if backup else 'no'}) ===\n")

    # Collect all result files
    result_files = []
    for dir_name in args.results_dirs:
        results_dir = Path(dir_name)
        if not results_dir.exists():
            print(f"Directory not found: {results_dir}")
            continue
        result_files.extend(results_dir.rglob("result.pt"))
        result_files.extend(results_dir.rglob("results.pt"))

    result_files.sort()

    if not result_files:
        print("No result files found.")
        return

    print(f"Found {len(result_files)} result file(s):\n")

    fixed = 0
    skipped = 0
    errors = 0

    for path in result_files:
        try:
            status = fix_file(path, apply=args.apply, backup=backup)
            print(f"  {status}")
            if status.startswith(("FIXED", "WOULD FIX")):
                fixed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")
            errors += 1

    print(f"\nSummary: {fixed} fixed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
