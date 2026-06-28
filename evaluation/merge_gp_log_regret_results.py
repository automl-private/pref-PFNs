#!/usr/bin/env python3
"""Merge GP log-regret result files produced by run_gp_log_regret.py."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


COMPATIBLE_METADATA_KEYS = (
    "n_grid",
    "budget",
    "n_init",
    "n_gp_functions",
    "n_bo_seeds",
    "gp_support",
    "grid_design",
    "grid_seed_offset",
    "gp_jitter",
    "gp_rff_num_features",
    "gp_opt_reference_size",
    "gp_rff_eval_batch_size",
    "qeubo_num_acqf_samples",
    "qeubo_max_fit_iter",
    "qeubo_max_fit_attempts",
    "qeubo_fit_hyperparams",
    "qeubo_continuous_num_restarts",
    "qeubo_continuous_raw_samples",
    "qeubo_continuous_maxiter",
    "eps",
    "input_dim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge several GP log-regret .pt files into one plotter-compatible result."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Result files from evaluation/run_gp_log_regret.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Merged output .pt file.",
    )
    parser.add_argument(
        "--on-conflict",
        choices=("error", "replace", "skip"),
        default="error",
        help="What to do when two inputs contain the same suite/method.",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help=(
            "Allow top-level run metadata differences such as budget or n_grid. "
            "Suite eval_hparams and benchmark metadata must still match."
        ),
    )
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Allow methods in the same suite to have different regret tensor shapes.",
    )
    parser.add_argument(
        "--require-same-suites",
        action="store_true",
        help="Require every input file to contain exactly the same suite names.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    result = torch.load(path, map_location="cpu")
    if not isinstance(result, dict):
        raise TypeError(f"{path} does not contain a dict result payload.")
    if "metadata" not in result or "suites" not in result:
        raise KeyError(f"{path} must contain top-level 'metadata' and 'suites' keys.")
    if not isinstance(result["suites"], Mapping):
        raise TypeError(f"{path}['suites'] must be a mapping.")
    return result


def check_metadata_compatible(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    mismatches: list[str] = []
    for key in COMPATIBLE_METADATA_KEYS:
        if reference.get(key) != current.get(key):
            mismatches.append(f"{key}: {reference.get(key)!r} != {current.get(key)!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"Benchmark metadata mismatch in {path}: {joined}")


def check_same_suites(reference_names: set[str], current_names: set[str], *, path: Path) -> None:
    if reference_names == current_names:
        return
    missing = sorted(reference_names - current_names)
    extra = sorted(current_names - reference_names)
    raise ValueError(
        f"Suite mismatch in {path}: missing={missing or 'none'}, extra={extra or 'none'}"
    )


def check_suite_metadata_compatible(
    existing: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    suite_name: str,
    path: Path,
) -> None:
    existing_eval = dict(existing.get("eval_hparams") or {})
    current_eval = dict(current.get("eval_hparams") or {})
    existing_benchmark = dict(existing.get("benchmark") or {})
    current_benchmark = dict(current.get("benchmark") or {})
    if existing_eval != current_eval or existing_benchmark != current_benchmark:
        raise ValueError(
            f"suite metadata mismatch for suite {suite_name!r} in {path}: "
            f"eval_hparams {existing_eval!r} != {current_eval!r}; "
            f"benchmark {existing_benchmark!r} != {current_benchmark!r}"
        )


def payload_shape(payload: Mapping[str, Any]) -> tuple[int, ...]:
    value = payload.get("simple_regret")
    if not isinstance(value, torch.Tensor):
        raise TypeError("method payload must contain tensor key 'simple_regret'.")
    return tuple(value.shape)


def check_method_shapes(suite_name: str, methods: Mapping[str, Mapping[str, Any]]) -> None:
    expected_shape: tuple[int, ...] | None = None
    expected_method: str | None = None
    mismatches: list[str] = []

    for method_name, payload in methods.items():
        shape = payload_shape(payload)
        if expected_shape is None:
            expected_shape = shape
            expected_method = method_name
            continue
        if shape != expected_shape:
            mismatches.append(
                f"{method_name}: {shape} differs from {expected_method}: {expected_shape}"
            )

    if mismatches:
        raise ValueError(
            f"Shape mismatch in suite {suite_name!r}. "
            "For fair curves, rerun with the same budget/n seeds/n GP functions. "
            + "; ".join(mismatches)
        )


def source_record(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    return {
        "path": str(path),
        "device": metadata.get("device"),
        "selected_methods": list(metadata.get("selected_methods", [])),
    }


def merge_results(
    inputs: Sequence[Path],
    *,
    on_conflict: str,
    allow_metadata_mismatch: bool,
    allow_shape_mismatch: bool,
    require_same_suites: bool,
) -> dict[str, Any]:
    loaded = [(path, load_result(path)) for path in inputs]
    if not loaded:
        raise ValueError("No input files were provided.")

    first_path, first_result = loaded[0]
    reference_metadata = first_result["metadata"]
    reference_suite_names = set(first_result["suites"].keys())

    merged: dict[str, Any] = {
        "metadata": copy.deepcopy(reference_metadata),
        "suites": {},
    }
    merged["metadata"]["device"] = "merged"
    merged["metadata"]["merged_from"] = []
    merged["metadata"]["selected_methods"] = []

    for path, result in loaded:
        if not allow_metadata_mismatch:
            check_metadata_compatible(reference_metadata, result["metadata"], path=path)
        if require_same_suites:
            check_same_suites(reference_suite_names, set(result["suites"].keys()), path=path)

        merged["metadata"]["merged_from"].append(source_record(path, result))

        for suite_name, suite in result["suites"].items():
            if "methods" not in suite:
                raise KeyError(f"{path} suite {suite_name!r} must contain methods.")
            if "eval_hparams" not in suite and "benchmark" not in suite:
                raise KeyError(
                    f"{path} suite {suite_name!r} must contain eval_hparams or benchmark metadata."
                )

            if suite_name not in merged["suites"]:
                merged["suites"][suite_name] = {
                    "methods": {},
                }
                if "eval_hparams" in suite:
                    merged["suites"][suite_name]["eval_hparams"] = copy.deepcopy(
                        suite["eval_hparams"]
                    )
                if "benchmark" in suite:
                    merged["suites"][suite_name]["benchmark"] = copy.deepcopy(suite["benchmark"])
            else:
                check_suite_metadata_compatible(
                    merged["suites"][suite_name],
                    suite,
                    suite_name=suite_name,
                    path=path,
                )

            merged_methods = merged["suites"][suite_name]["methods"]
            for method_name, payload in suite["methods"].items():
                if method_name in merged_methods:
                    if on_conflict == "error":
                        raise ValueError(
                            f"Duplicate method {method_name!r} in suite {suite_name!r} from {path}. "
                            "Use --on-conflict replace or skip if this is intentional."
                        )
                    if on_conflict == "skip":
                        continue

                merged_methods[method_name] = copy.deepcopy(payload)

    selected_methods: list[str] = []
    for suite in merged["suites"].values():
        for method_name in suite["methods"]:
            if method_name not in selected_methods:
                selected_methods.append(method_name)
    merged["metadata"]["selected_methods"] = selected_methods

    if not allow_shape_mismatch:
        for suite_name, suite in merged["suites"].items():
            check_method_shapes(suite_name, suite["methods"])

    return merged


def main() -> None:
    args = parse_args()
    merged = merge_results(
        args.inputs,
        on_conflict=args.on_conflict,
        allow_metadata_mismatch=args.allow_metadata_mismatch,
        allow_shape_mismatch=args.allow_shape_mismatch,
        require_same_suites=args.require_same_suites,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)

    method_names = merged["metadata"].get("selected_methods", [])
    suite_names = list(merged["suites"].keys())
    print(f"[merge] inputs={len(args.inputs)}")
    print(f"[merge] suites={len(suite_names)} methods={len(method_names)}")
    print(f"[merge] methods={json.dumps(method_names, ensure_ascii=False)}")
    print(f"[merge] saved {args.out}")


if __name__ == "__main__":
    main()
