#!/usr/bin/env python3
"""Plot GP paper-style log10(simple regret) results."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="fontconfig-")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GP log10(simple regret) curves.")
    parser.add_argument("--results", type=Path, default=Path("results/gp_log_regret/results.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/gp_log_regret"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/gp_log_regret/summary.csv"))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title-prefix", type=str, default="GP preference BO")
    return parser.parse_args()


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_")


def mean_stderr(log10_regret: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = log10_regret.detach().cpu().float().reshape(-1, log10_regret.shape[-1])
    counts = torch.isfinite(values).sum(dim=0)
    mean = torch.nanmean(values, dim=0)
    stderr = torch.zeros_like(mean)
    for idx in range(values.shape[-1]):
        finite = values[:, idx][torch.isfinite(values[:, idx])]
        if finite.numel() > 1:
            stderr[idx] = finite.std(unbiased=True) / math.sqrt(finite.numel())
    return mean.numpy(), stderr.numpy(), counts.numpy()


def display_label(method_name: str, metadata: Mapping) -> str:
    if metadata.get("kind") == "baseline":
        return method_name
    if metadata.get("is_ranking_baseline"):
        return f"{method_name} (ranking)"
    if metadata.get("is_in_domain") is False:
        return f"{method_name} (OOD)"
    return method_name


def suite_subtitle(suite: Mapping, *, compact: bool = False) -> str:
    benchmark = suite.get("benchmark")
    if benchmark:
        if compact:
            return (
                f"{benchmark.get('name')} | {benchmark.get('normalization')} | "
                f"noise={benchmark.get('noise_std')}"
            )
        return (
            f"benchmark={benchmark.get('name')}, "
            f"normalization={benchmark.get('normalization')}, "
            f"noise_std={benchmark.get('noise_std')}"
        )

    h = suite.get("eval_hparams", {})
    if compact:
        return f"l={h.get('lengthscale')}, os={h.get('outputscale')}, noise={h.get('noise_std')}"
    return (
        f"lengthscale={h.get('lengthscale')}, "
        f"outputscale={h.get('outputscale')}, noise_std={h.get('noise_std')}"
    )


def plot_suite(
    suite_name: str,
    suite: Mapping,
    *,
    out_path: Path,
    methods: Optional[Sequence[str]],
    dpi: int,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    plotted = 0
    for method_name, payload in suite["methods"].items():
        if methods is not None and method_name not in methods:
            continue
        y, err, counts = mean_stderr(payload["log10_regret"])
        x = np.arange(1, len(y) + 1)
        metadata = payload.get("metadata", {})
        linestyle = "--" if metadata.get("is_in_domain") is False else "-"
        line = ax.plot(
            x,
            y,
            label=display_label(method_name, metadata),
            linewidth=2.0,
            linestyle=linestyle,
        )[0]
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "no selected methods", transform=ax.transAxes, ha="center")

    ax.set_title(f"{title_prefix}\n{suite_subtitle(suite)}")
    ax.set_xlabel("# preference comparisons")
    ax.set_ylabel("log10(simple regret)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_all_suites(
    results: Mapping,
    *,
    out_path: Path,
    methods: Optional[Sequence[str]],
    dpi: int,
    title_prefix: str,
) -> None:
    suites = results["suites"]
    if len(suites) <= 1:
        return

    n = len(suites)
    cols = min(2, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9.0 * cols, 5.2 * rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (suite_name, suite) in zip(axes_flat, suites.items()):
        for method_name, payload in suite["methods"].items():
            if methods is not None and method_name not in methods:
                continue
            y, err, counts = mean_stderr(payload["log10_regret"])
            x = np.arange(1, len(y) + 1)
            metadata = payload.get("metadata", {})
            linestyle = "--" if metadata.get("is_in_domain") is False else "-"
            line = ax.plot(
                x,
                y,
                label=display_label(method_name, metadata),
                linewidth=1.8,
                linestyle=linestyle,
            )[0]
            ax.fill_between(
                x,
                y - err,
                y + err,
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(suite_subtitle(suite, compact=True))
        ax.set_xlabel("# comparisons")
        ax.set_ylabel("log10(simple regret)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    for ax in axes_flat[len(suites) :]:
        ax.axis("off")

    fig.suptitle(title_prefix)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    results: Mapping,
    *,
    summary_out: Path,
    methods: Optional[Sequence[str]],
) -> None:
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "suite",
                "eval_lengthscale",
                "eval_outputscale",
                "eval_noise_std",
                "benchmark_kind",
                "benchmark_name",
                "benchmark_normalization",
                "benchmark_noise_std",
                "method",
                "kind",
                "checkpoint",
                "config",
                "train_lengthscale",
                "train_outputscale",
                "train_noise_std",
                "is_in_domain",
                "is_ranking_baseline",
                "step",
                "mean_log10_regret",
                "stderr_log10_regret",
                "n",
            ]
        )
        for suite_name, suite in results["suites"].items():
            eval_h = suite.get("eval_hparams", {})
            benchmark = suite.get("benchmark") or {}
            for method_name, payload in suite["methods"].items():
                if methods is not None and method_name not in methods:
                    continue
                metadata = payload.get("metadata", {})
                train_h = metadata.get("train_hparams") or {}
                y, err, counts = mean_stderr(payload["log10_regret"])
                for step, mean, stderr, count in zip(
                    range(1, len(y) + 1),
                    y,
                    err,
                    counts,
                ):
                    writer.writerow(
                        [
                            suite_name,
                            eval_h.get("lengthscale"),
                            eval_h.get("outputscale"),
                            eval_h.get("noise_std"),
                            benchmark.get("kind"),
                            benchmark.get("name"),
                            benchmark.get("normalization"),
                            benchmark.get("noise_std"),
                            method_name,
                            metadata.get("kind"),
                            metadata.get("checkpoint"),
                            metadata.get("config"),
                            train_h.get("lengthscale"),
                            train_h.get("outputscale"),
                            train_h.get("noise_std"),
                            metadata.get("is_in_domain"),
                            metadata.get("is_ranking_baseline"),
                            step,
                            float(mean),
                            float(stderr),
                            int(count),
                        ]
                    )


def main() -> None:
    args = parse_args()
    results = torch.load(args.results, map_location="cpu")
    methods = args.methods if args.methods else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suite_name, suite in results["suites"].items():
        out_path = args.out_dir / f"{slugify(suite_name)}.png"
        plot_suite(
            suite_name,
            suite,
            out_path=out_path,
            methods=methods,
            dpi=args.dpi,
            title_prefix=args.title_prefix,
        )
        print(f"[plot] saved {out_path}")

    all_path = args.out_dir / "all_suites.png"
    plot_all_suites(
        results,
        out_path=all_path,
        methods=methods,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
    )
    if len(results["suites"]) > 1:
        print(f"[plot] saved {all_path}")

    write_summary(results, summary_out=args.summary_out, methods=methods)
    print(f"[summary] saved {args.summary_out}")


if __name__ == "__main__":
    main()
