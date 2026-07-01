#!/usr/bin/env python3
"""Plot GP paper-style simple-regret results."""

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


METHOD_COLORS = {
    "random": "#4C78A8",
    "qeubo": "#F58518",
    "qts": "#54A24B",
    "qei": "#E45756",
    "qnei": "#B279A2",
    "pfn": "#000000",
    "pfn_botorch": "#9D755D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot log10(mean simple regret) curves.")
    parser.add_argument("--results", type=Path, default=Path("results/gp_log_regret/results.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/gp_log_regret"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/gp_log_regret/summary.csv"))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title-prefix", type=str, default="GP preference BO")
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_")


def average_regret_stderr(simple_regret: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = simple_regret.detach().cpu().float().reshape(-1, simple_regret.shape[-1])
    counts = torch.isfinite(values).sum(dim=0)
    mean = torch.nanmean(values, dim=0)
    stderr = torch.zeros_like(mean)
    for idx in range(values.shape[-1]):
        finite = values[:, idx][torch.isfinite(values[:, idx])]
        if finite.numel() > 1:
            stderr[idx] = finite.std(unbiased=True) / math.sqrt(finite.numel())
    return mean.numpy(), stderr.numpy(), counts.numpy()


def plot_series(
    payload: Mapping,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    average_regret, err, counts = average_regret_stderr(payload["simple_regret"])
    y = np.log10(np.maximum(average_regret, eps))
    lower = np.log10(np.maximum(average_regret - err, eps))
    upper = np.log10(np.maximum(average_regret + err, eps))
    return y, lower, upper, counts


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
        if benchmark.get("kind") == "gp":
            h = suite.get("eval_hparams", {})
            input_dim = benchmark.get("input_dim")
            support = benchmark.get("support")
            rff_num_features = benchmark.get("rff_num_features")
            if compact:
                return (
                    f"GP prior | d={input_dim} | {support} | "
                    f"l={h.get('lengthscale')}, os={h.get('outputscale')}, "
                    f"noise={h.get('noise_std')}"
                )
            return (
                f"GP prior, input_dim={input_dim}, support={support}, "
                f"rff_num_features={rff_num_features}, "
                f"lengthscale={h.get('lengthscale')}, "
                f"outputscale={h.get('outputscale')}, noise_std={h.get('noise_std')}"
            )
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
    eps: float,
    method_colors: Mapping[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    plotted = 0
    for method_name, payload in suite["methods"].items():
        if methods is not None and method_name not in methods:
            continue
        y, lower, upper, counts = plot_series(payload, eps=eps)
        x = np.arange(1, len(y) + 1)
        metadata = payload.get("metadata", {})
        linestyle = "--" if metadata.get("is_in_domain") is False else "-"
        line = ax.plot(
            x,
            y,
            label=display_label(method_name, metadata),
            color=method_colors.get(method_name),
            linewidth=2.0,
            linestyle=linestyle,
        )[0]
        ax.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "no selected methods", transform=ax.transAxes, ha="center")

    ax.set_title(f"{title_prefix}\n{suite_subtitle(suite)}")
    ax.set_xlabel("# preference comparisons")
    ax.set_ylabel("log10(mean simple regret)")
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
    eps: float,
    method_colors: Mapping[str, str],
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
            y, lower, upper, counts = plot_series(payload, eps=eps)
            x = np.arange(1, len(y) + 1)
            metadata = payload.get("metadata", {})
            linestyle = "--" if metadata.get("is_in_domain") is False else "-"
            line = ax.plot(
                x,
                y,
                label=display_label(method_name, metadata),
                color=method_colors.get(method_name),
                linewidth=1.8,
                linestyle=linestyle,
            )[0]
            ax.fill_between(
                x,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(suite_subtitle(suite, compact=True))
        ax.set_xlabel("# comparisons")
        ax.set_ylabel("log10(mean simple regret)")
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
    eps: float,
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
                "benchmark_input_dim",
                "benchmark_support",
                "benchmark_rff_num_features",
                "benchmark_opt_reference_size",
                "method",
                "kind",
                "checkpoint",
                "config",
                "config_template_name",
                "method_input_dim",
                "train_lengthscale",
                "train_outputscale",
                "train_noise_std",
                "is_in_domain",
                "is_ranking_baseline",
                "step",
                "average_regret",
                "log10_average_regret",
                "stderr_regret",
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
                average_regret, regret_err, regret_counts = average_regret_stderr(payload["simple_regret"])
                log10_average_regret = np.log10(np.maximum(average_regret, eps))
                for step, avg_regret, log10_avg_regret, avg_regret_stderr, count in zip(
                    range(1, len(average_regret) + 1),
                    average_regret,
                    log10_average_regret,
                    regret_err,
                    regret_counts,
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
                            benchmark.get("input_dim"),
                            benchmark.get("support"),
                            benchmark.get("rff_num_features"),
                            benchmark.get("opt_reference_size"),
                            method_name,
                            metadata.get("kind"),
                            metadata.get("checkpoint"),
                            metadata.get("config"),
                            metadata.get("config_template_name"),
                            metadata.get("input_dim"),
                            train_h.get("lengthscale"),
                            train_h.get("outputscale"),
                            train_h.get("noise_std"),
                            metadata.get("is_in_domain"),
                            metadata.get("is_ranking_baseline"),
                            step,
                            float(avg_regret),
                            float(log10_avg_regret),
                            float(avg_regret_stderr),
                            int(count),
                        ]
                    )


def main() -> None:
    args = parse_args()
    results = torch.load(args.results, map_location="cpu")
    methods = args.methods if args.methods else None
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {}
    for suite in results["suites"].values():
        for method_name in suite["methods"]:
            if methods is not None and method_name not in methods:
                continue
            if method_name not in method_colors:
                fallback = fallback_colors[len(method_colors) % len(fallback_colors)]
                method_colors[method_name] = METHOD_COLORS.get(method_name, fallback)

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
            eps=args.eps,
            method_colors=method_colors,
        )
        print(f"[plot] saved {out_path}")

    all_path = args.out_dir / "all_suites.png"
    plot_all_suites(
        results,
        out_path=all_path,
        methods=methods,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
        eps=args.eps,
        method_colors=method_colors,
    )
    if len(results["suites"]) > 1:
        print(f"[plot] saved {all_path}")

    write_summary(results, summary_out=args.summary_out, methods=methods, eps=args.eps)
    print(f"[summary] saved {args.summary_out}")


if __name__ == "__main__":
    main()
