#!/usr/bin/env python3
"""
Run paper-style log10(simple regret) evaluation on GP preference tasks.

The GP benchmark hyperparameters come from one explicit PFN config. Baselines
can run without loading a PFN checkpoint; the optional PFN method is always the
single pair-score PFN agent named "pfn".

The plotted quantity is:

    log10(max_x f(x) - f(x_hat_t))

where x_hat_t is the method recommendation after t pairwise comparisons.
Regret is computed against the noiseless latent GP optimum/reference value.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="fontconfig-")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agents import (
    PairScorePFNAgent,
    QEIAgent,
    QNEIAgent,
    QEUBOAgent,
    QTSAgent,
    RandomAgent,
)
from evaluation.agents.base import PBOAgent
from evaluation.loop import run_bo_loop
from pfns.run_training_cli import load_config_from_python
from evaluation.oracle import EvalSuiteSpec, sample_gp_function, GaussianPreferenceOracle, GPHyperparameters


MULTIDIM_CHECKPOINT_RE = re.compile(r"^pref_gp_(\d+)d_(.+)$")
BASELINE_METHODS = ("random", "qeubo", "qts", "qei", "qnei")
PFN_METHOD = "pfn"
VALID_METHODS = BASELINE_METHODS + (PFN_METHOD,)


@dataclass(frozen=True)
class PFNSpec:
    checkpoint_path: Path
    config_path: Path
    train_hparams: GPHyperparameters
    prior_class: str
    input_dim: int = 1


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PFN checkpoints and baselines on GP log-regret suites."
    )
    parser.add_argument("--pfn-checkpoint", type=Path, default=None)
    parser.add_argument("--pfn-config", type=Path, default=None)
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/gp_log_regret/results.pt"))
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-gp-functions", type=int, default=5)
    parser.add_argument("--n-bo-seeds", type=int, default=10)
    parser.add_argument("--n-grid", type=int, default=500)
    parser.add_argument("--gp-rff-num-features", type=int, default=4096)
    parser.add_argument("--gp-opt-reference-size", type=int, default=65536)
    parser.add_argument(
        "--gp-opt-reference-seed-offset",
        type=int,
        default=None,
        help="Optional seed offset for continuous GP optimum reference sets.",
    )
    parser.add_argument("--gp-rff-eval-batch-size", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pfn-pair-batch-size", type=int, default=4096)
    parser.add_argument("--qeubo-num-acqf-samples", type=int, default=64)
    parser.add_argument("--qeubo-max-fit-iter", type=int, default=100)
    parser.add_argument("--qeubo-max-fit-attempts", type=int, default=20)
    parser.add_argument(
        "--qeubo-fit-hyperparams",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--qeubo-continuous-num-restarts", type=int, default=20)
    parser.add_argument("--qeubo-continuous-raw-samples", type=int, default=1024)
    parser.add_argument("--qeubo-continuous-maxiter", type=int, default=100)
    parser.add_argument("--gp-seed-offset", type=int, default=0)
    parser.add_argument("--oracle-seed-offset", type=int, default=10_000)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Method names to run, or 'all'. Valid methods: random qeubo qts qei qnei pfn.",
    )
    parser.add_argument("--exclude-methods", nargs="*", default=[])
    parser.add_argument("--save-every-method", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def infer_checkpoint_input_dim(name: str) -> Optional[int]:
    name = Path(str(name)).stem.removeprefix("pfn_")
    match = MULTIDIM_CHECKPOINT_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def load_hparams_from_config(config_path: Path) -> tuple[GPHyperparameters, str]:
    config = load_config_from_python(str(config_path), 0)
    prior = config.priors[0]
    hparams = GPHyperparameters(
        lengthscale=float(getattr(prior, "lengthscale")),
        outputscale=float(getattr(prior, "outputscale")),
        noise_std=float(getattr(prior, "noise_std")),
    )
    return hparams, type(prior).__name__


def requested_methods(args: argparse.Namespace) -> List[str]:
    if args.methods == ["all"]:
        selected = list(BASELINE_METHODS)
        if args.pfn_checkpoint is not None:
            selected.append(PFN_METHOD)
    else:
        requested = list(args.methods)
        unknown = sorted(set(requested) - set(VALID_METHODS))
        if unknown:
            raise ValueError(
                "Unknown methods "
                f"{unknown}. Valid methods are: {', '.join(VALID_METHODS)}."
            )
        selected = [name for name in VALID_METHODS if name in set(requested)]

    excluded = set(args.exclude_methods)
    unknown_excluded = sorted(excluded - set(VALID_METHODS))
    if unknown_excluded:
        raise ValueError(
            "Unknown excluded methods "
            f"{unknown_excluded}. Valid methods are: {', '.join(VALID_METHODS)}."
        )
    return [name for name in selected if name not in excluded]


def validate_pfn_args(args: argparse.Namespace, selected_methods: Sequence[str]) -> None:
    if args.input_dim < 1:
        raise ValueError(f"--input-dim must be positive, got {args.input_dim}.")
    if args.pfn_config is None:
        raise ValueError("--pfn-config is required; it defines the GP eval hyperparameters.")
    if PFN_METHOD in selected_methods and args.pfn_checkpoint is None:
        raise ValueError("--pfn-checkpoint is required when --methods includes pfn.")
    if args.pfn_checkpoint is None:
        return

    checkpoint_dim = infer_checkpoint_input_dim(args.pfn_checkpoint)
    if checkpoint_dim is not None and checkpoint_dim != int(args.input_dim):
        raise ValueError(
            f"Checkpoint name implies input_dim={checkpoint_dim}, "
            f"but --input-dim={args.input_dim}."
        )


def make_pfn_spec(
    args: argparse.Namespace,
    *,
    hparams: GPHyperparameters,
    prior_class: str,
) -> Optional[PFNSpec]:
    if args.pfn_checkpoint is None:
        return None
    return PFNSpec(
        checkpoint_path=args.pfn_checkpoint,
        config_path=args.pfn_config,
        train_hparams=hparams,
        prior_class=prior_class,
        input_dim=int(args.input_dim),
    )


def _replace_config_obj(obj, **updates):
    updates = {key: value for key, value in updates.items() if hasattr(obj, key)}
    if not updates:
        return obj
    if is_dataclass(obj):
        return replace(obj, **updates)
    for key, value in updates.items():
        setattr(obj, key, value)
    return obj


def apply_checkpoint_shape_overrides(config, spec: PFNSpec):
    if spec.input_dim <= 1:
        return config
    num_features = 2 * spec.input_dim
    model = _replace_config_obj(config.model, features_per_group=num_features)
    batch_shape_sampler = _replace_config_obj(
        config.batch_shape_sampler,
        min_num_features=num_features,
        max_num_features=num_features,
    )
    if is_dataclass(config):
        return replace(config, model=model, batch_shape_sampler=batch_shape_sampler)
    config.model = model
    config.batch_shape_sampler = batch_shape_sampler
    return config


def load_pfn_model(spec: PFNSpec, device: str):
    config = load_config_from_python(str(spec.config_path), 0)
    config = apply_checkpoint_shape_overrides(config, spec)
    model = config.model.create_model().to(device)
    model.eval()
    state = torch.load(spec.checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model


def build_eval_suite_specs(
    args: argparse.Namespace,
    eval_hparams: Sequence[GPHyperparameters],
) -> List[EvalSuiteSpec]:
    suites: List[EvalSuiteSpec] = []
    input_dim = int(args.input_dim)

    for hparams in eval_hparams:
        gp_functions = []
        for gp_idx in range(args.n_gp_functions):
            opt_reference_seed = None
            if args.gp_opt_reference_seed_offset is not None:
                opt_reference_seed = args.gp_opt_reference_seed_offset + gp_idx

            gp_functions.append(
                sample_gp_function(
                    input_dim=input_dim,
                    hparams=hparams,
                    seed=args.gp_seed_offset + gp_idx,
                    rff_num_features=args.gp_rff_num_features,
                    opt_reference_size=args.gp_opt_reference_size,
                    opt_reference_seed=opt_reference_seed,
                    rff_eval_batch_size=args.gp_rff_eval_batch_size,
                )
            )
        suites.append(
            EvalSuiteSpec(
                name=hparams.signature if input_dim == 1 else f"{hparams.signature}_d{input_dim}",
                functions=tuple(gp_functions),
                oracle_noise_std=hparams.noise_std,
                baseline_hparams=hparams,
                eval_hparams=hparams,
                benchmark={
                    "kind": "gp",
                    "input_dim": int(input_dim),
                    "support": "continuous_rff",
                    "gp_seed_offset": int(args.gp_seed_offset),
                    "rff_num_features": int(args.gp_rff_num_features),
                    "opt_reference_size": int(args.gp_opt_reference_size),
                    "opt_reference_seed_offset": args.gp_opt_reference_seed_offset,
                    "rff_eval_batch_size": int(args.gp_rff_eval_batch_size),
                },
            )
        )

    return suites


def make_agent_for_method(
    method_name: str,
    *,
    hparams: GPHyperparameters,
    args: argparse.Namespace,
    pfn_spec: Optional[PFNSpec],
    pfn_model,
    bo_seed: int,
) -> PBOAgent:
    if method_name == "random":
        return RandomAgent(seed=bo_seed)
    botorch_baselines = {
        "qeubo": QEUBOAgent,
        "qts": QTSAgent,
        "qei": QEIAgent,
        "qnei": QNEIAgent,
    }
    if method_name in botorch_baselines:
        return botorch_baselines[method_name](
            fit_hyperparams=args.qeubo_fit_hyperparams,
            max_fit_iter=args.qeubo_max_fit_iter,
            max_fit_attempts=args.qeubo_max_fit_attempts,
            gp_lengthscale=hparams.lengthscale,
            gp_outputscale=hparams.outputscale,
            num_acqf_samples=args.qeubo_num_acqf_samples,
            continuous_num_restarts=args.qeubo_continuous_num_restarts,
            continuous_raw_samples=args.qeubo_continuous_raw_samples,
            continuous_maxiter=args.qeubo_continuous_maxiter,
        )
    if method_name == PFN_METHOD:
        if pfn_spec is None or pfn_model is None:
            raise RuntimeError("PFN method was selected but no PFN checkpoint was loaded.")
        return PairScorePFNAgent(
            pfn_model,
            device=args.device,
            pair_batch_size=args.pfn_pair_batch_size,
            input_dim=pfn_spec.input_dim,
        )
    raise ValueError(f"Unknown method {method_name!r}.")


def hparams_equal(a: GPHyperparameters, b: GPHyperparameters) -> bool:
    return (
        math.isclose(a.lengthscale, b.lengthscale)
        and math.isclose(a.outputscale, b.outputscale)
        and math.isclose(a.noise_std, b.noise_std)
    )


def method_metadata(
    method_name: str,
    pfn_spec: Optional[PFNSpec],
    eval_hparams: Optional[GPHyperparameters],
    benchmark: Optional[Mapping],
) -> Dict:
    eval_hparams_dict = eval_hparams.as_dict() if eval_hparams is not None else None
    if method_name in BASELINE_METHODS:
        return {
            "method_name": method_name,
            "kind": "baseline",
            "checkpoint": None,
            "config": None,
            "train_hparams": None,
            "eval_hparams": eval_hparams_dict,
            "benchmark": dict(benchmark) if benchmark is not None else None,
            "is_in_domain": None,
            "is_ranking_baseline": False,
        }

    if method_name != PFN_METHOD:
        raise ValueError(f"Unknown method {method_name!r}.")
    if pfn_spec is None:
        raise RuntimeError("PFN metadata requested but no PFN spec was created.")
    spec = pfn_spec
    is_in_domain = False if eval_hparams is None else hparams_equal(spec.train_hparams, eval_hparams)
    return {
        "method_name": method_name,
        "kind": "pair_score_pfn",
        "checkpoint": str(spec.checkpoint_path),
        "config": str(spec.config_path),
        "prior_class": spec.prior_class,
        "input_dim": spec.input_dim,
        "train_hparams": spec.train_hparams.as_dict(),
        "eval_hparams": eval_hparams_dict,
        "benchmark": dict(benchmark) if benchmark is not None else None,
        "is_in_domain": is_in_domain,
        "is_ranking_baseline": False,
    }


def main() -> None:
    args = parse_args()
    torch_device = torch.device(args.device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    selected_methods = requested_methods(args)
    validate_pfn_args(args, selected_methods)
    print(f"selected_methods: {selected_methods}")
    if not selected_methods:
        raise RuntimeError("No methods selected for evaluation.")

    train_hparams, prior_class = load_hparams_from_config(args.pfn_config)
    eval_hparams = [train_hparams]
    pfn_spec = make_pfn_spec(args, hparams=train_hparams, prior_class=prior_class)

    suite_specs = build_eval_suite_specs(args, eval_hparams)
    if not suite_specs:
        raise RuntimeError("No benchmark suites selected for evaluation.")

    print("[setup] selected methods:", ", ".join(selected_methods))
    print("[setup] eval hparams:", ", ".join(h.signature for h in eval_hparams))
    print("[setup] input_dim:", args.input_dim)
    print("[setup] benchmark suites:", ", ".join(suite.name for suite in suite_specs))

    pfn_model = None
    if PFN_METHOD in selected_methods:
        assert pfn_spec is not None
        print(f"[load] pfn <- {pfn_spec.checkpoint_path}")
        pfn_model = load_pfn_model(pfn_spec, args.device)

    result = {
        "metadata": {
            "n_grid": args.n_grid,
            "budget": args.budget,
            "n_init": args.n_init,
            "n_gp_functions": args.n_gp_functions,
            "n_bo_seeds": args.n_bo_seeds,
            "gp_support": "continuous_rff",
            "gp_rff_num_features": args.gp_rff_num_features,
            "gp_opt_reference_size": args.gp_opt_reference_size,
            "gp_opt_reference_seed_offset": args.gp_opt_reference_seed_offset,
            "gp_rff_eval_batch_size": args.gp_rff_eval_batch_size,
            "qeubo_num_acqf_samples": args.qeubo_num_acqf_samples,
            "qeubo_max_fit_iter": args.qeubo_max_fit_iter,
            "qeubo_max_fit_attempts": args.qeubo_max_fit_attempts,
            "qeubo_fit_hyperparams": args.qeubo_fit_hyperparams,
            "qeubo_continuous_num_restarts": args.qeubo_continuous_num_restarts,
            "qeubo_continuous_raw_samples": args.qeubo_continuous_raw_samples,
            "qeubo_continuous_maxiter": args.qeubo_continuous_maxiter,
            "eps": args.eps,
            "device": args.device,
            "input_dim": args.input_dim,
            "pfn_checkpoint": str(args.pfn_checkpoint) if args.pfn_checkpoint is not None else None,
            "pfn_config": str(args.pfn_config),
            "selected_methods": selected_methods,
        },
        "suites": {},
    }

    for suite_spec in suite_specs:
        suite_name = suite_spec.name
        print(f"\n=== suite: {suite_name} ===")
        suite = {"methods": {}}
        if suite_spec.eval_hparams is not None:
            suite["eval_hparams"] = suite_spec.eval_hparams.as_dict()
        if suite_spec.benchmark is not None:
            suite["benchmark"] = suite_spec.benchmark

        for method_name in selected_methods:
            print(f"[suite={suite_name}] method={method_name}")
            n_functions = len(suite_spec.functions)
            simple_regret = torch.empty(n_functions, args.n_bo_seeds, args.budget)
            utility_at_recommendation = torch.empty_like(simple_regret)

            for function_idx, gp_function in enumerate(suite_spec.functions):
                for bo_seed in range(args.n_bo_seeds):
                    seed = args.oracle_seed_offset + function_idx * 100_000 + bo_seed

                    oracle = GaussianPreferenceOracle(
                        gp_function=gp_function,
                        noise_std=suite_spec.oracle_noise_std,
                        seed=seed,
                    )

                    agent = make_agent_for_method(
                        method_name,
                        hparams=suite_spec.baseline_hparams,
                        args=args,
                        pfn_spec=pfn_spec,
                        pfn_model=pfn_model,
                        bo_seed=bo_seed,
                    )

                    run = run_bo_loop(
                        agent=agent,
                        oracle=oracle,
                        budget=args.budget,
                        n_init=args.n_init,
                        seed=bo_seed,
                        n_grid=args.n_grid,
                        verbose=args.verbose,
                    )
                    sr = torch.tensor(run["simple_regret"], dtype=torch.float32)
                    utility = torch.tensor(
                        [oracle.f_at(x_hat) for x_hat in run["recommendations"]],
                        dtype=torch.float32,
                    )
                    simple_regret[function_idx, bo_seed] = sr
                    utility_at_recommendation[function_idx, bo_seed] = utility

                    print(
                        f"  function={function_idx:03d} seed={bo_seed:03d} "
                        f"final_regret={sr[-1].item():.6g}"
                    )

            log10_regret = torch.log10(torch.clamp(simple_regret, min=args.eps))
            suite["methods"][method_name] = {
                "simple_regret": simple_regret,
                "log10_regret": log10_regret,
                "utility_at_recommendation": utility_at_recommendation,
                "metadata": method_metadata(
                    method_name,
                    pfn_spec,
                    suite_spec.eval_hparams,
                    suite_spec.benchmark,
                ),
            }

            if args.save_every_method:
                result["suites"][suite_name] = suite
                atomic_torch_save(result, args.out)
                print(f"[save] {args.out}")

        result["suites"][suite_name] = suite
        atomic_torch_save(result, args.out)
        print(f"[save] {args.out}")

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
