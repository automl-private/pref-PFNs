#!/usr/bin/env python3
import argparse
import os
from dataclasses import fields
from pathlib import Path

import torch
from tqdm import tqdm

from pfns.priors.prior import Batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge batch_*.pt files into one memory-mapped training file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_batch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_path = args.output.resolve()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    batch_paths = sorted(source_dir.glob("batch_*.pt"))
    if not batch_paths:
        raise FileNotFoundError(f"No batch_*.pt files found in {source_dir}.")
    if output_path.parent == source_dir and output_path.name.startswith("batch_"):
        raise ValueError("The output inside source_dir must not start with 'batch_'.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite.")
    if temporary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{temporary_path} already exists. Use --overwrite.")

    first = load_batch(batch_paths[0])
    field_names = [field.name for field in fields(Batch)]
    if first.get("single_eval_pos") is None:
        raise ValueError(f"{batch_paths[0]} has no single_eval_pos.")

    merged = {}
    for name in field_names:
        value = first.get(name)
        if name == "single_eval_pos":
            merged[name] = torch.empty(len(batch_paths), dtype=torch.long)
        elif torch.is_tensor(value):
            merged[name] = torch.empty(
                (len(batch_paths), *value.shape),
                dtype=value.dtype,
            )
        else:
            merged[name] = value

    for batch_index, path in enumerate(
        tqdm(batch_paths, desc="Merging batches", unit="batch")
    ):
        payload = load_batch(path)
        for name in field_names:
            value = payload.get(name)
            target = merged[name]
            if name == "single_eval_pos":
                if value is None:
                    raise ValueError(f"{path} has no single_eval_pos.")
                target[batch_index] = int(value)
            elif torch.is_tensor(target):
                if not torch.is_tensor(value):
                    raise ValueError(f"{path}: field {name!r} is not a tensor.")
                if value.shape != target.shape[1:] or value.dtype != target.dtype:
                    raise ValueError(
                        f"{path}: field {name!r} has shape={tuple(value.shape)} "
                        f"and dtype={value.dtype}, expected shape={tuple(target.shape[1:])} "
                        f"and dtype={target.dtype}."
                    )
                target[batch_index].copy_(value)
            elif value != target:
                raise ValueError(f"{path}: field {name!r} differs between batches.")

    merged["format"] = "stacked_pregenerated_batches_v1"
    merged["num_batches"] = len(batch_paths)
    merged["source_dir"] = str(source_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_path.exists():
        temporary_path.unlink()
    torch.save(merged, temporary_path)
    os.replace(temporary_path, output_path)
    print(f"Saved {len(batch_paths)} batches to {output_path}")


if __name__ == "__main__":
    main()
