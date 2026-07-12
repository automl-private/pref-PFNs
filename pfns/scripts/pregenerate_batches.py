#!/usr/bin/env python3
import argparse
from dataclasses import fields
from pathlib import Path

import torch
from tqdm import tqdm

from pfns.priors.prior import Batch
from pfns.run_training_cli import load_config_from_python


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _batch_to_payload(batch: Batch) -> dict:
    return {
        field.name: _to_cpu(getattr(batch, field.name))
        for field in fields(Batch)
    }


def _merge_payloads(payloads: list[dict]) -> dict:
    if len(payloads) == 1:
        return payloads[0]

    merged = {}
    for field in fields(Batch):
        values = [payload[field.name] for payload in payloads]
        first = values[0]
        merged[field.name] = (
            torch.cat(values, dim=0)
            if torch.is_tensor(first) and first.ndim > 0
            else first
        )
    return merged


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pregenerate PFN training batches from an existing config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_file", type=str)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--sample-chunk-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_chunk_size < 1:
        raise ValueError("--sample-chunk-size must be positive.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_batches = list(out_dir.glob("batch_*.pt"))
    if existing_batches and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} already contains batch_*.pt files. Use --overwrite to replace them."
        )

    if args.overwrite:
        for path in existing_batches:
            path.unlink()

    config = load_config_from_python(args.config_file, args.config_index)
    get_batch = config.priors[0].create_get_batch_method()
    steps_per_epoch = config.steps_per_epoch
    num_batches = args.num_batches or (config.epochs * steps_per_epoch)

    metadata = {
        "config_file": args.config_file,
        "config_index": args.config_index,
        "num_batches": num_batches,
        "steps_per_epoch": steps_per_epoch,
        "device_used_for_generation": args.device,
        "sample_chunk_size": args.sample_chunk_size,
        "batch_shape_sampler": config.batch_shape_sampler,
    }
    torch.save(metadata, out_dir / "metadata.pt")

    for batch_idx in range(num_batches):
        epoch = batch_idx // steps_per_epoch + 1
        step = batch_idx % steps_per_epoch
        batch_shape = config.batch_shape_sampler.sample_batch_shape(epoch=epoch, step=step)
        kwargs = batch_shape.as_get_batch_kwargs()
        kwargs["device"] = args.device

        payloads = []
        with tqdm(
            total=batch_shape.batch_size,
            desc=f"Batch {batch_idx + 1}/{num_batches}",
            unit="sample",
            leave=False,
        ) as progress:
            for start in range(0, batch_shape.batch_size, args.sample_chunk_size):
                chunk_kwargs = dict(kwargs)
                chunk_kwargs["batch_size"] = min(
                    args.sample_chunk_size,
                    batch_shape.batch_size - start,
                )
                batch = get_batch(**chunk_kwargs)
                if batch.single_eval_pos is None:
                    batch.single_eval_pos = batch_shape.single_eval_pos
                payloads.append(_batch_to_payload(batch))
                progress.update(chunk_kwargs["batch_size"])

        payload = _merge_payloads(payloads)
        torch.save(payload, out_dir / f"batch_{batch_idx:08d}.pt")

        if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == num_batches:
            print(f"Saved {batch_idx + 1}/{num_batches} batches to {out_dir}")


if __name__ == "__main__":
    main()
