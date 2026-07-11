from dataclasses import dataclass, fields
from pathlib import Path
import random

import torch

from pfns.priors.prior import Batch, PriorConfig


def _load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class PregeneratedBatchPriorConfig(PriorConfig):
    data_dir: str
    shuffle: bool = True
    seed: int = 0
    cycle: bool = True
    strict_shapes: bool = True

    def create_get_batch_method(self):
        data_dir = Path(self.data_dir)
        batch_paths = sorted(data_dir.glob("batch_*.pt"))
        if not batch_paths:
            raise FileNotFoundError(f"No pregenerated batch_*.pt files found in {data_dir}.")

        order = list(range(len(batch_paths)))
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(order)

        state = {"index": 0}
        batch_field_names = {field.name for field in fields(Batch)}

        def get_batch(
            batch_size=2,
            seq_len=100,
            num_features=1,
            device="cpu",
            single_eval_pos=None,
            **kwargs,
        ):
            if state["index"] >= len(order):
                if not self.cycle:
                    raise StopIteration(f"Pregenerated batches in {data_dir} are exhausted.")
                state["index"] = 0
                if self.shuffle:
                    rng.shuffle(order)

            path = batch_paths[order[state["index"]]]
            state["index"] += 1

            payload = _load_payload(path)
            batch_kwargs = {
                name: payload.get(name)
                for name in batch_field_names
            }
            batch = Batch(**batch_kwargs)

            if self.strict_shapes:
                expected_x_shape = (batch_size, seq_len, num_features)
                if tuple(batch.x.shape) != expected_x_shape:
                    raise ValueError(
                        f"{path} has x.shape={tuple(batch.x.shape)}, expected {expected_x_shape}."
                    )

            return batch

        return get_batch
