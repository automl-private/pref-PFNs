from dataclasses import dataclass, fields
from pathlib import Path
import random

import torch

from pfns.priors.prior import Batch, PriorConfig


def _load_payload(path: Path, *, mmap: bool = False) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=mmap,
        )
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
        data_path = Path(self.data_dir)
        stacked_payload = None
        if data_path.is_file():
            stacked_payload = _load_payload(data_path, mmap=True)
            if stacked_payload.get("format") != "stacked_pregenerated_batches_v1":
                raise ValueError(f"{data_path} is not a stacked pregenerated batch file.")
            num_batches = int(stacked_payload["num_batches"])
            if stacked_payload["x"].shape[0] != num_batches:
                raise ValueError(
                    f"{data_path} contains {stacked_payload['x'].shape[0]} batches, "
                    f"but num_batches={num_batches}."
                )
            batch_paths = None
        else:
            batch_paths = sorted(data_path.glob("batch_*.pt"))
            if not batch_paths:
                raise FileNotFoundError(
                    f"No pregenerated batch_*.pt files found in {data_path}."
                )
            num_batches = len(batch_paths)

        order = list(range(num_batches))
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
                    raise StopIteration(
                        f"Pregenerated batches in {data_path} are exhausted."
                    )
                state["index"] = 0
                if self.shuffle:
                    rng.shuffle(order)

            batch_index = order[state["index"]]
            state["index"] += 1

            if stacked_payload is None:
                source = batch_paths[batch_index]
                payload = _load_payload(source)
                batch_kwargs = {
                    name: payload.get(name)
                    for name in batch_field_names
                }
            else:
                source = data_path
                batch_kwargs = {}
                for name in batch_field_names:
                    value = stacked_payload.get(name)
                    if torch.is_tensor(value):
                        value = value[batch_index]
                    batch_kwargs[name] = value
                batch_kwargs["single_eval_pos"] = int(
                    stacked_payload["single_eval_pos"][batch_index]
                )
            batch = Batch(**batch_kwargs)

            if self.strict_shapes:
                expected_x_shape = (batch_size, seq_len, num_features)
                if tuple(batch.x.shape) != expected_x_shape:
                    raise ValueError(
                        f"{source} has x.shape={tuple(batch.x.shape)}, "
                        f"expected {expected_x_shape}."
                    )

            return batch

        return get_batch
