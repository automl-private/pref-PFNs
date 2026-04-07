# We create a batch shape sampler, that determines the parts of the batch that determine how compute intensive it is.
# Putting this into one module is important to allow multi-gpu training, as we want to seed the sampling of this shape the same across workers
# s.t. they all take the same amount of time.

# It includes batch_size, seq_len, num_features, and single_eval_pos.

import random
import math
from dataclasses import dataclass
from typing import Optional

from pfns.base_config import BaseConfig


@dataclass
class BatchShape:
    batch_size: int
    seq_len: int
    num_features: int
    single_eval_pos: Optional[int] = None

    def as_get_batch_kwargs(self):
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_features": self.num_features,
            "single_eval_pos": self.single_eval_pos,
        }


def weighted_choice(min_val, max_val, B):
    assert 0 < B < 1, "B must be between 0 and 1"
    n = max_val - min_val + 1

    # Normalize factor (sum of geometric series)
    total = (1 - B**n) / (1 - B)

    # Sample uniform value
    r = random.random() * total

    # Invert cumulative distribution
    k = int(math.log(1 - r * (1 - B)) / math.log(B))

    return min_val + k


@dataclass(frozen=True)
class BatchShapeSamplerConfig(BaseConfig):
    batch_size: int = 32
    min_single_eval_pos: int = 0
    max_single_eval_pos: int = None
    base_for_exp_decay: float = 1.0 
    max_seq_len: int = 1000
    min_num_features: int = 1
    max_num_features: int = 16
    fixed_num_test_instances: Optional[int] = None

    seed: int = 42

    def sample_batch_shape(self, epoch: int, step: int) -> BatchShape:
        assert self.max_single_eval_pos is None or self.fixed_num_test_instances is None
        
        # Create deterministic seed based on epoch and step
        seed = self.seed + epoch * 10000 + step
        rng = random.Random(seed)

        # it seems to be beneficial to oversample small numbers of features
        num_features = rng.randint(self.min_num_features, self.max_num_features)

        max_single_eval_pos = self.max_seq_len - (self.fixed_num_test_instances if self.fixed_num_test_instances is not None else 1) if self.max_single_eval_pos is None else self.max_single_eval_pos

        if self.base_for_exp_decay < 1:
            # exp decay
            single_eval_pos = weighted_choice(
                self.min_single_eval_pos, 
                max_single_eval_pos,
                self.base_for_exp_decay)
        else:
            # uniform
            single_eval_pos = rng.randint(
                self.min_single_eval_pos,
                max_single_eval_pos,
            )

        seq_len = self.max_seq_len
        if self.fixed_num_test_instances is not None:
            seq_len = self.fixed_num_test_instances + single_eval_pos

        # future todo: adapt batch_size and num_features based on seq_len -> shrinking them for large seq_lens
        return BatchShape(
            batch_size=self.batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
        )
