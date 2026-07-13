from dataclasses import dataclass

from pfns.priors.pref.pref_gp_1d_qeubo_pool import (
    PrefGP1DqEUBOPoolPriorConfig,
    get_batch,
    make_gp_prior,
    sample_gp_batch,
)


@dataclass(frozen=True)
class PrefGP4DqEUBOPoolPriorConfig(PrefGP1DqEUBOPoolPriorConfig):
    pass
