from pfns.train import MainConfig
from pfns.optimizer import OptimizerConfig
from pfns.batch_shape_sampler import BatchShapeSamplerConfig
from pfns.model.transformer_config import TransformerConfig
from pfns.model.bar_distribution import BarDistributionConfig
from pfns.priors.pref.pref_gp_1d_regret import PrefGP1DRegretPriorConfig

# Continuous regression target distribution.
# Wide borders + full_support=True works well for unbounded GP targets.
borders = [- 1 * 10 ** (1 - i/25) for i in range(101)] + [0.001]
print(borders)

criterion = BarDistributionConfig(
    borders=borders,
    full_support=False,
)

model = TransformerConfig(
    criterion=criterion,
    emsize=128,
    nhid=256,
    nlayers=6,
    nhead=4,
    features_per_group=2,  # as we do not have attention between features, we should put them all in one group
    attention_between_features=False,
)

optimizer = OptimizerConfig(
    optimizer="adamw",
    lr=3e-4,
)

batch_shape_sampler = BatchShapeSamplerConfig(
    batch_size=100,
    base_for_exp_decay=0.95,
    min_single_eval_pos=0,
    max_single_eval_pos=99,  # restrict to small context sizes
    max_seq_len=100,
    min_num_features=2,
    max_num_features=2,   # 1D GP, but x in pair-comparisons get concatenated 
)

prior = PrefGP1DRegretPriorConfig(
    lengthscale=0.2,
    outputscale=1.0,
    noise_std=0.05,
)

config = MainConfig(
    priors=[prior],
    optimizer=optimizer,
    model=model,
    batch_shape_sampler=batch_shape_sampler,
    epochs=1000,
    steps_per_epoch=100,
    aggregate_k_gradients=1,
    n_targets_per_input=1,
    train_mixed_precision=False,
    scheduler="cosine_decay",
    warmup_epochs=100,
    train_state_dict_save_path="/work/dlclarge2/adriaens-pref-pfn/checkpoints/pfn_pref_gp_1d_regret_10M.pt",
    train_state_dict_load_path="/work/dlclarge2/adriaens-pref-pfn/checkpoints/pfn_pref_gp_1d_regret_10M.pt",
    validation_period=5,
    verbose=True,
    progress_bar=False,
    tensorboard_path="/work/dlclarge2/adriaens-pref-pfn/tb/pref_gp_1d_regret_10M",
    num_workers=0,
)