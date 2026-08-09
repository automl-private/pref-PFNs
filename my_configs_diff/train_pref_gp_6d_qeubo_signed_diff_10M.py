from pfns.train import MainConfig
from pfns.optimizer import OptimizerConfig
from pfns.batch_shape_sampler import BatchShapeSamplerConfig
from pfns.model.transformer_config import TransformerConfig
from pfns.model.bar_distribution import BarDistributionConfig
from pfns.priors.pref.pref_gp_qeubo_signed_diff import PrefGPqEUBOSignedDiffPriorConfig

gp_dim = 6
num_features = 3 * gp_dim

criterion = BarDistributionConfig(
    borders=[-5.0 + 0.05 * i for i in range(201)],
    full_support=True,
)

model = TransformerConfig(
    criterion=criterion,
    emsize=128,
    nhid=256,
    nlayers=6,
    nhead=4,
    features_per_group=num_features,
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
    max_single_eval_pos=99,
    max_seq_len=100,
    min_num_features=num_features,
    max_num_features=num_features,
)

prior = PrefGPqEUBOSignedDiffPriorConfig(
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
    train_state_dict_save_path=f"checkpoints_diff/pfn_pref_gp_{gp_dim}d_qeubo_signed_diff_10M.pt",
    train_state_dict_load_path=f"checkpoints_diff/pfn_pref_gp_{gp_dim}d_qeubo_signed_diff_10M.pt",
    validation_period=5,
    verbose=True,
    progress_bar=False,
    tensorboard_path=f"tb_diff/pref_gp_{gp_dim}d_qeubo_signed_diff_10M",
    num_workers=0,
)
