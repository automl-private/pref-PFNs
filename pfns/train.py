from __future__ import annotations

import importlib
import os
import time
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import base_config, provenance, utils
from .batch_shape_sampler import BatchShapeSamplerConfig
from .model.transformer_config import TransformerConfig
from .optimizer import OptimizerConfig
from .priors import data_loading, prior, utils as priors_utils
from .training_utils import (
    Metrics,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
    set_model_to,
    update_importance_sampling_infos,
)
from .utils import get_cosine_schedule_with_warmup, init_dist


@dataclass(frozen=True)
class MainConfig(base_config.BaseConfig):
    # Training configuration
    priors: tp.List[prior.PriorConfig]
    optimizer: OptimizerConfig

    # Model (includes criterion)
    model: TransformerConfig

    # Training
    batch_shape_sampler: BatchShapeSamplerConfig
    epochs: int = 10
    steps_per_epoch: int = 100
    aggregate_k_gradients: int = 1
    n_targets_per_input: int = 1
    train_mixed_precision: bool = True

    # LR Scheduler
    scheduler: str = "cosine_decay"
    warmup_epochs: int = 10

    # Checkpointing
    train_state_dict_save_path: tp.Optional[str] = None
    train_state_dict_load_path: tp.Optional[str] = None
    # Resuming continues training the CHECKPOINT's model, so a config mismatch means training
    # something other than what this config describes -- and overwriting the checkpoint with it.
    # `assert_resume_config_matches` refuses in that case; set this True to override deliberately,
    # which keeps the escape hatch visible in the config and therefore in the provenance record.
    allow_config_mismatch: bool = False

    # Validation
    test_priors: tp.List[prior.PriorConfig] | None = None
    validation_period: int | None = None

    # Logging
    verbose: bool = True
    progress_bar: bool = False
    tensorboard_path: tp.Optional[str] = None

    # Data loading
    dataloader_class: str | None = None
    num_workers: tp.Optional[int] = None


def train(
    c: MainConfig,
    device: str | None = None,
    reusable_config: bool = True,
    compile: bool = False,
    # Handy functions to override when not working with a standard file system
    save_object_function: tp.Callable | None = None,  # defaults to torch.save
    load_object_function: tp.Callable | None = None,  # defaults to torch.load
    check_path_exists_function: tp.Callable | None = None,  # defaults to os.path.exists
):
    if reusable_config:
        assert c.from_yaml(c.to_yaml()) == c, (
            "Config is not safe to use, got different config: "
            f"{c.from_yaml(c.to_yaml())=} vs {c=}"
        )

    # Arguments from original signature not in MainConfig are set to defaults here
    load_weights_from_this_state_dict = None

    total_start_time = time.time()

    default_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device is None:
        device = default_device
    using_dist, rank, device = init_dist(device)
    print(f"ALL: Using device {device}.")

    # Initialize Tensorboard writer
    writer = None
    if c.tensorboard_path is not None and rank == 0:
        writer = SummaryWriter(c.tensorboard_path)
        print(f"Tensorboard logging to: {c.tensorboard_path}")

    # Resolve dataloader_class string to actual class
    if c.dataloader_class is None:
        actual_dataloader_class = data_loading.StandardDataLoader
    else:
        parts = c.dataloader_class.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        try:
            module = importlib.import_module(module_path)
            actual_dataloader_class = getattr(module, class_name)
        except Exception as e:
            raise ImportError(
                f"Could not import dataloader_class '{c.dataloader_class}': {e}"
            ) from e

    def create_get_batch_method(priors: tp.List[prior.PriorConfig] | None):
        if not priors:
            raise ValueError("main_config.priors cannot be empty.")

        if len(priors) != 1:
            raise ValueError(
                "Currently only supporting a single prior. Later this should be a seqeunce that is called in order by wrapping."
            )

        if len(priors) == 1 and callable(priors[0]):  # Simplistic assumption
            get_batch_method_instance = priors[0]
        elif isinstance(priors[0], prior.PriorConfig):
            get_batch_method_instance = priors[0].create_get_batch_method()
        else:
            raise ValueError(
                "main_config.priors and main_config.test_priors must be a list of PriorConfig objects or a single callable."
            )

        return get_batch_method_instance

    get_batch_method_instance = create_get_batch_method(c.priors)
    if c.test_priors is not None:
        test_get_batch_method_instance = create_get_batch_method(c.test_priors)
    else:
        test_get_batch_method_instance = get_batch_method_instance

    current_extra_prior_kwargs_dict = {}
    if c.num_workers is not None:
        current_extra_prior_kwargs_dict["num_workers"] = c.num_workers

    data_loader = actual_dataloader_class(
        get_batch_method=get_batch_method_instance,  # Use the constructed/resolved instance
        batch_shape_sampler_function=c.batch_shape_sampler.sample_batch_shape,
        num_steps=c.steps_per_epoch,
        device=device,  # Pass the torch device object
        n_targets_per_input=c.n_targets_per_input,
        persistent_workers=True,  # can have persistent workers, as the dataset is counting the epochs itself here
        **current_extra_prior_kwargs_dict,
    )

    test_data_loader = actual_dataloader_class(
        get_batch_method=test_get_batch_method_instance,  # Use the constructed/resolved instance
        batch_shape_sampler_function=c.batch_shape_sampler.sample_batch_shape,
        num_steps=c.steps_per_epoch,
        device=device,  # Pass the torch device object
        n_targets_per_input=c.n_targets_per_input,
        persistent_workers=False,  # can't have persistent workers, otherwise the epoch count is not updated
        **current_extra_prior_kwargs_dict,
    )

    assert (
        c.model.features_per_group > 0 or c.model.features_per_group == -1
    ), "features_per_group must be > 0 or -1"

    model = c.model.create_model()
    criterion = model.criterion

    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters"
    )

    model.to(device)

    if compile:
        model = torch.compile(model)

    if hasattr(c.optimizer, "create_optimizer"):
        optimizer = c.optimizer.create_optimizer(model.parameters())
    else:
        raise ValueError("main_config.optimizer must have a 'create_optimizer' method")

    # Resolve scheduler string to function
    if c.scheduler == "cosine_decay":
        scheduler_fn = get_cosine_schedule_with_warmup
    else:
        assert c.scheduler == "constant", f"Scheduler {c.scheduler} not supported"
        scheduler_fn = None

    if scheduler_fn is None:
        scheduler = None
    else:
        scheduler = scheduler_fn(  # todo move warmup epochs into scheduler args, ideally as steps instead!?
            optimizer,
            c.warmup_epochs,
            c.epochs if c.epochs is not None else 100,
        )

    start_epoch = 1  # Default start epoch

    if should_load_checkpoint(c, check_path_exists_function=check_path_exists_function):
        # load_checkpoint needs the scheduler instance, not the factory
        start_epoch = load_checkpoint(  # load_checkpoint might return start_epoch
            model,
            optimizer,
            scheduler,
            c.train_state_dict_load_path,
            device,
            load_function=load_object_function,
            config=c,
            allow_config_mismatch=getattr(c, "allow_config_mismatch", False),
        )
    else:
        print(
            f"Checkpoint file {c.train_state_dict_load_path} not found or load/save paths are identical and file doesn't exist. Starting from scratch."
        )

    # set this before DDP
    data_loader.model = model
    test_data_loader.model = model

    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )

    scaler = GradScaler() if c.train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(data_loader)
    utils.check_compatibility(test_data_loader)

    total_loss = float("inf")
    try:
        for epoch in range(start_epoch, c.epochs + 1):
            epoch_start_time = time.time()
            try:
                epoch_result = train_or_evaluate_epoch(
                    c=c,
                    model=model,
                    optimizer=optimizer,
                    dl=data_loader,
                    device=device,
                    scaler=scaler,
                    criterion=criterion,
                    rank=rank,
                    using_dist=using_dist,
                    training=True,
                    writer=writer,
                    epoch=epoch,
                )
                total_loss = epoch_result.loss
                data_loader.importance_sampling_infos = (
                    epoch_result.importance_sampling_infos
                )

            except Exception as e:
                print("Invalid epoch encountered, skipping...")
                print(e)
                raise  # Re-raises the original exception with trace
            if c.validation_period is not None and (
                (epoch % c.validation_period == 0)
                or (epoch == c.epochs)
                or (epoch == 1)
            ):
                with torch.no_grad():
                    test_data_loader.epoch_count = (
                        epoch - 1
                    )  # -1 because the data_loader.__iter__ increases before the epoch
                    val_epoch_result = train_or_evaluate_epoch(
                        c=c,
                        model=model,
                        optimizer=optimizer,
                        dl=test_data_loader,
                        device=device,
                        scaler=None,
                        criterion=criterion,
                        rank=rank,
                        using_dist=using_dist,
                        training=False,
                        writer=None,  # Don't log step-level info for validation
                        epoch=epoch,
                    )
                    val_score_str = f"| eval mean loss {val_epoch_result.loss:5.2f} "

            else:
                val_score_str = ""

            epoch_time = time.time() - epoch_start_time
            if device.startswith("cuda"):
                max_gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                gpu_utilization = torch.cuda.utilization()
            else:
                max_gpu_mem_gb = None
                gpu_utilization = None
            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            if c.verbose:
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} | time: {epoch_time:5.2f}s | mean loss {epoch_result.loss:5.2f} "
                    + f"{val_score_str}"
                    + f"| lr {current_lr} "
                    + f"| data time {epoch_result.data_time:5.2f} step time {epoch_result.step_time:5.2f} "
                    + f"forward time {epoch_result.forward_time:5.2f} "
                    + f"| max gpu mem {f'{max_gpu_mem_gb:.1f}' if max_gpu_mem_gb is not None else 'N/A'} GiB "
                    + f"| gpu utilization {f'{gpu_utilization:.1f}' if gpu_utilization is not None else 'N/A'} %"
                    + f"| nan share {epoch_result.nan_share:5.2f} ignore share (for classification tasks) {epoch_result.ignore_share:5.4f} "
                )
                print("-" * 89)

            # Log epoch metrics to Tensorboard
            if writer:
                writer.add_scalar("epoch/train_loss", epoch_result.loss, epoch)
                writer.add_scalar("epoch/epoch_time", epoch_time, epoch)
                writer.add_scalar("epoch/data_time", epoch_result.data_time, epoch)
                writer.add_scalar("epoch/step_time", epoch_result.step_time, epoch)
                writer.add_scalar(
                    "epoch/forward_time", epoch_result.forward_time, epoch
                )
                writer.add_scalar("epoch/nan_share", epoch_result.nan_share, epoch)
                writer.add_scalar(
                    "epoch/ignore_share", epoch_result.ignore_share, epoch
                )

                # Log learning rate
                writer.add_scalar("epoch/learning_rate", current_lr, epoch)

                # Log GPU memory if available
                if device.startswith("cuda"):
                    writer.add_scalar("epoch/max_gpu_memory_gb", max_gpu_mem_gb, epoch)
                    writer.add_scalar("epoch/gpu_utilization", gpu_utilization, epoch)

                # Log validation loss if available
                if c.validation_period is not None and (
                    (epoch % c.validation_period == 0)
                    or (epoch == c.epochs)
                    or (epoch == 1)
                ):
                    writer.add_scalar("epoch/val_loss", val_epoch_result.loss, epoch)

            if scheduler is not None:
                scheduler.step()

            # Save model state dict after each epoch if path is provided (on rank 0)
            if c.train_state_dict_save_path is not None and rank == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    c.train_state_dict_save_path,
                    epoch,
                    config=c,
                    save_function=save_object_function,
                )

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        pass
    finally:
        if rank == 0:  # trivially true for non-parallel training
            set_model_to(model, optimizer, "eval")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
                data_loader = None
            if writer:
                writer.close()
            # Clean up distributed training
            if using_dist:
                torch.distributed.destroy_process_group()

    return {
        "total_loss": total_loss,
        "model": model.to("cpu"),
        "total_time": time.time() - total_start_time,
    }


# we could think about removing c as arg here to make the dep's clearer
def train_or_evaluate_epoch(
    c: MainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dl: priors_utils.DataLoader,
    device: str,
    scaler: GradScaler | None,
    criterion: torch.nn.Module,
    rank: int,
    using_dist: bool,
    training: bool = True,
    writer: SummaryWriter | None = None,
    epoch: int = 1,
):
    """
    Train or evaluate one epoch.
    """
    if training:
        assert optimizer is not None, "Optimizer must be provided for training"
    else:
        assert scaler is None, "Scaler must be None for evaluation"

    set_model_to(model, optimizer, "train" if training else "eval")

    metrics = Metrics(steps_per_epoch=len(dl))

    # Whether the prior does not return y, but but instead uses
    # a separate x_test and target.
    x_only_mode = c.model.x_only_mode

    importance_sampling_infos = []

    before_get_batch = time.time()
    assert (
        len(dl) % c.aggregate_k_gradients == 0
    ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."

    tqdm_iter = (
        tqdm(range(len(dl)), desc="Training Epoch")
        if rank == 0 and c.progress_bar
        else None
    )

    for batch_index, batch in enumerate(dl):
        batch: prior.Batch = batch  # for IDE support
        # batch.x.shape == (batch_size, seq_len, num_features)

        if x_only_mode:
            targets = batch.target.to(device)
        else:
            targets = batch.target_y.to(device)
        single_eval_pos = batch.single_eval_pos

        if tqdm_iter is not None:
            tqdm_iter.update()

        if using_dist and not (
            batch_index % c.aggregate_k_gradients == c.aggregate_k_gradients - 1
        ):
            potentially_no_sync_context = model.no_sync()
        else:
            potentially_no_sync_context = nullcontext()

        if training:
            potentially_no_grad_context = nullcontext()
        else:
            potentially_no_grad_context = torch.no_grad()

        with potentially_no_sync_context, potentially_no_grad_context:
            time_to_get_batch = time.time() - before_get_batch
            before_forward = time.time()
            try:
                with autocast(device.split(":")[0], enabled=scaler is not None):
                    if x_only_mode:
                        assert (
                            batch.target_y is None
                            and batch.y is None
                            and batch.y_style is None
                        ), "model.x_only_mode is not supported when y, target_y, or y_style are not None"
                        output = model(
                            x=batch.x.to(
                                device
                            ),  # shape: (batch_size, train_len, num_features)
                            test_x=batch.test_x.to(
                                device
                            ),  # shape: (batch_size, test_len, num_features)
                            y=None,
                            style=move_style_and_check_shape(
                                batch.style, batch.x, device
                            ),
                            only_return_standard_out=True,
                        )  # shape: (batch_size, test_len, num_groups)
                    else:
                        output = model(
                            x=batch.x.to(device),
                            y=batch.y[:, :single_eval_pos].to(device),
                            style=move_style_and_check_shape(
                                batch.style, batch.x, device
                            ),
                            y_style=move_y_style_and_check_shape(
                                batch.y_style, batch.y, device
                            ),
                            only_return_standard_out=True,
                        )  # shape: (batch_size, test_len)

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None and not x_only_mode:
                        targets = targets[
                            :, single_eval_pos:
                        ]  # shape: (batch_size, test_len)

                    loss, nan_share = compute_loss(
                        output, targets, criterion, c.n_targets_per_input, x_only_mode
                    )  # shape: (batch_size, test_len) | (batch_size, test_len, n_features)

                    loss_scaled = loss / c.aggregate_k_gradients

                if scaler:
                    loss_scaled = scaler.scale(loss_scaled)

                if training:
                    loss_scaled.backward()

                if batch_index % c.aggregate_k_gradients == c.aggregate_k_gradients - 1:
                    if scaler:
                        # we unscale s.t. we can clip grads right
                        scaler.unscale_(optimizer)

                    update_importance_sampling_infos(
                        importance_sampling_infos=importance_sampling_infos,
                        model=model,
                        optimizer=optimizer,
                        loss=loss.cpu().item(),
                        info_used_with_gradient_magnitudes=batch.info_used_with_gradient_magnitudes,
                    )

                    # todo i should run metrics on this
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )  # noop if no grads available

                    if batch.gradient_multipliers is not None:  # this None by default
                        assert (
                            training
                        ), "Gradient multipliers are only supported for training"
                        assert (
                            c.aggregate_k_gradients == 1
                        ), "Scaling grads is only supported if you don't do grad acc."
                        assert all(
                            batch.gradient_multipliers.view(-1)[0]
                            == batch.gradient_multipliers.view(-1)[i]
                            for i in range(batch.gradient_multipliers.numel())
                        ), "we don't scale losses for now to be able to try the interaction with gradient clipping, and thus we can only support the same scaler"
                        # todo make print to see that this is actually running
                        with torch.no_grad():
                            for w in model.parameters():
                                w.grad = w.grad * batch.gradient_multipliers.view(-1)[0]

                    if training:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                step_time = time.time() - before_forward

                metrics.update(
                    loss=loss,
                    nan_share=nan_share,
                    targets=targets,
                    forward_time=forward_time,
                    step_time=step_time,
                    time_to_get_batch=time_to_get_batch,
                )

            except Exception as e:
                print("Invalid step encountered, skipping...")
                print(e)
                raise (e)

        mean_loss = metrics.total_loss / (batch_index + 1)
        if tqdm_iter:
            tqdm_iter.set_postfix(
                {
                    "data_time": time_to_get_batch,
                    "step_time": step_time,
                    "mean_loss": mean_loss,
                }
            )

        # Log step-level metrics every 10 steps for training
        if writer and training and (batch_index % 10 == 0):
            global_step = (epoch - 1) * len(dl) + batch_index
            writer.add_scalar("step/data_time", time_to_get_batch, global_step)
            writer.add_scalar("step/step_time", step_time, global_step)
            writer.add_scalar("step/mean_loss", mean_loss, global_step)

        before_get_batch = time.time()

    return metrics.get_epoch_result(importance_sampling_infos)


def compute_loss(
    output: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    n_targets_per_input: int,
    x_only_mode: bool,
):
    """
    Compute the losses for the given output and targets.

    Args:
        output: The output of the model, shape (batch_size, num_eval_positions, n_out) | (batch_size, num_eval_positions, n_features, n_out)
        targets: The targets, shape (batch_size, num_eval_positions[, n_targets_per_input]) | (batch_size, num_eval_positions, n_features, n_targets_per_input)
        criterion: The criterion to use.
        n_targets_per_input: The number of targets per input.

    Returns:
        The losses, shape (batch_size, num_eval_positions)
    """
    if (
        len(output.shape) == 3
    ):  # else it is (batch_size, num_eval_positions, n_features, n_out)
        # Repeat output in the semi-last dimension n_targets_per_input times
        output = output.unsqueeze(2).expand(
            *output.shape[:2],
            n_targets_per_input,
            output.shape[-1],
        )

        if len(targets.shape) == 2:
            # This implies we only have a single target per input
            targets = targets.unsqueeze(2)

    assert targets.shape == output.shape[:-1], (
        f"Target shape {targets.shape} "
        f"does not match output shape {output.shape}."
        f"This might be because you are missing trailing "
        "1 dimension in the target."
    )

    if isinstance(criterion, nn.GaussianNLLLoss):
        assert (
            output.shape[-1] == 2
        ), "need to write a little bit of code to handle multiple regression targets at once"

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(
            mean_pred.flatten(),
            targets.flatten(),
            var=var_pred.flatten(),
        )
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        targets[torch.isnan(targets)] = -100
        losses = criterion(output.flatten(), targets.flatten())
        losses = losses.view(*targets.shape)
    elif isinstance(criterion, nn.CrossEntropyLoss):
        targets[torch.isnan(targets)] = -100
        losses = criterion(
            output.reshape(-1, len(criterion.weight)),
            targets.long().flatten(),
        )
    else:
        losses = criterion(output, targets.unsqueeze(-1))
    # mean over the last dimension (either features or target repetitions)
    if x_only_mode:
        loss = losses[~torch.isnan(targets)].mean()
        nan_share = torch.tensor(0.0)
    else:
        losses = losses.mean(-1)

        loss, nan_share = utils.torch_nanmean(
            losses.mean(
                1
            ),  # loss per sequence without nanmean, if any loss in a sequence is nan, the whole sequence is ignored
            return_nanshare=True,
        )  # loss and nan_share are both scalar tensors

    return loss, nan_share


def should_load_checkpoint(
    config: MainConfig, check_path_exists_function: tp.Callable | None = None
):
    if config.train_state_dict_load_path is None:
        return False
    if check_path_exists_function is None:
        check_path_exists_function = os.path.exists
    return (config.train_state_dict_save_path != config.train_state_dict_load_path) or (
        (config.train_state_dict_save_path == config.train_state_dict_load_path)
        and check_path_exists_function(config.train_state_dict_load_path)
    )


# Fields that may legitimately differ between the config that WROTE a checkpoint and the config
# resuming it. Everything else must match, or the resume is silently training a different model.
#   epochs                      - extending a run is the main legitimate reason to resume;
#   train_state_dict_*_path     - a resume may read one file and write another;
#   tensorboard_path            - logging destination is not part of the experiment;
#   device / num_workers        - execution environment, not the model or the data.
RESUME_ALLOWED_DIFFS = frozenset({
    "epochs", "train_state_dict_load_path", "train_state_dict_save_path",
    "tensorboard_path", "device", "num_workers", "progress_bar", "verbose",
})


def _flatten_config(obj, prefix=""):
    """Config -> {dotted key: repr}, so two configs can be diffed field by field."""
    out = {}
    d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", None)
    if d is None:
        return {prefix.rstrip("."): repr(obj)}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, (dict,)) or hasattr(v, "__dict__"):
            out.update(_flatten_config(v, key + "."))
        elif isinstance(v, (list, tuple)):
            # Recurse into lists of config objects (e.g. `priors`). A stored config holds them as
            # dicts and a live one as objects, so comparing reprs would always differ.
            for i, item in enumerate(v):
                if isinstance(item, dict) or hasattr(item, "__dict__"):
                    out.update(_flatten_config(item, f"{key}[{i}]."))
                else:
                    out[f"{key}[{i}]"] = repr(item)
        else:
            out[key] = repr(v)
    return out


def assert_resume_config_matches(checkpoint, config, path, allow_mismatch=False,
                                 action="RESUME", consequence=None):
    """Refuse to resume a checkpoint that was written by a DIFFERENT configuration.

    WHY THIS EXISTS [owner's request, 2026-08-18]. Resuming is silent and looks like success. On
    2026-08-18 three training jobs for new seed replicates were generated from another cell's config
    by substitution; the hardcoded `train_state_dict_load_path` came along unchanged, so all three
    loaded that cell's checkpoint, found it already at its final epoch, trained nothing, and re-saved
    it. Elapsed 13 seconds, exit code 0, `Final loss: inf` -- a job that reports success while doing
    the opposite of what was intended, and which would have overwritten a checkpoint other results
    depend on had the epoch differed.

    A resume is only meaningful if the checkpoint was produced by the same experiment. Comparing the
    stored config against the current one makes that explicit rather than assumed. Fields that
    legitimately change on a resume are listed in RESUME_ALLOWED_DIFFS.

    Set `allow_config_mismatch=True` in the config to override deliberately -- e.g. resuming after an
    intentional change -- which keeps the escape hatch but makes it appear in the config, and so in
    the provenance record, rather than being invisible.
    """
    stored = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if stored is None:
        print("WARNING: checkpoint carries no config; cannot verify it matches. Proceeding.")
        return
    a, b = _flatten_config(stored), _flatten_config(config)
    # Compare only keys present in BOTH, and ignore serialisation bookkeeping. A stored config is a
    # dict carrying `__config_type__` markers the live object lacks, and a field added to MainConfig
    # after a checkpoint was written is absent from it -- neither is an experiment difference, and
    # treating them as one would make the guard fire on every pre-existing checkpoint.
    keys = {k for k in set(a) & set(b)
            if not k.endswith("__config_type__")
            and k.split(".")[-1] not in RESUME_ALLOWED_DIFFS}
    diffs = sorted(k for k in keys if a[k] != b[k])
    if not diffs:
        return
    lines = "\n".join(f"    {k}:\n      checkpoint: {a.get(k)}\n      current:    {b.get(k)}"
                       for k in diffs[:12])
    more = f"\n    ... and {len(diffs) - 12} more" if len(diffs) > 12 else ""
    if consequence is None:
        consequence = (
            "  A resume continues training the checkpoint's model, so a config mismatch means you are\n"
            "  training something other than what this config describes -- and will overwrite the\n"
            "  checkpoint with it. If this is a fresh run, remove train_state_dict_load_path. If the\n"
            "  change is deliberate, set allow_config_mismatch=True.")
    msg = (f"REFUSING TO {action}: {path}\n"
           f"  was written by a DIFFERENT configuration ({len(diffs)} field(s) differ):\n"
           f"{lines}{more}\n" + consequence)
    if allow_mismatch:
        print("WARNING (allow_config_mismatch=True):\n" + msg)
        return
    raise ValueError(msg)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    train_state_dict_load_path,
    device,
    load_function: tp.Callable | None = None,
    config=None,
    allow_config_mismatch: bool = False,
):
    print(f"Loading checkpoint from {train_state_dict_load_path}")
    if load_function is None:
        load_function = torch.load
    try:
        checkpoint = load_function(train_state_dict_load_path, map_location=device)
        if config is not None:
            assert_resume_config_matches(
                checkpoint, config, train_state_dict_load_path, allow_config_mismatch)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New format with model, optimizer state and epoch
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
            # Fast-forward the scheduler to the correct epoch
            if scheduler is not None:
                for _ in range(start_epoch - 1):
                    scheduler.step()
            return start_epoch
        else:
            raise ValueError(
                f"Checkpoint does not contain 'model_state_dict' or 'optimizer_state_dict'. Checkpoint: {checkpoint}"
            )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e


def load_config(train_state_dict_load_path, load_function: tp.Callable | None = None):
    if load_function is None:
        load_function = torch.load
    checkpoint = load_function(train_state_dict_load_path, map_location="cpu")
    return MainConfig.from_dict(checkpoint["config"])


def _assert_safe_to_overwrite(path, config, allow_mismatch=False, load_function=None):
    """Refuse to overwrite a checkpoint belonging to a DIFFERENT experiment.

    Complements `assert_resume_config_matches`, which only fires when a run RESUMES. A config whose
    save path points at another experiment's checkpoint but which starts from scratch never loads it,
    so the resume guard never runs and the file is destroyed at the first periodic save. That is the
    worse of the two failures: a resume at least preserves the weights it continues from.

    Overwriting a checkpoint written by the SAME config is normal -- that is what periodic saving and
    re-running a config are. Only a mismatch is refused.
    """
    if load_function is None:
        load_function = torch.load
    if not os.path.exists(path):
        return
    try:
        existing = load_function(path, map_location="cpu")
    except Exception as e:
        print(f"WARNING: could not read existing checkpoint at {path} to verify overwrite ({e}).")
        return
    assert_resume_config_matches(
        existing, config, path, allow_mismatch=allow_mismatch, action="OVERWRITE",
        consequence=(
            "  This run is about to REPLACE a checkpoint produced by a different configuration, and\n"
            "  it never loaded it, so nothing else would have noticed. Point train_state_dict_save_path\n"
            "  at this run's own file. If the replacement is deliberate, set allow_config_mismatch=True."))


def save_checkpoint(
    model,
    optimizer,
    train_state_dict_save_path,
    epoch,
    config: MainConfig,
    save_function: tp.Callable | None = None,
):
    set_model_to(model, optimizer, "eval")
    save_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    # SAVE-SIDE GUARD, the complement of assert_resume_config_matches [owner, 2026-08-18].
    #
    # The load-side guard cannot see the most destructive case. A config whose SAVE path points at
    # another experiment's checkpoint while starting from scratch never loads it, so nothing compares
    # anything -- the file is simply overwritten at the first periodic save, and the original is gone.
    # Verified: should_load_checkpoint() returns False for exactly that configuration.
    #
    # Checked once, before the first write of a run, rather than on every save.
    if not getattr(save_checkpoint, "_verified", set()) or \
            train_state_dict_save_path not in save_checkpoint._verified:
        _assert_safe_to_overwrite(
            train_state_dict_save_path, config,
            allow_mismatch=getattr(config, "allow_config_mismatch", False))
        if not hasattr(save_checkpoint, "_verified"):
            save_checkpoint._verified = set()
        save_checkpoint._verified.add(train_state_dict_save_path)
    print(f"Saving checkpoint to {train_state_dict_save_path} (epoch {epoch})")
    try:
        # Save model state dict, optimizer state dict, and current epoch
        checkpoint = {
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config.to_dict(),
            # The config fixes the training distribution; this fixes the code that
            # realized it. See pfns/provenance.py.
            "provenance": provenance.provenance(),
        }
        if save_function is None:
            os.makedirs(os.path.dirname(train_state_dict_save_path), exist_ok=True)
            save_function = torch.save
        save_function(checkpoint, train_state_dict_save_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
