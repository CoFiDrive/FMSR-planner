import logging
import os
import warnings
from itertools import islice

import matplotlib
import torch
from beartype import beartype
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from fmsr.common.constants import SourceDataset
from fmsr.common.logging_config import setup_logging
from fmsr.data_loader.waymo_e2e_dataset import evaluate_waymo_e2e
from fmsr.training import training_utils_fmsr
from fmsr.training.logger import Logger

matplotlib.use("Agg")  # non-GUI backend for headless servers

# Disable TF32 during numerical-stability diagnosis on Ampere/Ada GPUs.
# These settings affect FP32 matrix multiplications and cuDNN convolutions.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

setup_logging()
LOG = logging.getLogger(__name__)

# Maximum L2 norm used to clip gradients before every optimizer update.
# A value of 1.0 is a conservative default for stable single-GPU training.
MAX_GRAD_NORM = 1.0

warnings.filterwarnings("error")
warnings.filterwarnings(
    "ignore", message="Grad strides do not match bucket view strides"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*epoch parameter.*")
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message="Implicitly cleaning up.*TemporaryDirectory.*",
)


class Trainer:
    def __init__(self):
        self.config = training_utils_fmsr.initialize_config()

        self.ssd_cache = training_utils_fmsr.initialize_training_session_cache(self.config)
        self.num_worker = training_utils_fmsr.initialize_torch(self.config)
        self.model_wrapper, self.cur_epoch = training_utils_fmsr.initialize_model(
            self.config
        )
        self.model = self.model_wrapper
        if isinstance(self.model_wrapper, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model_wrapper.module
        self.dataloader, self.sampler = training_utils_fmsr.initialize_dataloader(
            self.config, self.ssd_cache, self.num_worker
        )

        # Predict how long the training will take
        self.gradient_steps_per_epoch = int(len(self.dataloader))
        self.total_gradient_steps = self.gradient_steps_per_epoch * self.config.epochs

        if self.config.rank == 0:
            LOG.info(
                f"Training for {self.config.epochs} epochs, "
                f"{self.gradient_steps_per_epoch} gradient steps per epoch, "
                f"total {self.total_gradient_steps} gradient steps."
            )

        # Set up optimizer
        self.optimizer, self.scheduler, self.scaler, self.gradient_steps_skipped = (
            training_utils_fmsr.initialize_optimizer(
                model_wrapper=self.model_wrapper,
                model=self.model,
                config=self.config,
                gradient_steps_per_epoch=self.gradient_steps_per_epoch,
            )
        )

        training_utils_fmsr.save_config(self.config, self.config.rank)

        self.logger = Logger(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            continue_step=int(self.cur_epoch * self.gradient_steps_per_epoch),
            dataset=self.dataloader.dataset,
            dataloader=self.dataloader,
            total_gradient_steps=self.total_gradient_steps,
        )

    @beartype
    def schedule_loss_weights(self, epoch: int):
        """Schedule loss weights for different datasets based on the epoch."""
        self.detailed_loss_weights = {}
        if self.config.use_carla_data:
            carla_loss_weights = self.config.detailed_loss_weights(
                SourceDataset.CARLA, epoch
            )
            self.detailed_loss_weights.update(carla_loss_weights)
        if self.config.use_navsim_data:
            navsim_loss_weights = self.config.detailed_loss_weights(
                SourceDataset.NAVSIM, epoch
            )
            self.detailed_loss_weights.update(navsim_loss_weights)
        if self.config.use_waymo_e2e_data:
            waymo_e2e_loss_weights = self.config.detailed_loss_weights(
                SourceDataset.WAYMO_E2E_2025, epoch
            )
            self.detailed_loss_weights.update(waymo_e2e_loss_weights)

        total_loss_weight = sum(self.detailed_loss_weights.values())
        for key in self.detailed_loss_weights:
            self.detailed_loss_weights[key] /= (
                total_loss_weight  # Normalize to sum to 1
            )

    def train_loop(self):
        for epoch in range(self.cur_epoch, self.config.epochs):
            self.dataloader.dataset.shuffle(epoch)

            # Update sampler epoch
            self.sampler.update_batch_sizes(epoch)

            # Update loss weights
            self.schedule_loss_weights(epoch)

            # Training
            rfm_score = self.train()

            # Save model
            if self.config.rank == 0:
                self.save(rfm_score)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Ensure all processes sync here before next epoch

            self.cur_epoch += 1

    @beartype
    def train(self) -> float | None:
        self.model_wrapper.train()

        # Train loop
        for epoch_iteration, data in enumerate(
            tqdm(
                islice(self.dataloader, self.gradient_steps_per_epoch),
                total=self.gradient_steps_per_epoch,
                disable=self.config.rank != 0,
            )
        ):
            loss = torch.zeros(
                1, dtype=torch.float32, device=self.config.device
            )
            data["iteration"] = epoch_iteration
            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.config.torch_float_type,
                enabled=self.config.use_mixed_precision_training,
            ):
                # Forward pass
                predictions = self.model_wrapper(data=data)
                losses, log = self.model.compute_loss(
                    predictions=predictions, data=data
                )

                # Sum up losses
                for key, value in losses.items():
                    # add reg and cls loss
                    loss += self.detailed_loss_weights[key] * value.float().reshape(
                        1
                    )  # Reshape as sanity check if the loss is a scalar
                scaled_loss = {
                    key: self.detailed_loss_weights[key] * value
                    for key, value in losses.items()
                }
            # Detect NaN/Inf before backward. Updating the model with a
            # non-finite loss would contaminate the optimizer state.
            if not torch.isfinite(loss.detach()).all():
                bad_losses = {
                    key: value.detach().float().item()
                    for key, value in losses.items()
                    if torch.is_tensor(value)
                    and value.numel() == 1
                    and not torch.isfinite(value.detach()).all()
                }
                LOG.warning(
                    f"Skipping non-finite loss at epoch={self.cur_epoch}, "
                    f"iteration={epoch_iteration}, bad_losses={bad_losses}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.gradient_steps_skipped += 1
                continue

            # Backpropagate outside the autocast context.
            self.scaler.scale(loss).backward()

            # Gradients must be unscaled before their norm is measured and
            # clipped. With BF16 the scaler is disabled and this is a no-op.
            self.scaler.unscale_(self.optimizer)
            trainable_parameters = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=MAX_GRAD_NORM,
            )

            # Skip an update if gradients have already become NaN or Inf.
            # For FP16, scaler.update() also reduces the scale when needed.
            if not torch.isfinite(grad_norm):
                grad_norm_value = grad_norm.detach().float().item()
                LOG.warning(
                    f"Skipping non-finite gradient at epoch={self.cur_epoch}, "
                    f"iteration={epoch_iteration}, grad_norm={grad_norm_value}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.gradient_steps_skipped += 1
                continue

            # Gradient step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.gradient_steps_skipped += int(
                scale_before > self.scaler.get_scale()
            )  # Count how many times the scale changed
            if not (scale_before > self.scaler.get_scale()):
                self.scheduler.step(
                    self.gradient_steps_per_epoch * self.cur_epoch + epoch_iteration
                )
            if self.scaler.get_scale() > self.config.grad_scaler_max_grad_scale:
                self.scaler.update(new_scale=self.config.grad_scaler_max_grad_scale)

            self.logger.log_train(
                epoch_iteration=epoch_iteration,
                cur_epoch=self.cur_epoch,
                unscaled_loss=losses,
                scaled_loss=scaled_loss,
                data=data,
                step=int(self.cur_epoch * self.gradient_steps_per_epoch)
                + epoch_iteration,
                gradient_steps_skipped=self.gradient_steps_skipped,
                log=log,
                predictions=predictions,
            )
            self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.config.torch_float_type,
            enabled=self.config.use_mixed_precision_training,
        ):
            rfm_score = None
            if self.config.use_waymo_e2e_data:
                rfm_score = evaluate_waymo_e2e(self.model, self.config)
                self.logger.logs({"rfm": rfm_score})
        return rfm_score

    @beartype
    def save(self, rfm_score: float | None = None):
        if not self.config.save_model_checkpoint:
            return

        # Save best Waymo E2E model based on RFM score
        if self.config.use_waymo_e2e_data:
            potential_path = os.path.join(
                self.config.logdir,
                f"model_best_rfm_{self.cur_epoch}_{rfm_score:04f}.pth",
            )
            current_bests = os.listdir(self.config.logdir)
            current_bests = [f for f in current_bests if f.startswith("model_best_rfm")]
            new_best = False
            if len(current_bests) == 0 or all(
                rfm_score > float(f.split("_")[-1][:-4]) for f in current_bests
            ):
                torch.save(
                    self.model.state_dict(),
                    potential_path,
                )
                LOG.info(
                    f"New best Waymo E2E RFM score: {rfm_score:.4f}, saved model to {potential_path}"
                )
                new_best = True
            if new_best:
                for f in current_bests:
                    os.remove(os.path.join(self.config.logdir, f))

        if self.config.use_zero_redundancy and torch.cuda.device_count() > 1:
            # To save the whole optimizer we need to gather it on GPU 0.
            self.optimizer.consolidate_state_dict(0)

        # The parallel weights are named differently with the module.
        # We remove that, so that we can load the model with the same code.
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.logdir, f"model_{self.cur_epoch:04d}.pth"),
        )
        LOG.info(f"Saved model checkpoint for epoch {self.cur_epoch}.")
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.config.logdir, f"optimizer_{self.cur_epoch:04d}.pth"),
        )
        LOG.info(f"Saved optimizer checkpoint for epoch {self.cur_epoch}.")
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(self.config.logdir, f"scheduler_{self.cur_epoch:04d}.pth"),
        )
        LOG.info(f"Saved scheduler checkpoint for epoch {self.cur_epoch}.")
        if self.scaler is not None:
            torch.save(
                self.scaler.state_dict(),
                os.path.join(self.config.logdir, f"scaler_{self.cur_epoch:04d}.pth"),
            )
            LOG.info(f"Saved scaler checkpoint for epoch {self.cur_epoch}.")
        with open(
            os.path.join(
                self.config.logdir, f"gradient_steps_skipped_{self.cur_epoch:04d}.txt"
            ),
            "w",
        ) as f:
            f.write(str(self.gradient_steps_skipped))
        LOG.info(f"Saved gradient steps skipped for epoch {self.cur_epoch}.")

        # # Remove last epochs files to avoid accumulating storage
        # if self.cur_epoch > 0:
        #     last_model_file = os.path.join(
        #         self.config.logdir, f"model_{self.cur_epoch - 1:04d}.pth"
        #     )
        #     last_optimizer_file = os.path.join(
        #         self.config.logdir, f"optimizer_{self.cur_epoch - 1:04d}.pth"
        #     )
        #     last_scheduler_file = os.path.join(
        #         self.config.logdir, f"scheduler_{self.cur_epoch - 1:04d}.pth"
        #     )
        #     last_scaler_file = os.path.join(
        #         self.config.logdir, f"scaler_{self.cur_epoch - 1:04d}.pth"
        #     )
        #     last_gradient_steps_skipped_file = os.path.join(
        #         self.config.logdir,
        #         f"gradient_steps_skipped_{self.cur_epoch - 1:04d}.txt",
        #     )

        #     if (
        #         os.path.isfile(last_model_file)
        #         and self.cur_epoch not in self.config.epoch_checkpoints_keep
        #     ):
        #         os.remove(last_model_file)
        #         LOG.info(f"Removed model checkpoint for epoch {self.cur_epoch - 1}.")
        #     if os.path.isfile(last_optimizer_file):
        #         os.remove(last_optimizer_file)
        #         LOG.info(
        #             f"Removed optimizer checkpoint for epoch {self.cur_epoch - 1}."
        #         )
        #     if os.path.isfile(last_scheduler_file):
        #         os.remove(last_scheduler_file)
        #         LOG.info(
        #             f"Removed scheduler checkpoint for epoch {self.cur_epoch - 1}."
        #         )
        #     if os.path.isfile(last_scaler_file):
        #         os.remove(last_scaler_file)
        #         LOG.info(f"Removed scaler checkpoint for epoch {self.cur_epoch - 1}.")
        #     if os.path.isfile(last_gradient_steps_skipped_file):
        #         os.remove(last_gradient_steps_skipped_file)
        #         LOG.info(
        #             f"Removed gradient steps skipped for epoch {self.cur_epoch - 1}."
        #         )


@record  # Records error and tracebacks in case of failure
def main():
    training_utils_fmsr.increase_limit_file_descriptors()
    Trainer().train_loop()


if __name__ == "__main__":
    training_utils_fmsr.set_start_method()
    main()
