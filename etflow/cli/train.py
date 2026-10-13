"""Task-native Lightning PFM runner with fixed configuration and epoch resumes."""

import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from etflow.data.datamodule import BaseDataModule
from etflow.models.pfm_model import PosteriorFlow
from etflow.commons.pfm_artifacts import source_digest, run_lock, write_json


class Evidence(pl.Callback):
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        self.pending_rng = None
        self.times = []
        self.started = None

    def on_fit_start(self, trainer, module):
        if trainer.world_size != self.config["trainer"]["devices"]:
            raise RuntimeError("Distributed world size differs from the requested GPU/process count")
        self.fit_started = time.perf_counter()
        estimated_steps = trainer.estimated_stepping_batches
        trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch, self.config["seed"])
        if trainer.is_global_zero:
            info = {
                "source_sha256": source_digest(),
                "config": self.config,
                "train_molecules": len(trainer.datamodule.train_dataset),
                "val_molecules": len(trainer.datamodule.val_dataset),
                "world_size": trainer.world_size,
                "effective_batch": self.config["data"]["batch_size"]
                * trainer.world_size
                * trainer.accumulate_grad_batches,
                "estimated_optimizer_steps": estimated_steps,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            }
            (self.run_dir / "run.json").write_text(json.dumps(info, indent=2))

    def on_train_start(self, trainer, module):
        if self.pending_rng:
            state = self.pending_rng[trainer.global_rank]
            torch.set_rng_state(state["torch"].cpu())
            np.random.set_state(state["numpy"])
            random.setstate(state["python"])
            if module.device.type == "cuda" and "cuda" in state:
                torch.cuda.set_rng_state(state["cuda"].cpu())
            if "loader" in state:
                trainer.datamodule.dataloader_args["generator"].set_state(state["loader"].cpu())
            self.pending_rng = None

    def on_train_epoch_start(self, trainer, module):
        trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch, self.config["seed"])

    def on_train_epoch_end(self, trainer, module):
        # Set before the next iterator prefetches data through persistent workers.
        trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch + 1, self.config["seed"])

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if module.device.type == "cuda":
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if module.device.type == "cuda":
            torch.cuda.synchronize()
        if batch_idx >= 5:
            self.times.append(time.perf_counter() - self.started)

    def on_save_checkpoint(self, trainer, module, checkpoint):
        state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "loader": trainer.datamodule.dataloader_args["generator"].get_state(),
        }
        if module.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state()
        states = [None] * trainer.world_size
        if torch.distributed.is_initialized():
            torch.distributed.all_gather_object(states, state)
        else:
            states = [state]
        checkpoint["pfm_rng_states"] = states
        checkpoint["pfm_config"] = self.config

    def on_load_checkpoint(self, trainer, module, checkpoint):
        if checkpoint.get("pfm_config") != self.config:
            raise ValueError("Resume configuration differs from the checkpoint")
        self.pending_rng = checkpoint.get("pfm_rng_states")

    def on_fit_end(self, trainer, module):
        if trainer.is_global_zero:
            info = {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "fit_wall_seconds": time.perf_counter() - self.fit_started,
                "mean_microbatch_seconds": float(np.mean(self.times)) if self.times else None,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0,
            }
            (self.run_dir / "training_result.json").write_text(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop at this epoch boundary without changing the training schedule")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    settings = cfg["trainer"].copy()
    torch.set_num_threads(args.threads)
    if args.devices:
        settings["devices"] = args.devices
    if args.cpu:
        settings.update(accelerator="cpu", devices=1, precision="32-true")
    if settings["devices"] == 1:
        settings["strategy"] = "auto"
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(run_dir.iterdir()):
        raise ValueError("Run directory is not empty; resume or choose a new directory")
    cfg["trainer"] = settings
    cfg["data_dir"] = str(Path(args.data_dir).resolve())
    torch.set_float32_matmul_precision("highest")
    pl.seed_everything(cfg["seed"], workers=True)
    dm = BaseDataModule(
        data_dir=Path(cfg["data_dir"]),
        partition="qm9",
        dataloader_args={
            **cfg["data"],
            "pin_memory": not args.cpu,
            "persistent_workers": cfg["data"]["num_workers"] > 0,
            "generator": torch.Generator().manual_seed(cfg["seed"]),
        },
    )
    model = PosteriorFlow(**cfg["model_args"])
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        **settings,
        default_root_dir=str(run_dir),
        logger=CSVLogger(str(run_dir), name="metrics"),
        callbacks=[
            Evidence(cfg, run_dir),
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        enable_progress_bar=False,
    )
    if args.stop_after_epoch is not None:
        if not 1 <= args.stop_after_epoch < settings["max_epochs"]:
            raise ValueError("stop-after-epoch must precede configured max_epochs")
        class StopAtEpoch(pl.Callback):
            def on_train_epoch_end(self, trainer, module):
                if trainer.current_epoch + 1 >= args.stop_after_epoch:
                    trainer.should_stop = True
        trainer.callbacks.insert(0, StopAtEpoch())
    with run_lock(run_dir):
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
        trainer.save_checkpoint(str(run_dir / "checkpoints" / "final.ckpt"))


if __name__ == "__main__":
    main()
