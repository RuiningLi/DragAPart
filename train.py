"""
A minimal training script for DragAPart using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from omegaconf import OmegaConf
from typing import Dict
from pathlib import Path
from diffusers.models import AutoencoderKL
import os
from os import path as osp
from PIL import Image
import cv2

from networks import UNet2DDragConditionModel
from dataset import DragAMoveDataset
from diffusion import create_diffusion
from transformers import CLIPVisionModel
from accelerate import Accelerator
from accelerate.utils import set_seed

from inference import do_inference

import wandb


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def importance_sampling_fn(t: torch.Tensor, max_t: int, alpha = 0.5):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * torch.cos(np.pi * t / max_t))


@torch.no_grad()
def prepare_model_input_from_batch(batch, vae, clip_vit, device=None):
    if device is None: device = vae.device
    recon_rgb = batch["recon_pixel_values"].to(device)
    cond_rgb = batch["cond_pixel_values"].to(device)

    output = clip_vit(
        pixel_values=batch["clip_pixel_values"].squeeze(1).to(device), 
        output_hidden_states=True
    )
    cls_embedding = torch.stack(output.hidden_states, dim=1)[:, :, 0]

    recon_latent = vae.encode(recon_rgb).latent_dist.sample().mul_(0.18215)
    cond_latent = vae.encode(cond_rgb).latent_dist.sample().mul_(0.18215)
    return recon_latent, cond_latent, cls_embedding


def get_generator(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, no_dist=False):
    """
    Create a logger that writes to a log file and stdout.
    """
    if no_dist or dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)

    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(
    args,
    use_wandb: bool,
    image_size: int,
    results_dir: str,
    num_steps: int,
    global_batch_size: int,
    num_workers: int,
    log_every: int,
    ckpt_every: int,
    visualize_every: int,
    learning_rate: float,
    data_args: Dict,
    model_args: Dict,
    visualization_args: Dict,
    vae: str = "ema",
    random_seed: int = None,
    resume_checkpoint_path: str = None,
    importance_sampling: int = 0,
    learn_sigma: bool = False,
    
    flow_original_res: bool = False,
    one_sided_attn: bool = True,
):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    data_args["sample_size"] = image_size

    accelerator = Accelerator()
    if random_seed is not None:
        set_seed(random_seed, device_specific=True)
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(
            results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{results_dir}/*"))
        experiment_dir = f"{results_dir}/{experiment_index:03d}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        samples_dir = f"{experiment_dir}/samples"  # Stores samples from the model
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir, no_dist=accelerator.num_processes == 1)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Save a copy of the config file:
        OmegaConf.save(config, os.path.join(experiment_dir, "config.yaml"))

        if use_wandb:
            run = wandb.init(
                project="DragAPart", name=f"{experiment_index:03d}"
            )
    else:
        logger = create_logger(None)

    # Create model:
    assert (
        image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."

    latent_size = image_size // 8
    latent_channel = 4
    
    unet_additional_kwargs = {
        "sample_size": latent_size,
        "one_sided_attn": one_sided_attn,
    }
    unet_additional_kwargs.update(model_args)
    model = UNet2DDragConditionModel.from_pretrained_sd(
        "/scratch/shared/beegfs/ruining/checkpoint/stable-diffusion-v1-5",
        unet_additional_kwargs=unet_additional_kwargs,
    ).to(device)
    model.train()
    params = model.parameters()

    ema = deepcopy(model)
    if resume_checkpoint_path is not None:
        checkpoint_dict = torch.load(resume_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_dict["model"])
        opt.load_state_dict(checkpoint_dict["opt"])
        ema.load_state_dict(checkpoint_dict["ema"])
        if accelerator.is_main_process:
            logger.info(f"Loaded checkpoint from {resume_checkpoint_path}")

    ema = ema.to(device)
    requires_grad(ema, False)

    diffusion = create_diffusion(
        timestep_respacing="",
        learn_sigma=learn_sigma,
    )  # default: 1000 steps, linear noise schedule

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    clip_vit = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0)
    train_steps = 0 if resume_checkpoint_path is None else int(osp.basename(resume_checkpoint_path).split(".")[0])

    train_dataset = DragAMoveDataset(**data_args)
    loader = DataLoader(
        train_dataset,
        batch_size=int(global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=False,
    )

    if accelerator.is_main_process:
        val_data_args = deepcopy(data_args)
        val_data_args["dataset_root_folder"] = data_args["dataset_root_folder"].replace("train", "val")
        val_dataset = DragAMoveDataset(**val_data_args)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        val_generator = get_generator(val_loader)

    model, opt, loader = accelerator.prepare(model, opt, loader)
    logger.info(f"Dataset contains {len(train_dataset):,} different pairs")

    # Prepare models for training:
    update_ema(
        ema, model.module if accelerator.num_processes > 1 else model, decay=0
    )  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {num_steps} steps...")

    while True:
        for batch in loader:
            model.train()
            recon_latent, cond_latent, cls_embedding = prepare_model_input_from_batch(
                batch, vae, clip_vit, device=device
            )

            if importance_sampling != 0:
                candidates = torch.arange(diffusion.num_timesteps, device=device)
                probs = importance_sampling_fn(candidates, diffusion.num_timesteps)
                probs = probs / probs.sum()
                t = torch.multinomial(probs, recon_latent.shape[0], replacement=True).to(device)
                if importance_sampling < 0:
                    t = -t + diffusion.num_timesteps - 1
            else:
                t = torch.randint(0, diffusion.num_timesteps, (recon_latent.shape[0],), device=device)
            
            model_kwargs = dict(
                x_cond=cond_latent,
                hidden_cls=cls_embedding,
                drags=batch["drags"].to(device),
            )
            loss_dict = diffusion.training_losses(model, recon_latent, t, model_kwargs)
            loss = torch.nan_to_num(loss_dict["loss"], nan=0.).mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model.module if accelerator.num_processes > 1 else model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            log_dict = {}
            if train_steps % log_every == 0:
                torch.cuda.synchronize()

                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if accelerator.num_processes > 1:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / accelerator.num_processes
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                log_dict["train_loss"] = avg_loss
                log_dict["train_steps_per_sec"] = steps_per_sec

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if accelerator.num_processes > 1:
                    dist.barrier()
                
            if train_steps % visualize_every == 0:
                model.eval()
                if accelerator.is_main_process:
                    val_batch = next(val_generator)
                    _, val_cond_latent, val_cls_embedding = prepare_model_input_from_batch(
                        val_batch, vae, clip_vit, device=device
                    )

                    sample_latent = do_inference(
                        model.module if accelerator.num_processes > 1 else model,
                        diffusion,
                        val_cond_latent,
                        val_cls_embedding,
                        val_batch["drags"].to(device),
                        cfg_scale=visualization_args["cfg_scale"],
                        latent_size=latent_size,
                        latent_channel=latent_channel,
                    )

                    with torch.no_grad():
                        sample = vae.decode(sample_latent / 0.18215).sample[0]
                        
                    cond_image = np.ascontiguousarray(((val_batch["cond_pixel_values"][0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8))
                    for drag in val_batch["drags"][0]:
                        if drag.abs().sum() > 0:
                            x0, y0, x1, y1 = drag.tolist()
                            cond_image = cv2.arrowedLine(
                                cond_image, 
                                (int(x0 * image_size), int(y0 * image_size)), 
                                (int(x1 * image_size), int(y1 * image_size)), 
                                (255, 255, 0), 
                                3
                            )
                            cond_image = cv2.arrowedLine(
                                cond_image, 
                                (int(x0 * image_size), int(y0 * image_size)), 
                                (int(x1 * image_size), int(y1 * image_size)), 
                                (0, 0, 0), 
                                5
                            )

                    cond_image = Image.fromarray(cond_image)
                    ground_truth = Image.fromarray(((val_batch["recon_pixel_values"][0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8))
                    sample = Image.fromarray(((sample.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8))

                    cond_image.save(f"{samples_dir}/{train_steps:07d}_cond.png")
                    ground_truth.save(f"{samples_dir}/{train_steps:07d}_gt.png")
                    sample.save(f"{samples_dir}/{train_steps:07d}_sample.png")

                    log_dict["cond_images_eval"] = wandb.Image(cond_image)
                    log_dict["grouth_truth_eval"] = wandb.Image(ground_truth)
                    log_dict["samples_eval"] = wandb.Image(sample)

            if accelerator.is_main_process and use_wandb:
                wandb.log(log_dict)

            if train_steps >= num_steps:
                break

        if train_steps >= num_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")
    if accelerator.num_processes > 1:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(args, use_wandb=args.wandb, **config)
