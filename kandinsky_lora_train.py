import pyrallis
import logging
import math
import os
import copy
import shutil
from pathlib import Path
from enum import Enum, auto

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from dataclasses import asdict
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import diffusers
from diffusers import (
    AutoPipelineForText2Image,
    DDPMScheduler,
    UNet2DConditionModel,
    VQModel,
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    LoRAAttnAddedKVProcessor,
    AttnAddedKVProcessor,
    Attention,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available

import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import to_pil_image
from peft import LoraConfig, get_peft_model, PeftModel

from textual_inversion_config import KandinskyLoRAConfig
from models.kandinsky_pipelines import VQVAEImageProcessor
from datasets import get_cls_dataset_by_name
from utils.viz import create_image_pil_grid
import torch.nn as nn


logger = get_logger(__name__, log_level="INFO")


def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    clip_pixel_values = torch.stack(
        [example["clip_pixel_values"] for example in examples]
    )
    clip_pixel_values = clip_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()
    return {"pixel_values": pixel_values, "clip_pixel_values": clip_pixel_values}


def save_model_card(
    repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {base_model}
    tags:
    - kandinsky
    - text-to-image
    - diffusers
    - diffusers-training
    - lora
    inference: true
    ---
        """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


class LoRAAttnAddedKVProcessorDeprecated(LoRAAttnAddedKVProcessor):
    r"""
    Processor for implementing the LoRA attention mechanism with extra learnable key and value matrices for the text
    encoder.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __call__(
        self, attn: Attention, hidden_states: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        self_cls_name = self.__class__.__name__
        # deprecate(
        #     self_cls_name,
        #     "0.26.0",
        #     (
        #         f"Make sure use {self_cls_name[4:]} instead by setting"
        #         "LoRA layers to `self.{to_q,to_k,to_v,add_k_proj,add_v_proj,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
        #         " `LoraLoaderMixin.load_lora_weights`"
        #     ),
        # )
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        attn._modules.pop("processor")
        attn.processor = AttnAddedKVProcessor()
        return attn.processor(attn, hidden_states, **kwargs)


class LayerNames(Enum):
    xattn = auto()
    xattnp = auto()
    all_linear = auto()
    all_layers = auto()


def get_target_modules(mode: LayerNames, unet: UNet2DConditionModel):
    unet_modules = list(unet.named_modules())
    modules = []

    if mode == LayerNames.xattn:
        modules = ["to_k", "to_q", "to_v", "to_out.0"]

    elif mode == LayerNames.xattnp:
        modules = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]

    elif mode == LayerNames.all_linear:
        for name, module in unet_modules:
            if isinstance(module, nn.Linear):
                modules.append(name)

    elif mode == LayerNames.all_layers:
        for name, module in unet_modules:
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                modules.append(name)
    return modules


@pyrallis.wrap()
def main(cfg: KandinskyLoRAConfig):

    run_name = "debug"
    if cfg.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

        if cfg.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

    accelerator_project_config = ProjectConfiguration(
        total_limit=cfg.checkpoints_total_limit
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    project_suffix = cfg.dataset.name
    cfg.num_gpus = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(
            f"kandinsky-fine-tune-lora-{project_suffix}", config=asdict(cfg)
        )

        if cfg.report_to == "wandb":
            run_name = wandb.run.name

        cfg.output_dir = cfg.output_dir / run_name
        logging_dir = cfg.output_dir / cfg.logging_dir

        accelerator.project_configuration.project_dir = str(cfg.output_dir)
        accelerator.project_configuration.logging_dir = str(logging_dir)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

        if cfg.push_to_hub:
            repo_id = create_repo(
                repo_id=cfg.hub_model_id or Path(cfg.output_dir).name,
                exist_ok=True,
                token=cfg.hub_token,
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_decoder_model_name_or_path, subfolder="scheduler"
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        cfg.pretrained_prior_model_name_or_path, subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        cfg.pretrained_prior_model_name_or_path, subfolder="image_encoder"
    )
    prior = KandinskyV22PriorPipeline.from_pretrained(
        cfg.pretrained_prior_model_name_or_path
    )

    movq: VQModel = VQModel.from_pretrained(
        cfg.pretrained_decoder_model_name_or_path, subfolder="movq"
    )

    movq_processor = VQVAEImageProcessor()

    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_decoder_model_name_or_path, subfolder="unet"
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    movq.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    movq.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    prior.to(accelerator.device, dtype=weight_dtype)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        init_lora_weights="gaussian",
        # target_modules=get_target_modules(LayerNames.xattnp, unet),
        target_modules=get_target_modules(LayerNames.all_layers, unet),
    )

    # Add adapter and make sure the trainable params are in float32.
    lora_net = get_peft_model(copy.deepcopy(unet), unet_lora_config)
    # lora_net.encoder_hid_proj.requires_grad_(True)
    lora_layers = filter(lambda p: p.requires_grad, lora_net.parameters())

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_cls = torch.optim.AdamW
    scaled_lr = cfg.learning_rate * cfg.train_batch_size * cfg.num_gpus / (7 * 32)
    optimizer = optimizer_cls(
        lora_layers,
        lr=scaled_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            # transforms.CenterCrop((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    train_ds, val_ds = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[train_transform, val_transform]
    )

    if cfg.dataset.subset is not None:
        subset_indices = list(range(cfg.dataset.subset))
        train_ds = Subset(train_ds, subset_indices)

    def clip_collate_fn(examples):
        pixel_values = torch.stack([image for image, *label in examples])
        clip_pixel_values = image_processor(
            pixel_values, return_tensors="pt", do_rescale=False
        ).pixel_values

        return pixel_values, clip_pixel_values

    train_dataloader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        collate_fn=clip_collate_fn,
        num_workers=cfg.dataloader_num_workers,
    )

    val_dataloader = DataLoader(val_ds, batch_size=1, collate_fn=clip_collate_fn)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
    )
    # Prepare everything with our `accelerator`.
    val_dataloader, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        val_dataloader, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        cfg.train_batch_size
        * accelerator.num_processes
        * cfg.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_ds)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        lora_net.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_net):
                # Convert images to latent space
                images, clip_images = batch
                images = images.to(weight_dtype)
                clip_images.to(weight_dtype)

                images = movq_processor.preprocess(
                    images, height=cfg.dataset.img_size, width=cfg.dataset.img_size
                )
                latents = movq.encode(images).latents
                # image_embeds = image_encoder(clip_images).image_embeds

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                target = noise

                # Predict the noise residual and compute loss
                added_cond_kwargs = {"image_embeds": image_embeds}

                model_pred = lora_net(
                    noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs
                ).sample[:, :4]

                if cfg.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - cfg.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        cfg.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

        if accelerator.is_main_process:
            logger.info(f"Running validation... for {cfg.num_validation_images} images")
            # create pipeline
            pipeline = KandinskyV22Pipeline.from_pretrained(
                cfg.pretrained_decoder_model_name_or_path,
                unet=accelerator.unwrap_model(lora_net),
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=accelerator.device)
            if cfg.seed is not None:
                generator = generator.manual_seed(cfg.seed)

            samples, images = [], []
            for i, batch in enumerate(val_dataloader):

                if i >= cfg.num_validation_images:
                    break

                batch_images, clip_images = batch
                clip_images.to(weight_dtype)
                image_embeds = image_encoder(clip_images).image_embeds

                bsz = batch_images.shape[0]
                negative_image_embeds = prior.get_zero_embed(batch_size=bsz)

                samples.append(
                    pipeline(
                        image_embeds=image_embeds,
                        negative_image_embeds=negative_image_embeds,
                        generator=generator,
                    ).images[0]
                )
                images.append(batch_images)

            images = torch.cat(images)
            images_pil = [to_pil_image(image) for image in images]
            concat_samples = images_pil + samples

            # Save or display the image
            grid = create_image_pil_grid(concat_samples, cols=cfg.num_validation_images)

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log({"validation": [wandb.Image(grid)]})

            del pipeline
            torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lora_net = lora_net.to(torch.float32)
        logger.info(f"Saving LoRA weights in {cfg.output_dir}")
        lora_net.save_pretrained(cfg.output_dir)

        if cfg.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=cfg.pretrained_decoder_model_name_or_path,
                dataset_name=cfg.dataset.name,
                repo_folder=cfg.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=cfg.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        # Final inference
        # Load previous pipeline
        pipeline = KandinskyV22Pipeline.from_pretrained(
            cfg.pretrained_decoder_model_name_or_path, torch_dtype=weight_dtype
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        # pipeline.unet.load_attn_procs(cfg.output_dir)
        # pipeline.unet.from_pretrained(cfg.output_dir)
        logger.info(f"Loading LoRA weights from {cfg.output_dir}")
        lora_model = PeftModel.from_pretrained(pipeline.unet, cfg.output_dir)
        pipeline.unet = lora_model

        # run inference for one last time to check pipeline load
        generator = torch.Generator(device=accelerator.device)
        if cfg.seed is not None:
            generator = generator.manual_seed(cfg.seed)

        samples, images = [], []
        for i, batch in enumerate(val_dataloader):

            if i >= cfg.num_validation_images:
                break

            batch_images, clip_images = batch
            clip_images.to(weight_dtype)
            image_embeds = image_encoder(clip_images).image_embeds

            bsz = batch_images.shape[0]
            negative_image_embeds = prior.get_zero_embed(batch_size=bsz)

            samples.append(
                pipeline(
                    image_embeds=image_embeds,
                    negative_image_embeds=negative_image_embeds,
                    generator=generator,
                ).images[0]
            )
            images.append(batch_images)

        images = torch.cat(images)
        images_pil = [to_pil_image(image) for image in images]
        concat_samples = images_pil + samples

        # Save or display the image
        grid = create_image_pil_grid(concat_samples, cols=cfg.num_validation_images)

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log({"validation-pipe": [wandb.Image(grid)]})

    accelerator.end_training()


if __name__ == "__main__":
    main()
