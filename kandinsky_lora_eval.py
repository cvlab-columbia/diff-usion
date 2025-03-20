import PIL
import torch
import random
import pyrallis
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from diffusers.schedulers import DDIMInverseScheduler, DDIMScheduler, DDPMScheduler
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from peft import PeftModel
from textual_inversion_config import KandinskyEvalConfig
from datasets import get_cls_dataset_by_name
from utils.manipulations import (
    find_nearest_neighbors,
    compute_weighted_similarity_average,
)
from utils.viz import pil_grid, plot_grid, create_image_pil_grid, plot_grid_with_probs
from torchvision.transforms.functional import to_pil_image
from utils.templates import IMAGENET_CLIP_TEMPLATES
import numpy as np


@pyrallis.wrap()
def main(cfg: KandinskyEvalConfig):
    device_id = cfg.device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    prior: KandinskyV22PriorPipeline = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior"
    )

    pipeline = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder"
    )

    image_processor = CLIPImageProcessor.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
    )

    # Move models to GPU if available
    pipeline.to(device)
    image_encoder.to(device)
    prior.to(device)

    try:
        lora_model = PeftModel.from_pretrained(pipeline.unet, cfg.lora_weights_dir)
        pipeline.unet = lora_model
        print(f"loading lora weights from {cfg.lora_weights_dir}")
    except (TypeError, OSError):
        pass

    pos_embeds = torch.load(
        cfg.clip_image_embeds_dir / f"{cfg.embed_filenames[0]}.pt",
        map_location="cpu",
    ).to(device)
    neg_embeds = torch.load(
        cfg.clip_image_embeds_dir / f"{cfg.embed_filenames[1]}.pt",
        map_location="cpu",
    ).to(device)

    pos_embeds_mean = pos_embeds.mean(0).unsqueeze(0)
    neg_embeds_mean = neg_embeds.mean(0).unsqueeze(0)

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    _, val_ds = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[transform, transform]
    )

    data_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    manipulation_scale = 1.0
    all_samples = []
    for seed in range(1):
        for images, *_ in tqdm(data_loader):
            images = images.to(device)
            bsz = images.shape[0]
            negative_image_embeds = prior.get_zero_embed(batch_size=bsz).to(device)

            generator = torch.Generator(device="cpu").manual_seed(seed)

            with torch.no_grad():
                inputs = image_processor(
                    images=images, return_tensors="pt", do_rescale=False
                ).to(device)

                orig_image_embeds = image_encoder(**inputs).image_embeds
                # orig_image_embeds = prior(
                #     prompt=["a photo of a person"],
                #     generator=generator,
                #     num_images_per_prompt=bsz
                # ).image_embeds

                # image_embeds = orig_image_embeds + manipulation_scale * (
                #     pos_embeds_mean - neg_embeds_mean
                # )
                pos_embeds = prior(
                    prompt=[f"{x} old person" for x in IMAGENET_CLIP_TEMPLATES],
                    generator=generator,
                ).image_embeds

                neg_embeds = prior(
                    prompt=[f"{x} young person" for x in IMAGENET_CLIP_TEMPLATES],
                    generator=generator,
                ).image_embeds

                pos_embeds_mean = pos_embeds.mean(0)
                neg_embeds_mean = neg_embeds.mean(0)

                image_embeds = orig_image_embeds + manipulation_scale * (
                    pos_embeds_mean - neg_embeds_mean
                )

                # weighted average
                # pos_weighted = compute_weighted_similarity_average(
                #     orig_image_embeds, neg_embeds, temperature=0.05
                # )
                # image_embeds = pos_weighted

                # norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                #     dim=-1, keepdim=True
                # )

                # norm_neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
                # norm_neg_embeds_mean = norm_neg_embeds.mean(0)

                # image_embeds = norm_image_embeds + manipulation_scale * (
                #     pos_weighted - norm_neg_embeds_mean
                # )

                image_embeds = (
                    image_embeds
                    / image_embeds.norm(dim=-1, keepdim=True)
                    * orig_image_embeds.norm(dim=-1, keepdim=True)  # orig scaling term
                )

                samples = pipeline(
                    image_embeds=image_embeds,
                    negative_image_embeds=negative_image_embeds,
                    num_inference_steps=cfg.num_inference_steps,
                    output_type="pil",
                    generator=generator,
                ).images

                # images = pipeline(
                #     image_embeds=orig_image_embeds,
                #     negative_image_embeds=negative_image_embeds,
                #     num_inference_steps=cfg.num_inference_steps,
                #     output_type="pil",
                #     generator=generator,
                # ).images

            all_samples += samples
            break

    images_pil = [to_pil_image(image) for image in images]
    # images_pil = images
    concat_samples = images_pil + all_samples

    # Save or display the image
    grid = create_image_pil_grid(concat_samples, cols=cfg.batch_size)
    prefix = f"prior_real_age_manip{manipulation_scale}"
    save_path = cfg.output_dir / f"{prefix}.png"
    grid.save(save_path)


if __name__ == "__main__":
    main()
