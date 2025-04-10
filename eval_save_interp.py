import PIL
import math
import os
import torch
import random
import pyrallis
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from diffusers.schedulers import DDIMScheduler
from diffusers import KandinskyV22PriorPipeline
from models.kandinsky_pipelines import KandinskyV22PipelineWithInversion, ManipulateMode
from peft import PeftModel
from utils.metrics import ensemble_predict
from textual_inversion_config import KandinskyEvalConfig
from datasets import (
    get_cls_dataset_by_name,
    IMAGENET_CLIP_TEMPLATES,
    ImagesBase,
)
from utils.manipulations import (
    find_nearest_neighbors,
    compute_weighted_similarity_average,
)
from utils.viz import (
    pil_grid,
    plot_grid,
    create_image_pil_grid,
    plot_grid_with_probs,
    plot_new_row_with_probs,
)

import torch.nn.functional as F
import numpy as np


# reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


@pyrallis.wrap()
def main(cfg: KandinskyEvalConfig):
    device_id = cfg.device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(0)

    pipeline: KandinskyV22PipelineWithInversion = (
        KandinskyV22PipelineWithInversion.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder"
        )
    )
    # overrided when using EF-DDPM Inversion methods
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move models to GPU if available
    pipeline.to(device)
    pipeline.image_encoder.to(device)
    pipeline.prior.to(device)

    try:
        lora_model = PeftModel.from_pretrained(pipeline.unet, str(cfg.lora_weights_dir) + "/checkpoint-" + str(cfg.ckpt))
        pipeline.unet = lora_model
        print(f"loading lora weights from {cfg.lora_weights_dir}")
    except (OSError, TypeError):
        pass

    # load eval classifiers
    if cfg.eval_clf_weights.is_dir:
        classifiers = [
            torch.load(model_path, map_location="cpu").to(device)
            for model_path in cfg.eval_clf_weights.glob("*.pth")
        ]
    else:
        eval_clf = torch.load(cfg.eval_clf_weights, map_location="cpu").to(device)
        classifiers = [eval_clf]

    pos_embeds = torch.load(
        cfg.clip_image_embeds_dir / f"{cfg.embed_filenames[0]}.pt",
        map_location="cpu",
    ).to(device)
    neg_embeds = torch.load(
        cfg.clip_image_embeds_dir / f"{cfg.embed_filenames[1]}.pt",
        map_location="cpu",
    ).to(device)

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    cfg.dataset.file_list_path = cfg.file_list_paths[1]
    _, val_ds = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[transform, transform]
    )

    data_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    gs_targets = [4]
    t_skips = [0.7]
    t_skips = sorted(t_skips, reverse=True)
    manipulation_scales = [-1, 0, 1, 1.5, 2]  # eta in our equations
    modes = [ManipulateMode.cond_avg]

    cfg.output_dir = cfg.output_dir / Path("num_images_" + str(cfg.num_images)) / Path("interp_ckpt_" + str(cfg.ckpt))
    os.makedirs(cfg.output_dir, exist_ok=True)

    for images, label, img_path in tqdm(data_loader):
        images = images.to(device)

        with torch.no_grad():
            inputs = pipeline.image_processor(
                images=images, return_tensors="pt", do_rescale=False
            ).to(device)
            image_embeds = pipeline.image_encoder(**inputs).image_embeds
            orig_image_embeds = image_embeds.clone()

            norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                dim=-1, keepdim=True
            )
            norm_pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
            norm_neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
            norm_pos_embeds_mean = norm_pos_embeds.mean(0)
            norm_neg_embeds_mean = norm_neg_embeds.mean(0)

        inv_latents, zs = pipeline.ef_ddpm_inversion(
            source_embeds=orig_image_embeds,
            # source_embeds=None,
            image=images,
            generator=generator,
            guidance_scale=gs_targets[0],
        )

        sweep_samples = []
        for mode in modes:
            for manip in manipulation_scales:

                if mode == ManipulateMode.cond_avg:

                    # avg
                    image_embeds = norm_image_embeds + manip * (
                        norm_pos_embeds_mean - norm_neg_embeds_mean
                    )

                    # knn
                    # image_embeds = norm_image_embeds + manipulation_scale * (
                    #     norm_knn_pos_embeds - norm_image_embeds
                    # )

                    image_embeds = image_embeds * orig_image_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    # recon
                    # image_embeds = orig_image_embeds

                elif mode == ManipulateMode.target_cfg_avg:
                    bsz = orig_image_embeds.shape[0]
                    # image_embeds = [
                    #     orig_image_embeds,
                    #     pos_embeds_mean.repeat(bsz, 1),
                    #     neg_embeds_mean.repeat(bsz, 1),
                    # ]

                    #### kNN avg embeds ####
                    knn_pos_embeds = find_nearest_neighbors(
                        orig_image_embeds, pos_embeds, k=5
                    )
                    knn_neg_embeds = find_nearest_neighbors(
                        orig_image_embeds, neg_embeds, k=5
                    )
                    image_embeds = [
                        orig_image_embeds,
                        knn_pos_embeds.mean(1),
                        knn_neg_embeds.mean(1),
                    ]

                for gs_tar in gs_targets:
                    for t_skip in t_skips:
                        with torch.no_grad():
                            samples = pipeline.sample_ddpm(
                                latents=inv_latents,
                                zs=zs,
                                target_embeds=image_embeds,
                                guidance_scale=gs_tar,
                                T_skip=t_skip,
                                output_type="pt",
                                mode=mode,
                                manipulation_scale=manip,
                                image=images,
                            ).images
                            sweep_samples.append(samples)

            concat_samples = torch.cat([images] + sweep_samples)
            # compute probs
            classifiers_preds = ensemble_predict(classifiers, concat_samples)
            probs = classifiers_preds.probs

            # Save or display the image
            prefix = f"{Path(img_path[0]).name}".split(".")[0]
            save_path = (
                cfg.output_dir
                / f"{prefix}_gsi{gs_tar}_gs{gs_tar}_gsm{manipulation_scales[0]}_{manipulation_scales[-1]}_tskips{t_skips[0]}_{t_skips[-1]}.png"
            )
            save_path_probs = (
                cfg.output_dir
                / f"{prefix}_gsi{gs_tar}_gs{gs_tar}_gsm{manipulation_scales[0]}_{manipulation_scales[-1]}_tskips{t_skips[0]}_{t_skips[-1]}_probs.png"
            )

            # save input image
            # plot_grid_with_probs(
            #     sample=images,
            #     probs=None,
            #     save_path=cfg.output_dir / f"{prefix}.png",
            #     probs_only=True,
            #     nrow=1,
            #     padding=0,
            #     pad_value=1.0,
            #     prob_text_prefix="",
            # # )

            # plot_new_row_with_probs(
            #     sample=images, probs=None, save_path=cfg.output_dir / f"{prefix}.png"
            # )

            # # save input image with probs
            # # plot_grid_with_probs(
            # #     sample=images,
            # #     probs=probs[0].unsqueeze(0),
            # #     save_path=cfg.output_dir / f"{prefix}_probs.png",
            # #     probs_only=True,
            # #     nrow=1,
            # #     padding=0,
            # #     pad_value=1.0,
            # #     prob_text_prefix="",
            # # )

            # plot_new_row_with_probs(
            #     sample=images,
            #     probs=probs[0].unsqueeze(0),
            #     save_path=cfg.output_dir / f"{prefix}_probs.png",
            # )

            # save interp
            # plot_grid_with_probs(
            #     sample=concat_samples[1:],
            #     probs=None,
            #     save_path=save_path,
            #     probs_only=True,
            #     nrow=len(manipulation_scales),
            #     padding=10,
            #     pad_value=1.0,
            #     prob_text_prefix="",
            # )
            # plot_new_row_with_probs(
            #     sample=concat_samples[0:],
            #     probs=None,
            #     save_path=save_path,
            #     ncols=len(manipulation_scales)+1,
            # )

            # save interp with probs
            # plot_grid_with_probs(
            #     sample=concat_samples[1:],
            #     probs=probs[1:],
            #     save_path=save_path_probs,
            #     probs_only=True,
            #     nrow=len(manipulation_scales),
            #     padding=10,
            #     pad_value=1.0,
            #     prob_text_prefix="",
            # )
            plot_new_row_with_probs(
                sample=concat_samples,
                probs=probs,
                save_path=save_path_probs,
                ncols=len(manipulation_scales)+1,
            )


if __name__ == "__main__":
    main()
