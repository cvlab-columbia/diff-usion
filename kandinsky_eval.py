import PIL
import math
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
    Spawrious,
    IMAGENET_CLIP_TEMPLATES,
    ImagesBase,
)
from utils.manipulations import (
    find_nearest_neighbors,
    compute_weighted_similarity_average,
    top_k_pca_directions,
    project_embedding_svd,
)
from utils.viz import pil_grid, plot_grid, create_image_pil_grid, plot_grid_with_probs
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
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

    # load classifier
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

    data_embeds = torch.cat([pos_embeds, neg_embeds])

    pos_embeds_mean = pos_embeds.mean(0).unsqueeze(0)
    neg_embeds_mean = neg_embeds.mean(0).unsqueeze(0)

    k = 5
    pca_pos_directions = top_k_pca_directions(pos_embeds, k=k).T
    pca_neg_directions = top_k_pca_directions(neg_embeds, k=k).T

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    cfg.dataset.file_list_path = cfg.file_list_paths[0]
    _, val_ds = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[transform, transform]
    )
    # val_ds = ImagesBase(
    #     root_dir=cfg.dataset.image_dir / cfg.dataset.classes[0], transform=transform
    # )

    # datasets = Spawrious(
    #     benchmark="o2o_hard",
    #     root_dir=cfg.dataset.image_dir,
    #     train_transforms=transform,
    #     test_transforms=transform,
    #     split=0,
    # )
    # val_ds = datasets.get_test_dataset()

    # opposite_label = 0 if val_ds.labels[0] == 1 else 1
    opposite_label = 0
    if cfg.dataset.classes:
        if cfg.dataset.classes[0] == 1990:
            opposite_label = 0

    data_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    alpha = 0.8
    gs_targets = [4]
    t_skips = [0.8]
    t_skips = sorted(t_skips, reverse=True)
    manipulation_scales = [0.7]  # eta in our equations
    modes = [ManipulateMode.cond_avg]

    for images, *_ in tqdm(data_loader):
        images = images.to(device)
        bsz = images.shape[0]

        with torch.no_grad():
            negative_embeds = pipeline.prior.get_zero_embed(batch_size=bsz)
            inputs = pipeline.image_processor(
                images=images, return_tensors="pt", do_rescale=False
            ).to(device)
            image_embeds = pipeline.image_encoder(**inputs).image_embeds
            orig_image_embeds = image_embeds.clone()

            # projection
            proj_image_embeds = project_embedding_svd(image_embeds, pos_embeds, r=20)

            norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                dim=-1, keepdim=True
            )
            norm_pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
            norm_neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
            norm_pos_embeds_mean = norm_pos_embeds.mean(0)
            norm_neg_embeds_mean = norm_neg_embeds.mean(0)

        # inv_latents, zs = pipeline.ef_ddpm_inversion(
        #     source_embeds=orig_image_embeds,
        #     # source_embeds=None,
        #     image=images,
        #     generator=generator,
        #     guidance_scale=gs_targets[0],
        # )

        sweep_samples = []
        for mode in modes:
            for manip in manipulation_scales:
                # for pca_dir in pca_pos_directions:

                if mode == ManipulateMode.cond_avg:
                    # image_embeds = orig_image_embeds + manipulation_scale * (
                    #     pos_embeds_mean - neg_embeds_mean
                    # )
                    # knn_pos_embeds = find_nearest_neighbors(
                    #     orig_image_embeds, pos_embeds, k=10
                    # ).mean(1)
                    # norm_knn_pos_embeds = knn_pos_embeds / knn_pos_embeds.norm(
                    #     dim=-1, keepdim=True
                    # )

                    # avg
                    # image_embeds = norm_image_embeds + manip * (
                    #     norm_pos_embeds_mean - norm_neg_embeds_mean
                    # )
                    # image_embeds = norm_image_embeds + manip * (
                    #     norm_pos_embeds_mean - norm_image_embeds
                    # )

                    # knn
                    # image_embeds = norm_image_embeds + manipulation_scale * (
                    #     norm_knn_pos_embeds - norm_image_embeds
                    # )

                    # lda
                    # image_embeds = norm_image_embeds - manip * lda_edit_direction

                    # pca
                    # norm_pca_dir = pca_dir / pca_dir.norm(dim=-1, keepdim=True)
                    # image_embeds = norm_image_embeds + manip * norm_pca_dir

                    # proj
                    proj_image_embeds_norm = proj_image_embeds / proj_image_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    image_embeds = norm_image_embeds + manip * (
                        proj_image_embeds_norm - norm_image_embeds
                    )

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

                elif mode == ManipulateMode.sliders:
                    lora_model = PeftModel.from_pretrained(
                        pipeline.unet, cfg.lora_weights_dir
                    )
                    pipeline.unet = lora_model
                    print(f"loading lora weights from {cfg.lora_weights_dir}")
                    image_embeds = orig_image_embeds

                for gs_tar in gs_targets:
                    for t_skip in t_skips:
                        with torch.no_grad():
                            # samples = pipeline.sample_ddpm(
                            #     latents=inv_latents,
                            #     zs=zs,
                            #     target_embeds=image_embeds,
                            #     guidance_scale=gs_tar,
                            #     T_skip=t_skip,
                            #     output_type="pt",
                            #     mode=mode,
                            #     manipulation_scale=manip,
                            #     image=images,
                            # ).images
                            samples = pipeline.sample(
                                image_embeds=image_embeds,
                                negative_image_embeds=negative_embeds,
                                guidance_scale=gs_tar,
                                output_type="pt",
                                generator=generator,
                                initial_embeds=orig_image_embeds,
                                start_noise=1000,
                            ).images

                            sweep_samples.append(samples)
        break

    concat_samples = torch.cat([images] + sweep_samples)
    # concat_samples = torch.cat(sweep_samples)

    # compute probs
    # classifiers_preds = ensemble_predict(classifiers, concat_samples)
    # probs = classifiers_preds.probs
    probs = None

    # Save or display the image
    prefix = f"cat2dog_centered_proj_r20_rand_noise"
    save_path = (
        cfg.output_dir
        / f"{prefix}_gsi{gs_tar}_gs{gs_tar}_gsm{manipulation_scales[0]}_{manipulation_scales[-1]}_tskips{t_skips[0]}_{t_skips[-1]}.png"
    )
    plot_grid_with_probs(
        sample=concat_samples,
        probs=probs,
        save_path=save_path,
        probs_only=True,
        nrow=cfg.batch_size,
        # nrow=len(manipulation_scales),
    )


if __name__ == "__main__":
    main()
