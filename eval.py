import PIL
import math
import torch
import random
import pyrallis
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from diffusers.schedulers import DDIMScheduler
from models.kandinsky_pipelines import KandinskyV22PipelineWithInversion, ManipulateMode
from peft import PeftModel
from textual_inversion_config import KandinskyEvalConfig
from datasets import get_cls_dataset_by_name
from utils.metrics import compute_lpips_similarity
from utils.metrics import ensemble_predict
from utils.viz import plot_grid_with_probs
from utils.analyze_sweeps import analyze_results_like_baseline, analyze_results_from_best
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import numpy as np
import pandas as pd
from diffusers import (
    AutoPipelineForText2Image,
    DDPMScheduler,
    UNet2DConditionModel,
    VQModel,
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
)
import os

# reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def get_rows_for_report(
    img_paths: list[str],
    cls_prefix: str,
    experiment: str,
    lpips: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    avg_probs: torch.Tensor,
):
    rows = []
    for i, img_path in enumerate(img_paths):
        filename = f"generated_{cls_prefix}_{Path(img_path).name}"
        row = {
            "filename": filename,
            "experiment": experiment,
            "lpips": lpips[i].item(),
            "pred": preds[i].item(),
            "avg_pred": avg_probs[i].item(),
            "target": targets[i].item(),
        }
        rows.append(row)
    return rows


def get_direction_sign(idx: int):
    if idx == 0:
        sign = -1
    elif idx == 1:
        sign = 1
    else:
        raise ValueError("Currently two direction are supported in this script")
    return sign


# Add this function to check if a prediction has flipped
def has_prediction_flipped(orig_preds, new_preds):
    """Check if any prediction has flipped from one class to another."""
    return ((orig_preds.preds > 0.5) != (new_preds.preds > 0.5)).any().item()


@pyrallis.wrap()
def main(cfg: KandinskyEvalConfig):
    device_id = cfg.device
    print(f"using device {device_id}")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(0)

    pipeline: KandinskyV22PipelineWithInversion = (
        KandinskyV22PipelineWithInversion.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder"
        )
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior"
    )

    prior.to(device)

    # Move models to GPU if available
    pipeline.to(device)
    pipeline.image_encoder.to(device)


    # load eval classifiers
    if cfg.eval_clf_weights.is_dir:
        classifiers = [
            torch.load(model_path, map_location="cpu").to(device)
            for model_path in cfg.eval_clf_weights.glob("*.pth")
        ]
    else:
        eval_clf = torch.load(cfg.eval_clf_weights, map_location="cpu").to(device)
        classifiers = [eval_clf]


    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    cfg.dataset.file_list_path = cfg.file_list_paths[0]
    orig_classes = cfg.dataset.classes
    cfg.dataset.classes = [orig_classes[0]]
    _, val_ds_0 = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[transform, transform]
    )

    cfg.dataset.file_list_path = cfg.file_list_paths[1]
    cfg.dataset.classes = [orig_classes[1]]
    _, val_ds_1 = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[transform, transform]
    )

    data_loader_0 = DataLoader(val_ds_0, batch_size=cfg.batch_size)
    data_loader_1 = DataLoader(val_ds_1, batch_size=cfg.batch_size)
    data_loaders = [data_loader_0, data_loader_1]


    gs_inversion = 2
    gs_targets = [4]
    t_skips = list(np.linspace(0.9, 0.0, 10))
    manipulation_scales = [2]
    modes = [ManipulateMode.cond_avg]

    if cfg.lora_weights_dir is not None:
        ckpts = cfg.ckpt
    else:
        ckpts = [0]


    num_images = cfg.num_images
    

    for ckpt in ckpts:
        
        pos_embeds = torch.load(
        cfg.clip_image_embeds_dir / f"{cfg.embed_filenames[0]}.pt",
        map_location="cpu",
        ).to(device)[:num_images]
        neg_embeds = torch.load(
            cfg.clip_image_embeds_dir / f"{cfg.embed_filenames[1]}.pt",
            map_location="cpu",
        ).to(device)[:num_images]


        rows_all = []
        os.makedirs(cfg.output_dir / f"num_images_{num_images}", exist_ok=True)
        samples_save_dir = cfg.output_dir / f"num_images_{num_images}" / f"samples_ckpt_{ckpt}"
        samples_save_dir.mkdir(exist_ok=True, parents=True)
        pipeline: KandinskyV22PipelineWithInversion = (
        KandinskyV22PipelineWithInversion.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder"
        )
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior"
        )

        prior.to(device)

        # Move models to GPU if available
        pipeline.to(device)
        pipeline.image_encoder.to(device)
        try:
            if cfg.lora_weights_dir is not None:
                if ckpt != 0:   
                    newpath = cfg.lora_weights_dir / f"checkpoint-{ckpt}"
                    lora_model = PeftModel.from_pretrained(pipeline.unet, newpath)
                    pipeline.unet = lora_model
                    print(f"loading lora weights from {newpath}")
            else:
                print("No LoRA weights")
                
        except (OSError, TypeError) as e:
            print(f"Error loading LoRA weights: {e}")

        for i, data_loader in enumerate(data_loaders):
            for batch in data_loader:
                print(i)
                direction_sign = get_direction_sign(i)
                images, labels, img_paths = batch

                images = images.to(device)
                labels = labels.to(device)
                targets = labels

                with torch.no_grad():
                    inputs = pipeline.image_processor(
                        images=images, return_tensors="pt", do_rescale=False
                    ).to(device)
                    image_embeds = pipeline.image_encoder(**inputs).image_embeds
                    orig_image_embeds = image_embeds.clone()

                    # Get original predictions
                    orig_preds = ensemble_predict(classifiers, images)

                    norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    norm_pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
                    norm_neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
                    norm_pos_embeds_mean = norm_pos_embeds.mean(0)
                    norm_neg_embeds_mean = norm_neg_embeds.mean(0)

                    # as an p2p0, invert with zero prompt
                    inv_latents, zs = pipeline.ef_ddpm_inversion(
                        source_embeds=None, image=images, generator=generator
                    )
                    torch.cuda.empty_cache()

                    # Flag to track if we found a flip for this batch
                    found_flip = False
                    best_t_skip = None
                    best_samples = None
                    best_preds = None

                    for mode in modes:
                        if found_flip:
                            break
                            
                        for gs_tar in gs_targets:
                            if found_flip:
                                break
                                
                            for m_scale in manipulation_scales:
                                if found_flip:
                                    break
                                    
                                for t_skip in t_skips:
                                    t_skip = round(t_skip, 3)
                                    print(f"Trying t_skip: {t_skip}")

                                    if mode == ManipulateMode.cond_avg:
                                        image_embeds = (
                                            norm_image_embeds
                                            + direction_sign
                                            * m_scale
                                            * (norm_pos_embeds_mean - norm_neg_embeds_mean)
                                        )

                                        image_embeds = image_embeds * orig_image_embeds.norm(
                                            dim=-1, keepdim=True
                                        )

                                    elif mode == ManipulateMode.target_cfg_avg:
                                        bsz = orig_image_embeds.shape[0]
                                        image_embeds = [
                                            orig_image_embeds,
                                            pos_embeds.repeat(bsz, 1),
                                            neg_embeds.repeat(bsz, 1),
                                        ]

                                    elif mode == ManipulateMode.sliders:
                                        image_embeds = orig_image_embeds

                                    samples = pipeline.sample_ddpm(
                                        latents=inv_latents,
                                        zs=zs,
                                        target_embeds=image_embeds,
                                        guidance_scale=gs_tar,
                                        T_skip=t_skip,
                                        output_type="pt",
                                        mode=mode,
                                        manipulation_scale=m_scale,
                                    ).images

                                    # Check if predictions have flipped
                                    new_preds = ensemble_predict(classifiers, samples)
                                    flipped = has_prediction_flipped(orig_preds, new_preds)
                                    
                                    # for report
                                    lpips = compute_lpips_similarity(
                                        images, samples, reduction=None
                                    )

                                    experiment = f"skip_{t_skip}_manip_{m_scale}_cfgtar_{gs_tar}_mode_{mode}"
                                    experiment_rows = get_rows_for_report(
                                        img_paths=img_paths,
                                        cls_prefix=orig_classes[i],
                                        experiment=experiment,
                                        lpips=lpips,
                                        preds=new_preds.preds,
                                        avg_probs=new_preds.probs,
                                        targets=targets,
                                    )
                                    rows_all += experiment_rows

                                    # Save the samples
                                    for img_path, sample in zip(img_paths, samples):
                                        image = to_pil_image(sample)
                                        save_path = (
                                            samples_save_dir
                                            / f"{Path(img_path).name}_{experiment}.png"
                                        )
                                        image.save(save_path)
                                    
                                    # If we found a flip, save the best parameters and break
                                    if flipped:
                                        print(f"Found flip at t_skip: {t_skip}")
                                        found_flip = True
                                        best_t_skip = t_skip
                                        best_samples = samples
                                        best_preds = new_preds
                                        break
                    
                    # If we found a flip, add a special entry to indicate the best parameters
                    if found_flip:
                        best_experiment = f"BEST_skip_{best_t_skip}_manip_{m_scale}_cfgtar_{gs_tar}_mode_{mode}"
                        best_lpips = compute_lpips_similarity(
                            images, best_samples, reduction=None
                        )
                        
                        best_rows = get_rows_for_report(
                            img_paths=img_paths,
                            cls_prefix=f"BEST_{orig_classes[i]}",
                            experiment=best_experiment,
                            lpips=best_lpips,
                            preds=best_preds.preds,
                            avg_probs=best_preds.probs,
                            targets=targets,
                        )
                        rows_all += best_rows
                        
                        # Save the best samples with a special prefix
                        for img_path, sample in zip(img_paths, best_samples):
                            image = to_pil_image(sample)
                            save_path = (
                                samples_save_dir
                                / f"BEST_{Path(img_path).name}_{best_experiment}.png"
                            )
                            image.save(save_path)

        df = pd.DataFrame(rows_all)
        os.makedirs(cfg.output_dir / f"num_images_{num_images}", exist_ok=True)
        report_path = cfg.output_dir / f"num_images_{num_images}" / f"report_ckpt_{ckpt}.csv"
        print(f"report save in {str(report_path)}")
        #load the dataframe from report_path
        #df = pd.read_csv(report_path)
        df.to_csv(report_path)

        if cfg.analyze_results:
            log_path = report_path.parent / f"log_ckpt_{ckpt}.txt"
            samples_dir = cfg.output_dir / f"num_images_{num_images}" / f"samples_ckpt_{ckpt}"
            
            # Use the new function that analyzes based on BEST_ files
            results_dfs, flip_rates = analyze_results_from_best(df, samples_dir, log_path)
            
            # Save each manipulation's results to a separate CSV
            for manip_val, results_df in results_dfs.items():
                analyzed_report_path = report_path.parent / f"analyzed-manip{manip_val}-{report_path.name}"
                results_df.to_csv(analyzed_report_path)
                print(f"Manipulation {manip_val} flip rate: {flip_rates[manip_val]:.2%}")

if __name__ == "__main__":
    main()