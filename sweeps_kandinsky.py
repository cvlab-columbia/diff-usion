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
from classifiers.clip_classifier_train import CLIPClassifier
from textual_inversion_config import KandinskyEvalConfig
from datasets import get_cls_dataset_by_name, Spawrious
from utils.metrics import compute_lpips_similarity, compute_facenet_similarity
from utils.metrics import ensemble_predict
from utils.viz import plot_grid_with_probs, create_gif_from_sequence_of_batches
from utils.manipulations import find_nearest_neighbors
from utils.analyze_sweeps import analyze_results_like_baseline
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import numpy as np
import pandas as pd


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


@pyrallis.wrap()
def main(cfg: KandinskyEvalConfig):
    device_id = cfg.device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(0)

    pipeline: KandinskyV22PipelineWithInversion = (
        KandinskyV22PipelineWithInversion.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder"
        )
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move models to GPU if available
    pipeline.to(device)
    pipeline.image_encoder.to(device)

    try:
        lora_model = PeftModel.from_pretrained(pipeline.unet, cfg.lora_weights_dir)
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

    if cfg.clf_weights is not None:
        clip_clf: CLIPClassifier = torch.load(cfg.clf_weights, map_location="cpu").to(
            device
        )
        clip_clf.eval()

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

    lda_save_path = Path(
        "/proj/vondrick2/orr/projects/magnification/results/lda/butterfly/lda_torch_monarch_train_viceroy_train.pth"
    )
    lda_edit_direction = torch.load(lda_save_path, map_location="cpu").to(device)

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

    gs_inversion = 4
    gs_targets = [4]
    t_skips = list(np.linspace(0.0, 0.9, 10))
    manipulation_scales = [1.0, 1.5, 2.0]
    modes = [ManipulateMode.cond_avg]

    rows_all = []
    samples_save_dir = cfg.output_dir / "samples"
    samples_save_dir.mkdir(exist_ok=True, parents=True)

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

                norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                    dim=-1, keepdim=True
                )
                norm_pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
                norm_neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
                norm_pos_embeds_mean = norm_pos_embeds.mean(0)
                norm_neg_embeds_mean = norm_neg_embeds.mean(0)

                # as an p2p0, invert with zero prompt
                inv_latents, zs = pipeline.ef_ddpm_inversion(
                    source_embeds=orig_image_embeds,
                    # source_embeds=None,
                    image=images,
                    generator=generator,
                    guidance_scale=gs_inversion,
                )
                torch.cuda.empty_cache()

                for mode in modes:
                    for gs_tar in gs_targets:
                        for m_scale in manipulation_scales:
                            for t_skip in t_skips:
                                t_skip = round(t_skip, 3)

                                if mode == ManipulateMode.cond_avg:
                                    # unnormalized version
                                    # image_embeds = orig_image_embeds + m_scale * (
                                    #     pos_embeds - neg_embeds
                                    # )

                                    image_embeds = (
                                        norm_image_embeds
                                        + direction_sign
                                        * m_scale
                                        * (norm_pos_embeds_mean - norm_neg_embeds_mean)
                                    )

                                    # lda
                                    # image_embeds = (
                                    #     norm_image_embeds
                                    #     + direction_sign
                                    #     * m_scale
                                    #     * lda_edit_direction
                                    # )

                                    image_embeds = (
                                        image_embeds
                                        * orig_image_embeds.norm(dim=-1, keepdim=True)
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

                                # for report
                                classifiers_preds = ensemble_predict(
                                    classifiers, samples
                                )
                                lpips = compute_lpips_similarity(
                                    images, samples, reduction=None
                                )

                                experiment = f"skip_{t_skip}_manip_{m_scale}_cfgtar_{gs_tar}_mode_{mode}"
                                experiment_rows = get_rows_for_report(
                                    img_paths=img_paths,
                                    cls_prefix=orig_classes[i],
                                    experiment=experiment,
                                    lpips=lpips,
                                    preds=classifiers_preds.preds,
                                    avg_probs=classifiers_preds.probs,
                                    targets=targets,
                                )
                                rows_all += experiment_rows

                                for img_path, sample in zip(img_paths, samples):
                                    image = to_pil_image(sample)
                                    save_path = (
                                        samples_save_dir
                                        / f"{Path(img_path).name}_{experiment}.png"
                                    )
                                    image.save(save_path)

    df = pd.DataFrame(rows_all)
    report_path = cfg.output_dir / "report.csv"
    print(f"report save in {str(report_path)}")
    df.to_csv(report_path)

    if cfg.analyze_results:
        log_path = report_path.parent / "log.txt"
        results, flip_rate = analyze_results_like_baseline(df, log_path)
        analyzed_report_path = report_path.parent / f"analyzed-{report_path.name}"
        results.to_csv(analyzed_report_path)
        print(f"Flip rate: {flip_rate:.2%}")


if __name__ == "__main__":
    main()
