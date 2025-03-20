import torch
import pyrallis
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from textual_inversion.datasets_ti import get_class_from_string
from models.inversion_pipelines import CounterfactualSDPipeline
from diffusers.schedulers import DDIMScheduler
from peft import PeftModel
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import to_pil_image
from textual_inversion_config import SDEvalConfig
from utils.viz import plot_grid, plot_grid_with_probs
from utils.metrics import compute_lpips_similarity, ensemble_predict
from utils.analyze_sweeps import analyze_ddpmef_results
import pandas as pd
import numpy as np


def add_embeddings_to_pipeline(loaded_embeddings: dict, pipeline):
    tokens = list(loaded_embeddings.keys())
    embeddings = list(loaded_embeddings.values())

    # Add tokens to tokenizer
    num_added_tokens = pipeline.tokenizer.add_tokens(tokens)
    if num_added_tokens != len(tokens):
        raise ValueError(
            f"The tokenizer already contains the tokens {tokens}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    token_ids = pipeline.tokenizer.convert_tokens_to_ids(tokens)

    with torch.no_grad():
        for token_id, embedding in zip(token_ids, embeddings):
            pipeline.text_encoder.get_input_embeddings().weight.data[
                token_id
            ] = embedding

    return pipeline


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


@pyrallis.wrap()
def main(cfg: SDEvalConfig):
    device_id = cfg.device
    model_id = "CompVis/stable-diffusion-v1-4"

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(0)

    pipeline: CounterfactualSDPipeline = CounterfactualSDPipeline.from_pretrained(
        model_id
    )
    # overrided when using EF-DDPM Inversion methods
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # load LoRA if exist
    try:
        lora_model = PeftModel.from_pretrained(pipeline.unet, cfg.lora_weights_dir)
        pipeline.unet = lora_model
        print(f"loading lora weights from {cfg.lora_weights_dir}")
    except (OSError, TypeError):
        pass

    # load TI embeddings if exist
    learned_embeds = None
    tokenizer = pipeline.tokenizer
    placeholder_tokens = cfg.dataset.classes
    if cfg.text_embeds_path is not None:
        learned_embeds = torch.load(cfg.text_embeds_path)
        placeholder_tokens = list(learned_embeds.keys())
        # Register the embeddings
        pipeline = add_embeddings_to_pipeline(learned_embeds, pipeline)
        tokenizer = pipeline.tokenizer
        print(f"loading TI embeds from {cfg.text_embeds_path}")

    # load eval classifiers
    if cfg.eval_clf_weights.is_dir:
        classifiers = [
            torch.load(model_path, map_location="cpu").to(device)
            for model_path in cfg.eval_clf_weights.glob("*.pth")
        ]
    else:
        eval_clf = torch.load(cfg.eval_clf_weights, map_location="cpu").to(device)
        classifiers = [eval_clf]

    # Move models to GPU if available
    pipeline.to(device)

    val_transforms = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    val_ds_0 = get_class_from_string(cfg.dataset.name)(
        root_dir=cfg.dataset.image_dir,
        split="val",
        tokenizer=tokenizer,
        placeholder_token=placeholder_tokens,
        num_vectors=cfg.num_vectors,
        file_list_path=cfg.file_list_paths[0],
        classes=[cfg.dataset.classes[0]],
        transform=val_transforms,
    )
    val_ds_1 = get_class_from_string(cfg.dataset.name)(
        root_dir=cfg.dataset.image_dir,
        split="val",
        tokenizer=tokenizer,
        placeholder_token=placeholder_tokens,
        num_vectors=cfg.num_vectors,
        file_list_path=cfg.file_list_paths[1],
        classes=[cfg.dataset.classes[1]],
        transform=val_transforms,
    )

    if cfg.dataset.subset is not None:
        subset_indices = list(range(cfg.dataset.subset))
        val_ds_0 = Subset(val_ds_0, subset_indices)
        val_ds_1 = Subset(val_ds_1, subset_indices)

    if learned_embeds is not None:
        prompt_0 = " ".join(val_ds_0.placeholder_token[: cfg.num_vectors])
        prompt_1 = " ".join(val_ds_1.placeholder_token[cfg.num_vectors :])
    else:
        prompt_0, prompt_1 = cfg.dataset.classes

    data_loader_0 = DataLoader(val_ds_0, batch_size=cfg.batch_size)
    data_loader_1 = DataLoader(val_ds_1, batch_size=cfg.batch_size)

    data_loaders = [data_loader_0, data_loader_1]
    prompts = [prompt_0, prompt_1]

    gs_source = 3.5
    gs_targets = [15]
    t_skips = list(np.linspace(0.0, 0.9, 10))

    rows_all = []
    samples_save_dir = cfg.output_dir / "samples"
    samples_save_dir.mkdir(exist_ok=True, parents=True)

    for i, data_loader in enumerate(data_loaders):
        batch = next(iter(data_loader))
        images = batch["pixel_values"].to(device)
        labels = batch["label"]
        # targets = flip/unflip labels
        # targets = 1 - labels
        targets = labels
        img_paths = batch["img_path"]

        src_prompts = [f"a photo of a {prompts[i]}"] * len(images)
        target_prompts = [f"a photo of a {prompts[1-i]}"] * len(images)

        with torch.no_grad():
            inv_latents, zs = pipeline.ef_ddpm_inversion(
                prompt=src_prompts,
                image=images,
                guidance_scale=gs_source,
                generator=generator,
            )

            for gs_tar in gs_targets:
                for t_skip in t_skips:
                    t_skip = round(t_skip, 3)
                    samples = pipeline.sample_ddpm(
                        latents=inv_latents,
                        zs=zs,
                        prompt=target_prompts,
                        guidance_scale=gs_tar,
                        T_skip=t_skip,
                        generator=generator,
                    ).images

                    # for report
                    classifiers_preds = ensemble_predict(classifiers, samples)
                    lpips = compute_lpips_similarity(images, samples, reduction=None)

                    experiment = f"skip_{t_skip}_cfgtar_{gs_tar}"
                    experiment_rows = get_rows_for_report(
                        img_paths=img_paths,
                        cls_prefix=cfg.dataset.classes[i],
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
                            samples_save_dir / f"{Path(img_path).name}_{experiment}.png"
                        )
                        image.save(save_path)

    df = pd.DataFrame(rows_all)
    report_path = cfg.output_dir / "report.csv"
    print(f"report save in {str(report_path)}")
    df.to_csv(report_path)

    if cfg.analyze_results:
        log_path = report_path.parent / "log.txt"
        results, flip_rate = analyze_ddpmef_results(df, log_path)
        analyzed_report_path = report_path.parent / f"analyzed-{report_path.name}"
        results.to_csv(analyzed_report_path)
        print(f"Flip rate: {flip_rate:.2%}")


if __name__ == "__main__":
    main()
