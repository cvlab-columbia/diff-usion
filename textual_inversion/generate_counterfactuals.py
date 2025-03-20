import torch
import pyrallis
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from textual_inversion.datasets_ti import get_class_from_string
from models.inversion_pipelines import CounterfactualSDPipeline
from diffusers.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms.v2 as transforms
from textual_inversion_config import SDEvalConfig
from utils.viz import plot_grid, plot_grid_with_probs


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

    learned_embeds = None
    tokenizer = pipeline.tokenizer
    placeholder_tokens = cfg.dataset.classes
    if cfg.text_embeds_path is not None:
        learned_embeds = torch.load(cfg.text_embeds_path)
        placeholder_tokens = list(learned_embeds.keys())
        # Register the embeddings
        pipeline = add_embeddings_to_pipeline(learned_embeds, pipeline)
        tokenizer = pipeline.tokenizer

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
        classes=[cfg.dataset.classes[0]],
        transform=val_transforms,
    )
    val_ds_1 = get_class_from_string(cfg.dataset.name)(
        root_dir=cfg.dataset.image_dir,
        split="val",
        tokenizer=tokenizer,
        placeholder_token=placeholder_tokens,
        num_vectors=cfg.num_vectors,
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
    t_skips = [0.6]

    for i, data_loader in enumerate(data_loaders):
        batch = next(iter(data_loader))
        images = batch["pixel_values"].to(device)
        src_prompts = [f"a photo of a {prompts[i]}"] * len(images)
        target_prompts = [f"a photo of a {prompts[1-i]}"] * len(images)

        with torch.no_grad():
            inv_latents, zs = pipeline.ef_ddpm_inversion(
                prompt=src_prompts,
                image=images,
                guidance_scale=gs_source,
                generator=generator,
            )

            sweep_samples = []
            for gs_tar in gs_targets:
                for t_skip in t_skips:
                    samples = pipeline.sample_ddpm(
                        latents=inv_latents,
                        zs=zs,
                        prompt=target_prompts,
                        guidance_scale=gs_tar,
                        T_skip=t_skip,
                        generator=generator,
                    ).images
                    sweep_samples.append(samples)

        concat_samples = torch.cat(
            [images] + sweep_samples[: cfg.num_validation_images]
        )

        # Save or display the image
        prefix = f"{prompts[i]}"
        save_path = (
            cfg.output_dir
            / f"{prefix}_gs{gs_targets[0]}_tskips{t_skips[0]}_{t_skips[-1]}.png"
        )
        plot_grid(
            sample=concat_samples,
            save_path=save_path,
            nrow=cfg.batch_size,
        )


if __name__ == "__main__":
    main()
