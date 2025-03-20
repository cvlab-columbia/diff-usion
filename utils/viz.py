import math
import torch
from PIL import Image
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def plot_grid(
    sample: torch.Tensor,
    save_path: Path,
    nrow: int = 4,
    max_samples: Optional[int] = None,
):
    sample = sample.detach().cpu()[:max_samples, ...]
    grid = make_grid(sample, nrow=nrow)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    im = Image.fromarray(grid)
    im.save(save_path)
    return im


def plot_grid_with_probs(
    sample: torch.Tensor,
    probs: torch.Tensor,  # New argument for probability scores
    save_path: Path,
    nrow: int = 4,
    padding: int = 2,
    class_idx: Optional[int] = None,
    max_samples: Optional[int] = None,
    probs_only: bool = False,
    prob_text_prefix: str = "Prob:",
    pad_value: float = 0.0,
):
    sample = sample.detach().cpu()[:max_samples, ...]
    if probs is not None:
        probs = probs.detach().cpu()[:max_samples, ...]
    use_class_idx = class_idx is not None

    # Create the grid
    grid = make_grid(sample, nrow=nrow, padding=padding, pad_value=pad_value)
    grid = grid.permute(1, 2, 0)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)

    # Convert to PIL Image
    im = Image.fromarray(grid)
    draw = ImageDraw.Draw(im)

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 48)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the size of each image in the grid
    _, _, width, height = sample.shape

    # Add text to each image in the grid
    for i in range(min(len(sample), nrow * (len(sample) // nrow))):
        row = i // nrow
        col = i % nrow
        x = col * (width + padding)  # +2 for the gap between images
        y = row * (height + padding)

        if probs is not None:
            prob = probs[i]
            if use_class_idx is False:
                class_idx = torch.argmax(prob).item()
                if len(prob.shape) == 0:
                    max_prob = prob.item()
                else:
                    max_prob = prob[class_idx].item()

            if probs_only is True:
                text = f"{prob_text_prefix} {max_prob:.2f}"
            else:
                text = f"Class: {class_idx}\nProb: {max_prob:.2f}"

            # Add a semi-transparent background for better readability
            text_bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(text_bbox, fill=(255, 255, 255, 128))

            # Draw the text
            draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))

    im.save(save_path)
    return im


def plot_new_row_with_probs(
    sample: torch.Tensor,
    probs: torch.Tensor,
    save_path: Path,
    nrows: int = 1,
    ncols: int = 1,
    dpi: float = 300,
    base_fontsize: int = 24,
    max_samples: Optional[int] = None,
):
    sample = sample.detach().cpu()[:max_samples, ...].permute(0, 2, 3, 1)
    if probs is not None:
        probs = probs.detach().cpu()[:max_samples, ...]

    width, height = sample.shape[1:3]
    figsize = (width / dpi, height / dpi)  # Convert pixels to inches
    fontsize = int(base_fontsize * 100 / dpi)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1])
    )
    if nrows == 1 and ncols == 1:
        # single image
        sample = sample[0]
        axes.imshow(sample)
        axes.axis("off")

        if probs is not None:
            probs = probs[0]
            prob = probs.item()
            # Display probability in top-left corner
            axes.text(
                0.02,
                0.98,
                f"{prob:.2f}",
                transform=axes.transAxes,
                ha="left",
                va="top",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
                fontsize=fontsize,
            )
    else:
        # single row
        for i, img in enumerate(sample):
            axes[i].imshow(img)
            axes[i].axis("off")

            if probs is not None:
                prob = probs[i].item()
                # Display probability in top-left corner
                axes[i].text(
                    0.02,
                    0.98,
                    f"{prob:.2f}",
                    transform=axes[i].transAxes,
                    ha="left",
                    va="top",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
                    fontsize=fontsize,
                )

    # Save the figure
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def plot_logits_and_predictions(
    logits: torch.Tensor, probs: torch.Tensor, save_path: Path, max_samples: int = 8
):
    num_samples = min(logits.shape[0], max_samples)
    num_classes = logits.shape[1]

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        try:
            logit_values = logits[i].detach().cpu().numpy()
            prob_values = probs.detach().cpu().numpy()[i]
        except:
            print(1)

        prediction = np.argmax(prob_values)

        # Plot logits
        ax[0].bar(
            np.arange(num_classes),
            logit_values,
            color="blue",
            alpha=0.6,
            label="Logits",
        )
        ax[0].set_xticks(np.arange(num_classes))
        ax[0].set_xlabel("Classes")
        ax[0].set_ylabel("Logits")
        ax[0].set_title(f"Sample {i + 1} - Prediction: Class {prediction}")
        ax[0].legend()

        # Highlight the predicted class
        ax[1].bar(
            np.arange(num_classes),
            prob_values,
            color="red",
            alpha=0.6,
            label="Probs",
        )
        ax[1].set_xticks(np.arange(num_classes))
        ax[1].set_xlabel("Classes")
        ax[1].set_ylabel("Probs")
        ax[1].set_title(f"Sample {i + 1} - Prediction: Class {prediction}")
        ax[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return fig


def create_gif_from_images(image_objects, output_path, duration=500):
    """
    Create a GIF from a list of PIL Image objects.

    Args:
    image_objects (list of PIL.Image.Image): List of PIL Image objects.
    output_path (str): Path to save the output GIF.
    duration (int): Duration of each frame in milliseconds.
    """
    # Ensure all images are in RGB mode
    images = [img.convert("RGB") for img in image_objects]

    # Save images as a GIF
    images[0].save(
        output_path, save_all=True, append_images=images[1:], duration=duration, loop=0
    )


def create_gif_from_sequence_of_batches(
    image_batches: list[torch.Tensor],
    gif_path: Path,
    duration: int = 500,
    save_input: bool = False,
):
    """
    Create a GIF from a sequence of batches of images.

    Parameters:
    - image_batches (list of torch.Tensor): A list where each element is a batch of images (shape: B x H x W x C).
    - gif_path (str): Path to save the output GIF.
    - duration (int): Duration of each frame in milliseconds.
    """
    frames = []  # To store GIF frames

    for batch in image_batches:
        # Ensure the batch is a torch tensor
        batch = batch.detach().cpu()
        bsz = batch.shape[0]

        # Concatenate images in the batch row-wise
        grid = make_grid(batch, nrow=bsz)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        combined_row = Image.fromarray(grid)

        # Convert the row to a PIL Image and append to frames
        frames.append(combined_row)

    # Save as a GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    if save_input is True:
        # Save input row
        input_row_path = gif_path.parent / f"input_{gif_path.stem}.png"
        frames[0].save(input_row_path)


def plot_probs_bar(probs: list[float]):
    num_vecs = len(probs)
    fig = plt.figure(figsize=(3, 6))
    ind = list(range(num_vecs))
    plt.bar(ind, probs)
    plt.xticks(ind, list(map(lambda x: str(x), ind)))
    return fig


def pil_grid(image1: Image, image2: Image) -> Image:
    # Get the width and height of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with a width equal to the sum of the two images' widths
    # and a height equal to the maximum height of the two images
    new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))

    # Paste the first image on the left side
    new_image.paste(image1, (0, 0))

    # Paste the second image on the right side
    new_image.paste(image2, (width1, 0))
    return new_image


def create_image_pil_grid(
    images, cols=1, padding=1, input_padding: Optional[int] = None
):
    """
    Create a grid of images using only Pillow.

    Args:
    images (list): List of PIL Image objects
    cols (int, optional): Number of columns in the grid
    padding (int, optional): Padding between images and around the grid

    Returns:
    PIL.Image: A new image containing the grid of input images
    """
    # Ensure all images are the same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Resize all images to the maximum dimensions
    resized_images = [img.resize((max_width, max_height)) for img in images]

    # Calculate grid dimensions
    rows = math.ceil(len(images) / cols)
    grid_width = cols * max_width + (cols + 1) * padding
    if input_padding is not None:
        grid_width += input_padding
    grid_height = rows * max_height + (rows + 1) * padding

    # Create a new image with white background
    grid_img = Image.new("RGB", (grid_width, grid_height), color="white")

    # Paste images into the grid
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        x = col * (max_width + padding) + padding
        if input_padding is not None:
            if col > 0:
                x = col * (max_width + padding) + padding + input_padding
        y = row * (max_height + padding) + padding
        grid_img.paste(img, (x, y))

    return grid_img
