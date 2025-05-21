import os
import torch
import pyrallis
import shutil
from typing import Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from textual_inversion_config import KandinskyEvalConfig
from utils.metrics import ensemble_predict

import numpy as np
import pandas as pd


def add_text_to_image(image, text):
    """Add text to the top of an image"""
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    # Get a font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
    except:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_width = draw.textlength(text, font=font)
    x = (image.width - text_width) // 2
    y = 10

    # Add white text with black outline for visibility
    outline_color = "black"
    text_color = "white"
    for adj in range(-2, 3):
        for adj2 in range(-2, 3):
            draw.text((x + adj, y + adj2), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)

    return image


def format_manip(value):
    """
    Format manipulation values according to specific rules:
    - Converts 1.0 to "1"
    - Keeps 1.5 as "1.5"
    - Converts 2.0 to "2"

    Args:
        value (float): The value to format

    Returns:
        str: The formatted string representation
    """
    # Check if the value is effectively an integer (e.g., 1.0, 2.0)
    if float(value).is_integer():
        return str(int(float(value)))
    # Otherwise keep the decimal representation
    return str(float(value))


def create_gifs(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_dir: Path,
    target_class: int,
    output_dir: Path,
    samples_dir: Path,
    use_predictions: bool = True,
    eval_clf_dir: Optional[Path] = None,
):
    """Create GIFs from original and best generated images"""

    # Load classifiers if using predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifiers = []
    if use_predictions:
        eval_clf_dir = eval_clf_dir / Path(str(dataset_name))
        classifiers = [
            torch.load(model_path, map_location=device)
            for model_path in eval_clf_dir.glob("*.pth")
        ]
        for clf in classifiers:
            clf.eval()

    # Set up paths based on dataset
    orig_dir = dataset_dir
    if dataset_name == "afhq":
        classes = ["dog", "cat"]
    elif dataset_name == "kikibouba":
        classes = ["kiki", "bouba"]
    elif dataset_name == "butterfly":
        classes = ["Monarch", "Viceroy"]
    elif dataset_name == "kermany":
        classes = ["DRUSEN", "NORMAL"]
    elif dataset_name == "madsane":
        classes = ["mad", "sane"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create output directory if it doesn't exist
    gif_dir = Path(output_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Filter dataframe for target class
    df_filtered = df[df["target"] == target_class]
    # filter only with filename that have 'BEST' in it
    df_filtered = df_filtered[df_filtered["filename"].str.contains("BEST")]

    # Process each row in the filtered dataframe
    for index, row in df_filtered.iterrows():
        filename = row["filename"]
        # Extract the original filename
        if "generated_" in filename:
            parts = filename.split("_", 3)
            if len(parts) >= 3:
                original_filename = parts[3]

                # Get original image path
                if target_class == 0:  # class0 to class1
                    orig_path = orig_dir / classes[0] / original_filename
                else:  # class1 to class0
                    orig_path = orig_dir / classes[1] / original_filename

                # Find the best image for this file
                best_file_pattern = f"BEST_*{original_filename}*"
                best_files = list(samples_dir.glob(best_file_pattern))
                if best_files and orig_path.exists():
                    best_path = best_files[0]  # Take the first match if multiple exist

                    # Load images
                    if orig_path.suffix == ".npy":
                        orig_img = np.load(orig_path)
                        if len(orig_img.shape) == 2:  # Handle grayscale images
                            orig_img = np.stack([orig_img] * 3, axis=2)
                        orig_img = Image.fromarray(orig_img).convert("RGB")
                    else:
                        orig_img = Image.open(orig_path).convert("RGB")

                    gen_img = Image.open(best_path).convert("RGB")

                    if use_predictions:
                        # Get classifier predictions
                        transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Resize((512, 512)),
                            ]
                        )

                        orig_img_tensor = transform(orig_img).to(device)[None]
                        orig_pred = ensemble_predict(
                            classifiers, orig_img_tensor
                        ).probs.item()
                        gen_img_tensor = transform(gen_img).to(device)[None]
                        gen_pred = ensemble_predict(
                            classifiers, gen_img_tensor
                        ).probs.item()
                    else:
                        # Use real labels
                        orig_pred = target_class
                        gen_pred = 1 - target_class  # opposite class

                    # Add class labels
                    orig_img = add_text_to_image(
                        orig_img.resize((512, 512)), f"Prob: {orig_pred:.2f}"
                    )
                    gen_img = add_text_to_image(
                        gen_img.resize((512, 512)), f"Prob: {gen_pred:.2f}"
                    )

                    # Create and save gif
                    gif_path = (
                        gif_dir
                        / f"{original_filename.replace('.png', '.gif').replace('.npy', '.gif').replace('.jpg', '.gif').replace('.jpeg', '.gif')}"
                    )

                    orig_img.save(
                        gif_path,
                        save_all=True,
                        append_images=[gen_img],
                        duration=1000,  # 1 second per frame
                        loop=0,
                    )
                    print(f"Created gif: {gif_path}")


def compress_gifs_folder(folder_path):
    """Compress the entire gifs folder into a zip file"""
    folder_path = Path(folder_path)
    zip_path = folder_path.with_suffix(".zip")

    print(f"Compressing {folder_path} to {zip_path}")
    shutil.make_archive(str(folder_path), "zip", folder_path)
    print(f"Compression complete. Zip file saved at: {zip_path}")


@pyrallis.wrap()
def main(cfg: KandinskyEvalConfig):
    """Main function that reads parameters from the config file"""
    dataset_name = cfg.dataset.name
    image_dir = cfg.dataset.image_dir
    eval_clf_dir = cfg.eval_clf_weights
    num_images = cfg.num_images
    os.makedirs("gifs", exist_ok=True)

    # Process each checkpoint
    for ckpt in cfg.ckpt:
        print(f"Processing checkpoint {ckpt}")

        # Path to the samples directory
        samples_dir = Path(
            f"{cfg.output_dir}/num_images_{num_images}/samples_ckpt_{ckpt}"
        )

        # Path to the report CSV
        report_path = Path(
            f"{cfg.output_dir}/num_images_{num_images}/report_ckpt_{ckpt}.csv"
        )

        # Check if report exists
        if not report_path.exists():
            print(f"Report file not found: {report_path}")
            continue

        # Load the report
        df = pd.read_csv(report_path)

        # Create output directory for this checkpoint
        output_base_dir = Path(f"gifs/{dataset_name}_ckpt_{ckpt}")

        # Process each target class
        for target in [0, 1]:
            output_dir = output_base_dir / f"target_{target}"
            create_gifs(
                df=df,
                dataset_name=dataset_name,
                dataset_dir=image_dir,
                target_class=target,
                output_dir=output_dir,
                ckpt=ckpt,
                samples_dir=samples_dir,
                use_predictions=True,
                eval_clf_dir=eval_clf_dir,
            )


if __name__ == "__main__":
    main()
