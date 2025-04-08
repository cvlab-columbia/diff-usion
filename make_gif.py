import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import pdb
import shutil
from datasets import get_cls_dataset_by_name
from textual_inversion_config import DatasetConfig
import torchvision.transforms.v2 as transforms_v2
import random
from utils.metrics import ensemble_predict


def create_comparison_grid(df, df_sd, df_sd_ti, df_sliders,dataset_name, target_class, output_path, images_per_row=10, ckpt=1000, manip_scale=0.0):


    df_pred = pd.read_csv(f"results/eval/kandinsky_sweeps/reports/{dataset_name}_ckpt_{ckpt}/analyzed-manip{manip_scale}-report.csv", index_col=0)
    df_sd_pred = pd.read_csv(f"results/eval/sd_sweeps/efddpm/{dataset_name}/file_list_alex_lpips/report.csv", index_col=0)
    df_sd_ti_pred = pd.read_csv(f"results/eval/sd_sweeps/ti-efddpm/{dataset_name}/file_list_alex_lpips/report.csv", index_col=0)
    df_sliders_pred = pd.read_csv(f"results/eval/sliders_sweeps/reports/{dataset_name}/report.csv", index_col=0)


    
    # Filter dataframes for target class
    df_filtered = df[df['target'] == target_class]
    df_sd_filtered = df_sd[df_sd['target'] == target_class]
    df_sd_ti_filtered = df_sd_ti[df_sd_ti['target'] == target_class]

    
    #pdb.set_trace()
    
    """Create a grid comparing different methods"""
    
    # Set up paths based on dataset
    if dataset_name == "afhq":
        orig_dir = Path("/proj/vondrick4/datasets/data/afhq/val")
        classes = ['dog', 'cat']
    elif dataset_name == "kikibouba":
        orig_dir = Path("/proj/vondrick4/mia/kiki_bouba_v2_split/val")
        classes = ['kiki', 'bouba']
    elif dataset_name == "butterfly":
        orig_dir = Path("/proj/vondrick/datasets/magnification/butterfly/test")
        classes = ["Monarch", "Viceroy"]
    elif dataset_name == "kermany":
        orig_dir = Path("/proj/vondrick/datasets/magnification/kermany/test")
        classes = ["DRUSEN", "NORMAL"]   
    elif dataset_name == "madsane":
        orig_dir = Path("/proj/vondrick/datasets/magnification/black_holes/10K/test")
        classes = ["mad", "sane"]
    elif dataset_name == "inaturalist":
        orig_dir = Path('/proj/vondrick2/utkarsh/datasets/iNat2021/val')
        classes = ["6372", "6375"]
    
    # ... (other dataset paths)
    
    # Filter dataframes for target class
    df_filtered = df[df['target'] == target_class]
    df_sd_filtered = df_sd[df_sd['target'] == target_class]
    df_sd_ti_filtered = df_sd_ti[df_sd_ti['target'] == target_class]
    if target_class == 0:
        df_sliders_filtered = df_sliders[df_sliders['target'] == 1]
    else:
        df_sliders_filtered = df_sliders[df_sliders['target'] == 0]
    

    
    # Create figure with gridspec
    fig = plt.figure(figsize=(50, 8.5))
    gs = fig.add_gridspec(12, images_per_row, height_ratios=[1,4, 1,4, 1,4, 1,4, 1,4, 1,4])
    axes = np.empty((12, images_per_row), dtype=object)
    
    # Create all axes
    for i in range(12):
        for j in range(images_per_row):
            axes[i,j] = fig.add_subplot(gs[i, j])
    
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.05)

    # Set up headers
    headers = ['Input', 'DDPM-EF', 'TIME', 'TI+DDPM-EF', 'Sliders', 'Ours']
    
    for i, header in enumerate(headers):
        # Clear the header row
        for j in range(images_per_row):
            axes[i*2, j].axis('off')
        
        # Add header text to first column
        axes[i*2, 0].text(-0.1, 0.5, header, 
                         horizontalalignment='right',
                         verticalalignment='center',
                         fontsize=12,
                         fontweight='bold')
    for idx, (filename, row) in enumerate(df_filtered.iterrows()):
        if idx >= images_per_row:
            break
            
        if target_class == 0:  # dog to cat
            base_filename = filename.replace(f"generated_{classes[0]}_", "")
            orig_path = orig_dir / classes[0] / base_filename
        else:  # cat to dog
            base_filename = filename.replace(f"generated_{classes[1]}_", "")
            orig_path = orig_dir / classes[1] / base_filename
        
        # Original image
        if ".npy" in str(orig_path):
            orig_img = np.load(orig_path)
            if len(orig_img.shape) == 2:
                orig_img =orig_img[:,:,None].repeat(3, axis=2)
            #orig_img = Image.fromarray(img_array)
        else:
            orig_img = Image.open(orig_path).convert('RGB')

        axes[1, idx].imshow(orig_img)
        axes[1, idx].axis('off')
        
        # DDPM-EF (using parameters from df_sd)
        if filename in df_sd_filtered.index:
            skip = df_sd_filtered.loc[filename, 'skip']
            cfgtar = int(df_sd_filtered.loc[filename, 'cfgtar'])
            #get row where filename = 'filename' and skip = skip
            temp_row = df_sd_pred[df_sd_pred['filename'] == filename]
            temp_row = temp_row[temp_row['experiment'] == f"skip_{skip}_cfgtar_{cfgtar}"]
            pred = temp_row['avg_pred'].values[0] if not temp_row.empty else None
            if dataset_name == "mad_sane":
                ddpmef_path = Path(f"results/eval/sd_sweeps/efddpm/madsane_ckpt/file_list_alex_lpips/samples") / f"{base_filename}_skip_{skip}_cfgtar_{cfgtar}.png"
            else:
                ddpmef_path = Path(f"results/eval/sd_sweeps/efddpm/{dataset_name}/file_list_alex_lpips/samples") / f"{base_filename}_skip_{skip}_cfgtar_{cfgtar}.png"
            # if ddpmef_path.exists():
            #     ddpmef_img = Image.open(ddpmef_path).convert('RGB')
            #     axes[3, idx].imshow(ddpmef_img)
            if ddpmef_path.exists():
                ddpmef_img = Image.open(ddpmef_path).convert('RGB')
                axes[3, idx].imshow(ddpmef_img)
                if pred is not None:
                    axes[3, idx].text(0.5, 0.95, f"Pred: {pred:.2f}", 
                                    horizontalalignment='center',
                                    verticalalignment='top',
                                    transform=axes[3, idx].transAxes,
                                    color='white',
                                    fontsize=10)
        axes[3, idx].axis('off')
        
        # TIME
        time_path = Path(f"/proj/vondrick2/mia/TIME/renamed_output/{dataset_name}/Results/exp-2/CC/all_CF") / f"generated_{base_filename.replace('.npy','.png')}"
        
        if time_path.exists():
            time_img = Image.open(time_path).convert('RGB')
            axes[5, idx].imshow(time_img)
        axes[5, idx].axis('off')
        
        # TI+DDPM-EF (using parameters from df_sd_ti)
        if filename in df_sd_ti_filtered.index:
            skip = df_sd_ti_filtered.loc[filename, 'skip']
            cfgtar = int(df_sd_ti_filtered.loc[filename, 'cfgtar'])
            if dataset_name == "madsane":
                ti_path = Path(f"results/eval/sd_sweeps/ti-efddpm/madsane/file_list_alex_lpips/samples") / f"{base_filename}_skip_{skip}_cfgtar_{cfgtar}.png"
            else:
                ti_path = Path(f"results/eval/sd_sweeps/ti-efddpm/{dataset_name}/file_list_alex_lpips/samples") / f"{base_filename}_skip_{skip}_cfgtar_{cfgtar}.png"
            if ti_path.exists():
                ti_img = Image.open(ti_path).convert('RGB')
                axes[7, idx].imshow(ti_img)
        axes[7, idx].axis('off')
        

        # Sliders
        #pdb.set_trace()
        if filename in df_sliders_filtered.index:
            rank = int(df_sliders_filtered.loc[filename, 'rank'])
            n = int(df_sliders_filtered.loc[filename, 'n'])
            temp_row = df_sliders_pred[df_sliders_pred['filename'] == filename]
            
            if dataset_name == "kikibouba":
                exp_name = f"config_a1.0_r{rank}_n{n}_dataset_kiki_bouba_v2_split_alpha1.0_rank{rank}_noxattn_last.pt"
            elif dataset_name == "madsane":
                exp_name = f"config_a1.0_r{rank}_n{n}_dataset_10K_alpha1.0_rank{rank}_noxattn_last.pt"
            else:
                exp_name = f"config_a1.0_r{rank}_n{n}_dataset_{dataset_name}_alpha1.0_rank{rank}_noxattn_last.pt"

            temp_row = temp_row[temp_row['experiment'] == exp_name + f"/target_{target_class}/2"]
            pred = temp_row['avg_pred'].values[0] if not temp_row.empty else None


            sliders_path = Path(f"/proj/vondrick2/mia/sliders/results/{dataset_name}/{exp_name}/target_{target_class}/2") / base_filename.replace('.npy','.png')
            #pdb.set_trace()
            if sliders_path.exists():
                sliders_img = Image.open(sliders_path).convert('RGB')
                axes[9, idx].imshow(sliders_img)
                if pred is not None:
                    axes[9, idx].text(0.5, 0.95, f"Pred: {pred:.2f}", 
                                    horizontalalignment='center',
                                    verticalalignment='top',
                                    transform=axes[9, idx].transAxes,
                                    color='white',
                                    fontsize=10)
        axes[9, idx].axis('off')
        
        # Ours (using parameters from df)
        skip = row['tksip']
        manip = row['manip']
        if manip == 1.0 or manip == 2.0:
            manip = int(manip)
        cfgtar = int(row['gs_tar'])

        if dataset_name == "madsane":
            ours_path = Path(f"results/eval/kandinsky_sweeps/reports/madsane_ckpt_{ckpt}/samples") / f"{base_filename}_skip_{skip}_manip_{manip}_cfgtar_{cfgtar}_mode_ManipulateMode.cond_avg.png"
        else:
            ours_path = Path(f"results/eval/kandinsky_sweeps/reports/{dataset_name}_ckpt_{ckpt}/samples") / f"{base_filename}_skip_{skip}_manip_{manip}_cfgtar_{cfgtar}_mode_ManipulateMode.cond_avg.png"
        if ours_path.exists():
            ours_img = Image.open(ours_path).convert('RGB')
            axes[11, idx].imshow(ours_img)
        axes[11, idx].axis('off')
        

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def add_text_to_image(image, text):
    """Add text to the top of an image"""
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    # Get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Calculate text size and position
    text_width = draw.textlength(text, font=font)
    x = (image.width - text_width) // 2
    y = 10
    
    # Add white text with black outline for visibility
    outline_color = 'black'
    text_color = 'white'
    for adj in range(-2, 3):
        for adj2 in range(-2, 3):
            draw.text((x+adj, y+adj2), text, font=font, fill=outline_color)
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

def create_gifs(df, dataset_name, target_class, output_dir, ckpt=2000, manip_scale=2.0, use_predictions=True):
    """Create GIFs from original and best generated images"""
    
    # Load classifiers if using predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifiers = []
    if use_predictions:
        eval_clf_dir = Path("/proj/vondrick2/mia/diff-usion/results/ensemble") / Path(str(dataset_name))
        classifiers = [
            torch.load(model_path, map_location=device)
            for model_path in eval_clf_dir.glob("*.pth")
        ]
        for clf in classifiers:
            clf.eval()

    # Set up paths based on dataset
    if dataset_name == "afhq":
        orig_dir = Path("/proj/vondrick4/datasets/data/afhq/val")
        classes = ['dog', 'cat']
    elif dataset_name == "kikibouba":
        orig_dir = Path("/proj/vondrick4/mia/kiki_bouba_v2_split/val")
        classes = ['kiki', 'bouba']
    elif dataset_name == "butterfly":
        orig_dir = Path("/proj/vondrick/datasets/magnification/butterfly/test")
        classes = ["Monarch", "Viceroy"]
    elif dataset_name == "kermany":
        orig_dir = Path("/proj/vondrick/datasets/magnification/kermany/test")
        classes = ["DRUSEN", "NORMAL"]   
    elif dataset_name == "madsane":
        orig_dir = Path("/proj/vondrick/datasets/magnification/black_holes/10K/test")
        classes = ["mad", "sane"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create output directory if it doesn't exist
    gif_dir = Path(output_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to the samples directory
    samples_dir = Path(f"results/eval/kandinsky_sweeps/reports/{dataset_name}_ckpt_num_images/num_images_10000/samples_ckpt_{ckpt}")
    
    # Filter dataframe for target class
    
    df_filtered = df[df['target'] == target_class]
    
    # Process each row in the filtered dataframe
    for filename, row in df_filtered.iterrows():
        # Extract the original filename
        if "generated_" in filename:
            #import pdb; pdb.set_trace()
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
                    if orig_path.suffix == '.npy':
                        orig_img = np.load(orig_path)
                        if len(orig_img.shape) == 2:  # Handle grayscale images
                            orig_img = np.stack([orig_img] * 3, axis=2)
                        orig_img = Image.fromarray(orig_img).convert('RGB')
                    else:
                        orig_img = Image.open(orig_path).convert('RGB')
                    
                    gen_img = Image.open(best_path).convert('RGB')
                    
                    if use_predictions:
                        # Get classifier predictions
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((512, 512)),
                        ])
                        
                        orig_img_tensor = transform(orig_img).to(device)[None]
                        orig_pred = ensemble_predict(classifiers, orig_img_tensor).probs.item()
                        gen_img_tensor = transform(gen_img).to(device)[None]
                        gen_pred = ensemble_predict(classifiers, gen_img_tensor).probs.item()
                    else:
                        # Use real labels
                        orig_pred = target_class
                        gen_pred = 1 - target_class  # opposite class
                    
                    # Add class labels
                    orig_img = add_text_to_image(orig_img.resize((512,512)), f"Prob: {orig_pred:.2f}")
                    gen_img = add_text_to_image(gen_img.resize((512,512)), f"Prob: {gen_pred:.2f}")
                    
                    # Create and save gif
                    gif_path = gif_dir / f"{original_filename.replace('.png', '.gif').replace('.npy', '.gif').replace('.jpg', '.gif').replace('.jpeg', '.gif')}"
                    
                    orig_img.save(
                        gif_path,
                        save_all=True,
                        append_images=[gen_img],
                        duration=1000,  # 1 second per frame
                        loop=0
                    )
                    print(f"Created gif: {gif_path}")

def create_class_transition_gifs(df, dataset_name, output_dir, use_predictions=True, ckpt=1000, manip_scale=0.0):
    """Create gifs transitioning between class 0 and class 1 inputs from training set"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if dataset_name == "afhq":
        cfg = DatasetConfig(name=dataset_name, image_dir=Path("/proj/vondrick4/datasets/data/afhq"), classes=['dog', 'cat'])
    elif dataset_name == "kikibouba":
        cfg = DatasetConfig(name=dataset_name, image_dir=Path("/proj/vondrick4/mia/kiki_bouba_v2_split"), classes=['kiki', 'bouba'])
    elif dataset_name == "butterfly":
        cfg = DatasetConfig(name=dataset_name, image_dir=Path("/proj/vondrick/datasets/magnification/butterfly"), classes=["Monarch", "Viceroy"])
    elif dataset_name == "kermany":     
        cfg = DatasetConfig(name=dataset_name, image_dir=Path("/proj/vondrick/datasets/magnification/kermany"), classes=["DRUSEN", "NORMAL"])
    elif dataset_name == "madsane":
        cfg = DatasetConfig(name="mad_sane", image_dir=Path("/proj/vondrick/datasets/magnification/black_holes/10K"), classes=["mad", "sane"])
        cfg.classes = ["mad", "sane"]
    elif dataset_name == "inaturalist":
        cfg = DatasetConfig(name="inaturalist", image_dir=Path("/proj/vondrick2/utkarsh/datasets/iNat2021"), classes=["6372", "6375"])
        cfg.classes = [6372, 6375]
    # Create transform
    transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((512, 512)),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ])
    
    # Get training dataset
    cfg.file_list_path = None
    train_dataset, _ = get_cls_dataset_by_name(cfg, [transform, transform])
    print("Got training dataset")
    
    # Get the first 50 images from each class directly from dataset's stored lists
    labels = np.array(train_dataset.labels)
    class0_mask = labels == 0
    class1_mask = labels == 1
    
    # Get the first 50 indices for each class
    class0_indices = np.where(class0_mask)[0]
    class1_indices = np.where(class1_mask)[0]    
    # Get images using indices
    class0_images = [(train_dataset[i][0], train_dataset[i][2]) for i in class0_indices[:50]]
    class1_images = [(train_dataset[i][0], train_dataset[i][2]) for i in class1_indices[:50]]
    
    print(f"Collected {len(class0_images)} images for class 0")
    print(f"Collected {len(class1_images)} images for class 1")

    eval_clf_dir = Path("/proj/vondrick2/mia/magnificationold/results/ensemble") / Path(str(dataset_name))

    classifiers = [
        torch.load(model_path, map_location=device)
        for model_path in eval_clf_dir.glob("*.pth")
    ]

    # Create gifs for the first 50 pairs
    for idx in range(50):
        img0, path0 = class0_images[idx]
        img1, path1 = class1_images[idx]

        img0_tensor = img0[None].to(device)
        img1_tensor = img1[None].to(device)

      
        # Get classifier predictions
        if use_predictions:
            img0_pred = ensemble_predict(classifiers, img0_tensor).preds.item()
            img1_pred = ensemble_predict(classifiers, img1_tensor).preds.item()
        else:
            img0_pred = 0
            img1_pred = 1

        
        # Convert tensor to PIL Image
        img0_pil = transforms_v2.ToPILImage()(img0)
        img1_pil = transforms_v2.ToPILImage()(img1)


        
        # Add class labels
        img0_with_text = add_text_to_image(img0_pil, f"Class {img0_pred}")
        img1_with_text = add_text_to_image(img1_pil, f"Class {img1_pred}")
        
        # Create and save gif
        gif_path = output_dir / f"{idx}.gif"
        img0_with_text.save(
            gif_path,
            save_all=True,
            append_images=[img1_with_text],
            duration=1000,
            loop=0
        )
        
        # Save original images to a single null folder with mixed ordering
        null_dir = output_dir.parent / "null"
        null_dir.mkdir(exist_ok=True, parents=True)
        img0_pil.save(null_dir / f"{idx*2}.png")     # Even indices for class 0
        img1_pil.save(null_dir / f"{idx*2+1}.png")   # Odd indices for class 1
    
    # Save test images from both classes in a single folder with random ordering
    test_dir = output_dir.parent / "test"
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Get images for test sets (indices 50-74 for 25 images per class)
    test_indices_0 = class0_indices[50:75]  # 25 images
    test_indices_1 = class1_indices[50:75]  # 25 images
    
    # Collect all test images and their labels
    test_images = []
    for test_idx in test_indices_0:
        img, _ = train_dataset[test_idx][0], train_dataset[test_idx][2]
        test_images.append((img, 0))  # class 0
        
    for test_idx in test_indices_1:
        img, _ = train_dataset[test_idx][0], train_dataset[test_idx][2]
        test_images.append((img, 1))  # class 1
    
    # Randomly shuffle the images
    random.seed(42)  # for reproducibility
    random.shuffle(test_images)
    
    # Save images with numerical indices and create answer key
    answer_key = {}
    for idx, (img, label) in enumerate(test_images):
        img_pil = transforms_v2.ToPILImage()(img)
        save_path = test_dir / f"{idx}.png"
        img_pil.save(save_path)
        answer_key[idx] = label
    
    # Save answer key to text file
    with open(test_dir.parent / "answer_key.txt", "w") as f:
        f.write("image_index,class_label\n")  # header
        for idx, label in sorted(answer_key.items()):
            f.write(f"{idx},{label}\n")
        
    print(f"Saved 50 randomly shuffled test images (25 per class) in {test_dir}")
    print(f"Answer key saved in {test_dir.parent}/answer_key.txt")
    print(f"Saved 100 mixed null images (50 per class) in {null_dir}")

def compress_gifs_folder(folder_path):
    """Compress the entire gifs folder into a zip file"""
    folder_path = Path(folder_path)
    zip_path = folder_path.with_suffix('.zip')
    
    print(f"Compressing {folder_path} to {zip_path}")
    shutil.make_archive(str(folder_path), 'zip', folder_path)
    print(f"Compression complete. Zip file saved at: {zip_path}")


if __name__ == "__main__":
    dataset_name = "butterfly"
    manip_scale = 2.0
    num_images = 10000
    ckpt = 1900
    os.makedirs("gifs", exist_ok=True)

    df = pd.read_csv(f"results/eval/kandinsky_sweeps/reports/{dataset_name}_ckpt_num_images/num_images_{num_images}/analyzed-manip{manip_scale}-report_ckpt_{ckpt}.csv", index_col=0)

    for target in [0, 1]:
        output_dir = Path(f"gifs/{dataset_name}_manip_{manip_scale}_ckpt_{ckpt}/target_{target}")
        create_gifs(df, dataset_name, target, output_dir,ckpt=ckpt, manip_scale=manip_scale, use_predictions=True)
