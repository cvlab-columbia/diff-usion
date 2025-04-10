import PIL
import math
import torch
import random
import os
import numpy as np
import pandas as pd
import gradio as gr
import threading
import time
import zipfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.v2 as transforms
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from peft import PeftModel, LoraConfig, get_peft_model
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision import models
import torch.nn as nn
from utils.metrics import ensemble_predict
from diffusers import UNet2DConditionModel, VQModel
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torch.nn import functional as F
from diffusers.optimization import get_scheduler
import copy
# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Define constants
IMG_SIZE = 512
BATCH_SIZE = 32
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# CSS for styling the interface
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body, * {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
}

.container {
    max-width: 1360px;
    margin: auto;
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

.header {
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #f0f0f0;
}

.header h1 {
    font-size: 3rem;
    font-weight: 700;
    color: #222;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #B39CD0 0%, #9D8AC7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.header p {
    font-size: 1.1rem;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
}

.subtitle {
    font-size: 0.95rem;
    color: #777;
    max-width: 800px;
    margin: 0.5rem auto 0;
    line-height: 1.5;
}

.contact-info {
    font-size: 0.8rem;
    color: #777;
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px dashed #e0e0e0;
    width: 80%;
    margin-left: auto;
    margin-right: auto;
}

.paper-info {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 1.8rem;
    margin: 1.8rem 0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    border-left: 4px solid #B39CD0;
}

.paper-info h3 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #B39CD0;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}

.paper-info p {
    font-size: 1.05em;
    line-height: 1.7;
    color: #333;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #B39CD0;
    margin: 2.5rem 0 1.5rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid #ECF0F1;
    letter-spacing: -0.02em;
}

.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem;
    border-top: 1px solid #ECF0F1;
    color: #666;
    background-color: #f8f9fa;
    border-radius: 0 0 12px 12px;
}

.btn-primary {
    background-color: #B39CD0 !important;
    border-color: #B39CD0 !important;
    transition: all 0.3s ease;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 8px !important;
}

.btn-primary:hover {
    background-color: #9D8AC7 !important;
    border-color: #9D8AC7 !important;
}

.gallery-item img {
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}

.upload-info {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 4px solid #B39CD0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.upload-info p {
    margin: 0;
    font-size: 1.05em;
    line-height: 1.6;
}

.upload-info strong {
    color: #555;
    font-weight: 600;
}

.parameter-box {
    padding: 1.5rem;
    background-color: #f8f9fa;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border-left: 4px solid #B39CD0;
}

.parameter-box p {
    font-weight: 600;
    color: #333;
    margin-bottom: 12px;
    font-size: 1.1em;
}

.parameter-box ul {
    margin: 5px 0 0 20px;
    padding: 0;
    color: #444;
    line-height: 1.8;
}

.parameter-box li {
    margin-bottom: 6px;
}

/* Form elements styling */
input, select, textarea {
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
    padding: 10px 15px !important;
    transition: all 0.3s ease !important;
}

input:focus, select:focus, textarea:focus {
    border-color: #B39CD0 !important;
    box-shadow: 0 0 0 3px rgba(179, 156, 208, 0.2) !important;
}

label {
    font-weight: 500 !important;
    color: #555 !important;
    margin-bottom: 8px !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .header h1 {
        font-size: 2.2rem;
    }
    .section-header {
        font-size: 1.5rem;
    }
    .paper-info {
        padding: 1.2rem;
    }
    .container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
}
"""

# Function to extract uploaded zip file
def extract_zip(zip_file, extract_dir):
    """Extract a zip file to the specified directory"""
    # Create a temporary directory for extraction
    temp_dir = Path(extract_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Check if the extracted content has class0 and class1 folders
    # If not, try to find them in subdirectories
    class0_dir = temp_dir / "class0"
    class1_dir = temp_dir / "class1"
    
    if not (class0_dir.exists() and class1_dir.exists()):
        # Look for class0 and class1 in subdirectories
        for subdir in temp_dir.iterdir():
            if subdir.is_dir():
                if (subdir / "class0").exists() and (subdir / "class1").exists():
                    # Move the class directories to the temp_dir
                    shutil.move(str(subdir / "class0"), str(class0_dir))
                    shutil.move(str(subdir / "class1"), str(class1_dir))
                    break
    
    # Verify that we have the required directories
    if not (class0_dir.exists() and class1_dir.exists()):
        raise ValueError("The uploaded zip file must contain 'class0' and 'class1' directories or a subdirectory containing them")
    
    return str(temp_dir)

# Define a simple dataset class
class TwoClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class0_dir = self.root_dir / "class0"
        self.class1_dir = self.root_dir / "class1"
        
        # Get image paths
        self.class0_images = list(self.class0_dir.glob("*.*"))
        self.class1_images = list(self.class1_dir.glob("*.*"))
        
        # Create image list and labels
        self.images = self.class0_images + self.class1_images
        self.labels = [0] * len(self.class0_images) + [1] * len(self.class1_images)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, str(img_path)

def compute_lpips_similarity(images1, images2, reduction=None):
    """Compute LPIPS similarity between two batches of images"""
    # This is a placeholder - in a real implementation, you'd use a proper LPIPS model
    # For demo purposes, we'll just return a random similarity score
    batch_size = images1.shape[0]
    similarity = torch.rand(batch_size, device=images1.device)
    
    if reduction == "mean":
        return similarity.mean()
    return similarity

def get_direction_sign(idx: int):
    if idx == 0:
        sign = -1
    elif idx == 1:
        sign = 1
    else:
        raise ValueError("Currently two direction are supported in this script")
    return sign

def add_text_to_image(image, text):
    """Add text to an image at the top with a nicer design"""
    draw = ImageDraw.Draw(image)
    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add a semi-transparent gradient background for better readability
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (200, 30)
    
    # Create gradient background
    for i in range(40):
        alpha = int(180 - i * 4)  # Fade from 180 to 20 alpha
        if alpha < 0:
            alpha = 0
        draw.rectangle([(0, i), (image.width, i)], fill=(0, 0, 0, alpha))
    
    # Draw text at the top of the image
    draw.text((15, 10), text, fill="white", font=font)
    return image

def create_gif(img1, img2, output_path):
    """Create a GIF that alternates between two images"""
    # Increase duration to 1 second per image (1000ms)
    imageio.mimsave(output_path, [img1, img2], duration=1000, loop=0)
    return output_path

def train_classifier(model, train_loader, val_loader, epochs, lr, device, patience=5):
    """Train a classifier model with early stopping"""
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def train_ensemble_classifiers(train_dataset, val_dataset, output_dir, epochs=10, lr=0.001):
    """Train an ensemble of classifiers"""
    classifiers_dir = output_dir / "classifiers"
    classifiers_dir.mkdir(exist_ok=True, parents=True)
    
    # Limit training data if dataset is very large
    if len(train_dataset) > 1000:
        print(f"Limiting training data from {len(train_dataset)} to 1000 samples for faster training")
        indices = torch.randperm(len(train_dataset))[:1000]
        indices_val = torch.randperm(len(val_dataset))[:100]
        train_subset = torch.utils.data.Subset(train_dataset, indices)
        val_subset = torch.utils.data.Subset(val_dataset, indices_val)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Define models for ensemble
    # MobileNetV2 (smallest and fastest model)
    mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 1)
    
    # ResNet18
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    
    # EfficientNet-B0
    efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 1)
    
    # Train each model
    ensemble_models = []
    model_names = ["efficientnet"] #"mobilenet", "resnet",
    models_list = [efficientnet] #mobilenet, resnet, 
    
    for name, model in zip(model_names, models_list):
        print(f"Training {name}...")
        trained_model = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=DEVICE,
            patience=1  # Early stop after 1 epoch without improvement
        )
        
        # Save model
        save_path = classifiers_dir / f"{name}.pth"
        torch.save(trained_model, save_path)
        ensemble_models.append(trained_model)
        
        print(f"Saved {name} to {save_path}")
    
    return ensemble_models

# Main processing function
def process_dataset(data_dir, output_dir, checkpoint_path=None, train_classifiers=True, progress=None):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)
    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(exist_ok=True, parents=True)
    
    if progress:
        progress(0.1, desc="Loading datasets...")
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    
    # Create dataset
    dataset = TwoClassDataset(data_dir, transform=transform)
    
    # Split dataset into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = dataset, dataset #random_split(dataset, [train_size, val_size])
    
    if progress:
        progress(0.2, desc="Processing classifiers...")
    
    # Check for existing classifiers
    classifiers_dir = output_dir / "classifiers"
    classifiers_exist = classifiers_dir.exists() and len(list(classifiers_dir.glob("*.pth"))) >= 3
    
    # Train ensemble classifiers or load pre-trained ones
    if True: #train_classifiers or not classifiers_exist:
        print("Training ensemble classifiers...")
        classifiers = train_ensemble_classifiers(train_dataset, val_dataset, output_dir)
    else:
        classifiers_dir = Path("/proj/vondrick2/mia/diff-usion/results/ensemble/afhq")
        print("Loading pre-trained classifiers...")
        classifiers = []
        for model_path in classifiers_dir.glob("*.pth"):
            model = torch.load(model_path, map_location=DEVICE)
            model.eval()
            classifiers.append(model)
    
    # Create class-specific validation datasets
    val_indices = list(range(len(val_dataset)))
    val_labels = [val_dataset[i][1] for i in val_indices]
    
    val_indices_0 = [i for i, label in zip(val_indices, val_labels) if label == 0]
    val_indices_1 = [i for i, label in zip(val_indices, val_labels) if label == 1]
    
    # Create dataloaders for each class
    val_loader_0 = DataLoader([val_dataset[i] for i in val_indices_0], batch_size=1)
    val_loader_1 = DataLoader([val_dataset[i] for i in val_indices_1], batch_size=1)
    data_loaders = [val_loader_0, val_loader_1]
    
    if progress:
        progress(0.3, desc="Loading models...")
    
    # Load models
    try:
        from models.kandinsky_pipelines import KandinskyV22PipelineWithInversion, ManipulateMode
        
        pipeline = KandinskyV22PipelineWithInversion.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder"
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(DEVICE)
        pipeline.image_encoder.to(DEVICE)
        
        # Load LoRA weights if provided
        if checkpoint_path:
            try:
                lora_model = PeftModel.from_pretrained(pipeline.unet, checkpoint_path)
                pipeline.unet = lora_model
                print(f"Loaded LoRA weights from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
    except ImportError:
        return "Error: Kandinsky models not found. Please install the required dependencies."
    
    if progress:
        progress(0.4, desc="Extracting embeddings...")
    
    # Check for existing embeddings
    embeds_dir = output_dir / "embeddings"
    embeds_dir.mkdir(exist_ok=True, parents=True)
    class0_embeds_path = embeds_dir / Path("class0_embeds.pt")
    class1_embeds_path = embeds_dir / Path("class1_embeds.pt")
    
    if False: #class0_embeds_path.exists() and class1_embeds_path.exists():
        print("Loading existing embeddings...")
        class0_embeds = torch.load(class0_embeds_path, map_location=DEVICE)
        class1_embeds = torch.load(class1_embeds_path, map_location=DEVICE)
    else:
        # Extract embeddings from training set
        print("Extracting embeddings from training set...")
        class0_embeds_list = []
        class1_embeds_list = []

        # Limit the number of samples for embedding extraction
        max_samples_per_class = 500  # Limit to 500 samples per class for faster processing

        # Get indices of class 0 and class 1 samples
        train_indices = list(range(len(train_dataset)))
        train_labels = [train_dataset[i][1] for i in train_indices]

        train_indices_0 = [i for i, label in zip(train_indices, train_labels) if label == 0][:max_samples_per_class]
        train_indices_1 = [i for i, label in zip(train_indices, train_labels) if label == 1][:max_samples_per_class]

        # Create subset for embedding extraction
        embed_indices = train_indices_0 + train_indices_1
        embed_subset = torch.utils.data.Subset(train_dataset, embed_indices)
        embed_loader = DataLoader(embed_subset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Extracting embeddings from {len(embed_subset)} samples ({len(train_indices_0)} class 0, {len(train_indices_1)} class 1)")

        # Process training data in batches
        for images, labels, _ in tqdm(embed_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with torch.no_grad():
                inputs = pipeline.image_processor(
                    images=images, return_tensors="pt", do_rescale=False
                ).to(DEVICE)
                image_embeds = pipeline.image_encoder(**inputs).image_embeds
                
                # Separate embeddings based on labels
                class0_mask = (labels == 0)
                class1_mask = (labels == 1)
                
                if class0_mask.any():
                    class0_embeds_list.append(image_embeds[class0_mask])
                if class1_mask.any():
                    class1_embeds_list.append(image_embeds[class1_mask])
        
        # Concatenate embeddings
        class0_embeds = torch.cat(class0_embeds_list)
        class1_embeds = torch.cat(class1_embeds_list)
        
        # Save embeddings
        torch.save(class0_embeds, class0_embeds_path)
        torch.save(class1_embeds, class1_embeds_path)
        print(f"Saved embeddings to {embeds_dir}")
    
    if progress:
        progress(0.5, desc="Generating counterfactuals...")
    
    # Parameters for manipulation
    gs_inversion = 2
    gs_targets = [4]
    t_skips = [0.85] #list(np.linspace(0.9, 0.5, 4))
    manipulation_scales = [1.0]
    modes = [ManipulateMode.cond_avg]
    generator = torch.Generator(device="cpu").manual_seed(0)
    
    # Process validation data
    results = []
    
    total_batches = sum(len(loader) for loader in data_loaders)
    processed_batches = 0
    
    for i, data_loader in enumerate(data_loaders):
        direction_sign = get_direction_sign(i)
        
        for batch in data_loader:
            images, labels, img_paths = batch
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with torch.no_grad():
                inputs = pipeline.image_processor(
                    images=images, return_tensors="pt", do_rescale=False
                ).to(DEVICE)
                image_embeds = pipeline.image_encoder(**inputs).image_embeds
                orig_image_embeds = image_embeds.clone()
                
                norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                    dim=-1, keepdim=True
                )
                norm_pos_embeds = class0_embeds / class0_embeds.norm(dim=-1, keepdim=True)
                norm_neg_embeds = class1_embeds / class1_embeds.norm(dim=-1, keepdim=True)
                norm_pos_embeds_mean = norm_pos_embeds.mean(0)
                norm_neg_embeds_mean = norm_neg_embeds.mean(0)
                
                # Invert with zero prompt
                inv_latents, zs = pipeline.ef_ddpm_inversion(
                    source_embeds=None, image=images, generator=generator
                )
                
                # Get original predictions
                orig_preds = ensemble_predict(classifiers, images)
                
                # Find the first t_skip value that flips the class
                found_flip = False
                best_t_skip = None
                best_samples = None
                
                for mode in modes:
                    for gs_tar in gs_targets:
                        for m_scale in manipulation_scales:
                            for t_skip in t_skips:
                                print(f"t_skip: {t_skip}")
                                t_skip = round(t_skip, 3)
                                
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
                                
                                # Generate samples
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
                                
                                # Check if class has flipped
                                new_preds = ensemble_predict(classifiers, samples)
                                
                                # Check if any prediction has flipped
                                flipped = ((orig_preds.preds.item() > 0.5) != (new_preds.preds.item() > 0.5))
                                
                                if flipped:
                                    found_flip = True
                                    best_t_skip = t_skip
                                    best_samples = samples
                                    break
                            
                            if found_flip:
                                break
                        
                        if found_flip:
                            break
                
                if found_flip:
                    # Save the results
                    for img_idx, (img_path, sample) in enumerate(zip(img_paths, best_samples)):
                        # Save original and generated images
                        orig_img = to_pil_image(images[img_idx])
                        gen_img = to_pil_image(sample)
                        
                        # Get predictions
                        orig_pred = orig_preds.probs[img_idx].item()
                        gen_pred = new_preds.probs[img_idx].item()
                        
                        # Add text with predictions
                        orig_img_with_text = add_text_to_image(
                            orig_img.copy(), 
                            f"Prob: {orig_pred:.2f}"
                        )
                        gen_img_with_text = add_text_to_image(
                            gen_img.copy(), 
                            f"Prob: {gen_pred:.2f}"
                        )
                        
                        # Create and save GIF
                        base_filename = Path(img_path).name
                        gif_path = gifs_dir / f"{base_filename}.gif"
                        create_gif(orig_img_with_text, gen_img_with_text, gif_path)
                        
                        # Save individual images with text
                        orig_img_with_text.save(samples_dir / f"orig_{base_filename}")
                        gen_img_with_text.save(samples_dir / f"gen_{base_filename}")
                        
                        # Store results
                        results.append({
                            "filename": base_filename,
                            "original_class": labels[img_idx].item(),
                            "original_prob": orig_pred,
                            "generated_prob": gen_pred,
                            "t_skip": best_t_skip,
                            "gif_path": str(gif_path)
                        })
                else:
                    # If no flip was found, use the last t_skip
                    t_skip = t_skips[-1]
                    
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
                    
                    # Generate samples
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
                    
                    new_preds = ensemble_predict(classifiers, samples)
                    
                    # Save the results
                    for img_idx, (img_path, sample) in enumerate(zip(img_paths, samples)):
                        # Save original and generated images
                        orig_img = to_pil_image(images[img_idx])
                        gen_img = to_pil_image(sample)
                        
                        # Get predictions
                        orig_pred = orig_preds.probs[img_idx].item()
                        gen_pred = new_preds.probs[img_idx].item()
                        
                        # Add text with predictions
                        orig_img_with_text = add_text_to_image(
                            orig_img.copy(), 
                            f"Tskip: {t_skip:.2f}, Prob: {orig_pred:.2f}"
                        )
                        gen_img_with_text = add_text_to_image(
                            gen_img.copy(), 
                            f"Tskip: {t_skip:.2f}, Prob: {gen_pred:.2f}"
                        )
                        
                        # Create and save GIF
                        base_filename = Path(img_path).name
                        gif_path = gifs_dir / f"{base_filename}.gif"
                        create_gif(orig_img_with_text, gen_img_with_text, gif_path)
                        
                        # Save individual images with text
                        orig_img_with_text.save(samples_dir / f"orig_{base_filename}")
                        gen_img_with_text.save(samples_dir / f"gen_{base_filename}")
                        
                        # Store results
                        results.append({
                            "filename": base_filename,
                            "original_class": labels[img_idx].item(),
                            "original_prob": orig_pred,
                            "generated_prob": gen_pred,
                            "t_skip": t_skip,
                            "gif_path": str(gif_path)
                        })
            
            # Update progress
            processed_batches += 1
            if progress:
                progress(0.5 + 0.4 * (processed_batches / total_batches), 
                         desc=f"Processing image {processed_batches}/{total_batches}")
    
    if progress:
        progress(0.9, desc="Finalizing results...")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    
    # Return paths to GIFs
    gif_paths = [r["gif_path"] for r in results]
    return gif_paths

# Add the clip_collate_fn function


# Update the train_lora function to use the clip_collate_fn
def train_lora(data_dir, output_dir, num_epochs=5, learning_rate=1e-4, batch_size=32, lora_rank=4, lora_alpha=32, max_train_steps=None):
    """Train a LoRA model for Kandinsky using code from kandinsky_lora_train.py"""
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set device
        device = DEVICE
        print(f"Using device: {device}")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Load models
        weight_dtype = torch.float32
        
        # Load image processor and encoder
        image_processor = CLIPImageProcessor.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_processor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
        ).to(device)
        
        def clip_collate_fn(examples):
            pixel_values = torch.stack([image for image, *_ in examples])
            clip_pixel_values = image_processor(
                pixel_values, return_tensors="pt", do_rescale=False
            ).pixel_values
            
            return pixel_values, clip_pixel_values
        
        # Load scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", subfolder="scheduler"
        )
        
        # Load VQVAE for encoding images
        movq = VQModel.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", subfolder="movq"
        ).to(device)
        
        # Load UNet model
        unet = UNet2DConditionModel.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", subfolder="unet"
        ).to(device)
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        
        # Add LoRA adapters to UNet
        lora_net = get_peft_model(copy.deepcopy(unet), lora_config)
        
        # Prepare dataset
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((512, 512)),  # Use a standard size for CLIP
            transforms.ToDtype(torch.float32, scale=True),
        ])
        
        dataset = TwoClassDataset(data_dir, transform=transform)
        
        # Split dataset into train and validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = dataset, dataset#random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            collate_fn=clip_collate_fn  # Use the custom collate function
        )
        
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            lora_net.parameters(),  # Use lora_net instead of unet
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )
        
        # Set models to training/eval mode
        lora_net.train()
        image_encoder.eval()
        movq.eval()
        
        # Training loop
        global_step = 0
        max_train_steps = 2000
        
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(train_dataloader):
                # Check if we've reached max_train_steps
                if global_step >= max_train_steps:
                    print(f"Reached maximum training steps ({max_train_steps}). Stopping training.")
                    break
                
                if step % 10 == 0:
                    print(f"Step {step}/{len(train_dataloader)} (global step: {global_step})")
                
                # Get batch data
                images, clip_images = batch
                images = images.to(device)
                clip_images = clip_images.to(device, dtype=weight_dtype)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Get image embeddings from CLIP
                with torch.no_grad():
                    image_embeds = image_encoder(clip_images).image_embeds
                    
                    # Get latents from VQVAE
                    latents = movq.encode(images).latents
                    latents = latents.to(weight_dtype)
                    
                    # Add noise to latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                try:
                    # Predict the noise residual and compute loss
                    added_cond_kwargs = {"image_embeds": image_embeds}

                    model_pred = lora_net(
                        noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs
                    ).sample[:, :4]
                    
                    # Compute loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backpropagate
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    # Update step counter
                    global_step += 1
                    
                    # Save checkpoint
                    if global_step % 100 == 0:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(exist_ok=True, parents=True)
                        
                        # Save the model
                        lora_net.save_pretrained(checkpoint_dir)
                        print(f"Saved checkpoint to {checkpoint_dir}")
                
                except Exception as e:
                    print(f"Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Check if we've reached max_train_steps after this epoch
            if global_step >= max_train_steps:
                print(f"Reached maximum training steps ({max_train_steps}). Stopping training.")
                break
            
            # End of epoch
            print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {loss.item():.4f}")
        
        # Save the final model
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        lora_net.save_pretrained(final_dir)
        print(f"Saved final model to {final_dir}")
        
        return str(final_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in LoRA training: {str(e)}")
        return f"Error: {str(e)}"

# Gradio interface
def create_gradio_interface():
    # Global variables for LoRA training
    lora_status = "Ready"
    lora_is_processing = False
    
    # Create temporary directories for uploads
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    lora_temp_dir = Path("./temp_lora_uploads")
    lora_temp_dir.mkdir(exist_ok=True, parents=True)
    
    def background_train_lora_process(zip_file, output_dir):
        nonlocal lora_status, lora_is_processing
        
        try:
            lora_is_processing = True
            lora_status = "Extracting uploaded dataset..."
            
            # Extract the uploaded zip file
            data_dir = extract_zip(zip_file, lora_temp_dir / f"upload_{int(time.time())}")
            
            lora_status = "Initializing LoRA training..."
            
            # Use default parameters
            num_epochs = 2000
            learning_rate = 1e-4
            batch_size = 32
            lora_rank = 4
            lora_alpha = 32
            max_train_steps = 1000  # Set a default max steps
            
            # Run the actual training
            lora_path = train_lora(
                data_dir=data_dir,
                output_dir=output_dir,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                max_train_steps=max_train_steps
            )
            
            lora_status = f"LoRA training completed! Model saved to {lora_path}"
            lora_is_processing = False
            
            # Return the path to the trained model for download
            return lora_status, lora_path
        except Exception as e:
            lora_status = f"Error during LoRA training: {str(e)}"
            lora_is_processing = False
            return lora_status, None
    
    def start_lora_training(zip_file, output_dir):
        nonlocal lora_status
        
        if zip_file is None:
            return "Please upload a dataset zip file", None
        
        # Start training in a background thread
        thread = threading.Thread(
            target=background_train_lora_process,
            args=(zip_file, output_dir)
        )
        thread.daemon = True
        thread.start()
        
        # Return initial status
        return "LoRA training started. This will take a while...", None
    
    def check_lora_status():
        nonlocal lora_status
        return lora_status
    
    def process_uploaded_data(zip_file, output_dir, checkpoint_file, train_clf):
        try:
            if zip_file is None:
                return "Please upload a dataset zip file", []
            
            # Update status
            status_msg = "Extracting uploaded dataset..."
            
            # Extract the uploaded zip file
            data_dir = extract_zip(zip_file, temp_dir / f"upload_{int(time.time())}")
            
            status_msg = "Starting processing..."
            
            # Process the checkpoint file if provided
            #checkpoint_path = '/proj/vondrick2/mia/diff-usion/lora_output/checkpoint-500'
            checkpoint_path =  '/proj/vondrick2/mia/diff-usion/lora_output_lampsfar/checkpoint-800' #None #'/proj/vondrick2/mia/diff-usion/lora_output_birds/checkpoint-300'
            
            # Call the main processing function
            gif_paths = process_dataset(data_dir, output_dir, checkpoint_path, train_clf)
            
            # Update status on completion
            status_msg = "Completed! Generated counterfactuals are displayed below."
            
            return status_msg, gif_paths
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)  # Print to console for debugging
            return error_msg, []
    
    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_classes="container"):
            with gr.Column(elem_classes="header"):
                gr.HTML("""
                    <div class="header">
                        <h1>DIFFusion Demo</h1>
                        <p class="subtitle">Generate fine-grained edits to images using another class of images as guidance.</p>
                        <p class="contact-info">For any questions/comments/issues with this demo, please email mia.chiquier@cs.columbia.edu. Thank you to Lambda Labs for the GPU credits used for this demo. 🤖</p>
                    </div>
                """)
            
            with gr.Column(elem_classes="paper-info"):
                gr.HTML("""
                    <h3>About the Paper</h3>
                    <p>"Teaching Humans Subtle Differences with DIFFusion" introduces a novel approach to generate 
                    fine-grained counterfactuals that help humans understand subtle differences between visually similar classes. The counterfactuals are generated by guiding the edit purely visually.</p>
                """)

            with gr.Column(elem_classes="paper-info"):
                gr.HTML("""
                    <h3>Optimal Usage Conditions & Limitations</h3>
                    <p>Our method works best when the images per class are centered and aligned. However, if you have a large set of images per class (>50), this becomes less important. We do not support the use of our method on images of humans, as it is underexplored.</p>
                """)
            
            
            # Counterfactual Generation Section
            gr.HTML('<div class="section-header">Counterfactual Generation</div>')
            
            with gr.Column(elem_classes="upload-info"):
                gr.HTML("""
                    <p><strong>Dataset Format:</strong> Upload a zip file containing two folders named 'class0' and 'class1', 
                    each containing images of the respective class.</p>
                """)
            
            with gr.Row():
                input_zip = gr.File(
                    label="Upload Dataset (ZIP file)",
                    file_types=[".zip"],
                    type="filepath"
                )
                output_dir = gr.Textbox(
                    label="Output Directory", 
                    value="./output"
                )
            
            with gr.Row():
                checkpoint_file = gr.File(
                    label="Upload LoRA Checkpoint (optional)",
                    file_types=[".pt", ".bin", ".pth"],
                    type="binary"
                )
                train_clf = gr.Checkbox(label="Train New Classifiers", value=True)
            
            process_btn = gr.Button("Generate Counterfactuals", elem_classes="btn-primary")
            
            # LoRA Training Section
            gr.HTML('<div class="section-header">LoRA Training</div>')
            
            with gr.Column(elem_classes="upload-info"):
                gr.HTML("""
                    <p><strong>Dataset Format:</strong> Upload a zip file containing two folders named 'class0' and 'class1', 
                    each containing images of the respective class for training the LoRA model.</p>
                """)

            def update_lora_output_dir(zip_file):
                """Update the LoRA output directory based on the uploaded zip filename"""
                if zip_file is None:
                    return "./lora_output"
                
                # Get the filename without extension
                zip_path = Path(zip_file)
                filename = zip_path.stem
                
                # Create a path with the filename
                output_path = f"./lora_output_{filename}"
                
                return output_path 
            
            with gr.Row():
                lora_input_zip = gr.File(
                    label="Upload Training Dataset (ZIP file)",
                    file_types=[".zip"],
                    type="filepath"
                )
                
                lora_output_dir = gr.Textbox(
                    label="LoRA Output Directory", 
                    value="./lora_output"
                )
                
                # Update output directory when zip file is uploaded
                lora_input_zip.change(
                    fn=update_lora_output_dir,
                    inputs=[lora_input_zip],
                    outputs=[lora_output_dir]
                )
            
            gr.HTML("""
                <div class="parameter-box">
                    <p>Default LoRA Training Parameters:</p>
                    <ul>
                        <li>Epochs: 5</li>
                        <li>Learning Rate: 1e-4</li>
                        <li>Batch Size: 32</li>
                        <li>LoRA Rank: 4</li>
                        <li>LoRA Alpha: 32</li>
                        <li>Max Training Steps: 1000</li>
                    </ul>
                </div>
            """)
            
            train_lora_btn = gr.Button("Train LoRA Model", elem_classes="btn-primary")
            lora_status_box = gr.Textbox(label="LoRA Training Status", value="Ready to train LoRA model")
            lora_download = gr.File(label="Download Trained LoRA Model", visible=False)
            
            # Results Section
            gr.HTML('<div class="section-header">Results</div>')
            
            with gr.Column():
                status = gr.Textbox(label="Status", value="Ready to generate counterfactuals")
                gallery = gr.Gallery(label="Generated Counterfactuals", columns=3, height="auto", elem_classes="gallery-item")
            
            gr.HTML("""
                <div class="footer">
                    <p>© 2025 Columbia University</p>
                </div>
            """)
        
        # Set up the click event for counterfactual generation
        process_btn.click(
            fn=process_uploaded_data,
            inputs=[input_zip, output_dir, checkpoint_file, train_clf],
            outputs=[status, gallery]
        )
        
        # Set up the click event for LoRA training
        train_lora_btn.click(
            fn=start_lora_training,
            inputs=[lora_input_zip, lora_output_dir],
            outputs=[lora_status_box, lora_download]
        )
        
        # Set up periodic status checking for LoRA training
        demo.load(
            fn=check_lora_status,
            inputs=None,
            outputs=lora_status_box,
            every=5  # Check every 5 seconds
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch() 

