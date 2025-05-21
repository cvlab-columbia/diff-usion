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
import glob
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
import subprocess
import tarfile
import queue
import hashlib
import json
from models.kandinsky_pipelines import KandinskyV22PipelineWithInversion, ManipulateMode


# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Define constants
IMG_SIZE = 512
BATCH_SIZE = 32
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Add these global variables after the imports and before the CSS definition
# Global variables for LoRA training
lora_status = "Ready"
lora_is_processing = False

# Global variables for generation control
generation_should_stop = False
classifier_should_stop = False  # New flag for classifier training
embedding_should_stop = False   # New flag for embedding encoding
lora_should_stop = False        # New flag for LoRA training
generation_queue = queue.Queue()
is_processing = False  # Add this to prevent multiple simultaneous processes

# Create temporary directories for uploads
temp_dir = Path("./temp_uploads")
temp_dir.mkdir(exist_ok=True, parents=True)

lora_temp_dir = Path("./temp_lora_uploads")
lora_temp_dir.mkdir(exist_ok=True, parents=True)

# Create a global queue for real-time updates
result_queue = queue.Queue()
displayed_results = []  # Keep track of all displayed results

# Add these global variables at the top of your file
total_images_to_process = 0
images_processed = 0

# Add these global variables after the existing ones
displayed_results_class0_to_class1 = []  # Results for class 0 to class 1
displayed_results_class1_to_class0 = []  # Results for class 1 to class 0

# Add global variables for caching
CACHE_DIR = Path("./cached_results")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

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

/* Hide the output directory */
.hidden-element {
    display: none !important;
}

/* Additional CSS for better alignment */
.container {
    padding: 0 1.5rem;
}

.main-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.results-container {
    margin-top: 0;
    padding-top: 0;
}

.full-width-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #f0f0f0;
    text-align: center;
}

.content-row {
    display: flex;
    gap: 2rem;
}

.sidebar {
    min-width: 250px;
    padding-right: 1.5rem;
}

.section-header {
    margin-top: 0;
}

.tabs-container {
    margin-top: 1rem;
}

.gallery-container {
    margin-top: 1rem;
}

/* Hide the output directory */
.hidden-element {
    display: none !important;
}

.gallery-item img {
    object-fit: contain !important;
    height: 200px !important;
    width: auto !important;
}

/* Force GIFs to restart when tab is selected */
.tabs-container .tabitem[style*="display: block"] .gallery-container img {
    animation: none;
    animation: reload-animation 0.1s;
}

@keyframes reload-animation {
    0% { opacity: 0.99; }
    100% { opacity: 1; }
}
"""

# Add to your global variables
current_cache_key = None
is_using_default_params = False

# Update the EXAMPLE_DATASETS to include direct dataset paths, embeddings, and classifiers
EXAMPLE_DATASETS = [
    {
        "name": "butterfly",
        "display_name": "Butterfly (Monarch vs Viceroy)",
        "description": "Dataset containing images of Monarch and Viceroy butterflies for counterfactual generation",
        "path": "./data/butterfly.tar.gz",
        "direct_dataset_path": "./data/butterflygrad",
        "checkpoint_path": "./output/lora/butterfly/copper-forest-49/checkpoint-1800",
        "embeddings_path": "./data/results/clip_image_embeds/butterfly",
        "classifier_path": "./data/results/ensemble/butterfly"
    },
    {
        "name": "retina",
        "display_name": "Retina (Normal vs Drusen)",
        "description": "Dataset containing normal retina images and retinas with drusen",
        "path": "compressed_datasets/retina.zip",
        "checkpoint_path": "./data/lora_output_retina/checkpoint-500",
        "embeddings_path": "./data/precomputed/retina/embeddings",
        "classifier_path": "./data/precomputed/retina/classifiers"
    },
    {
        "name": "lamp",
        "display_name": "Lamps",
        "description": "Dataset containing images of table lamps and floor lamps",
        "path": "compressed_datasets/lampsfar.zip",
        "checkpoint_path": "./data/lora_output_lampsfar/checkpoint-800",
        "embeddings_path": "./data/results/clip_image_embeds/lampsfar",
        "classifier_path": "./data/results/ensemble/lampsfar"
    },
    {
        "name": "couches",
        "display_name": "Couches",
        "description": "Dataset containing images of chairs and floor",
        "path": "compressed_datasets/couches.zip",
        "embeddings_path": "./data/results/clip_image_embeds/couches",
        "checkpoint_path": "./data/lora_output/couches/checkpoint-1000",
    }
]

# Function to get available example datasets
def get_example_datasets():
    """Get list of available example datasets"""
    return [dataset["name"] for dataset in EXAMPLE_DATASETS]

# Function to get example dataset info
def get_example_dataset_info(name):
    """Get information about an example dataset"""
    for dataset in EXAMPLE_DATASETS:
        if dataset["name"] == name:
            return dataset
    return None

#Function to check if we're using default parameters
def is_using_default_params(dataset_name, custom_tskip, num_images_per_class):
    """Check if we're using default parameters for the given dataset"""
    if dataset_name is None:
        return False
    if "butterfly" in dataset_name.lower():
        return (custom_tskip == 70 or custom_tskip == "70") and num_images_per_class == 10
    elif "lamp" in dataset_name.lower():
        return (custom_tskip == 85 or custom_tskip == "85") and num_images_per_class == 10
    elif "couch" in dataset_name.lower():
        return (custom_tskip == 85 or custom_tskip == "85") and num_images_per_class == 10
    return False

# # Function to get the output directory - either cache or regular output
# def get_output_directory(dataset_name, is_default_params, cache_key):
#     """Get the appropriate output directory based on parameters"""
#     if is_default_params:
#         # Use cache directory
#         cache_path = CACHE_DIR / cache_key
#         cache_path.mkdir(exist_ok=True, parents=True)
        
#         # Create dataset-specific directory
#         dataset_dir = cache_path / dataset_name.replace(" ", "_").lower()
#         dataset_dir.mkdir(exist_ok=True, parents=True)
        
#         # Create class-specific directories
#         class0_to_class1_dir = dataset_dir / "class0_to_class1"
#         class1_to_class0_dir = dataset_dir / "class1_to_class0"
#         class0_to_class1_dir.mkdir(exist_ok=True, parents=True)
#         class1_to_class0_dir.mkdir(exist_ok=True, parents=True)
        
#         # Create context directory
#         context_dir = dataset_dir / "context"
#         context_dir.mkdir(exist_ok=True, parents=True)
        
#         return dataset_dir, class0_to_class1_dir, class1_to_class0_dir, context_dir
#     else:
#         # Use regular output directory
#         output_dir = Path(f"./results/{dataset_name.replace(' ', '_').lower()}")
#         output_dir.mkdir(exist_ok=True, parents=True)
        
#         # Create gifs directory with class-specific subdirectories
#         gifs_dir = output_dir / "gifs"
#         gifs_dir.mkdir(exist_ok=True, parents=True)
        
#         class0_to_class1_dir = gifs_dir / "class0_to_class1"
#         class1_to_class0_dir = gifs_dir / "class1_to_class0"
#         class0_to_class1_dir.mkdir(exist_ok=True, parents=True)
#         class1_to_class0_dir.mkdir(exist_ok=True, parents=True)
        
#         # Create context directory
#         context_dir = output_dir / "context"
#         context_dir.mkdir(exist_ok=True, parents=True)
        
#         return output_dir, class0_to_class1_dir, class1_to_class0_dir, context_dir


def has_prediction_flipped(orig_preds, new_preds):
    """Check if any prediction has flipped from one class to another."""
    return ((orig_preds.preds > 0.5) != (new_preds.preds > 0.5)).any().item()

# Function to extract uploaded zip or tar.gz file
def extract_archive(archive_file, extract_dir):
    """Extract a zip or tar.gz file to the specified directory"""
    # Create a temporary directory for extraction
    temp_dir = Path(extract_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check file extension
    file_path = Path(archive_file)
    
    if file_path.suffix.lower() == '.zip':
        # Extract the zip file
        with zipfile.ZipFile(archive_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    elif file_path.name.endswith('.tar.gz') or file_path.name.endswith('.tgz'):
        # Extract the tar.gz file
        with tarfile.open(archive_file, 'r:gz') as tar_ref:
            tar_ref.extractall(temp_dir)
    else:
        raise ValueError(f"Unsupported archive format: {file_path.suffix}. Please use .zip or .tar.gz")
    
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
        raise ValueError("The uploaded archive must contain 'class0' and 'class1' directories or a subdirectory containing them")
    
    return str(temp_dir)

# Function to handle cached results (placeholder implementation)
def get_cached_result_info(name):
    """Get information about a cached result (placeholder)"""
    # This is a placeholder - in a real implementation, you'd store and retrieve cached results
    return None

# Modify the TwoClassDataset class to accept num_samples_per_class as a parameter
class TwoClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples_per_class=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class0_dir = self.root_dir / "class0"
        self.class1_dir = self.root_dir / "class1"
        
        # Get image paths
        self.class0_images = list(self.class0_dir.glob("*.*"))
        self.class1_images = list(self.class1_dir.glob("*.*"))
        
        # Limit the number of samples per class if specified
        if num_samples_per_class is not None:
            self.class0_images = self.class0_images[:num_samples_per_class]
            self.class1_images = self.class1_images[:num_samples_per_class]
        
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
    """Create a GIF that alternates between two images with elegant labels"""
    # Create copies of the images to avoid modifying the originals
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    # Add labels to the images
    draw1 = ImageDraw.Draw(img1_copy)
    draw2 = ImageDraw.Draw(img2_copy)
    
    try:
        # Use a larger font size for better visibility
        font = ImageFont.truetype("arial.ttf", 36)  # Increased from 28 to 36
    except:
        font = ImageFont.load_default()
    
    # Add a subtle shadow effect for better visibility
    padding = 15
    
    # Original image - add text with shadow effect
    # First draw shadow/outline
    for offset in [(1,1), (-1,1), (1,-1), (-1,-1)]:
        draw1.text(
            (padding + offset[0], padding + offset[1]),
            "Original",
            fill=(0, 0, 0, 180),
            font=font
        )
    
    # Then draw the main text
    draw1.text(
        (padding, padding),
        "Original",
        fill=(255, 255, 255, 230),
        font=font
    )
    
    # Generated image - add text with shadow effect
    # First draw shadow/outline
    for offset in [(1,1), (-1,1), (1,-1), (-1,-1)]:
        draw2.text(
            (padding + offset[0], padding + offset[1]),
            "Generated",
            fill=(0, 0, 0, 180),
            font=font
        )
    
    # Then draw the main text
    draw2.text(
        (padding, padding),
        "Generated",
        fill=(255, 255, 255, 230),
        font=font
    )
    
    # Increase duration to 1 second per image (1000ms)
    imageio.mimsave(output_path, [img1_copy, img2_copy], duration=1000, loop=0)
    return output_path

def train_classifier(model, train_loader, val_loader, epochs, lr, device, patience=5):
    """Train a classifier model with early stopping"""
    global classifier_should_stop
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Check if we should stop
        if classifier_should_stop:
            print("Classifier training cancelled")
            break
            
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

def train_ensemble_classifiers(train_dataset, val_dataset, classifiers_dir, epochs=10, lr=0.001):
    """Train an ensemble of classifiers"""

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

# Modify the update_progress_status function to be more informative
def update_progress_status():
    """Update the progress status for the counterfactual generation"""
    global images_processed, total_images_to_process, is_processing
    
    if not is_processing:
        if images_processed > 0:
            return f"Processing complete. Generated {images_processed} counterfactual images."
        return "Ready to process images."
    
    if total_images_to_process == 0:
        return "Preparing to process images..."
    
    percentage = (images_processed / total_images_to_process) * 100
    return f"Progress: {images_processed}/{total_images_to_process} images processed ({percentage:.1f}%)"

# Add function to cancel generation
def cancel_generation():
    """Cancel all ongoing processes"""
    global generation_should_stop, classifier_should_stop, embedding_should_stop, lora_should_stop
    
    # Set all stop flags
    generation_should_stop = True
    classifier_should_stop = True
    embedding_should_stop = True
    lora_should_stop = True
    
    return "All processes have been requested to stop. This may take a moment to complete."

def save_results_to_cache(output_dir, cache_key):
    """Save generated results to cache directory"""
    cache_path = CACHE_DIR / cache_key
    cache_path.mkdir(exist_ok=True, parents=True)
    
    # Copy gifs directory
    output_gifs_dir = Path(output_dir) / "gifs"
    cache_gifs_dir = cache_path / "gifs"
    
    if output_gifs_dir.exists():
        # Remove existing cache if it exists
        if cache_gifs_dir.exists():
            shutil.rmtree(cache_gifs_dir)
        
        # Copy the new results, maintaining subdirectory structure
        shutil.copytree(output_gifs_dir, cache_gifs_dir)
    
    # Copy context images if they exist
    output_context_dir = Path(output_dir) / "context"
    cache_context_dir = cache_path / "context"
    
    if output_context_dir.exists():
        if cache_context_dir.exists():
            shutil.rmtree(cache_context_dir)
        shutil.copytree(output_context_dir, cache_context_dir)

# Update the process_with_selected_dataset function to handle the new directory structure
def process_with_selected_dataset(zip_file, output_dir, dataset_display_name, checkpoint_path=None, train_clf=True, 
                                 is_direct_path=False, direct_path=None, embeddings_path=None, 
                                 classifier_path=None, use_classifier_stopping=True, custom_tskip=85,
                                 num_images_per_class=5):
    
    global current_cache_key
    """Process using either the uploaded file or the selected dataset, with caching"""
    print(f"Processing with dataset: {dataset_display_name}")
    
    # If a custom dataset is uploaded, don't use cache
    if zip_file is not None:
        print(f"Using uploaded file: {zip_file}")
        return process_uploaded_data(
            zip_file, output_dir, checkpoint_path, train_clf, 
            is_direct_path, direct_path, embeddings_path, 
            classifier_path, use_classifier_stopping, custom_tskip,
            num_images_per_class
        )
    
    # Find the selected dataset
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        if dataset["display_name"] == dataset_display_name:
            selected_dataset = dataset
            break
    
    if not selected_dataset:
        print("Error: No dataset selected")
        return "No dataset selected", [], [], [], "Error: No dataset selected", None, None
    
    # Get dataset paths from the selected dataset
    dataset_path = selected_dataset.get("path", None)
    checkpoint_path = selected_dataset.get("checkpoint_path", checkpoint_path)
    direct_path = selected_dataset.get("direct_dataset_path", None)
    is_direct_path = direct_path is not None
    embeddings_path = selected_dataset.get("embeddings_path", embeddings_path)
    classifier_path = selected_dataset.get("classifier_path", classifier_path)
    
    # Check if we're using default parameters
    is_default_params = is_using_default_params(dataset_display_name, custom_tskip, num_images_per_class)
    
    #import pdb; pdb.set_trace()
    if is_default_params:
        print("Using default parameters, checking cache...")
        
        # Generate cache key
        cache_key = get_cache_key(
            selected_dataset["name"], checkpoint_path, train_clf, 
            embeddings_path, classifier_path, use_classifier_stopping, 
            custom_tskip, num_images_per_class
        )
        
        # Check if cache exists
        #import pdb; pdb.set_trace()
        cache_path = CACHE_DIR / cache_key
        dataset_dir = cache_path / "gifs"
        #import pdb; pdb.set_trace()
        if cache_path.exists() and dataset_dir.exists():
            current_cache_key = cache_key
            print(f"Found cached results for key: {cache_key}")
            
            # Get paths to class-specific directories
            class0_to_class1_dir = dataset_dir / "class0_to_class1"
            class1_to_class0_dir = dataset_dir / "class1_to_class0"
            context_dir = cache_path/ "context"
            
            # Get all GIF paths
            class0_to_class1_gifs = list(class0_to_class1_dir.glob("*.gif")) if class0_to_class1_dir.exists() else []
            class1_to_class0_gifs = list(class1_to_class0_dir.glob("*.gif")) if class1_to_class0_dir.exists() else []
            
            # Sort the GIFs by filename for consistent ordering
            class0_to_class1_gifs.sort(key=lambda p: p.name)
            class1_to_class0_gifs.sort(key=lambda p: p.name)
            
            # Get context images
            class0_context = context_dir / "class0_sample.jpg" if (context_dir / "class0_sample.jpg").exists() else None
            class1_context = context_dir / "class1_sample.jpg" if (context_dir / "class1_sample.jpg").exists() else None
            
            # Convert paths to strings
            class0_to_class1_paths = [str(p) for p in class0_to_class1_gifs]
            class1_to_class0_paths = [str(p) for p in class1_to_class0_gifs]
            all_gifs = class0_to_class1_paths + class1_to_class0_paths
            
            # Update the global gallery variables
            global displayed_results, displayed_results_class0_to_class1, displayed_results_class1_to_class0
            displayed_results = all_gifs
            displayed_results_class0_to_class1 = class0_to_class1_paths
            displayed_results_class1_to_class0 = class1_to_class0_paths
            
            # Return cached results
            return (
                "Using cached results for default parameters.", 
                displayed_results,
                displayed_results_class0_to_class1,
                displayed_results_class1_to_class0,
                "Loaded from cache", 
                str(class0_context) if class0_context else None, 
                str(class1_context) if class1_context else None
            )
    
    # If not using cache or cache doesn't exist, process normally
    print("Processing dataset normally...")
    
    # Process the dataset
    result = process_uploaded_data(
        dataset_path, output_dir, checkpoint_path, train_clf, 
        is_direct_path, direct_path, embeddings_path, 
        classifier_path, use_classifier_stopping, custom_tskip,
        num_images_per_class, selected_dataset["name"] # Pass the dataset name
    )
    
    return result

def process_uploaded_data(zip_file, output_dir, checkpoint_path=None, train_clf=True, 
                         is_direct_path=False, direct_path=None, embeddings_path=None, 
                         classifier_path=None, use_classifier_stopping=True, custom_tskip=None,
                         num_images_per_class=5, dataset_name=None):
    """Process an uploaded dataset"""
    global is_processing, generation_should_stop
    
    if is_processing:
        return "Already processing a dataset. Please wait or cancel the current process.", [], [], [], "Already processing", None, None
    
    generation_should_stop = False
    is_processing = True
    
    # Create output directory and context directory
    output_dir = Path(output_dir)
    context_dir = output_dir / "context"
    output_dir.mkdir(exist_ok=True, parents=True)
    context_dir.mkdir(exist_ok=True, parents=True)
    
    def save_context_images(dataset_path):
        """Helper function to save context images from a dataset path"""
        class0_dir = dataset_path / "class0"
        class1_dir = dataset_path / "class1"
        
        # Get first image from each class
        class0_img = next(class0_dir.glob("*.*"))
        class1_img = next(class1_dir.glob("*.*"))
        
        # Save context images
        shutil.copy(class0_img, context_dir / "class0_sample.jpg")
        shutil.copy(class1_img, context_dir / "class1_sample.jpg")
        
        return str(context_dir / "class0_sample.jpg"), str(context_dir / "class1_sample.jpg")
    
    try:
        if is_direct_path and direct_path:
            dataset_dir = Path(direct_path)
            # Save context images before starting thread
            class0_context, class1_context = save_context_images(dataset_dir)
            
            # Start the background processing thread
            thread = threading.Thread(
                target=background_process,
                args=(dataset_dir, output_dir, checkpoint_path, train_clf, 
                      embeddings_path, classifier_path, use_classifier_stopping, 
                      custom_tskip, num_images_per_class, dataset_name)
            )
            thread.daemon = True
            thread.start()
            
            return "Processing dataset from direct path...", [], [], [], "Starting processing...", class0_context, class1_context
        
        elif zip_file:
            # Create a unique temporary directory
            temp_id = int(time.time())
            extract_dir = temp_dir / f"upload_{temp_id}"
            extract_dir.mkdir(exist_ok=True, parents=True)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the correct directory structure
            class0_dir = extract_dir / "class0"
            class1_dir = extract_dir / "class1"
            
            if not (class0_dir.exists() and class1_dir.exists()):
                # Try to find class0 and class1 directories in subdirectories
                subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                for subdir in subdirs:
                    if (subdir / "class0").exists() and (subdir / "class1").exists():
                        extract_dir = subdir
                        break
                else:
                    is_processing = False
                    return "Error: Invalid directory structure", [], [], [], "Error", None, None
            
            # Save context images before starting thread
            class0_context, class1_context = save_context_images(extract_dir)
            
            # Start the background processing thread
            thread = threading.Thread(
                target=background_process,
                args=(extract_dir, output_dir, checkpoint_path, train_clf, 
                      embeddings_path, classifier_path, use_classifier_stopping, 
                      custom_tskip, num_images_per_class, dataset_name)
            )
            thread.daemon = True
            thread.start()
            
            return "Processing uploaded dataset...", [], [], [], "Starting processing...", class0_context, class1_context
        
        else:
            is_processing = False
            return "Error: No dataset provided", [], [], [], "Error", None, None
            
    except Exception as e:
        is_processing = False
        return f"Error processing dataset: {str(e)}", [], [], [], "Error", None, None

def get_embeddings(train_dataset, val_dataset, embeddings_path):
    """Get CLIP embeddings for both classes and save them"""
    device = DEVICE
    embeddings_path = Path(embeddings_path)
    embeddings_path.mkdir(exist_ok=True, parents=True)

    # Initialize CLIP models
    image_processor = CLIPImageProcessor.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
    ).to(device)

    # Create data loader
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_0_embeds_list = []
    class_1_embeds_list = []

    # Process batches
    for images, labels, _ in tqdm(data_loader, desc="Computing CLIP embeddings"):
        images = images.to(device)
        labels = labels.to(device)

        # Process images through CLIP
        inputs = image_processor(
            images=images, return_tensors="pt", do_rescale=False
        ).to(device)

        with torch.no_grad():
            image_embeds = image_encoder(**inputs).image_embeds
            
            # Separate embeddings based on labels
            class_0_mask = (labels == 0)
            class_1_mask = (labels == 1)
            
            class_0_embeds_list.append(image_embeds[class_0_mask])
            class_1_embeds_list.append(image_embeds[class_1_mask])

    # Concatenate all embeddings
    class0_embeddings = torch.cat(class_0_embeds_list)
    class1_embeddings = torch.cat(class_1_embeds_list)

    # Save embeddings
    torch.save(class0_embeddings, embeddings_path / "class0_embeddings.pt")
    torch.save(class1_embeddings, embeddings_path / "class1_embeddings.pt")

    return embeddings_path
# Update the background_process function
def background_process(dataset_dir, output_dir, checkpoint_path=None, train_clf=True, 
                      embeddings_path=None, classifier_path=None, use_classifier_stopping=True,
                      custom_tskip=None, num_images_per_class=5, dataset_name=None):
    """Process a dataset in the background"""
    global is_processing, generation_should_stop, classifier_should_stop, embedding_should_stop
  
    # Process validation images to generate counterfactuals
    results = []
    processed_batches = 0

    
   

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create gifs directory with class-specific subdirectories
    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(exist_ok=True, parents=True)
    
    class0_to_class1_dir = gifs_dir / "class0_to_class1"
    class1_to_class0_dir = gifs_dir / "class1_to_class0"
    class0_to_class1_dir.mkdir(exist_ok=True, parents=True)
    class1_to_class0_dir.mkdir(exist_ok=True, parents=True)
    
    # Create context directory
    context_dir = output_dir / "context"
    context_dir.mkdir(exist_ok=True, parents=True)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((512, 512)),  # Use a standard size for CLIP
        transforms.ToDtype(torch.float32, scale=True),
    ])
    
    dataset = TwoClassDataset(dataset_dir, transform=transform, num_samples_per_class=num_images_per_class)
    dataset_large = TwoClassDataset(dataset_dir, transform=transform, num_samples_per_class=1000)

    val_loader_single = DataLoader(dataset, batch_size=1, shuffle=False)
    if classifier_path is None and train_clf: #train new calssifers button is clicked :
        #get outputdir from dataset_dir but tack it on to output dir
        classifier_path = Path(output_dir) /  "classifiers" / dataset_dir.name 
        print(f"Training classifiers for {dataset_dir.name}")
        train_ensemble_classifiers(dataset_large, dataset, classifier_path)

    if embeddings_path is None:
        embeddings_path = Path(output_dir) / "embeddings" / dataset_dir.name 
        print(f"Training embeddings for {dataset_dir.name}")
        get_embeddings(dataset_large, dataset_large, embeddings_path)
        device = DEVICE

    #import pdb; pdb.set_trace()
    if train_clf:
        classifiers = [
                torch.load(model_path, map_location="cpu").to(DEVICE)
                for model_path in Path(classifier_path).glob("*.pth")
            ]

        print(f"Loaded {len(classifiers)} classifiers from {classifier_path}")
    
    embeddings_path = Path(embeddings_path)
    class0_embeddings = torch.load(embeddings_path / "class0_embeddings.pt", map_location=DEVICE)
    class1_embeddings = torch.load(embeddings_path / "class1_embeddings.pt", map_location=DEVICE)

   

    generator = torch.Generator(device=DEVICE).manual_seed(0)

    pipeline: KandinskyV22PipelineWithInversion = (
        KandinskyV22PipelineWithInversion.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder"
        )
        ).to(DEVICE)
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    if checkpoint_path is not None:
        lora_model = PeftModel.from_pretrained(pipeline.unet, checkpoint_path)
        pipeline.unet = lora_model
        print(f"loading lora weights from {checkpoint_path}")
    pipeline.to(DEVICE)
    pipeline.image_encoder.to(DEVICE)
    # Define t_skips based on user input
    if custom_tskip is not None:
        # Convert from percentage (0-100) to decimal (0-1)
        t_skip_value = float(custom_tskip) / 100.0
        t_skips = [t_skip_value]
        print(f"Using custom t_skip value: {t_skip_value}")
    else:
        t_skips = [0.9, 0.8, 0.7, 0.6, 0.5]  # Default t_skip values
    
    with torch.no_grad():
        for images, labels, img_paths in tqdm(val_loader_single, desc="Generating counterfactuals"):
            # Check if cancellation was requested
            if generation_should_stop:
                print("Generation cancelled by user")
                gif_paths = [r["gif_path"] for r in results]  # Return whatever we've generated so far
                is_processing = False  # Release the processing flag
                return gif_paths
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get original predictions
            if train_clf:
                orig_preds = ensemble_predict(classifiers, images)
            else:
                orig_preds = None
            
            
            inputs = pipeline.image_processor(
                images=images, return_tensors="pt", do_rescale=False
            ).to(DEVICE)
            image_embeds = pipeline.image_encoder(**inputs).image_embeds
            orig_image_embeds = image_embeds.clone()

            # Get original predictions

            norm_image_embeds = orig_image_embeds / orig_image_embeds.norm(
                dim=-1, keepdim=True
            )
            norm_pos_embeds = class0_embeddings / class0_embeddings.norm(dim=-1, keepdim=True)
            norm_neg_embeds = class1_embeddings / class1_embeddings.norm(dim=-1, keepdim=True)
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

            modes = [ManipulateMode.cond_avg]
            gs_targets = [4.0]
            manipulation_scales = [2.0]
            direction_sign = -1.0 if labels.item() == 0 else 1.0

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
                            # Check for cancellation again
                            if generation_should_stop:
                                print("Generation cancelled by user")
                                gif_paths = [r["gif_path"] for r in results]
                                is_processing = False  # Release the processing flag
                                return gif_paths
                                
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
                            if train_clf:
                                new_preds = ensemble_predict(classifiers, samples)
                                flipped = has_prediction_flipped(orig_preds, new_preds)
                            else:
                                flipped = True
                            
                                            
                            
                            # If we found a flip, save the best parameters and break
                            if flipped:
                                print(f"Found flip at t_skip: {t_skip}")
                                found_flip = True
                            
                                # Save the results
                                for img_idx, (img_path, sample) in enumerate(zip(img_paths, samples)):
                                    # Save original and generated images
                                    orig_img = to_pil_image(images[img_idx])
                                    gen_img = to_pil_image(sample)
               
              
                                    # Create and save GIF with synchronized timing
                                    base_filename = Path(img_path).name
                                    # Determine output directory based on label
                                    if labels[img_idx].item() == 0:
                                        # Class 0 to Class 1
                                        gif_path = class0_to_class1_dir / f"{base_filename}.gif"
                                    else:
                                        # Class 1 to Class 0
                                        gif_path = class1_to_class0_dir / f"{base_filename}.gif"
                                    
                                    # For Class 0 → Class 1 (when labels[img_idx].item() == 0)
                                    if labels[img_idx].item() == 0:
                                        # Original first, then generated
                                        create_gif(orig_img, gen_img, gif_path)
                                    else:  # For Class 1 → Class 0
                                        # Original first, then generated (same order for consistency)
                                        create_gif(orig_img, gen_img, gif_path)
                                    
                                    # Add the GIF path to the result queue with direction information and context images
                                    if labels[img_idx].item() == 0:
                                        # For Class 0 → Class 1
                                        result_queue.put((str(gif_path), labels[img_idx].item()))
                                    else:
                                        # For Class 1 → Class 0
                                        result_queue.put((str(gif_path), labels[img_idx].item()))
                                        
                                    
                                   
                                    break
                                break
                            
    if is_using_default_params(dataset_name, custom_tskip, num_images_per_class):
        # Generate cache key based on parameters
        #import pdb; pdb.set_trace()
        print("SAVING TO CACHE")
        cache_key = get_cache_key(
            dataset_name,
            checkpoint_path,
            train_clf,
            embeddings_path,
            classifier_path,
            use_classifier_stopping,
            custom_tskip,
            num_images_per_class
        )
        # Save to cache
        save_results_to_cache(output_dir, cache_key)
        # Update current cache key for check_for_new_results
        global current_cache_key
        current_cache_key = cache_key

    # Return paths to GIFs
    print(result_queue, "return_queue")

    is_processing = False  # Release the processing flag
    return result_queue

# Update the train_lora function to use the clip_collate_fn
def train_lora(data_dir, output_dir, num_epochs=5, learning_rate=1e-4, batch_size=32, lora_rank=4, lora_alpha=32, max_train_steps=None):
    """Train a LoRA model for Kandinsky using code from kandinsky_lora_train.py"""
    global lora_should_stop
    
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

        #modify num_epochs so we reach max_train_steps
        num_epochs = max_train_steps // len(train_dataloader)
        import pdb; pdb.set_trace()
        
        for epoch in range(num_epochs):
            # Check if we should stop
            if lora_should_stop:
                print("LoRA training cancelled")
                return "LoRA training was cancelled"
                
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
def process_and_clear(input_zip, output_dir, example_datasets_dropdown, checkpoint_path_state, train_clf, 
                     is_direct_path_state, direct_path_state, embeddings_path_state, 
                     classifier_path_state, use_classifier_stopping, custom_tskip,
                     num_images_per_class):
    """Clear folders first, then process the dataset"""
    # Clear folders first
    clear_output_folders()
    
    # Then process the dataset
    return process_with_selected_dataset(
        input_zip, output_dir, example_datasets_dropdown, checkpoint_path_state, train_clf,
        is_direct_path_state, direct_path_state, embeddings_path_state,
        classifier_path_state, use_classifier_stopping, custom_tskip,
        num_images_per_class
    )
def update_example_images(dataset_name):
    """Update the example images based on the selected dataset"""
    # Find the dataset info
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        if dataset["name"] == dataset_name:
            selected_dataset = dataset
            break
    
    if selected_dataset:
        # Get the context images from the dataset's directory
        dataset_dir = selected_dataset.get("direct_dataset_path")
        if dataset_dir:
            class0_img = next((str(p) for p in Path(dataset_dir).glob("class0/*.*")), None)
            class1_img = next((str(p) for p in Path(dataset_dir).glob("class1/*.*")), None)
            return class0_img, class1_img
    
    return None, None

# Add a state variable to store the direct dataset path
direct_path_state = gr.State(None)
# Map display names back to internal names (add this back)
def get_name_from_display(display_name):
    for dataset in EXAMPLE_DATASETS:
        if dataset["display_name"] == display_name:
            return dataset["name"]
    return None

# Modify the use_selected_dataset function
def use_selected_dataset(display_name):
    name = get_name_from_display(display_name)
    if not name:
        print("No dataset name found")
        return None, None, False, None, None, None
    
    dataset_info = get_example_dataset_info(name)
    
    # Check if there's a direct dataset path available
    if dataset_info and "direct_dataset_path" in dataset_info and os.path.exists(dataset_info["direct_dataset_path"]):
        print(f"Using direct dataset path: {dataset_info['direct_dataset_path']}")
        # Return paths for direct dataset, checkpoint, embeddings, and classifiers
        return None, dataset_info["checkpoint_path"], True, dataset_info["direct_dataset_path"], \
               dataset_info.get("embeddings_path"), dataset_info.get("classifier_path")
    elif dataset_info and os.path.exists(dataset_info["path"]):
        # Return the archive path and other paths
        return dataset_info["path"], dataset_info["checkpoint_path"], False, None, \
               dataset_info.get("embeddings_path"), dataset_info.get("classifier_path")
    return None, None, False, None, None, None
def reset_galleries():
    """Reset all galleries when changing datasets or parameters"""
    global displayed_results, displayed_results_class0_to_class1, displayed_results_class1_to_class0
    global current_cache_key  # Also reset the cache key
    
    displayed_results = []
    displayed_results_class0_to_class1 = []
    displayed_results_class1_to_class0 = []
    current_cache_key = None  # Reset the cache key
    
    # Clear the result queue if it exists
    while not result_queue.empty():
        result_queue.get()
        
    return [], [], [], "Galleries reset"
def clear_output_folders():
    """Delete the output/gifs and output/context folders and their contents"""
    import shutil
    from pathlib import Path
    
    # Folders to clear
    folders = ["gifs", "context"]
    
    for folder in folders:
        folder_path = Path("./output") / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"Deleted {folder_path}")

def create_gradio_interface():
    # Create temporary directories for uploads
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    clear_output_folders()

    
    lora_temp_dir = Path("./temp_lora_uploads")
    lora_temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Get initial list of example datasets
    example_datasets = get_example_datasets()
    
    with gr.Blocks(css=css) as demo:
        # Add the header at the top level to span across all columns
        with gr.Row(elem_classes="full-width-header"):
            with gr.Column():
                gr.HTML("""
                    <div class="header">
                        <h1>DIFFusion Demo</h1>
                        <p class="subtitle">Generate fine-grained edits to images using another class of images as guidance.</p>
                    </div>
                """)
        
        # Main content row with sidebar, config column and results column
        with gr.Row(elem_classes="content-row"):
            # Sidebar for example datasets
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.HTML('<div class="section-header">Example Datasets</div>')
                
                # Create a dropdown for example datasets
                example_datasets_dropdown = gr.Dropdown(
                    choices=[dataset["display_name"] for dataset in EXAMPLE_DATASETS],
                    value=next((dataset["display_name"] for dataset in EXAMPLE_DATASETS if "lamp" in dataset["display_name"].lower()), None),  # Set lamp as default
                    label="Example Datasets",
                    info="Select a pre-loaded dataset to use"
                )
                
                # Add dataset descriptions directly in the dropdown info
                dataset_descriptions = {dataset["display_name"]: dataset.get("description", "") for dataset in EXAMPLE_DATASETS}
                
                # Add some spacing
                gr.HTML("<div style='height: 20px;'></div>")
                
                # Add a hidden state for the dataset description (we'll still update it but not display it)
                dataset_description = gr.Textbox(visible=False)
            
            # Main content area
            with gr.Column(scale=2, elem_classes="main-container"):
                # Paper info and configuration
                with gr.Column():
                    with gr.Column(elem_classes="paper-info"):
                        gr.HTML("""
                            <h3>DIFFusion Marketing Demo</h3>
                            <p>Ever struggled to describe a visual transformation in AI image editing? Same. It's often difficult to identify the exact transformation you want to perform. This insight led us to develop an image-guided method that enables precise edits by learning from class differences, rather than requiring predefined descriptions. Our method automatically identifies and applies the transformation, making it especially useful in science where researchers can visualize subtle differences between similar-looking image categories. While science was our focus, I realized the method is also ideal for marketing. Product differences can be difficult to articulate verbally, and we don't always know how to best adapt aesthetic elements to showcase new products effectively. This demo was created for marketers to explore these capabilities with their own product imagery. Of course, if you have a science dataset, this demo works great for that too.</p>
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
                            label="Upload Custom Dataset (ZIP or TAR.GZ file)",
                            file_types=[".zip", ".tar.gz", ".tgz"],
                            type="filepath"
                        )
                        # Hide the output directory by using elem_classes
                        output_dir = gr.Textbox(
                            label="Output Directory", 
                            value="./output",
                            elem_classes="hidden-element"
                        )
                    
                    with gr.Row():
                        gr.HTML('<div class="section-header">LoRA Training</div>')
                    
                    with gr.Column(elem_classes="upload-info"):
                        gr.HTML("""
                            <p><strong>Dataset Format:</strong> Upload a zip file containing two folders named 'class0' and 'class1', 
                            each containing images of the respective class for training the LoRA model.</p>
                        """)
                    
                    with gr.Row():
                        lora_output_dir = gr.Textbox(
                            label="LoRA Output Directory", 
                            value="./lora_output"
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
                    train_clf = gr.Checkbox(label="Train New Classifiers", value=False)
                    
                    with gr.Row():
                        use_classifier_stopping = gr.Checkbox(
                            label="Use Classifier for Early Stopping", 
                            value=False,
                            info="If checked, will stop at the first t_skip value that flips the prediction"
                        )
                        
                        custom_tskip = gr.Slider(
                            minimum=0, 
                            maximum=100, 
                            value=85, 
                            step=5, 
                            label="Custom T-Skip Value (0-100)", 
                            info="Enter a value between 0-100 to use a specific t_skip (as percentage). Leave empty to use default values.",
                            visible=True
                        )
                    
                    # Add a text box for number of images per class
                    with gr.Row():
                        num_images_per_class = gr.Number(
                            label="Number of Images per Class", 
                            value=10,
                            minimum=1,
                            maximum=100,
                            step=1,
                            info="Specify how many images to process from each class"
                        )
                    
                    with gr.Row():
                        process_btn = gr.Button("Generate Counterfactuals", elem_classes="btn-primary")
                        cancel_btn = gr.Button("Cancel Generation", elem_classes="btn-primary")
                    
                    # Status for the main column
                    status = gr.Textbox(label="Status", value="Ready to generate counterfactuals")
            
            # Results column
            with gr.Column(scale=2, elem_classes="results-container"):
                # Class Examples section header - MOVED HERE
                gr.HTML('<div class="section-header">Class Examples</div>')
                
                # Class example images - MOVED HERE
                with gr.Row():
                    class0_context_image = gr.Image(label="Class 0 Example", type="filepath", height=256)
                    class1_context_image = gr.Image(label="Class 1 Example", type="filepath", height=256)
                
                # Results section header
                gr.HTML('<div class="section-header">Results</div>')
                
                # Add tabs for different direction signs - make "All Results" the default tab
                with gr.Tabs(elem_classes="tabs-container") as result_tabs:
                    with gr.TabItem("All Results"):
                        gallery = gr.Gallery(
                            label="Generated Images",
                            show_label=False,
                            elem_id="gallery_all",
                            columns=4,  # Show 4 images per row
                            rows=None,  # Let it adjust rows automatically
                            height="auto",
                            allow_preview=True,
                            preview=False,
                            object_fit="contain"
                        )

                    with gr.TabItem("Class 0 → Class 1"):
                        gallery_class0_to_class1 = gr.Gallery(
                            label="Class 0 to Class 1",
                            show_label=False,
                            elem_id="gallery_0to1",
                            columns=4,  # Show 4 images per row
                            rows=None,  # Let it adjust rows automatically
                            height="auto",
                            allow_preview=True,
                            preview=True,
                            object_fit="contain"
                        )

                    with gr.TabItem("Class 1 → Class 0"):
                        gallery_class1_to_class0 = gr.Gallery(
                            label="Class 1 to Class 0",
                            show_label=False,
                            elem_id="gallery_1to0",
                            columns=4,  # Show 4 images per row
                            rows=None,  # Let it adjust rows automatically
                            height="auto",
                            allow_preview=True,
                            preview=True,
                            object_fit="contain"
                        )
                    # with gr.TabItem("All Results"):
                    #     gallery = gr.Gallery(
                    #     columns=[3],
                    #     rows=[3], 
                    #     height="auto",
                    #     allow_preview=True,  # Make sure this is enabled
                    #     preview=True,        # Try setting this explicitly
                    #     object_fit="contain"  # Try different fit modes
                    # )
                    
                    # with gr.TabItem("Class 0 → Class 1"):
                    #     gallery_class0_to_class1 = gr.Gallery(
                    #         columns=[3],
                    #     rows=[3], 
                    #         height="auto", 
                    #         allow_preview=True,  # Make sure this is enabled
                    #     preview=True,        # Try setting this explicitly
                    #     object_fit="contain"  # Try different fit modes
                    #     )
                    
                    # with gr.TabItem("Class 1 → Class 0"):
                    #     gallery_class1_to_class0 = gr.Gallery(
                    #         columns=[3],
                    #     rows=[3], 
                    #         height="auto", 
                    #         allow_preview=True,  # Make sure this is enabled
                    #     preview=True,        # Try setting this explicitly
                    #     object_fit="contain"  # Try different fit modes
                    #     )
                
                # Add a progress status box in the results column
                progress_status = gr.Textbox(
                    label="Progress", 
                    value="Ready to process",
                    interactive=False
                )
        
        # Define state variables inside the function
        checkpoint_path_state = gr.State(None)
        is_direct_path_state = gr.State(False)
        direct_path_state = gr.State(None)
        embeddings_path_state = gr.State(None)
        classifier_path_state = gr.State(None)
        custom_tskip_state = gr.State(None)  # Add this state variable

        # Set up the cancel button click handler
        cancel_btn.click(
            fn=cancel_generation,
            inputs=None,
            outputs=status
        )

                # Update the change event to include the slider
        input_zip.change(
            fn=handle_zip_upload,
            inputs=input_zip,
            outputs=[
                dataset_description, 
                checkpoint_path_state, 
                is_direct_path_state, 
                direct_path_state, 
                embeddings_path_state, 
                classifier_path_state, 
                custom_tskip_state,
                custom_tskip,  # slider
                lora_output_dir  # Add lora output dir
            ]
        ).then(  # Reset galleries when new zip is uploaded
            fn=reset_galleries,
            inputs=None,
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status]
        ).then(  # Clear example dataset selection
            fn=lambda: gr.update(value=None),
            inputs=None,
            outputs=example_datasets_dropdown
        )
        
        # # Set up the dropdown change event to update the dataset description and paths
        # example_datasets_dropdown.change(
        #     fn=update_dataset_info,
        #     inputs=example_datasets_dropdown,
        #     outputs=[dataset_description, checkpoint_path_state, is_direct_path_state, direct_path_state, 
        #              embeddings_path_state, classifier_path_state, custom_tskip_state]
        # )
        

        # # Add this direct connection to set custom_tskip based on dataset name
        # example_datasets_dropdown.change(
        #     fn=set_custom_tskip_for_dataset,
        #     inputs=example_datasets_dropdown,
        #     outputs=custom_tskip
        # )
        
        

        # #change cache key when dataset changes
        # example_datasets_dropdown.change(
        #     fn=change_cache_key,
        #     inputs=[example_datasets_dropdown, num_images_per_class, train_clf, use_classifier_stopping, custom_tskip],
        #     outputs=None
        # )
        example_datasets_dropdown.change(
            fn=reset_galleries,  # Reset first
            inputs=None,
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status]
        ).then(  # Update dataset info
            fn=update_dataset_info,
            inputs=example_datasets_dropdown,
            outputs=[dataset_description, checkpoint_path_state, is_direct_path_state, direct_path_state, 
                    embeddings_path_state, classifier_path_state, custom_tskip_state]
        ).then(  # Set custom t-skip
            fn=set_custom_tskip_for_dataset,
            inputs=example_datasets_dropdown,
            outputs=custom_tskip
        ).then(  # Change cache key
            fn=change_cache_key,
            inputs=[example_datasets_dropdown, num_images_per_class, train_clf, use_classifier_stopping, custom_tskip],
            outputs=None
        ).then(  # Update example images
            fn=lambda display_name: update_example_images(get_name_from_display(display_name)),
            inputs=example_datasets_dropdown,
            outputs=[class0_context_image, class1_context_image]
        )
            
        # Set up single click event that does both operations in order
        process_btn.click(
            fn=process_and_clear,
            inputs=[
                input_zip, output_dir, example_datasets_dropdown, checkpoint_path_state, train_clf, 
                is_direct_path_state, direct_path_state, embeddings_path_state, 
                classifier_path_state, use_classifier_stopping, custom_tskip,
                num_images_per_class
            ],
            outputs=[status, gallery, gallery_class0_to_class1, gallery_class1_to_class0, 
                    progress_status, class0_context_image, class1_context_image]
        )

        
        # Set up the click event for LoRA training
        train_lora_btn.click(
            fn=start_lora_training,
            inputs=[input_zip, lora_output_dir],
            outputs=[lora_status_box]
        )
        
        # Set up periodic status checking for LoRA training
        demo.load(
            fn=check_lora_status,
            inputs=None,
            outputs=lora_status_box,
            every=5  # Check every 5 seconds
        )
        
        # Add a periodic refresh for the galleries
        # Add a periodic refresh for the galleries
        demo.load(
            fn=lambda display_name="", custom_tskip=85, num_images=5: check_for_new_results(
                get_name_from_display(display_name), custom_tskip, num_images
            ),
            inputs=[example_datasets_dropdown, custom_tskip, num_images_per_class],
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status, class0_context_image, class1_context_image],
            every=1  # Check every second
        )
                
        # Add this event handler:
        example_datasets_dropdown.change(
            fn=reset_galleries,
            inputs=None,
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status]
        )
    
    return demo

# Add the start_lora_training function
def start_lora_training(zip_file, output_dir):
    """Start the LoRA training process in a background thread"""
    global lora_status, lora_is_processing
    
    if lora_is_processing:
        return "LoRA training is already in progress. Please wait for it to complete.", None
    
    if zip_file is None:
        return "Error: No zip file uploaded", None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract the zip file
    try:
        data_dir = extract_archive(zip_file, lora_temp_dir / f"lora_upload_{int(time.time())}")
    except Exception as e:
        return f"Error extracting zip file: {str(e)}", None
    
    # Start training in a background thread
    lora_status = "Starting LoRA training..."
    lora_is_processing = True
    
    thread = threading.Thread(
        target=background_train_lora_process,
        args=(zip_file, output_dir)
    )
    thread.daemon = True
    thread.start()
    
    return "LoRA training started. This may take a while...", None

def background_train_lora_process(zip_file, output_dir):
    """Background process for LoRA training"""
    global lora_status, lora_is_processing, lora_should_stop
    
    try:
        # Reset the stop flag at the start of a new process
        lora_should_stop = False
        
        # Extract the zip file
        data_dir = extract_archive(zip_file, lora_temp_dir / f"lora_upload_{int(time.time())}")
        
        # Update status
        lora_status = "Extracting dataset completed. Starting LoRA training..."
        
        # Train the LoRA model
        result = train_lora(data_dir, output_dir)
        
        if result.startswith("Error"):
            lora_status = f"LoRA training failed: {result}"
        else:
            lora_status = f"LoRA training completed successfully. Model saved to: {result}"
    except Exception as e:
        lora_status = f"Error in LoRA training: {str(e)}"
    finally:
        lora_is_processing = False
        lora_should_stop = False  # Reset the flag when done
        
def handle_zip_upload(input_zip):
    """Handle when a zip file is uploaded"""
    if input_zip is None:
        return None, None, None, None, None, None, None, gr.update(), gr.update()  # Add lora dir update
    
    try:
        # Extract the zip file to a temporary directory
        temp_dir = Path("./temp_uploads") / f"upload_{int(time.time())}"
        data_dir = extract_archive(input_zip, temp_dir)
        
        # Get the original filename without extension
        zip_filename = Path(input_zip).stem
        if zip_filename.endswith('.tar'):  # Handle .tar.gz files
            zip_filename = Path(zip_filename).stem
            
        # Create lora output dir path based on filename
        lora_dir = f"./lora_output/{zip_filename}"
        
        # Verify the required folders exist
        class0_dir = Path(data_dir) / "class0"
        class1_dir = Path(data_dir) / "class1"
        
        if not (class0_dir.exists() and class1_dir.exists()):
            raise ValueError("Uploaded zip must contain 'class0' and 'class1' folders")
        
        # Reset all paths and states for custom upload
        return (
            "",  # description
            None,  # checkpoint_path
            True,  # is_direct_path
            str(data_dir),  # direct_path
            None,  # embeddings_path
            None,  # classifier_path
            85,    # custom_tskip state
            gr.update(value=85),  # Update slider value
            gr.update(value=lora_dir)  # Update lora output dir
        )
    except Exception as e:
        print(f"Error processing zip file: {str(e)}")
        return None, None, None, None, None, None, None, gr.update(), gr.update()
# Make sure the temp directory exists
temp_dir = Path("./temp_uploads")
temp_dir.mkdir(exist_ok=True, parents=True)

def check_lora_status():
    """Check the status of LoRA training"""
    global lora_status
#     return lora_status
# def check_for_new_results():
#     """Check for new results in the directories and update the galleries"""
#     global displayed_results, displayed_results_class0_to_class1, displayed_results_class1_to_class0, images_processed
#     global current_cache_key, is_using_default_params
    
#     # Clear existing results
#     displayed_results = []
#     displayed_results_class0_to_class1 = []
#     displayed_results_class1_to_class0 = []
    
#     # Determine which directory to check based on parameters
#     if is_using_default_params and current_cache_key:
#         base_dir = CACHE_DIR / current_cache_key
#     else:
#         base_dir = Path("./output")
    
#     if base_dir.exists():
#         # Check class0 to class1 directory
#         class0_to_class1_dir = base_dir / "gifs" / "class0_to_class1"
#         if class0_to_class1_dir.exists():
#             for gif_path in class0_to_class1_dir.glob("*.gif"):
#                 gif_str = str(gif_path)
#                 displayed_results_class0_to_class1.append(gif_str)
#                 displayed_results.append(gif_str)
        
#         # Check class1 to class0 directory
#         class1_to_class0_dir = base_dir / "gifs" / "class1_to_class0"
#         if class1_to_class0_dir.exists():
#             for gif_path in class1_to_class0_dir.glob("*.gif"):
#                 gif_str = str(gif_path)
#                 displayed_results_class1_to_class0.append(gif_str)
#                 displayed_results.append(gif_str)
    
#     # Update images_processed based on total GIFs found
#     images_processed = len(displayed_results)
    
#     # Generate the progress status text
#     progress_text = update_progress_status()
    
#     return (
#         displayed_results,
#         displayed_results_class0_to_class1,
#         displayed_results_class1_to_class0,
#         progress_text
#     )


def check_for_new_results(dataset_name, custom_tskip, num_images_per_class):
    """Check for new results in the directories and update the galleries"""
    global displayed_results, displayed_results_class0_to_class1, displayed_results_class1_to_class0, images_processed
    global current_cache_key, is_using_default_params
    
    # First, gather all GIFs into temporary lists
    temp_all_results = []
    temp_class0_to_class1 = []
    temp_class1_to_class0 = []
    
    # Determine which directory to check

    if is_using_default_params(dataset_name, custom_tskip, num_images_per_class) and current_cache_key:
        base_dir = CACHE_DIR / current_cache_key
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = Path("./output")
    
    if base_dir.exists():
        # Gather all GIFs first
        class0_to_class1_dir = base_dir / "gifs" / "class0_to_class1"
        class1_to_class0_dir = base_dir / "gifs" / "class1_to_class0"
        
        # Get all GIFs from class0_to_class1
        if class0_to_class1_dir.exists():
            gifs = sorted(list(class0_to_class1_dir.glob("*.gif")))
            temp_class0_to_class1.extend([str(p) for p in gifs])
            temp_all_results.extend([str(p) for p in gifs])
        
        # Get all GIFs from class1_to_class0
        if class1_to_class0_dir.exists():
            gifs = sorted(list(class1_to_class0_dir.glob("*.gif")))
            temp_class1_to_class0.extend([str(p) for p in gifs])
            temp_all_results.extend([str(p) for p in gifs])

        #get context images
        # Get context images correctly
        class0_context = next((str(p) for p in (base_dir / "context").glob("class0*")), None)
        class1_context = next((str(p) for p in (base_dir / "context").glob("class1*")), None)
    
    # Only update the global variables and return once we have all results
    displayed_results = temp_all_results
    displayed_results_class0_to_class1 = temp_class0_to_class1
    displayed_results_class1_to_class0 = temp_class1_to_class0
    
    # Update images_processed based on total GIFs found
    images_processed = len(displayed_results)
    
    # Generate the progress status text
    progress_text = update_progress_status()
    
    return (
        displayed_results,
        displayed_results_class0_to_class1,
        displayed_results_class1_to_class0,
        progress_text,
        class0_context,
        class1_context,
    )
# Add a function to update dataset info when dropdown changes
def update_dataset_info(dataset_display_name):
    """Update dataset description and paths when dropdown changes"""
    # Find the selected dataset
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        if dataset["display_name"] == dataset_display_name:
            selected_dataset = dataset
            break
    
    if not selected_dataset:
        return "No dataset selected", None, False, None, None, None, None
    
    # Get dataset description
    description = selected_dataset.get("description", "No description available")
    
    # Get paths
    checkpoint_path = selected_dataset.get("checkpoint_path", None)
    direct_path = selected_dataset.get("direct_dataset_path", None)
    is_direct_path = direct_path is not None
    embeddings_path = selected_dataset.get("embeddings_path", None)
    classifier_path = selected_dataset.get("classifier_path", None)
    
    # Set default custom_tskip based on dataset
    custom_tskip = None
    if "butterfly" in dataset_display_name.lower():
        custom_tskip = 70  # Set to 70 for butterfly
    elif "lamp" in dataset_display_name.lower():
        custom_tskip = 85  # Set to 85 for lamp
    
    print(f"Setting custom_tskip to {custom_tskip} for dataset {dataset_display_name}")
    
    return description, checkpoint_path, is_direct_path, direct_path, embeddings_path, classifier_path, custom_tskip



# Function to generate a cache key based on parameters
def get_cache_key(dataset_name, checkpoint_path, train_clf, embeddings_path, 
                 classifier_path, use_classifier_stopping, custom_tskip, num_images_per_class):
    """Generate a unique cache key based on the processing parameters"""
    # Create a dictionary of parameters
    params = {
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "train_clf": train_clf,
        "embeddings_path": str(embeddings_path),
        "classifier_path": str(classifier_path),
        "use_classifier_stopping": use_classifier_stopping,
        "custom_tskip": custom_tskip,
        "num_images_per_class": num_images_per_class
    }
    
    # Convert to JSON string and hash
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

def change_cache_key(dataset_name, num_images_per_class, train_clf, use_classifier_stopping, custom_tskip):
    """Change the cache key based on the selected dataset"""
    global current_cache_key
    
    # Find the selected dataset from EXAMPLE_DATASETS
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        if dataset["display_name"] == dataset_name:
            selected_dataset = dataset
            break
    
    if not selected_dataset:
        print(f"No dataset found for name: {dataset_name}")
        return
    
    # Get all parameters from the selected dataset
    checkpoint_path = selected_dataset.get("checkpoint_path", None)
    direct_path = selected_dataset.get("direct_dataset_path", None)
    embeddings_path = selected_dataset.get("embeddings_path", None)
    classifier_path = selected_dataset.get("classifier_path", None)
    

    
    # Generate and set the cache key
    current_cache_key = get_cache_key(
        selected_dataset["name"],  # Use internal name instead of display name
        checkpoint_path,
        train_clf,
        embeddings_path,
        classifier_path,
        use_classifier_stopping,
        custom_tskip,
        num_images_per_class
    )
    print(selected_dataset["name"], checkpoint_path, train_clf, embeddings_path, classifier_path, use_classifier_stopping, custom_tskip, num_images_per_class)
    #import pdb; pdb.set_trace()

# Function to check if cached results exist
def check_cache(cache_key):
    """Check if cached results exist for the given key"""
    cache_path = CACHE_DIR / cache_key
    return cache_path.exists() and (cache_path / "gifs").exists()

# Add this function to create context images for each class
def create_context_image(image_paths, output_path, title, preferred_index=0):
    """Create a context image showing samples from a class
    
    Args:
        image_paths: List of paths to images in the class
        output_path: Where to save the context image
        title: Title for the image
        preferred_index: Index of the preferred image to use (default: 0)
    """
    if not image_paths:
        # Create a blank image if no samples are available
        img = Image.new('RGB', (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        draw.text((256, 256), "No samples available", fill=(80, 80, 80), font=font, anchor="mm")
        img.save(output_path)
        return
    
    # Use the preferred index if available, otherwise use the first image
    img_index = min(preferred_index, len(image_paths) - 1)
    img = Image.open(image_paths[img_index]).convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS)
    
    # Add title
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    # Draw a semi-transparent background for the title
    draw.rectangle([(0, 0), (img.width, 50)], fill=(0, 0, 0, 180))
    
    # Save the context image
    img.save(output_path)

# Fix the update_custom_tskip function
def update_custom_tskip(tskip_value):
    """Update the custom_tskip input field with the value from the state"""
    print(f"Updating custom_tskip input with value: {tskip_value}")
    if tskip_value is None:
        return ""
    return str(tskip_value)  # Convert to string for the text input

# Add this function to directly set the custom_tskip based on dataset name
def set_custom_tskip_for_dataset(dataset_name):
    """Set the custom_tskip value based on the selected dataset"""
    if dataset_name is None:
        return 85
    if "butterfly" in dataset_name.lower():
        return 70
    elif "lamp" in dataset_name.lower():
        return 85
    else:
        return 85



# Add this to the interface if you want a button to manually save to cache
if __name__ == "__main__":
    # Uncomment this line to save current results to cache
    #save_current_results_to_cache()
    
    demo = create_gradio_interface()
    demo.launch()

# Add these functions at the top of the file, after the imports and global variables
# but before any other function definitions

# 