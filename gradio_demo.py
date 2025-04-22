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
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
import tarfile
import queue
import hashlib
import json


# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Define constants
IMG_SIZE = 512
BATCH_SIZE = 32
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
        "path": "/proj/vondrick/datasets/magnification/butterfly.tar.gz",
        "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/example_images/butterfly",
        "checkpoint_path": "/proj/vondrick2/mia/magnificationold/output/lora/butterfly/copper-forest-49/checkpoint-1800",
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/butterfly",
        "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/butterfly",
        "class_names": ["class0", "class1"]
    },
        {
        "name": "afhq",
        "display_name": "Cats vs. Dogs (AFHQ)",
        "description": "Dataset containing images of table lamps and floor lamps",
        "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/example_images/afhq",
        "checkpoint_path": None,
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/afhq",
        "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/afhq",
        "class_names": ["class0", "class1"]
    },
    {
        "name": "lamp",
        "display_name": "Lamps",
        "description": "Dataset containing images of table lamps and floor lamps",
        "path": "compressed_datasets/lampsfar.zip",
        "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/example_images/lamps",
        "checkpoint_path": "/proj/vondrick2/mia/diff-usion/lora_output_lampsfar/checkpoint-800",
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/lampsfar",
        "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/lampsfar",
        "class_names": ["class0", "class1"]
    },
    {
        "name": "couches",
        "display_name": "Couches",
        "description": "Dataset containing images of chairs and floor",
        "path": "compressed_datasets/couches.zip",
        "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/example_images/couches",
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/couches",
        "checkpoint_path": "/proj/vondrick2/mia/diff-usion/lora_output/couches/checkpoint-1000",
        "class_names": ["class0", "class1"]
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
        #import pdb; pdb.set_trace()
        self.transform = transform
        if 'kermany' in str(self.root_dir):
            #import pdb; pdb.set_trace()
            self.class0_dir = self.root_dir / "NORMAL"
            self.class1_dir = self.root_dir / "DRUSEN"
        elif 'kiki_bouba' in str(self.root_dir):
            self.class0_dir = self.root_dir / "kiki"
            self.class1_dir = self.root_dir / "bouba"
        elif 'afhq' in str(self.root_dir):
            self.class0_dir = self.root_dir / "dog"
            self.class1_dir = self.root_dir / "cat"
        else:
            self.class0_dir = self.root_dir / "class0"
            self.class1_dir = self.root_dir / "class1"
        
        # Get image paths
        #import pdb; pdb.set_trace()
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
                                 manip_val=2):
    
    print(f"\nProcessing with dataset: {dataset_display_name}")
    
    # Find the selected dataset
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        if dataset["display_name"] == dataset_display_name:
            selected_dataset = dataset
            break
    
    if not selected_dataset:
        print("Error: No dataset selected")
        return "No dataset selected", [], [], [], "Error: No dataset selected", None, None
    
    # Generate cache key
    cache_key = get_cache_key(
        selected_dataset["name"], checkpoint_path, False, embeddings_path, 
        classifier_path, use_classifier_stopping, custom_tskip, 
        manip_val, 
    )
    
    print(f"Generated cache key: {cache_key}")
    
    # Check if cache exists
    cache_path = CACHE_DIR / cache_key
    dataset_dir = cache_path / "gifs"
    print(f"Looking for cache in: {cache_path}")
    print(f"Looking for gifs in: {dataset_dir}")
    print(f"Cache exists: {cache_path.exists()}")
    print(f"Gifs dir exists: {dataset_dir.exists()}")

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

        status_message = f"Using cached results with t-skip={custom_tskip}, manip_scale={manip_val}"
        
        # Return cached results
        return (
            "Using cached results for default parameters.", 
            displayed_results,
            displayed_results_class0_to_class1,
            displayed_results_class1_to_class0,
            status_message, 
            str(class0_context) if class0_context else None, 
            str(class1_context) if class1_context else None
        )

    else:
        print("No cached results found, processing dataset...")
        return "No cached results found, processing dataset...", [], [], [], "No cached results found, processing dataset...", None, None
    return 



# def process_and_clear(example_datasets_dropdown, checkpoint_path_state, 
#                      is_direct_path_state, direct_path_state, embeddings_path_state, 
#                      classifier_path_state, use_classifier_stopping, custom_tskip,
#                      manip_val):
#     """Clear folders first, then process the dataset"""
#     # Clear folders first
#     clear_output_folders()
    
#     # Then process the dataset
#     return process_with_selected_dataset(
#         None,  # input_zip (always None)
#         "./output",  # output_dir (hardcoded)
#         example_datasets_dropdown, 
#         checkpoint_path_state,
#         False,  # train_clf (always False)
#         is_direct_path_state,
#         direct_path_state,
#         embeddings_path_state,
#         classifier_path_state,
#         use_classifier_stopping,
#         custom_tskip,
#         manip_val
#     )

def process_and_clear(example_datasets_dropdown, checkpoint_path_state, 
                     is_direct_path_state, direct_path_state, embeddings_path_state, 
                     classifier_path_state, use_classifier_stopping, custom_tskip,
                     manip_val):
    """Clear galleries first, then process the dataset"""
    # Clear galleries but keep example images
    clear_output_folders()
    
    # Process the dataset
    result = process_with_selected_dataset(
        None,  # input_zip (always None)
        "./output",  # output_dir (hardcoded)
        example_datasets_dropdown, 
        checkpoint_path_state,
        False,  # train_clf (always False)
        is_direct_path_state,
        direct_path_state,
        embeddings_path_state,
        classifier_path_state,
        use_classifier_stopping,
        custom_tskip,
        manip_val
    )
    
    # Return all outputs except example images
    return (
        result[1],  # gallery
        result[2],  # gallery_class0_to_class1
        result[3],  # gallery_class1_to_class0
        result[4],  # progress_status    # Don't update class1_context_image
    )

def update_example_images(dataset_display_name):
    """Update the example images based on the selected dataset"""
    print(f"\nUpdating example images for {dataset_display_name}")
    
    # Find the dataset info
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        print(f"Checking dataset: {dataset['display_name']}", dataset_display_name)
        if dataset["display_name"] == dataset_display_name:
            selected_dataset = dataset
            print(f"Selected dataset: {selected_dataset}")
            break

    class_names = selected_dataset.get("class_names", None)
    
    if selected_dataset:
        dataset_dir = selected_dataset.get("direct_dataset_path")
        print(f"Dataset directory: {dataset_dir}")
        
        if dataset_dir:
            # Debug: List all files in the directory
            print("Contents of directory:")
            for path in Path(dataset_dir).rglob("*"):
                print(f"  {path}")
            
            # Try to find class0 and class1 images

            class0_path = Path(dataset_dir) / class_names[0]
            class1_path = Path(dataset_dir) / class_names[1]
            print(f"Looking in class0: {class0_path}")
            print(f"Looking in class1: {class1_path}")
            
            class0_img = next((str(p) for p in Path(dataset_dir).glob(f"{class_names[0]}/*.*")), None)
            class1_img = next((str(p) for p in Path(dataset_dir).glob(f"{class_names[1]}/*.*")), None)
            
            print(f"Found images:\nclass0={class0_img}\nclass1={class1_img}")
            return class0_img, class1_img
    
    print("No images found")
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
                        <p class="contact-info">For any questions/comments/issues with this demo, please email mia.chiquier@cs.columbia.edu.🤖</p>
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
                    
                    # with gr.Column(elem_classes="upload-info"):
                    #     gr.HTML("""
                    #         <p><strong>Dataset Format:</strong> Upload a zip file containing two folders named 'class0' and 'class1', 
                    #         each containing images of the respective class.</p>
                    #     """)
                    
                    # with gr.Row():
                    #     input_zip = gr.File(
                    #         label="Upload Custom Dataset (ZIP or TAR.GZ file)",
                    #         file_types=[".zip", ".tar.gz", ".tgz"],
                    #         type="filepath"
                    #     )
                    #     # Hide the output directory by using elem_classes
                    #     output_dir = gr.Textbox(
                    #         label="Output Directory", 
                    #         value="./output",
                    #         elem_classes="hidden-element"
                    #     )
                    
                    # with gr.Row():
                    #     gr.HTML('<div class="section-header">LoRA Training</div>')
                    
                    # with gr.Column(elem_classes="upload-info"):
                    #     gr.HTML("""
                    #         <p><strong>Dataset Format:</strong> Upload a zip file containing two folders named 'class0' and 'class1', 
                    #         each containing images of the respective class for training the LoRA model.</p>
                    #     """)
                    
                    # with gr.Row():
                    #     lora_output_dir = gr.Textbox(
                    #         label="LoRA Output Directory", 
                    #         value="./lora_output"
                    #     )
                    
                    #     gr.HTML("""
                    #         <div class="parameter-box">
                    #             <p>Default LoRA Training Parameters:</p>
                    #             <ul>
                    #                 <li>Epochs: 5</li>
                    #                 <li>Learning Rate: 1e-4</li>
                    #                 <li>Batch Size: 32</li>
                    #                 <li>LoRA Rank: 4</li>
                    #                 <li>LoRA Alpha: 32</li>
                    #                 <li>Max Training Steps: 1000</li>
                    #             </ul>
                    #         </div>
                    #     """)
                    
                    # train_lora_btn = gr.Button("Train LoRA Model", elem_classes="btn-primary")
                    # lora_status_box = gr.Textbox(label="LoRA Training Status", value="Ready to train LoRA model")
                    # train_clf = gr.Checkbox(label="Train New Classifiers", value=False)
                    
                    with gr.Row():
                        use_classifier_stopping = gr.State(False)# 
                        
                        custom_tskip = gr.Dropdown(
                        choices=[55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                        value=85,  # default value
                        label="Custom T-Skip Value",
                        info="Select a t-skip value",
                        visible=True
                    )
                    
                    # Add a text box for number of images per class
                    with gr.Row():
                        manip_val = gr.Dropdown(
                        choices=[1.0, 1.5, 2.0],
                        value=2.0,  # default value
                        label="Manip scale",
                        info="Select a manip scale",
                        visible=True
                        )
                        #
                    
                    with gr.Row():
                        process_btn = gr.Button("Generate Counterfactuals", elem_classes="btn-primary")
                        cancel_btn = gr.Button("Cancel Generation", elem_classes="btn-primary")
                    
                    # Status for the main column
                    #status = gr.Textbox(label="Status", value="Ready to generate counterfactuals")
            
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

               

                default_dataset = next((dataset["display_name"] for dataset in EXAMPLE_DATASETS if "lamps" in dataset["display_name"].lower()), None)
                if default_dataset:
                    # Initial load of example images
                    class0_img, class1_img = update_example_images(default_dataset)
                    if class0_img and class1_img:
                        class0_context_image.value = class0_img  # Directly set the value
                        class1_context_image.value = class1_img

                print(f"Class 0 image: {class0_context_image.value}")
                print(f"Class 1 image: {class1_context_image.value}")
                
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
        #set the default to these to be those for the lamp dataset  
        default_dataset = next((dataset for dataset in EXAMPLE_DATASETS if "lamp" in dataset["display_name"].lower()), None)
        if default_dataset:
            checkpoint_path_state = gr.State(default_dataset["checkpoint_path"])
            is_direct_path_state = gr.State(False)
            direct_path_state = gr.State(None)
            embeddings_path_state = gr.State(default_dataset["embeddings_path"])
            classifier_path_state = gr.State(default_dataset["classifier_path"])

        process_btn.click(
            fn=process_and_clear,
            inputs=[
                example_datasets_dropdown, checkpoint_path_state, 
                is_direct_path_state, direct_path_state, embeddings_path_state, 
                classifier_path_state, use_classifier_stopping, custom_tskip,
                manip_val
            ],
            outputs=[
                gallery,                    # Make sure these variables are all defined
                gallery_class0_to_class1,   # and not None
                gallery_class1_to_class0, 
                progress_status
            ]  # Removed 'status' since it wasn't defined
        )

        # Set up the cancel button click handler
        cancel_btn.click(
            fn=cancel_generation,
            inputs=None,
            outputs=None
        )

        num_images_per_class = gr.State(10)

        example_datasets_dropdown.change(
            fn=reset_galleries,  # Reset galleries but not example images
            inputs=None,
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status]
        ).then(  # Update dataset info
            fn=update_dataset_info,
            inputs=example_datasets_dropdown,
            outputs=[dataset_description, checkpoint_path_state, is_direct_path_state, direct_path_state, 
                    embeddings_path_state, classifier_path_state, custom_tskip]
        ).then(  # Set custom t-skip
            fn=set_custom_tskip_for_dataset,
            inputs=example_datasets_dropdown,
            outputs=custom_tskip
        ).then(  # Change cache key
            fn=change_cache_key,
            inputs=[example_datasets_dropdown, num_images_per_class, use_classifier_stopping, custom_tskip],
            outputs=None
        ).then(  # Update example images
            fn=update_example_images,
            inputs=example_datasets_dropdown,
            outputs=[class0_context_image, class1_context_image]
        ).then(  # Automatically generate counterfactuals when dataset changes
            fn=process_and_clear,
            inputs=[
                example_datasets_dropdown, checkpoint_path_state, 
                is_direct_path_state, direct_path_state, embeddings_path_state, 
                classifier_path_state, use_classifier_stopping, custom_tskip,
                manip_val
            ],
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, 
                     progress_status]
        )

        # Load initial example images and generate counterfactuals for default dataset (Lamps)
        demo.load(
            fn=update_example_images,
            inputs=example_datasets_dropdown,
            outputs=[class0_context_image, class1_context_image]
        ).then(  # Initial counterfactual generation
            fn=process_and_clear,
            inputs=[
                example_datasets_dropdown, checkpoint_path_state, 
                is_direct_path_state, direct_path_state, embeddings_path_state, 
                classifier_path_state, use_classifier_stopping, custom_tskip,
                manip_val
            ],
            outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, 
                     progress_status]
        )

        # example_datasets_dropdown.change(
        #     fn=reset_galleries,  # Reset first
        #     inputs=None,
        #     outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status]
        # ).then(  # Update dataset info
        #     fn=update_dataset_info,
        #     inputs=example_datasets_dropdown,
        #     outputs=[dataset_description, checkpoint_path_state, is_direct_path_state, direct_path_state, 
        #             embeddings_path_state, classifier_path_state, custom_tskip_state]
        # ).then(  # Set custom t-skip
        #     fn=set_custom_tskip_for_dataset,
        #     inputs=example_datasets_dropdown,
        #     outputs=custom_tskip
        # ).then(  # Change cache key
        #     fn=change_cache_key,
        #     inputs=[example_datasets_dropdown, manip_val, use_classifier_stopping, custom_tskip],
        #     outputs=None
        # ).then(  # Update example images
        #     fn=lambda display_name: update_example_images(display_name),
        #     inputs=example_datasets_dropdown,
        #     outputs=[class0_context_image, class1_context_image]
        # )
      

        # process_btn.click(
        #     fn=process_and_clear,
        #     inputs=[
        #         example_datasets_dropdown, checkpoint_path_state, 
        #         is_direct_path_state, direct_path_state, embeddings_path_state, 
        #         classifier_path_state, use_classifier_stopping, custom_tskip,
        #         manip_val
        #     ],
        #     outputs=[status, gallery, gallery_class0_to_class1, gallery_class1_to_class0, 
        #             progress_status, class0_context_image, class1_context_image]
        # )

        
        # # Set up the click event for LoRA training
        # train_lora_btn.click(
        #     fn=start_lora_training,
        #     inputs=[input_zip, lora_output_dir],
        #     outputs=[lora_status_box]
        # )
        
        # # Set up periodic status checking for LoRA training
        # demo.load(
        #     fn=check_lora_status,
        #     inputs=None,
        #     outputs=lora_status_box,
        #     every=5  # Check every 5 seconds
        # )
        
        # Add a periodic refresh for the galleries
        # Add a periodic refresh for the galleries

                
        # Add this event handler:
        # example_datasets_dropdown.change(
        #     fn=reset_galleries,
        #     inputs=None,
        #     outputs=[gallery, gallery_class0_to_class1, gallery_class1_to_class0, progress_status]
        # )
    
    return demo



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
                 classifier_path, use_classifier_stopping, custom_tskip, manip_val):
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
        "manip_val": float(manip_val)
    }
    print(f"Params: {params}")
    
    # Convert to JSON string and hash
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()
def change_cache_key(dataset_name, manip_val, use_classifier_stopping, custom_tskip):
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
    embeddings_path = selected_dataset.get("embeddings_path", None)
    classifier_path = selected_dataset.get("classifier_path", None)

    
    # Generate and set the cache key
    current_cache_key = get_cache_key(
        selected_dataset["name"],  # Use internal name instead of display name
        checkpoint_path,
        False,  # train_clf is always False
        embeddings_path,
        classifier_path,
        use_classifier_stopping,
        custom_tskip,
        manip_val
    )

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

if __name__ == "__main__":
    # Uncomment this line to save current results to cache
    #save_current_results_to_cache()
    
    demo = create_gradio_interface()
    demo.launch()

# Add these functions at the top of the file, after the imports and global variables
# but before any other function definitions

# 
