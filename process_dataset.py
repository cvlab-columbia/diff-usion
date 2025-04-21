import itertools
from pathlib import Path
import json
import hashlib
import time
import torch
from models.kandinsky_pipelines import KandinskyV22PipelineWithInversion
import shutil
from gradio_demo_simple import background_process_nodefault
import numpy as np

# EXAMPLE_DATASETS = [
#     {
#         "name": "lamp",
#         "display_name": "Lamps",
#         "description": "Dataset containing images of table lamps and floor lamps",
#         "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/lampsfar",
#         "checkpoint_path": "/proj/vondrick2/mia/diff-usion/lora_output_lampsfar/checkpoint-800",
#         "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/lampsfar",
#         "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/lampsfar"
#     }
# ]

#/proj/vondrick/datasets/magnification/kermany

EXAMPLE_DATASETS = [
    {
        "name": "kermany",
        "display_name": "Retina",
        "description": "Dataset containing images of table lamps and floor lamps",
        "direct_dataset_path": "/proj/vondrick/datasets/magnification/kermany/train",
        "class_names": ["DRUSEN", "NORMAL"],
        "checkpoint_path": "/proj/vondrick2/mia/diff-usion/output/lora/kermany/balmy-snowball-84/checkpoint-2000",
        "embeddings_path": "/proj/vondrick2/mia/magnificationold/results/clip_image_embeds/kermany",
        "classifier_path": None
    }
]


# EXAMPLE_DATASETS = [
#     {
#         "name": "kikibouba",
#         "display_name": "KikiBouba",
#         "description": "Dataset containing images of table lamps and floor lamps",
#         "direct_dataset_path": "/proj/vondrick4/mia/kiki_bouba_v2_split/train",
#         "checkpoint_path": None,
#         "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/kikibouba",
#         "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/kikibouba"
#     }
# ]

# EXAMPLE_DATASETS = [
#     {
#         "name": "afhq",
#         "display_name": "Cats vs. Dogs (AFHQ)",
#         "description": "Dataset containing images of table lamps and floor lamps",
#         "direct_dataset_path": "/proj/vondrick2/orr/projects/stargan-v2/data/afhq/train",
#         "checkpoint_path": None,
#         "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/afhq",
#         "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/afhq"
#     }
# ]

# Update the EXAMPLE_DATASETS to include direct dataset paths, embeddings, and classifiers
# EXAMPLE_DATASETS = [
#     {
#         "name": "butterfly",
#         "display_name": "Butterfly (Monarch vs Viceroy)",
#         "description": "Dataset containing images of Monarch and Viceroy butterflies for counterfactual generation",
#         "path": "/proj/vondrick/datasets/magnification/butterfly.tar.gz",
#         "direct_dataset_path": "/proj/vondrick/datasets/magnification/butterflygrad",
#         "checkpoint_path": "/proj/vondrick2/mia/magnificationold/output/lora/butterfly/copper-forest-49/checkpoint-1800",
#         "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/butterfly",
#         "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/butterfly"
#     }
# ]


# EXAMPLE_DATASETS = [
#     {
#         "name": "couches",
#         "display_name": "Couches",
#         "description": "Dataset containing images of chairs and floor",
#         "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/couches",
#         "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/couches",
#         "checkpoint_path": "/proj/vondrick2/mia/diff-usion/lora_output/couches/checkpoint-1000",
#     }
# ]



def generate_cache_key(dataset_name, checkpoint_path, embeddings_path, 
                      classifier_path, use_classifier_stopping, custom_tskip, manip_val):
    """Generate a unique cache key based on the processing parameters"""
    params = {
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "train_clf": False,
        "embeddings_path": str(embeddings_path),
        "classifier_path": str(classifier_path),
        "use_classifier_stopping": use_classifier_stopping,
        "custom_tskip": custom_tskip,
        "manip_val": manip_val
    }
    #import pdb; pdb.set_trace()
    print(params)
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


import os
def process_dataset(dataset_name, manip_val, t_skip, use_clf_stop, output_dir, device="cuda:3"):
    """Process a dataset with given parameters"""
    # Find the dataset info
    selected_dataset = None
    for dataset in EXAMPLE_DATASETS:
        if dataset["name"] == dataset_name:
            selected_dataset = dataset
            break
    
    if not selected_dataset:
        print(f"Dataset {dataset_name} not found!")
        return False
    
    try:
        # Create output directories
        if use_clf_stop:
            t_skip = None
        
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        gifs_dir = output_dir / "gifs"
        os.makedirs(gifs_dir, exist_ok=True)
        gifs_dir.mkdir(parents=True, exist_ok=True)
        class0_to_class1_dir = gifs_dir / "class0_to_class1"
        class1_to_class0_dir = gifs_dir / "class1_to_class0"
        class0_to_class1_dir.mkdir(exist_ok=True, parents=True)
        class1_to_class0_dir.mkdir(exist_ok=True, parents=True)
        
        # Get paths from dataset
        checkpoint_path = selected_dataset.get("checkpoint_path")
        embeddings_path = selected_dataset.get("embeddings_path")
        classifier_path = selected_dataset.get("classifier_path")
        
        # Process the dataset using background_process
        result = background_process_nodefault(
            dataset_dir=selected_dataset.get("direct_dataset_path"),
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            train_clf=False,  # Always False now
            embeddings_path=embeddings_path,
            classifier_path=classifier_path,
            use_classifier_stopping=use_clf_stop,
            custom_tskip=t_skip,
            manip_val=manip_val,
            dataset_name=dataset_name,
            device=device
        )
        
        # Create a completed.txt file to indicate successful processing
        (output_dir / "completed.txt").touch()
        return True
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return False

def precache_dataset(dataset, t_skip_options=[15], manip_val=[1.0], device="cuda:3"):
    """Generate caches for all parameter combinations for a dataset"""
    # Parameter combinations to try
    num_images_options = [10]  # Different numbers of images
    classifier_stopping_options = [False]  # Whether to use classifier stopping
    num_images = 10
    
    # Get dataset paths
    checkpoint_path = dataset.get("checkpoint_path")
    embeddings_path = dataset.get("embeddings_path")
    classifier_path = dataset.get("classifier_path")
    
    # Generate all combinations
    combinations = list(itertools.product(
        manip_val,
        t_skip_options,
        classifier_stopping_options
    ))
    
    print(f"\nProcessing dataset: {dataset['name']}")
    print(f"Total combinations to process: {len(combinations)}")
    print(combinations)
    
    for manip, t_skip, use_clf_stop in combinations:
        # Generate cache key
        cache_key = generate_cache_key(
            dataset["name"],
            checkpoint_path,
            embeddings_path,
            classifier_path,
            use_clf_stop,
            t_skip,
            manip
        )
        
        cache_dir = Path("./cached_results2") / cache_key


        
        # Skip if cache already exists
        if cache_dir.exists() and (cache_dir / "completed.txt").exists():
            print(f"Cache exists for {dataset['name']} with params: images={num_images}, t_skip={t_skip}, clf_stop={use_clf_stop}")
            continue
            
        print(f"\nGenerating cache for {dataset['name']}")
        print(f"Parameters: images={num_images}, t_skip={t_skip}, clf_stop={use_clf_stop}")
        
        # Process the dataset
        success = process_dataset(
            dataset_name=dataset["name"],
            manip_val=manip,
            t_skip=t_skip,
            use_clf_stop=use_clf_stop,
            output_dir=cache_dir,
            device=device
        )
        
        if success:
            print("Cache generation successful")
        else:
            print("Cache generation failed")
            # Optionally clean up failed cache
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        
        # Optional: Add delay between processing to prevent overload
        time.sleep(5)

def main():
    # Create cache directory
    cache_dir = Path("./cached_results2")
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each dataset
    for dataset in EXAMPLE_DATASETS:
        precache_dataset(dataset, t_skip_options=np.arange(50,100,5).tolist(), manip_val=[1.0, 1.5, 2.0], device="cuda:0")

if __name__ == "__main__":
    main()