import itertools
from pathlib import Path
import json
import hashlib
import shutil



EXAMPLE_DATASETS = [
    {
        "name": "afhq",
        "display_name": "Cats vs. Dogs (AFHQ)",
        "description": "Dataset containing images of table lamps and floor lamps",
        "direct_dataset_path": "/proj/vondrick2/orr/projects/stargan-v2/data/afhq/train",
        "checkpoint_path": None,
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/afhq",
        "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/afhq"
    },
        {
        "name": "butterfly",
        "display_name": "Butterfly (Monarch vs Viceroy)",
        "description": "Dataset containing images of Monarch and Viceroy butterflies for counterfactual generation",
        "path": "/proj/vondrick/datasets/magnification/butterfly.tar.gz",
        "direct_dataset_path": "/proj/vondrick/datasets/magnification/butterflygrad",
        "checkpoint_path": "/proj/vondrick2/mia/magnificationold/output/lora/butterfly/copper-forest-49/checkpoint-1800",
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/butterfly",
        "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/butterfly"
    },
    {
        "name": "couches",
        "display_name": "Couches",
        "description": "Dataset containing images of chairs and floor",
        "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/couches",
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/couches",
        "checkpoint_path": "/proj/vondrick2/mia/diff-usion/lora_output/couches/checkpoint-1000",
    },
        {
        "name": "lamp",
        "display_name": "Lamps",
        "description": "Dataset containing images of table lamps and floor lamps",
        "direct_dataset_path": "/proj/vondrick2/mia/diff-usion/lampsfar",
        "checkpoint_path": "/proj/vondrick2/mia/diff-usion/lora_output_lampsfar/checkpoint-800",
        "embeddings_path": "/proj/vondrick2/mia/diff-usion/results/clip_image_embeds/lampsfar",
        "classifier_path": "/proj/vondrick2/mia/diff-usion/results/ensemble/lampsfar"
    }
]
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
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

def generate_cache_list(dataset, t_skip_options, manip_val_options):
    """Generate list of cache keys for a dataset"""
    cache_keys = []
    classifier_stopping_options = [False]
    
    # Get dataset paths
    checkpoint_path = dataset.get("checkpoint_path")
    embeddings_path = dataset.get("embeddings_path")
    classifier_path = dataset.get("classifier_path")
    
    # Generate all combinations
    combinations = list(itertools.product(
        manip_val_options,
        t_skip_options,
        classifier_stopping_options
    ))
    
    for manip, t_skip, use_clf_stop in combinations:
        cache_key = generate_cache_key(
            dataset["name"],
            checkpoint_path,
            embeddings_path,
            classifier_path,
            use_clf_stop,
            t_skip,
            manip
        )
        cache_keys.append(cache_key)
    
    return cache_keys

def main():
    # Generate cache list
    all_cache_keys = []
    t_skip_options = np.arange(50, 100, 5).tolist()
    manip_val_options = [1.0, 1.5, 2.0]
    
    for dataset in EXAMPLE_DATASETS:
        cache_keys = generate_cache_list(dataset, t_skip_options, manip_val_options)
        all_cache_keys.extend(cache_keys)
    
    # Write cache keys to file
    with open('cached_list.txt', 'w') as f:
        for key in all_cache_keys:
            f.write(f"{key}\n")
    
    # Create copy of specified cache folders
    source_dir = Path("./cached_results")
    target_dir = Path("./cached_results_copy")
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # Read cache list
    with open('cached_list.txt', 'r') as f:
        wanted_caches = set(line.strip() for line in f)
    
    # Copy only the specified cache folders
    for cache_dir in source_dir.iterdir():
        if cache_dir.name in wanted_caches:
            print(f"Copying {cache_dir.name}")
            shutil.copytree(cache_dir, target_dir / cache_dir.name, dirs_exist_ok=True)

if __name__ == "__main__":
    import numpy as np
    main()