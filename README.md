# Teaching Humans Subtle Differences with *DIFF*usion
<!-- <a href="https://openreview.net/forum?id=rm9ewAwLTR&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dthecvf.com%2FICCV%2F2025%2FConference%2FAuthors%23your-submissions)"><img src="https://img.shields.io/badge/arXiv-2308.02669-b31b1b.svg" height=20.5></a> -->
<a href="https://diff-usion.cs.columbia.edu/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

Official implementation of **Teaching Humans Subtle Differences with *DIFF*usion**.

![](assets/teaser.png)
We illustrate the counterfactual results from our methods on the Butterfly dataset, the Black
Hole dataset, and the Retina dataset. In the Butterfly dataset, the Viceroy has a cross-sectional line (${\color{yellow}\text{yellow}}$), a smaller head with less dots
(${\color{magenta}\text{magenta}}$), and more “scaley” dots (${\color{blue}\text{blue}}$), compared to the Monarch. In the Black Hole dataset, SANE has more uniform wisps (${\color{yellow}\text{yellow}}$)
and less of a prominent photon ring (${\color{blue}\text{blue}}$) as compared to MAD, with these distinguishing features discovered through our method rather
than known a priori. In the Retina dataset, normal retinas lack the horizontal line bumps (${\color{yellow}\text{yellow}}$) present in retinas with drusen.


## Setup
Create conda environment:
```bash
$ conda create -p /proj/vondrick4/mia/condaenvs/diff-usion python=3.9
$ conda activate diff-usion
```
Clone and install requirements:
```bash
$ git clone https://github.com/cvlab-columbia/diff-usion
$ cd diff-usion
$ pip install -r requirements.txt
```

## Gradio demo 

We've made a gradio demo that automatically runs our method end-to-end given just your dataset (extracts embeds, trains classifiers, finetune's the diffusion model, and runs our arithmetic edit!). To run it in reasonable time, it requires GPUs, so we've provided the gradio demo code for you to run on your own machine. Alternatively, you can run the code yourself via the next sections.

To run the gradio demo, run:
```bash
$ python gradio_diff-usion_demo.py
```

## Reproducing our results 

First, download the datasets: [AFHQ](https://www.kaggle.com/datasets/dimensi0n/afhq-512), [Retina](https://www.kaggle.com/datasets/paultimothymooney/kermany2018). The [Butterfly](https://drive.google.com/file/d/1AFp4t0ykNqOpYcxFeLJBQgOIk5jYaSwE/view?usp=sharing) dataset was obtained from iNaturalist and the [KikiBouba](https://drive.google.com/file/d/17ibF3tzFiZrMb9ZnpYlLEh-xmWkPJpNH/view?usp=drive_link) dataset was generated from the [Kiki Bouba generative model](https://github.com/TAU-VAILab/kiki-bouba) repository, but we have provided the compressed folders in the google drive links for ease of use. We are not able to release the black hole dataset publicly. 

Next, download the fine-tuned lora weights, clip embeddings, and ensemble classifiers from [here](https://drive.google.com/file/d/1pSI9gh9nD74A3O7CRDw4iD3fL2Boj9Hk/view?usp=sharing). Place the `results` folder in the root directory. 

Then, run the following commands to reproduce our results:
```bash
$ python eval.py --config_path configs/edit/retina.yaml
$ python make_gif.py --config_path configs/edit/retina.yaml
```

You can also reproduce our results on our datasets without downloading anything beyond the datasets by reproducing the clip embeddings, ensemble classifiers, and lora weights yourself via the instructions in the next section.

## Usage on your own dataset 
### Dataset format
To use your own dataset, add a dataset class to the `datasets.py` file. This dataset should return a tuple of (image, label, filename). There should only be two classes, and the labels should be 0 or 1. You will also need to modify the `get_cls_dataset_by_name` function to return your dataset, and create a config file for each folder in the `configs` folder: one for fine-tuning lora (optional) the diffusion model (`lora`), one for training the ensemble classifiers (`ensemble`), and one for editing (`edit`). You can copy the structure from the existing configs for the Retina dataset. 

### (Optional) Domain Tuning on YOUR dataset
To finetune our diffusion decoder on a new dataset, modify the example config file according to your dataset location and run:
```bash
$ python kandinsky_lora_train.py --config_path configs/lora/retina.yaml
```

### Locate source and target CLIP embedding files (or create your own)
To save CLIP features for YOUR own dataset, run:
```bash
$ python scripts/save_embeds.py
```

### Locate ensemble classifiers (or train your own)
To train our set of ensemble classifiers, run:
```bash
$ python scripts/ensemble_train.py --config_path configs/ensemble/retina.yaml
```

Note that classifiers are not strictly necessary to generate fine-grained edits. The classifier is used to determine the highest tskip while also still flipping the prediction from the classifiers, but you could also simply pick the tskip that you prefer.

### Modify config file
The image editing config file is in `configs/edit` folder, and we have provided an example one for the retina dataset.

### Run inference
```bash
$ python eval.py --config_path configs/edit/retina.yaml
```

### Create GIFs from results
```bash
$ python make_gif.py --config_path configs/edit/retina.yaml
```

### Create interpolations
You might also want to see how the strength of the manipulation scalar affects the interpolation. To do this, run:
```bash
$ python eval_save_interp.py --config_path configs/edit/retina.yaml
```

## Citation
If our code or models aided your research, please cite our [paper](https://arxiv.org/pdf/2504.08046):
```
@misc{chiquier2025teaching,
  title={Teaching Humans Subtle Differences with DIFFusion},
  author={Chiquier, Mia and Avrech, Orr and Gandelsman, Yossi and Feng, Berthy and Bouman, Katherine and Vondrick, Carl},
  journal={arXiv preprint arXiv:2504.08046},
  year={2025}
}		
```
