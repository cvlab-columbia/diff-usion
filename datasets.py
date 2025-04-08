import os
import sys
import json
import torch
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, random_split
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from utils.templates import INSTRUCT_PREFIX_TEMPLATES, IMAGENET_CLIP_TEMPLATES
from textual_inversion_config import InstructInversionBPTTConfig, DatasetConfig
from collections import Counter 
from io import StringIO


class AFHQ(Dataset):
    ALL_CLASSES = ["dog", "cat"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
        num_samples: int = 10000,
    ):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dog_dir = self.split_dir / self.ALL_CLASSES[0]
        self.cat_dir = self.split_dir / self.ALL_CLASSES[1]

        self.images_0 = list(self.dog_dir.rglob("*.[jpg jpeg]*"))
        self.images_1 = list(self.cat_dir.rglob("*.[jpg jpeg]*"))

        self.images_0 = self.images_0[:num_samples]
        self.images_1 = self.images_1[:num_samples]

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.images_0 = [x for x in self.images_0 if x.name in files_list]
            self.images_1 = [x for x in self.images_1 if x.name in files_list]

        if classes is not None and len(classes) == 1:
            if classes[0] == self.ALL_CLASSES[0]:
                self.image_list = self.images_0
                self.labels = [0] * len(self.image_list)
            elif classes[0] == self.ALL_CLASSES[1]:
                self.image_list = self.images_1
                self.labels = [1] * len(self.image_list)
        else:
            self.image_list = self.images_0 + self.images_1
            self.labels = [0] * len(self.images_0) + [1] * len(self.images_1)

        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class TextualInversionAFHQ(AFHQ):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        placeholder_str: Optional[list[str]] = None,
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, split, classes, transform)
        if placeholder_str is None:
            if classes is None:
                placeholder_str = self.ALL_CLASSES
            else:
                placeholder_str = classes
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]
        image, prompt = get_sample_with_prompts(
            idx,
            self.image_list,
            self.placeholder_str[label],
            self.use_prefix,
            self.transform,
        )
        return image, prompt, label, str(img_path)




class KikiBouba(Dataset):
    ALL_CLASSES = ["kiki", "bouba"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
        num_samples: int = 10000,
    ):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0][:num_samples]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1][:num_samples]

        self.images_0 = list(self.dir_0.rglob("*.jpg"))
        self.images_1 = list(self.dir_1.rglob("*.jpg"))

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.images_0 = [x for x in self.images_0 if x.name in files_list]
            self.images_1 = [x for x in self.images_1 if x.name in files_list]

        if classes is not None and len(classes) == 1:
            if classes[0] == self.ALL_CLASSES[0]:
                self.image_list = self.images_0
                self.labels = [0] * len(self.image_list)
            elif classes[0] == self.ALL_CLASSES[1]:
                self.image_list = self.images_1
                self.labels = [1] * len(self.image_list)
        else:
            self.image_list = self.images_0 + self.images_1
            self.labels = [0] * len(self.images_0) + [1] * len(self.images_1)

        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)

class KikiBouba(Dataset):
    ALL_CLASSES = ["kiki", "bouba"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
        num_samples: int = 10000,
    ):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0][:num_samples]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1][:num_samples]

        self.images_0 = list(self.dir_0.rglob("*.jpg"))
        self.images_1 = list(self.dir_1.rglob("*.jpg"))

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.images_0 = [x for x in self.images_0 if x.name in files_list]
            self.images_1 = [x for x in self.images_1 if x.name in files_list]

        if classes is not None and len(classes) == 1:
            if classes[0] == self.ALL_CLASSES[0]:
                self.image_list = self.images_0
                self.labels = [0] * len(self.image_list)
            elif classes[0] == self.ALL_CLASSES[1]:
                self.image_list = self.images_1
                self.labels = [1] * len(self.image_list)
        else:
            self.image_list = self.images_0 + self.images_1
            self.labels = [0] * len(self.images_0) + [1] * len(self.images_1)

        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class TextualInversionKikiBouba(KikiBouba):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        placeholder_str: Optional[list[str]] = None,
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, split, classes, transform)
        if placeholder_str is None:
            if classes is None:
                placeholder_str = self.ALL_CLASSES
            else:
                placeholder_str = classes
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]
        image, prompt = get_sample_with_prompts(
            idx,
            self.image_list,
            self.placeholder_str[label],
            self.use_prefix,
            self.transform,
        )
        return image, prompt, label, str(img_path)


class Butterfly(Dataset):
    ALL_CLASSES = ["Monarch", "Viceroy"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        num_samples: int = 10000,
        transform=None,
    ):
        self.MAX_SAMPLES = num_samples
        if split == "val":
            split = "test"
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1]

        self.images_0 = list(self.dir_0.rglob("*.[jpg jpeg]*"))[: self.MAX_SAMPLES]
        self.images_1 = list(self.dir_1.rglob("*.[jpg jpeg]*"))[: self.MAX_SAMPLES]

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.images_0 = [x for x in self.images_0 if x.name in files_list]
            self.images_1 = [x for x in self.images_1 if x.name in files_list]

        if classes is not None and len(classes) == 1:
            if classes[0] == self.ALL_CLASSES[0]:
                self.image_list = self.images_0
                self.labels = [0] * len(self.image_list)
            elif classes[0] == self.ALL_CLASSES[1]:
                self.image_list = self.images_1
                self.labels = [1] * len(self.image_list)
            else:
                raise ValueError(f"Unknown classes: {classes}")
        else:
            self.image_list = self.images_0 + self.images_1
            self.labels = [0] * len(self.images_0) + [1] * len(self.images_1)

        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class TextualInversionButterfly(Butterfly):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        placeholder_str: Optional[list[str]] = None,
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, split, classes, transform)
        if placeholder_str is None:
            if classes is None:
                placeholder_str = self.ALL_CLASSES
            else:
                placeholder_str = classes
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]
        image, prompt = get_sample_with_prompts(
            idx,
            self.image_list,
            self.placeholder_str[label],
            self.use_prefix,
            self.transform,
        )
        return image, prompt, label, str(img_path)


class BlackHolesMadSane(Dataset):
    ALL_CLASSES = ["mad", "sane"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
        num_samples: int = 10000,
    ):
        if split == "val":
            split = "test"
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1]

        self.images_0 = list(self.dir_0.rglob("*.npy"))[:num_samples]
        self.images_1 = list(self.dir_1.rglob("*.npy"))[:num_samples]

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.images_0 = [x for x in self.images_0 if f"{x.stem}.png" in files_list]
            self.images_1 = [x for x in self.images_1 if f"{x.stem}.png" in files_list]

        if classes is not None and len(classes) == 1:
            if classes[0] == self.ALL_CLASSES[0]:
                self.image_list = self.images_0
                self.labels = [0] * len(self.image_list)
            elif classes[0] == self.ALL_CLASSES[1]:
                self.image_list = self.images_1
                self.labels = [1] * len(self.image_list)
            else:
                raise ValueError(f"Unknown classes: {classes}")
        else:
            self.image_list = self.images_0 + self.images_1
            self.labels = [0] * len(self.images_0) + [1] * len(self.images_1)

        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = np.load(img_path)
        # Scale the image to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=-1)
        # make it RGB
        image = np.repeat(image, 3, axis=-1)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class TextualInversionBlackHolesMadSane(BlackHolesMadSane):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        placeholder_str: Optional[list[str]] = None,
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, split, classes, transform)
        if placeholder_str is None:
            if classes is None:
                placeholder_str = self.ALL_CLASSES
            else:
                placeholder_str = classes
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix

    def __getitem__(self, idx):
        prompt = convert_placeholders_to_prompt(self.placeholder_str, self.use_prefix)

        img_path = self.image_list[idx]
        image = np.load(img_path)
        # Scale the image to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=-1)
        # make it RGB
        image = np.repeat(image, 3, axis=-1)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, prompt, label, str(img_path)


class Kermany(Dataset):
    ALL_CLASSES = ["DRUSEN", "NORMAL"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
        num_samples: int = 10000,
    ):
        if split == "val":
            split = "test"
        self.MAX_SAMPLES = num_samples
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1]
        self.images_0 = list(self.dir_0.rglob("*.[jpg jpeg]*"))[: self.MAX_SAMPLES]
        self.images_1 = list(self.dir_1.rglob("*.[jpg jpeg]*"))[: self.MAX_SAMPLES]

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.images_0 = [x for x in self.images_0 if x.name in files_list]
            self.images_1 = [x for x in self.images_1 if x.name in files_list]

        if classes is not None and len(classes) == 1:
            if classes[0] == self.ALL_CLASSES[0]:
                self.image_list = self.images_0
                self.labels = [0] * len(self.image_list)
            elif classes[0] == self.ALL_CLASSES[1]:
                self.image_list = self.images_1
                self.labels = [1] * len(self.image_list)
            else:
                raise ValueError(f"Unknown classes: {classes}")
        else:
            self.image_list = self.images_0 + self.images_1
            self.labels = [0] * len(self.images_0) + [1] * len(self.images_1)

        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)


def convert_to_np(img_path: Path):
    image = Image.open(img_path).convert("RGB")
    return np.array(image).transpose(2, 0, 1)


def process_image(img_path: Path):
    image = Image.open(img_path).convert("RGB")
    return image


def convert_placeholders_to_prompt(
    placeholders_list: Union[str, list[str]], use_prefix: bool = False
):
    if isinstance(placeholders_list, str):
        prompt = placeholders_list
    elif len(placeholders_list) == 1:
        prompt = placeholders_list[0]
    else:
        prompt = " ".join(placeholders_list)
    if use_prefix is True:
        # prefix = random.choice(INSTRUCT_PREFIX_TEMPLATES)
        prefix = random.choice(IMAGENET_CLIP_TEMPLATES)
        prompt_prefix = f"{prefix} "
        prompt = prompt_prefix + prompt

    return prompt


def get_sample_with_prompts(
    idx: int,
    image_list: list,
    placeholder_str: list[str],
    use_prefix: bool,
    transform=None,
):
    img_path = image_list[idx]
    image = process_image(img_path)

    if transform:
        image = transform(image)

    prompt = convert_placeholders_to_prompt(placeholder_str, use_prefix)
    return image, prompt


def get_dataset(dataset_name: str, **kwargs):
    """
    Dynamically select and instantiate the desired dataset class using getattr().

    Args:
    dataset_name (str): Name of the dataset (e.g 'AFHQ', 'FFHQ', etc.)
    **kwargs: Additional arguments to pass to the dataset constructor

    Returns:
    Dataset: An instance of the selected dataset class
    """
    try:
        dataset_class = getattr(sys.modules[__name__], dataset_name)
        return dataset_class(**kwargs)
    except AttributeError:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_ti_dataset_by_name(cfg: InstructInversionBPTTConfig, dataset_transforms: list):
    train_transform, val_transform = dataset_transforms
    if cfg.dataset.name == "afhq":
        dataset = ConcatDataset(
            [
                TextualInversionAFHQ(
                    root_dir=cfg.dataset.image_dir,
                    placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
                    transform=train_transform,
                    use_prefix=cfg.dataset.use_prefix,
                    split="train",
                )
            ]
            * cfg.dataset.repeats
        )
        eval_dataset = TextualInversionAFHQ(
            root_dir=cfg.dataset.image_dir,
            placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
            transform=val_transform,
            use_prefix=cfg.dataset.use_prefix,
            split="val",
        )

    elif cfg.dataset.name == "kikibouba":
        dataset = ConcatDataset(
            [
                TextualInversionKikiBouba(
                    root_dir=cfg.dataset.image_dir,
                    placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
                    transform=train_transform,
                    use_prefix=cfg.dataset.use_prefix,
                    split="train",
                )
            ]
            * cfg.dataset.repeats
        )
        eval_dataset = TextualInversionKikiBouba(
            root_dir=cfg.dataset.image_dir,
            placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
            transform=val_transform,
            use_prefix=cfg.dataset.use_prefix,
            split="val",
        )

    elif cfg.dataset.name == "imagenet":
        train_dataset = TextualInversionImageNetLabels(
            root_dir=cfg.dataset.image_dir,
            synset_ids=cfg.dataset.synset_ids,
            transform=train_transform,
            split="train",
        )
        dataset = ConcatDataset([train_dataset] * cfg.dataset.repeats)
        eval_dataset = TextualInversionImageNetLabels(
            root_dir=cfg.dataset.image_dir,
            synset_ids=cfg.dataset.synset_ids,
            transform=val_transform,
            split="val",
        )
        cfg.diffusion.embedding_config.placeholder_strings = (
            train_dataset.placeholder_str
        )
        cfg.diffusion.embedding_config.initializer_words = ["cat"] * len(
            train_dataset.placeholder_str
        )

    elif cfg.dataset.name == "yearbook":
        dataset = ConcatDataset(
            [
                TextualInversionYearbook(
                    root_dir=cfg.dataset.image_dir,
                    placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
                    classes=cfg.dataset.classes,
                    transform=train_transform,
                    use_prefix=cfg.dataset.use_prefix,
                    split="train",
                )
            ]
            * cfg.dataset.repeats
        )
        eval_dataset = TextualInversionYearbook(
            root_dir=cfg.dataset.image_dir,
            placeholder_str=cfg.diffusion.embedding_config.placeholder_strings,
            classes=cfg.dataset.classes,
            transform=val_transform,
            use_prefix=cfg.dataset.use_prefix,
            split="val",
        )

    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
    return dataset, eval_dataset, cfg


def get_cls_dataset_by_name(cfg: DatasetConfig, dataset_transforms: list):
    train_transform, val_transform = dataset_transforms
    
    if cfg.name == "afhq":
        dataset = AFHQ(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, num_samples=cfg.num_samples
        )

        eval_dataset = AFHQ(
            cfg.image_dir,
            transform=val_transform,
            split="val",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )


    elif cfg.name == "inaturalist":
        
        dataset = INatDatasetJoint(
            cfg.image_dir, transform=train_transform, cindset =cfg.classes, split="train", train_subset=cfg.train_subset)
        
        
        eval_dataset = INatDatasetJoint(
            cfg.image_dir,
            transform=val_transform,
            cindset=cfg.classes, 
            split="val",
        )

    elif cfg.name == "kermany":
        dataset = Kermany(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, num_samples=cfg.num_samples
        )
        eval_dataset = Kermany(
            cfg.image_dir,
            transform=val_transform,
            split="val",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "celebahq":
        dataset = CelebAHQ(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, num_samples=cfg.num_samples
        )
        eval_dataset = CelebAHQ(
            cfg.image_dir,
            transform=val_transform,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "celebahqyoung":
        dataset = CelebAHQYoung(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, 
        )
        cfg.file_list_path = None
        eval_dataset = CelebAHQYoung(
            cfg.image_dir,
            transform=val_transform,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "kikibouba":
        dataset = KikiBouba(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, num_samples=cfg.num_samples
        )
        eval_dataset = KikiBouba(
            cfg.image_dir,
            transform=val_transform,
            split="val",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "celebahqyoung":
        dataset = CelebAHQYoung(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes
        )
        eval_dataset = CelebAHQYoung(
            cfg.image_dir,
            transform=val_transform,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "imagenet":
        dataset = ImageNetDataset(
            root_dir=cfg.image_dir,
            transform=train_transform,
            split="train",
            synset_ids=cfg.synset_ids,
        )
        eval_dataset = ImageNetDataset(
            root_dir=cfg.image_dir,
            transform=val_transform,
            split="val",
            synset_ids=cfg.synset_ids,
        )

  

    elif cfg.name == "celeba":
        dataset = CelebA(
            root_dir=cfg.image_dir,
            transform=train_transform,
            split="train",
            file_list_path=cfg.file_list_path,
            classes=cfg.classes,
        )
        eval_dataset = CelebA(
            root_dir=cfg.image_dir,
            transform=val_transform,
            split="val",
            classes=cfg.classes,
        )

    elif cfg.name == "butterfly":
        dataset = Butterfly(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, num_samples=cfg.num_samples
        )
        eval_dataset = Butterfly(
            cfg.image_dir,
            transform=val_transform,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "mad_sane":
        dataset = BlackHolesMadSane(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes, num_samples=cfg.num_samples
        )
        eval_dataset = BlackHolesMadSane(
            cfg.image_dir,
            transform=val_transform,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")
    return dataset, eval_dataset


def get_t2i_dataset_by_name(cfg: DatasetConfig, dataset_transforms: list):
    train_transform, val_transform = dataset_transforms
    if cfg.name == "afhq":
        dataset = TextualInversionAFHQ(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=train_transform,
            use_prefix=cfg.use_prefix,
            split="train",
        )
        eval_dataset = TextualInversionAFHQ(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=val_transform,
            use_prefix=cfg.use_prefix,
            split="val",
        )

    elif cfg.name == "kikibouba":
        dataset = TextualInversionKikiBouba(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=train_transform,
            use_prefix=cfg.use_prefix,
            split="train",
        )
        eval_dataset = TextualInversionKikiBouba(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=val_transform,
            use_prefix=cfg.use_prefix,
            split="val",
        )

    elif cfg.name == "yearbook":
        dataset = TextualInversionYearbook(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=train_transform,
            use_prefix=cfg.use_prefix,
            split="train",
        )
        eval_dataset = TextualInversionYearbook(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=val_transform,
            use_prefix=cfg.use_prefix,
            split="val",
        )

    elif cfg.name == "butterfly":
        dataset = TextualInversionButterfly(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=train_transform,
            use_prefix=cfg.use_prefix,
            split="train",
        )
        eval_dataset = TextualInversionButterfly(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=val_transform,
            use_prefix=cfg.use_prefix,
            split="val",
        )

    elif cfg.name == "mad_sane":
        dataset = TextualInversionBlackHolesMadSane(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=train_transform,
            use_prefix=cfg.use_prefix,
            split="train",
        )
        eval_dataset = TextualInversionBlackHolesMadSane(
            root_dir=cfg.image_dir,
            placeholder_str=None,
            classes=cfg.classes,
            transform=val_transform,
            use_prefix=cfg.use_prefix,
            split="val",
        )

    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")
    return dataset, eval_dataset


def read_txt(filepath: Path):
    lines = []
    with open(filepath, "r") as file:
        # Read the file line by line
        for line in file:
            # Remove whitespace at the end of each line
            line = line.strip()
            # Process the line
            lines.append(line)
    return lines


def read_file_list(filepath: Path):
    lines = read_txt(filepath)[1:]
    filenames = [str(x).replace("generated_", "") for x in lines]
    return filenames


# Parse the LOC_synset_mapping.txt to create a mapping dictionary
mapping_file = "files/LOC_synset_mapping.txt"
synset_to_index = {}
synset_to_name = {}
with open(mapping_file, "r") as f:
    for index, line in enumerate(f):
        synset_id, _ = line.strip().split(" ", 1)
        synset_to_index[synset_id] = line.strip().split(" ", 1)
        synset_to_name[synset_id] = line.strip()[9:]


class ImageNetDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, synset_ids: List[str], transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = root_dir / split
        self.transform = transform
        self.image_list = []
        self.labels = []
        self.list_of_names = []
        self.list_of_synset_id = []

        iter = 0
        # if not synset_ids:
        #     synset_ids = synset_ids #os.listdir(self.dataset_dir) ['n02123159','n02099601'] # ['n02124075','n02102480'] #

        self.num_classes = len(synset_ids)
        # Traverse the root_dir to get all image file paths and their corresponding synset IDs
        for class_idx, synset_id in enumerate(synset_ids):
            synset_dir = os.path.join(self.dataset_dir, synset_id)
            if iter >= self.num_classes:
                break
            if os.path.isdir(synset_dir):
                self.list_of_names.append(synset_to_name[synset_id])
                self.list_of_synset_id.append(synset_id)
                if os.path.isdir(synset_dir):
                    iter = iter + 1
                    for img_name in os.listdir(synset_dir):
                        if img_name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                        ):
                            img_path = os.path.join(synset_dir, img_name)
                            self.image_list.append(img_path)
                            self.labels.append(class_idx)
                            print(img_path, class_idx)
        # self.list_of_names = ["tiger cat", "Egyptian cat", "golden retriever", "Sussex spaniel"]
        print(self.list_of_names)
        indices = list(range(len(self.image_list)))
        random.shuffle(indices)
        self.image_list = [self.image_list[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = torch.tensor(self.labels[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TextualInversionImageNet(ImageNetDataset):
    def __init__(self, root_dir: Path, split: str, synset_ids, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, split, synset_ids, transform)
        self.file_path = "files/vocab.json"

        # Open the JSON file and load its contents into a dictionary
        with open(self.file_path, "r") as file:
            data_dict = json.load(file)
        self.tokens = list(data_dict.keys())
        self.placeholder_str = [self.tokens[i] for i in range(self.num_classes)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        prompt = convert_placeholders_to_prompt(self.placeholder_str)
        return image, prompt


class TextualInversionImageNetLabels(ImageNetDataset):
    def __init__(self, root_dir: Path, split: str, synset_ids, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, split, synset_ids, transform)
        self.file_path = "files/vocab.json"

        # Open the JSON file and load its contents into a dictionary
        with open(self.file_path, "r") as file:
            data_dict = json.load(file)
        self.tokens = list(data_dict.keys())
        self.placeholder_str = [self.tokens[i] for i in range(self.num_classes)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        prompt = convert_placeholders_to_prompt(self.placeholder_str)
        return image, prompt, label


class ImageEdits(Dataset):
    def __init__(self, images_dir: Path, edits_dir: Path, transform=None) -> None:
        self.images_dir = images_dir
        self.edits_dir = edits_dir
        self.transform = transform

        image_files = []
        for ext in ("*.jpg", "*.png"):
            image_files.extend(self.images_dir.rglob(ext))
        self.image_list = image_files

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img_edit_path = self.edits_dir / img_path.name

        image = convert_to_np(img_path)
        image_edit = convert_to_np(img_edit_path)

        images = np.concatenate([image, image_edit])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1

        if self.transform:
            images = self.transform(images)

        image, image_edit = images.chunk(2)
        return image, image_edit


class TextualInversionEdits(ImageEdits):
    def __init__(
        self,
        images_dir: Path,
        edits_dir: Path,
        placeholder_str: list[str],
        transform=None,
    ) -> None:
        super().__init__(images_dir, edits_dir, transform)
        self.placeholder_str = placeholder_str

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img_edit_path = self.edits_dir / img_path.name

        image = convert_to_np(img_path)
        image_edit = convert_to_np(img_edit_path)

        images = np.concatenate([image, image_edit])
        images = torch.tensor(images)

        if self.transform:
            images = self.transform(images)

        images = images / 255.0
        image, image_edit = images.chunk(2)

        prompt = convert_placeholders_to_prompt(self.placeholder_str)
        return image, image_edit, prompt


class ImagesBase(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        image_files = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
                image_files.extend(self.root_dir.rglob(ext))
        self.image_list = image_files

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image


class TextualInversionImagesBase(ImagesBase):
    def __init__(
        self,
        root_dir: Path,
        placeholder_str: list[str],
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, transform)
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix

    def __getitem__(self, idx):
        image, prompt = get_sample_with_prompts(
            idx, self.image_list, self.placeholder_str, self.use_prefix, self.transform
        )
        return image, prompt


class ZeroInversionAFHQ(AFHQ):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        placeholder_str: list[str],
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, split, transform)
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix
        self.captions_dir = root_dir.parent / "captions"

    def __getitem__(self, idx):
        image, prompt = get_sample_with_prompts(
            idx, self.image_list, self.placeholder_str, self.use_prefix, self.transform
        )
        label = self.labels[idx]

        img_path = Path(self.image_list[idx])
        relative_dir = img_path.relative_to(self.root_dir).parent
        caption_path = self.captions_dir / relative_dir / f"{img_path.stem}.json"
        with open(caption_path, "r") as fp:
            caption = json.load(fp)

        return image, prompt, label, caption, str(img_path)


class CelebAHQ(Dataset):
    ALL_CLASSES = ["serious", "smiling"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            root_dir (Path): Path to CelebAHQ dataset root
            split (str): 'train', 'val', or 'test'
            classes (list[str]): ["smiling"] or ["not_smiling"] or both
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Read annotation files
        with open(self.root_dir / 'CelebAMask-HQ-attribute-anno.txt', 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' + datastr.replace('  ', ' ')

        with open(self.root_dir / 'CelebA-HQ-to-CelebA-mapping.txt', 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']
        mapstr = ' '.join(mapstr)

        # Process dataframes
        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(self.root_dir / 'list_eval_partition.csv')
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        
        # Fix column names for merging
        mapping_df['image_id'] = mapping_df['orig_file'].str.replace('.jpg', '')
        partition_df['image_id'] = partition_df['filename'].str.replace('.jpg', '')
        partition_df = pd.merge(mapping_df, partition_df, on='image_id')

        # Map split name to partition number
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"Split must be one of {list(split_map.keys())}")
            
        # Filter data by split
        self.data = data[partition_df['partition'] == split_map[split]]
        if num_samples is not None:
            self.data = self.data.sample(n=num_samples, random_state=42)
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)
        
        # Smiling attribute is at index 31 (0-based)
        smile_labels = self.data.iloc[:, 2:].to_numpy()[:, 31]
        
        # Create image list based on classes
        all_images = [(f"{idx}", label) 
                     for idx, label in zip(self.data['idx'], smile_labels)]
        
        # Filter by file list if provided
        if file_list_path is not None:
            with open(file_list_path, 'r') as f:
                files_list = set(line.strip() for line in f.readlines())
            all_images = [(img, label) for img, label in all_images if img in files_list]

        self.image_list, self.labels = zip(*all_images)
        self.image_list = [self.root_dir / 'CelebA-HQ-img' / img for img in self.image_list]
        
        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes
        
        print(f"Loaded {len(self.image_list)} images for {split} split")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print the distribution of classes in the loaded dataset"""
        class_counts = Counter(self.labels)
        print(f"Not Smiling: {class_counts[0]} images")
        print(f"Smiling: {class_counts[1]} images")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)
    

class CelebAHQYoung(Dataset):
    ALL_CLASSES = ["old", "young"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
    ):
        """
        Args:
            root_dir (Path): Path to CelebAHQ dataset root
            split (str): 'train', 'val', or 'test'
            classes (list[str]): ["smiling"] or ["not_smiling"] or both
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Read annotation files
        with open(self.root_dir / 'CelebAMask-HQ-attribute-anno.txt', 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' + datastr.replace('  ', ' ')

        with open(self.root_dir / 'CelebA-HQ-to-CelebA-mapping.txt', 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']
        mapstr = ' '.join(mapstr)

        # Process dataframes
        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(self.root_dir / 'list_eval_partition.csv')
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        
        # Fix column names for merging
        mapping_df['image_id'] = mapping_df['orig_file'].str.replace('.jpg', '')
        partition_df['image_id'] = partition_df['filename'].str.replace('.jpg', '')
        partition_df = pd.merge(mapping_df, partition_df, on='image_id')

        # Map split name to partition number
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"Split must be one of {list(split_map.keys())}")
            
        # Filter data by split
        self.data = data[partition_df['partition'] == split_map[split]]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)
        
        # Age attribute is at index 39 (0-based)
        age_labels = self.data.iloc[:, 2:].to_numpy()[:, 39]
        
        # Create image list based on classes
        all_images = [(f"generated_{idx}", label) 
                     for idx, label in zip(self.data['idx'], age_labels)]
        
        # Filter by file list if provided
        if file_list_path is not None:
            with open(file_list_path, 'r') as f:
                files_list = set(line.strip() for line in f.readlines())
            #import pdb; pdb.set_trace()
            all_images = [(img, label) for img, label in all_images if img in files_list]

        # else:
        self.image_list, self.labels = zip(*all_images)
        self.image_list = [self.root_dir / 'CelebA-HQ-img' / img.replace('generated_', '') for img in self.image_list]
        
        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes
        
        print(f"Loaded {len(self.image_list)} images for {split} split")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print the distribution of classes in the loaded dataset"""
        class_counts = Counter(self.labels)
        print(f"Not Smiling: {class_counts[0]} images")
        print(f"Smiling: {class_counts[1]} images")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)

class CelebA(Dataset):

    id_to_cls = [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.image_dir = root_dir / "img_align_celeba"
        self.transform = transform

        labels_path = root_dir / "list_attr_celeba.csv"
        split_file_path = root_dir / "list_eval_partition.csv"
        split_df = pd.read_csv(split_file_path)
        if "train" in split:
            split_idx = 0
        elif "val" in split:
            split_idx = 1
        elif "test" in split:
            split_idx = 2
        split_indices = split_df.index[split_df["partition"] == split_idx].to_list()
        split_filenames = split_df.loc[split_indices, "image_id"].to_list()

        labels_df = pd.read_csv(labels_path)
        labels_df = labels_df.loc[split_indices]
        if classes:
            self.num_classes = len(classes)
            self.classes = classes
        else:
            self.num_classes = len(labels_df.columns) - 1
            self.classes = self.id_to_cls

        mask = labels_df == 1
        labels = mask.apply(
            lambda row: [labels_df.columns.get_loc(col) - 1 for col in row.index[row]],
            axis=1,
        )
        self.labels = labels.to_list()

        image_list, labels = [], []
        for cls_idx, query_cls in enumerate(self.classes):
            # support class negation with a -1 identifier
            neg_id = query_cls.split("-1_")[0] == ""
            if neg_id is True:
                query_cls = query_cls.split("-1_")[-1]
                query_idx = self.cls_to_id[query_cls]
                filt_image_files_labels = [
                    (self.image_dir / filename, label)
                    for filename, label in zip(split_filenames, self.labels)
                    if query_idx not in label
                ]
            else:
                query_idx = self.cls_to_id[query_cls]
                filt_image_files_labels = [
                    (self.image_dir / filename, label)
                    for filename, label in zip(split_filenames, self.labels)
                    if query_idx in label
                ]

            filt_image_files, filt_labels = zip(*filt_image_files_labels)
            image_list += filt_image_files
            labels += [cls_idx] * len(filt_labels)

        self.image_list = image_list
        self.labels = labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)

        # for multi-label, targets are one-hots
        # to be consumed by BCE loss
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



class INatDatasetJoint(Dataset):
    def __init__(self, root_dir, cindset=[6372, 6375], split='train', transform=None, train_subset=None):
        self.root_dir = root_dir
        self.split = split
        self.train_subset = train_subset  # Can be 'train1', 'train2', or None
        
        # Use base split (train or val) for directory structure
        base_split = 'train' if split.startswith('train') else split
        self.imgdir = os.path.join(root_dir, base_split)
        self.bbox_dir = os.path.join(root_dir, f"{base_split}_bbox")

        self.metafile = os.path.join(root_dir, base_split+'.json')
        self.transform = transform

        with open(self.metafile) as ifd:
            self.metadata = json.load(ifd)

        self.images = {tmp['id']: tmp for tmp in self.metadata['images']}
        self.temp = [tmp['class'] for tmp in self.metadata['categories']]
        self.num_classes = len(cindset)
        
        self.classes = [tmp['common_name'] for tmp in self.metadata['categories'] if tmp['id'] in cindset]
        self.scientific_name = [tmp['name'] for tmp in self.metadata['categories']  if tmp['id'] in cindset]
        combined_name = set(['{} ({})'.format(sn, cn) for sn, cn in zip(self.scientific_name, self.classes)])
        self.classinds = [tmp['id'] for tmp in self.metadata['categories'] if tmp['id'] in cindset]
        self.classindsboth = [tmp['id'] for tmp in self.metadata['categories'] if tmp['id'] in [6372, 6375]]

        self.annotation = [tmp for tmp in self.metadata['annotations'] if tmp["category_id"] in cindset]
        imgset = set([tmp["image_id"] for tmp in self.annotation])
        self.images = {k:v for k,v in self.images.items() if k in imgset}
        
        # If this is a train subset, select appropriate portion of images
        if self.train_subset:
            # Convert images dict to list for easier splitting
            images_list = list(self.images.items())
            # Sort by key for reproducibility
            images_list.sort(key=lambda x: x[0])
            
            if self.train_subset == 'train1':
                # Take first 200 images
                images_list = images_list[:200]
            elif self.train_subset == 'train2':
                # Take next 100 images
                images_list = images_list[200:300]
            
            # Convert back to dict
            self.images = dict(images_list)
            
            # Update annotations to only include selected images
            selected_image_ids = set(self.images.keys())
            self.annotation = [ann for ann in self.annotation if ann["image_id"] in selected_image_ids]

        with open(os.path.join(self.root_dir, 'cbd_descriptors.json')) as ifd:
            self.cbd_descriptors = json.load(ifd)
        self.cbd_descriptors = {k:v for k,v in self.cbd_descriptors.items() if k in combined_name}
        self.common_name = {i:name for i, name in enumerate(self.classes)}
        self.classindmap = {i:ind for ind, i in enumerate(self.classindsboth)}
        print(f"Split: {split}, Subset: {train_subset if train_subset else 'None'}, Images: {len(self.images)}")
        self.generated = [0 for _ in range(len(self.images))]

        assert len(self.annotation)==len(self.images)
        assert len(self.cbd_descriptors)==len(self.classes)
        self.image_list = [os.path.join(self.root_dir, self.images[anno["image_id"]]['file_name']) for anno in self.annotation]
        self.labels = [self.classindmap[int(anno["category_id"])] for anno in self.annotation]
        
    def __len__(self):
        return len(self.image_list)
    

    def resize_bbox(self, bbox, original_size, new_size):
        """
        Resize bbox coordinates based on image resize ratio
        bbox: [x1, y1, x2, y2]
        original_size: (height, width)
        new_size: (height, width)
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate resize ratio
        h_ratio = new_size[0] / original_size[0]
        w_ratio = new_size[1] / original_size[1]
        
        # Apply ratio to bbox coordinates
        new_x1 = int(x1 * w_ratio)
        new_y1 = int(y1 * h_ratio)
        new_x2 = int(x2 * w_ratio)
        new_y2 = int(y2 * h_ratio)
        
        return [new_x1, new_y1, new_x2, new_y2]
    

    def expand_bbox(self, bbox, padding, image_shape):
        """
        Expand bbox by adding fixed padding on all sides while keeping it within image bounds
        bbox: [x1, y1, x2, y2]
        padding: int (number of pixels to add on each side)
        image_shape: (height, width, channels)
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding to all sides
        new_x1 = x1 - padding
        new_y1 = y1 - padding
        new_x2 = x2 + padding
        new_y2 = y2 + padding
        
        # Clip to image bounds
        new_x1 = max(0, min(new_x1, image_shape[2]))
        new_y1 = max(0, min(new_y1, image_shape[1]))
        new_x2 = max(0, min(new_x2, image_shape[2]))
        new_y2 = max(0, min(new_y2, image_shape[1]))

    
        return [new_x1, new_y1, new_x2, new_y2]

    def __getitem__(self, idx):
        anno = self.annotation[idx]
        img = self.images[anno["image_id"]]
        label = self.classindmap[int(anno["category_id"])]
        
        # Load image
        if self.generated[idx] == 1:
            image_path = os.path.join("/proj/vondrick2/mia/magnification", img['file_name'])
        else:
            image_path = os.path.join(self.root_dir, img['file_name'])
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size[::-1]

        if self.transform:
            image = self.transform(image)
            new_size = tuple(image.shape[1:])
        
        return image, label, img['file_name']

    def get_desc_byid(self, id):
        name = self.scientific_name[id]
        common_name = self.classes[id]
        key = '{} ({})'.format(name, common_name)
        return self.cbd_descriptors[key]
