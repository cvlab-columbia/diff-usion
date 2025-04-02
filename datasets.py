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


class Birds(Dataset):
    def __init__(self, root_dir: Path, split: str, transform=None):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform

        self.car_images = list(self.split_dir.rglob("*.jpg"))

        self.image_list = self.car_images
        self.labels = [0] * len(self.car_images)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class TextualInversionBirds(Birds):
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

    def __getitem__(self, idx):
        image, prompt = get_sample_with_prompts(
            idx, self.image_list, self.placeholder_str, self.use_prefix, self.transform
        )
        label = self.labels[idx]
        return image, prompt, label


class Cars(Dataset):
    def __init__(self, root_dir: Path, split: str, transform=None):
        self.root_dir = root_dir
        self.newstr = "cars_" + split
        self.split_dir = self.root_dir / self.newstr
        self.transform = transform

        self.car_images = list(self.split_dir.rglob("*.jpg"))

        self.image_list = self.car_images
        self.labels = [0] * len(self.car_images)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class TextualInversionCars(Cars):
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

    def __getitem__(self, idx):
        image, prompt = get_sample_with_prompts(
            idx, self.image_list, self.placeholder_str, self.use_prefix, self.transform
        )
        label = self.labels[idx]
        return image, prompt, label


class AFHQ(Dataset):
    ALL_CLASSES = ["dog", "cat"]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dog_dir = self.split_dir / self.ALL_CLASSES[0]
        self.cat_dir = self.split_dir / self.ALL_CLASSES[1]

        self.images_0 = list(self.dog_dir.rglob("*.[jpg jpeg]*"))
        self.images_1 = list(self.cat_dir.rglob("*.[jpg jpeg]*"))

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


class FFHQ(Dataset):
    def __init__(
        self,
        root_dir: Path,
        classes: list = ["15-19", "70-120"],
        labels_path: Optional[Path] = None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.image_dir = root_dir / "images"
        self.transform = transform
        if labels_path is None:
            labels_path = root_dir / "ffhq_aging_labels.csv"

        labels_df = pd.read_csv(labels_path)
        class_images = []
        for cls_name in classes:
            image_names_from_labels = labels_df[labels_df["age_group"] == cls_name][
                "image_number"
            ].to_list()
            class_images.append(image_names_from_labels)

        self.image_list = [
            self.get_image_path(x) for sublist in class_images for x in sublist
        ]
        labels = []
        for i, sublist in enumerate(class_images):
            labels += [i] * len(sublist)
        self.labels = labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_image_path(self, image_name: Union[str, int]):
        return self.image_dir / f"{str(image_name).zfill(5)}.png"


class TextualInversionFFHQ(FFHQ):
    def __init__(
        self,
        root_dir: Path,
        placeholder_str: list[str],
        classes: list = ["15-19", "70-120"],
        labels_path: Optional[Path] = None,
        use_prefix: bool = False,
        transform=None,
    ):
        super().__init__(root_dir, classes, labels_path, transform)
        self.placeholder_str = placeholder_str
        self.use_prefix = use_prefix

    def __getitem__(self, idx):
        image, prompt = get_sample_with_prompts(
            idx, self.image_list, self.placeholder_str, self.use_prefix, self.transform
        )
        label = self.labels[idx]
        return image, prompt, label


class Yearbook(Dataset):
    ALL_CLASSES = [
        "1930",
        "1940",
        "1950",
        "1960",
        "1970",
        "1980",
        "1990",
        "2000",
        "2010",
    ]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        gender: str = "F",
        transform=None,
    ):
        """
        classes should be a list of decade strings, e.g 1960, 1990, 2010, etc.
        If None, take all images from all decades.
        """
        self.root_dir = root_dir
        self.image_dir = root_dir / gender
        self.transform = transform
        if classes is None:
            classes = self.ALL_CLASSES
        self.num_classes = len(classes)
        self.classes = classes

        split_file = self.root_dir / f"{split}_{gender}.txt"
        split_lines = read_txt(split_file)

        full_images_list = [self.root_dir / str(x).split(" ")[0] for x in split_lines]
        image_list, labels = [], []
        for image_path in full_images_list:
            sample_year = image_path.name.split("_")[0]
            for i, class_year in enumerate(classes):
                if (
                    int(sample_year) >= int(class_year)
                    and int(sample_year) < int(class_year) + 10
                ):
                    image_list.append(image_path)
                    labels.append(i)

        self.image_list = image_list
        self.labels = labels

        if file_list_path is not None:
            files_list = read_file_list(file_list_path)
            self.image_list = [x for x in self.image_list if x.name in files_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class TextualInversionYearbook(Yearbook):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        placeholder_str: Optional[list[str]] = None,
        use_prefix: bool = False,
        gender: str = "F",
        transform=None,
    ):
        super().__init__(
            root_dir=root_dir,
            split=split,
            classes=classes,
            gender=gender,
            transform=transform,
        )
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
    ):
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1]

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
    MAX_SAMPLES = int(1e4)

    def __init__(
        self,
        root_dir: Path,
        split: str,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
    ):
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
    ):
        if split == "val":
            split = "test"
        self.root_dir = root_dir
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.dir_0 = self.split_dir / self.ALL_CLASSES[0]
        self.dir_1 = self.split_dir / self.ALL_CLASSES[1]

        self.images_0 = list(self.dir_0.rglob("*.npy"))
        self.images_1 = list(self.dir_1.rglob("*.npy"))

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
    MAX_SAMPLES = int(1e4)

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
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes
        )

        eval_dataset = AFHQ(
            cfg.image_dir,
            transform=val_transform,
            split="val",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "kikibouba":
        dataset = KikiBouba(
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes
        )
        eval_dataset = KikiBouba(
            cfg.image_dir,
            transform=val_transform,
            split="val",
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

    elif cfg.name == "yearbook":
        dataset = Yearbook(
            root_dir=cfg.image_dir,
            transform=train_transform,
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
            split="train",
        )
        eval_dataset = Yearbook(
            root_dir=cfg.image_dir,
            transform=val_transform,
            classes=cfg.classes,
            split="val",
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
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes
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
            cfg.image_dir, transform=train_transform, split="train", classes=cfg.classes
        )
        eval_dataset = BlackHolesMadSane(
            cfg.image_dir,
            transform=val_transform,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
        )

    elif cfg.name == "kermany":
        dataset = Kermany(
            cfg.image_dir, split="train", classes=cfg.classes, transform=train_transform
        )

        eval_dataset = Kermany(
            cfg.image_dir,
            split="test",
            classes=cfg.classes,
            file_list_path=cfg.file_list_path,
            transform=val_transform,
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
        recursive: bool = True,
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        image_files = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            if recursive:
                image_files.extend(self.root_dir.rglob(ext))
            else:
                image_files.extend(self.root_dir.glob(ext))
        self.image_list = image_files

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = process_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)


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


def _get_combinations(benchmark_type: str) -> Tuple[dict, dict]:
    combinations = {
        "o2o_easy": (
            ["desert", "jungle", "dirt", "snow"],
            ["dirt", "snow", "desert", "jungle"],
            "beach",
        ),
        "o2o_medium": (
            ["mountain", "beach", "dirt", "jungle"],
            ["jungle", "dirt", "beach", "snow"],
            "desert",
        ),
        # "o2o_hard": (
        #     ["beach", "snow"],
        #     ["snow", "desert"],
        #     "beach",
        # ),
        "o2o_hard": (
            ["jungle", "mountain", "snow", "desert"],
            ["mountain", "snow", "desert", "jungle"],
            "beach",
        ),
        "m2m_hard": (
            ["dirt", "jungle", "snow", "beach"],
            ["snow", "beach", "dirt", "jungle"],
            None,
        ),
        "m2m_easy": (
            ["desert", "mountain", "dirt", "jungle"],
            ["dirt", "jungle", "mountain", "desert"],
            None,
        ),
        "m2m_medium": (
            ["beach", "snow", "mountain", "desert"],
            ["desert", "mountain", "beach", "snow"],
            None,
        ),
    }
    if benchmark_type not in combinations:
        raise ValueError("Invalid benchmark type")
    group, test, filler = combinations[benchmark_type]
    return build_combination(benchmark_type, group, test, filler)


def build_combination(benchmark_type, group, test, filler=None):
    total = 3168
    combinations = {}
    if "m2m" in benchmark_type:
        counts = [total, total]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[1], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[0], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[3], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[2], counts[1])],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[1]],
            ("dachshund",): [test[1], test[0]],
            ("labrador",): [test[2], test[3]],
            ("corgi",): [test[3], test[2]],
        }
    else:
        counts = [int(0.97 * total), int(0.87 * total)]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[0], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[1], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[2], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[3], counts[1])],
            ("bulldog", "dachshund", "labrador", "corgi"): [
                (filler, total - counts[0]),
                (filler, total - counts[1]),
            ],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[0]],
            ("dachshund",): [test[1], test[1]],
            ("labrador",): [test[2], test[2]],
            ("corgi",): [test[3], test[3]],
        }
    return combinations
