import random
import importlib
from PIL import Image
from pathlib import Path
from typing import Optional
from transformers import CLIPTokenizer
from datasets import (
    AFHQ,
    KikiBouba,
    CelebA,
    Butterfly,
    BlackHolesMadSane,
    Kermany,
    IMAGENET_CLIP_TEMPLATES,
)
import numpy as np


class TextualInversionMixin:
    def get_sample_with_prompts(self, img_path: Path, label: int):
        example = {}
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.base_dataset.transform:
            image = self.base_dataset.transform(image)

        placeholder_string = self.get_class_prompts()[label]
        text = f"{random.choice(self.templates)} {placeholder_string}"

        if self.context_prompt is not None:
            text = f"{text} {self.context_prompt}"

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["text"] = text

        example["pixel_values"] = image
        return example

    def get_class_prompts(self):
        class_prompts = []
        for i in range(2):
            cls_idx = int(i * self.num_vectors)
            placeholder_string = " ".join(
                self.placeholder_token[cls_idx : cls_idx + self.num_vectors]
            )
            class_prompts.append(placeholder_string)
        return class_prompts


class BaseTextualDataset(TextualInversionMixin):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        tokenizer: CLIPTokenizer,
        placeholder_token: list[str],
        num_vectors: int,
        base_dataset: type,
        learnable_property: str = "object",  # [object, style]
        repeats: int = 1,
        context_prompt: Optional[str] = None,
        classes: Optional[list[str]] = None,
        file_list_path: Optional[Path] = None,
        transform=None,
    ):
        self.base_dataset = base_dataset(
            root_dir, split, classes, file_list_path, transform
        )

        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.placeholder_token = placeholder_token
        self.num_vectors = num_vectors
        self.context_prompt = context_prompt  # e.g "in the context of sks"

        self.num_images = len(self.base_dataset.image_list)
        self._length = self.num_images

        if split == "train":
            self._length = self.num_images * repeats

        self.templates = IMAGENET_CLIP_TEMPLATES

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        img_path = self.base_dataset.image_list[idx]
        label = self.base_dataset.labels[idx]
        example = self.get_sample_with_prompts(img_path=img_path, label=label)
        example["img_path"] = str(img_path)
        example["label"] = label
        return example


class TextAFHQ(BaseTextualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, base_dataset=AFHQ)


class TextKikiBouba(BaseTextualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, base_dataset=KikiBouba)


class TextButterfly(BaseTextualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, base_dataset=Butterfly)


class TextBlackHolesMadSane(BaseTextualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, base_dataset=BlackHolesMadSane)

    def get_sample_with_prompts(self, img_path: Path, label: int):
        example = {}

        image = np.load(img_path)
        # Scale the image to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=-1)
        # make it RGB
        image = np.repeat(image, 3, axis=-1)

        if self.base_dataset.transform:
            image = self.base_dataset.transform(image)

        placeholder_string = self.get_class_prompts()[label]
        text = f"{random.choice(self.templates)} {placeholder_string}"

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["pixel_values"] = image

        return example


class TextKermany(BaseTextualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, base_dataset=Kermany)


class TextCelebA(BaseTextualDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, base_dataset=CelebA)


def get_class_from_string(dataset_name: str):
    """Dynamically load a class from a string."""
    module_name = "textual_inversion.datasets_ti"
    if dataset_name == "afhq":
        class_name = "TextAFHQ"
    elif dataset_name == "kikibouba":
        class_name = "TextKikiBouba"
    elif dataset_name == "butterfly":
        class_name = "TextButterfly"
    elif dataset_name == "mad_sane":
        class_name = "TextBlackHolesMadSane"
    elif dataset_name == "kermany":
        class_name = "TextKermany"
    elif dataset_name == "celeba":
        class_name = "TextCelebA"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
