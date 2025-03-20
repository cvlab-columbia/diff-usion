import torch
import pyrallis
from tqdm import tqdm
from shutil import copy
from pathlib import Path
from typing import Callable
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet18_Weights
from torchvision import models
from datasets import Spawrious
from textual_inversion_config import ClassifierTrainConfig
import torch.nn as nn
import numpy as np
import torchvision.transforms.v2 as transforms


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    log_dir: Path,
    patience: int,
    device: torch.device,
    model_name: str,
    loss_fn: Callable,
    save_val_paths: bool = False,
) -> Path:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = np.inf
    early_stop_counter = 0

    for epoch in range(epochs):
        print(epoch)
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for batch in tqdm(train_loader):
            images, labels, *_ = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # Determine if it's binary or multi-class classification
            if outputs.shape[-1] == 1:  # Binary classification
                labels = labels.unsqueeze(1) if labels.dim() == 1 else labels
                loss = loss_fn(outputs, labels.float())  # BCEWithLogitsLoss
                preds = torch.round(torch.sigmoid(outputs))
            else:  # Multi-class classification
                loss = loss_fn(outputs, labels.long())  # CrossEntropyLoss
                preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(preds == labels.float())

        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_acc = train_corrects.double() / len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels, *_ = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # Determine if it's binary or multi-class classification
                if outputs.shape[-1] == 1:  # Binary classification
                    labels = labels.unsqueeze(1) if labels.dim() == 1 else labels
                    loss = loss_fn(outputs, labels.float())  # BCEWithLogitsLoss
                    preds = torch.round(torch.sigmoid(outputs))
                else:  # Multi-class classification
                    loss = loss_fn(outputs, labels.long())  # CrossEntropyLoss
                    preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.float())

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_path = log_dir / f"{model_name}_{epoch}.pth"
            torch.save(model, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break

    saved_paths = []
    if save_val_paths is True:
        with torch.no_grad():
            for batch in val_loader:
                images, labels, bgs, image_paths = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # Determine if it's binary or multi-class classification
                if outputs.shape[-1] == 1:  # Binary classification
                    labels = labels.unsqueeze(1) if labels.dim() == 1 else labels
                    loss = loss_fn(outputs, labels.float())  # BCEWithLogitsLoss
                    preds = torch.round(torch.sigmoid(outputs))
                else:  # Multi-class classification
                    loss = loss_fn(outputs, labels.long())  # CrossEntropyLoss
                    preds = torch.argmax(outputs, dim=1)

                # Save images by pred
                for i, pred in enumerate(preds.detach().cpu().flatten()):
                    img_path = Path(image_paths[i])
                    saved_paths.append((img_path, pred.item()))

    return Path(save_path), saved_paths


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_incorrect: bool = False,
) -> list[Path]:
    # Test loop
    model.eval()
    test_corrects = 0
    saved_paths = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels, bgs, image_paths = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Determine if it's binary or multi-class classification
            if outputs.shape[-1] == 1:  # Binary classification
                labels = labels.unsqueeze(1) if labels.dim() == 1 else labels
                preds = torch.round(torch.sigmoid(outputs))
            else:  # Multi-class classification
                preds = torch.argmax(outputs, dim=1)

            test_corrects += torch.sum(preds == labels.float())

            if save_incorrect is True:
                # Save incorrectly classified images
                incorrect_indices = (preds != labels).cpu().numpy().flatten()
                for i, incorrect in enumerate(incorrect_indices):
                    if incorrect:
                        img_path = Path(image_paths[i])
                        saved_paths.append(img_path)

    test_acc = test_corrects.double() / len(test_loader.dataset)

    print(f"Test Acc: {test_acc:.4f}")
    return saved_paths


@pyrallis.wrap()
def main(cfg: ClassifierTrainConfig):
    torch.manual_seed(42)

    log_dir = cfg.log_dir
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    lr = cfg.lr
    patience = cfg.patience  # Early stopping patience
    device_id = cfg.device
    val_split = 0.1

    save_val_paths = True
    save_test_incorrect = True

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    datasets = Spawrious(
        benchmark="o2o_hard",
        root_dir=cfg.dataset.image_dir,
        train_transforms=train_transform,
        test_transforms=test_transform,
        split=0,
    )

    # train/val split
    train_ds = datasets.get_train_dataset()
    val_size = int(len(train_ds) * val_split)
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - val_size, val_size]
    )
    test_ds = datasets.get_test_dataset()
    num_classes = len(datasets.class_list)

    # Define the ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if num_classes == 2:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

    model_name = model.__class__.__name__
    print(f"Train {model_name}")
    model_path, saved_val_paths = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        log_dir=log_dir,
        patience=patience,
        device=device,
        model_name=model_name,
        loss_fn=loss_fn,
        save_val_paths=save_val_paths,
    )

    print("Training complete")
    final_model = torch.load(model_path).to(device)

    print("Test")
    saved_paths = test_model(
        final_model, test_loader, device, save_incorrect=save_test_incorrect
    )

    print("Copy Miss-classified Test Files")
    for img_path in saved_paths:
        save_dir = cfg.log_dir / "test"
        save_dir.mkdir(exist_ok=True, parents=True)
        class_name = img_path.name.split("_")[1]
        save_cls_dir = save_dir / class_name
        save_cls_dir.mkdir(exist_ok=True)

        copy(img_path, save_cls_dir / img_path.name)

    print("Copy Predicted Validation Files")
    for img_path, pred in saved_val_paths:
        save_dir = cfg.log_dir / "val" / "images"
        save_dir.mkdir(exist_ok=True, parents=True)

        class_name = datasets.class_list[int(pred)]
        save_cls_dir = save_dir / class_name
        save_cls_dir.mkdir(exist_ok=True)

        copy(img_path, save_cls_dir / img_path.name)


if __name__ == "__main__":
    main()
