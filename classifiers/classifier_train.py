import torch
import pyrallis
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision import models
from datasets import get_cls_dataset_by_name
from textual_inversion_config import ClassifierTrainConfig
import torch.nn as nn
import numpy as np
import torchvision.transforms.v2 as transforms


def add_linear_layer(model, num_classes):
    # Get the number of features in the last layer
    if hasattr(model, "fc"):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, num_classes)
        else:
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture")

    return model


@pyrallis.wrap()
def main(cfg: ClassifierTrainConfig):
    torch.manual_seed(42)

    log_dir = cfg.log_dir
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    lr = cfg.lr
    patience = cfg.patience  # Early stopping patience
    device_id = cfg.device

    best_val_loss = np.inf
    early_stop_counter = 0

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    train_ds, val_ds = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[train_transform, val_transform]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=64
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=16)

    # Model
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model = add_linear_layer(model, train_ds.num_classes)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.to(device)

    for epoch in range(epochs):
        print(epoch)
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(preds == labels)

        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_acc = train_corrects.double() / len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels)

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
            save_path = log_dir / f"model_{epoch}.pth"
            torch.save(model, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break

    print("Training complete")


if __name__ == "__main__":
    main()
