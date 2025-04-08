import torch
import pyrallis
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision import models
#import datasets from parent folder

from datasets import get_cls_dataset_by_name 
from textual_inversion_config import ClassifierTrainConfig
import torch.nn as nn
import numpy as np
import torchvision.transforms.v2 as transforms
import pdb
import os

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    log_dir: Path,
    patience: int,
    device: torch.device,
    model_name: str
):
    loss_fn = nn.BCEWithLogitsLoss()  # Suitable for binary classification with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = np.inf
    early_stop_counter = 0
    with open(log_dir / f"{model_name}_log.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Device: {device}\n")

        for epoch in range(epochs):
            print(epoch)
            model.train()
            train_loss = 0.0
            train_corrects = 0

            for images, labels, paths in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                if not len(outputs.shape) == len(labels.shape):
                    labels = labels.unsqueeze(1)
                preds = torch.round(torch.sigmoid(outputs))
                loss = loss_fn(outputs, labels.float())
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
                for images, labels, paths in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    if not len(outputs.shape) == len(labels.shape):
                        labels = labels.unsqueeze(1)
                    preds = torch.round(torch.sigmoid(outputs))

                    loss = loss_fn(outputs, labels.float())
                    val_loss += loss.item() * images.size(0)
                    val_corrects += torch.sum(preds == labels.float())

            val_loss /= len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            #write to text file in log_dir
            f.write(f"Epoch {epoch+1}/{epochs}, "
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} \n")

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


@pyrallis.wrap()
def main(cfg: ClassifierTrainConfig):
    torch.manual_seed(42)

    log_dir = cfg.log_dir
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    lr = cfg.lr
    patience = cfg.patience  # Early stopping patience
    device_id = cfg.device
    normalize = False
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ]
    )

    # cfg.dataset.file_list_path =  None #cfg.file_list_paths[0]
    # orig_classes = cfg.dataset.classes
    # cfg.dataset.classes = [orig_classes[0]]
    # _, val_ds_0 = get_cls_dataset_by_name(
    #     cfg.dataset, dataset_transforms=[val_transform, val_transform]
    # )

    # cfg.dataset.file_list_path = None #cfg.file_list_paths[1]
    # cfg.dataset.classes = [orig_classes[1]]
    cfg.dataset.file_list_path = None
    _, val_ds = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[val_transform, val_transform]
    )
    cfg.dataset.file_list_paths = cfg.file_list_paths
    #cfg.dataset.classes = orig_classes

    #add val_ds_0 and val_ds_1 together

    
    cfg.dataset.num_samples = 1000 #num_samples
    cfg.log_dir = log_dir 
    os.makedirs(cfg.log_dir, exist_ok=True)
    \
    train_ds, _ = get_cls_dataset_by_name(
        cfg.dataset, dataset_transforms=[train_transform, val_transform]
    )

    #val_ds = torch.utils.data.ConcatDataset([val_ds_0, val_ds_1])




    # Define the MobileNetV2 model
    mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 1)  # Binary output
    mobilenet = mobilenet.to(device)

    # Define the ResNet18 model
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)  # Binary output
    resnet = resnet.to(device)

    # Define the EfficientNet-B0 model
    efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    efficientnet.classifier[1] = nn.Linear(
        efficientnet.classifier[1].in_features, 1
    )  # Binary output
    efficientnet = efficientnet.to(device)

    # Store models in a list for easy access
    ensemble_models = [resnet, efficientnet, mobilenet] 

    # Train each model in the ensemble
    for model in ensemble_models:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=32
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=16)

        model_name = model.__class__.__name__
        print(f"Train {model_name}")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            log_dir=cfg.log_dir,
            patience=patience,
            device=device,
            model_name=model_name
        )

    print("Training complete")


if __name__ == "__main__":
    main()