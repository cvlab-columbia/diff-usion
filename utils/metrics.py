import torch
from scipy import linalg
from dataclasses import dataclass
from torchvision.models.inception import inception_v3, Inception_V3_Weights
import numpy as np
import torch.nn as nn
import lpips
from typing import Optional, Union
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F


@dataclass
class EnsemblePredictOutput:
    probs: torch.Tensor
    preds: torch.Tensor


def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0


def compute_lpips_similarity(
    images_1: torch.Tensor, images_2: torch.Tensor, reduction: Optional[str] = None
):
    device = images_1.device
    # Load the LPIPS model (use vgg for better perceptual similarity, alex for speed)
    lpips_model = lpips.LPIPS() # alexnet by default
    lpips_model.eval()

    # Ensure model is using the right device
    lpips_model = lpips_model.to(device)

    # lpips model expects images in range [-1,1]
    img1_tensor = normalize(images_1)
    img2_tensor = normalize(images_2)

    # Compute the LPIPS similarity score
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor)

    distance = distance.flatten()
    if reduction == "mean":
        distance = distance.mean()
    return distance



def calculate_fid(
    real_images: torch.Tensor, fake_images: torch.Tensor, inception_model: nn.Module
):
    def get_activations(images, model):
        model.eval()
        with torch.no_grad():
            activations = model(images)[0]
        return activations.cpu().numpy().reshape(images.shape[0], -1)

    real_activations = get_activations(real_images, inception_model)
    fake_activations = get_activations(fake_images, inception_model)

    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(
        real_activations, rowvar=False
    )
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(
        fake_activations, rowvar=False
    )

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


def load_inception_model():
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception_model.fc = nn.Identity()
    inception_model.eval()
    return inception_model


def ensemble_predict(
    models: list[nn.Module], images: torch.Tensor
) -> EnsemblePredictOutput:
    if len(models) == 0:
        raise RuntimeError(
            "Classifiers list is empty, cannot evaluate. Check model loading"
        )

    # Stack predictions
    outputs = torch.zeros(len(images), device=images.device)

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs += torch.sigmoid(model(images).squeeze())  # Sum up probabilities

    # Average probabilities and apply threshold
    avg_outputs = outputs / len(models)
    final_preds = (
        avg_outputs > 0.5
    ).float()  # Binary prediction based on average probability
    output = EnsemblePredictOutput(probs=avg_outputs, preds=final_preds)
    return output
