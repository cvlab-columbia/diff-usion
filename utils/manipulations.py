import torch
from typing import Optional


def find_nearest_neighbors(
    input_embed: torch.Tensor, pool_embeds: torch.Tensor, k: int
):
    # Normalize the input and pool embeddings to unit vectors for cosine similarity
    input_embed = input_embed / input_embed.norm(dim=-1, keepdim=True)
    pool_embeds = pool_embeds / pool_embeds.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarities = torch.matmul(input_embed, pool_embeds.T)

    # Get the top k indices with the highest similarity values
    top_k_similarities, top_k_indices = torch.topk(similarities, k=k, largest=True)
    top_k_embeds = pool_embeds[top_k_indices]

    return top_k_embeds


def compute_weighted_similarity_average(
    input_embed: torch.Tensor, pool_embeds: torch.Tensor, temperature: int = 1
):

    # Normalize embeddings to unit vectors
    input_embed = input_embed / input_embed.norm(dim=-1, keepdim=True)
    pool_embeds = pool_embeds / pool_embeds.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarities = torch.matmul(input_embed, pool_embeds.T)

    # Convert similarities to weights (ensure positive weights)
    weights = torch.relu(similarities)  # Clip negative similarities to zero (optional)
    # weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize to sum to 1 (optional)
    weights = torch.softmax(weights / temperature, dim=-1)

    # Compute weighted average
    weighted_average = torch.matmul(weights, pool_embeds)
    weighted_average = weighted_average / weighted_average.norm(dim=-1, keepdim=True)

    return weighted_average


def top_k_pca_directions(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Finds the top k PCA directions of a tensor matrix of CLIP image embeddings.

    Args:
        X (torch.Tensor): Tensor of shape (N, D) with image embeddings.
        k (int): Number of principal components to extract.

    Returns:
        torch.Tensor: Top k PCA directions of shape (D, k).
    """
    # Step 1: Center the data (subtract mean)
    X_centered = X - X.mean(dim=0, keepdim=True)

    # Step 2: Compute the covariance matrix (D x D)
    cov_matrix = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Step 3: Eigen decomposition (or use SVD)
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)  # eigvecs: (D, D)

    # Step 4: Get top-k eigenvectors (largest eigenvalues)
    top_k_indices = torch.argsort(eigvals, descending=True)[:k]
    top_k_eigenvectors = eigvecs[:, top_k_indices]  # Shape (D, k)

    return top_k_eigenvectors


def project_embedding_svd(
    input_embedding: torch.Tensor,
    pool_embeddings: torch.Tensor,
    r: Optional[int] = None,
):
    """
    Projects an input embedding onto the subspace spanned by the pool of embeddings using SVD.

    Args:
        input_embedding (torch.Tensor): A tensor of shape (B, D) representing the input embeddings.
        pool_embeddings (torch.Tensor): A tensor of shape (N, D) where each row is an embedding in the subspace.

    Returns:
        torch.Tensor: The projected input embedding onto the subspace of pool_embeddings, shape (D,).
    """
    # Center the data (mean subtraction)
    # input_centered = input_embedding - input_embedding.mean(dim=1, keepdim=True)
    # pool_embeddings_centered = pool_embeddings - pool_embeddings.mean(dim=0)

    # Perform SVD on the pool embeddings
    U, S, Vt = torch.svd(pool_embeddings)
    if r is not None:
        Vt = Vt[:, :r]

    # Project the input embedding onto the subspace spanned by V (right singular vectors)
    # projected_embedding = Vt.T @ (Vt @ Vt.T).inverse() @ input_centered.T
    projected_embedding = input_embedding @ (Vt @ Vt.T)

    return projected_embedding
