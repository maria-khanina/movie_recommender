import torch
import torch.nn.functional as F


def build_interaction_matrix(
    user_indices: torch.Tensor,
    movie_indices: torch.Tensor,
    max_user_id: int,
    max_movie_id: int,
) -> torch.Tensor:
    
    interaction_matrix = torch.zeros(
        max_user_id + 1, 
        max_movie_id + 1, 
        dtype=torch.bool,
    )
    
    interaction_matrix[user_indices, movie_indices] = True
    
    return interaction_matrix


def info_nce(
    user_vectors: torch.Tensor,
    movie_vectors: torch.Tensor,
    user_ids: torch.Tensor,
    movie_ids: torch.Tensor,
    movie_log_freqs: torch.Tensor | None = None,
    interaction_matrix: torch.Tensor | None = None,
    check_interactions=True,
    temperature=1.0,
) -> torch.Tensor:
    
    logits = torch.matmul(user_vectors, movie_vectors.T) / temperature

    if movie_log_freqs is not None:
        batch_movie_log_freqs = movie_log_freqs[movie_ids]
        logits = logits - batch_movie_log_freqs.unsqueeze(0)
    
    device = logits.device

    eye_mask = torch.eye(
        logits.size(0), 
        dtype=torch.bool, 
        device=device,
    )
    
    if check_interactions and interaction_matrix is not None:
        batch_mask = interaction_matrix[user_ids][:, movie_ids].to(
            device,
        )
        collision_mask = batch_mask & ~eye_mask
    else:
        movie_mask = (
            movie_ids.unsqueeze(1) == movie_ids.unsqueeze(0)
        )
        user_mask = (
            user_ids.unsqueeze(1) == user_ids.unsqueeze(0)
        )
        collision_mask = (movie_mask | user_mask) & ~eye_mask
        
    logits.masked_fill_(
        collision_mask, 
        -1e9,
    )
    
    targets = torch.arange(
        logits.size(0), 
        device=device,
    )
    
    return F.cross_entropy(
        logits, 
        targets,
    )
