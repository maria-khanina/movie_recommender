import torch
import torch.nn.functional as F

from logic.two_towers import TwoTowerModel
from logic.recommender_dl import move_tensors


def get_all_movie_embeddings(
    two_towers: TwoTowerModel,
    all_movie_ids: torch.Tensor,
    all_genres: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    
    movie_vectors = two_towers.movie_tower(
        all_movie_ids.to(device),
        all_genres.to(device),
    )
    
    if two_towers.normalize_outputs:
        movie_vectors = F.normalize(
            movie_vectors, 
            p=2, 
            dim=1,
        )
        
    return movie_vectors


def get_batch_user_embeddings(
    two_towers: TwoTowerModel,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    
    user_vectors = two_towers.user_tower(
        tensors["user_ids"],
        tensors["ages"],
        tensors["genders"],
        tensors["jobs"],
    )

    if two_towers.normalize_outputs:
        user_vectors = F.normalize(
            user_vectors, 
            p=2, 
            dim=1,
        )
        
    return user_vectors


def calculate_batch_hits_macro(
    user_vectors: torch.Tensor,
    all_movie_vectors: torch.Tensor,
    all_movie_ids: torch.Tensor,
    target_movie_ids: torch.Tensor,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    similarity_scores = torch.matmul(
        user_vectors, 
        all_movie_vectors.T,
    )
    
    _, top_k = torch.topk(
        similarity_scores, 
        k=k, 
        dim=1,
    )
    
    target_movie_ids = target_movie_ids.view(-1, 1)
    
    retrieved_movie_ids = all_movie_ids[top_k].to(
        device,
    )
    
    match = (retrieved_movie_ids == target_movie_ids)
    
    return match.sum(dim=1)


def collect_user_hits(
    two_towers: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    var_key_map: dict[str, str],
    all_movie_vectors: torch.Tensor,
    all_movie_ids: torch.Tensor,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    users: list[torch.Tensor] = []
    hits: list[torch.Tensor] = []
    
    for batch in dataloader:
        tensors = move_tensors(
            var_key_map, 
            batch, 
            device,
        )
        
        user_vectors = get_batch_user_embeddings(
            two_towers=two_towers,
            tensors=tensors,
        )
        
        hits_per_row = calculate_batch_hits_macro(
            user_vectors=user_vectors,
            all_movie_vectors=all_movie_vectors,
            all_movie_ids=all_movie_ids,
            target_movie_ids=tensors["movie_ids"],
            k=k,
            device=device,
        )
        
        users.append(batch["real_user_id"].to(device))
        hits.append(hits_per_row)
        
    users = torch.cat(users)
    hits = torch.cat(hits).float()
    
    return users, hits


def compute_user_recalls(
    users: torch.Tensor,
    hits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    user_totals = torch.bincount(users)
    user_hits = torch.bincount(users, weights=hits)
    
    existing_mask = user_totals > 0
    user_ids = torch.nonzero(existing_mask).squeeze(-1)
    
    user_recalls = user_hits[user_ids] / user_totals[user_ids]
    
    return user_ids, user_recalls


def split_metrics(
    user_ids: torch.Tensor,
    user_recalls: torch.Tensor,
    cold_user_ids: torch.Tensor,
) -> dict[str, float]:
    cold_users_mask = torch.isin(user_ids, cold_user_ids)
    
    cold_recalls = user_recalls[cold_users_mask]
    warm_recalls = user_recalls[~cold_users_mask]
    
    return {
        "global": user_recalls.mean().item(),
        "warm": warm_recalls.mean().item(),
        "cold": cold_recalls.mean().item(),
    }


def recall_at_k_macro(
    two_towers: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    var_key_map: dict[str, str],
    all_movie_ids: torch.Tensor,
    all_genres: torch.Tensor,
    device: torch.device,
    cold_user_ids: torch.Tensor,
    k: int = 100,
) -> dict[str, float]:
    two_towers.eval()
    
    with torch.no_grad():
        all_movie_vectors = get_all_movie_embeddings(
            two_towers=two_towers,
            all_movie_ids=all_movie_ids,
            all_genres=all_genres,
            device=device,
        )
        
        users, hits = collect_user_hits(
            two_towers=two_towers,
            dataloader=dataloader,
            var_key_map=var_key_map,
            all_movie_vectors=all_movie_vectors,
            all_movie_ids=all_movie_ids,
            k=k,
            device=device,
        )
            
    user_ids, user_recalls = compute_user_recalls(users, hits)
    
    return split_metrics(user_ids, user_recalls, cold_user_ids)
