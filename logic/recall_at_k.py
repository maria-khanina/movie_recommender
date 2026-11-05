from typing import Literal

import torch
import torch.nn.functional as F

from two_towers import TwoTowerModel


def get_movie_embeddings(
    two_towers: TwoTowerModel,
    movie_ids: torch.Tensor,
    genres: torch.Tensor,
) -> torch.Tensor:
    
    movie_vectors = two_towers.movie_tower(
        movie_ids,
        genres,
    )

    if two_towers.normalize_outputs:
        movie_vectors = F.normalize(
            movie_vectors, 
            p=2, 
            dim=1,
        )
        
    return movie_vectors


def get_user_embeddings(
    two_towers: TwoTowerModel,
    user_ids: torch.Tensor,
    ages: torch.Tensor,
    genders: torch.Tensor,
    jobs: torch.Tensor,
) -> torch.Tensor:
    
    user_vectors = two_towers.user_tower(user_ids, ages, genders, jobs)

    if two_towers.normalize_outputs:
        user_vectors = F.normalize(user_vectors, p=2, dim=1)
        
    return user_vectors


def calculate_batch_hits_macro(
    user_vectors: torch.Tensor,
    all_movie_vectors: torch.Tensor,
    all_movie_ids: torch.Tensor,
    liked_movie_ids: torch.Tensor,
    max_k: int,
    device: torch.device,
) -> torch.Tensor:
    similarity_scores = torch.matmul(
        user_vectors, 
        all_movie_vectors.T,
    )
    
    _, top_k = torch.topk(
        similarity_scores, 
        k=max_k, 
        dim=1,
    )

    liked_movie_ids = liked_movie_ids.view(-1, 1)
    retrieved_movie_ids = all_movie_ids[top_k].to(device)
    match = (retrieved_movie_ids == liked_movie_ids)
    
    return match


def collect_user_hits(
    two_towers: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    var_key_map: dict[str, str],
    all_movie_vectors: torch.Tensor,
    all_movie_ids: torch.Tensor,
    max_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    users: list[torch.Tensor] = []
    matches: list[torch.Tensor] = []

    for batch in dataloader:
        user_ids = batch[var_key_map["user_ids"]].to(device)
        ages = batch[var_key_map["ages"]].to(device)
        genders = batch[var_key_map["genders"]].to(device)
        jobs = batch[var_key_map["jobs"]].to(device)
        
        liked_movie_ids = batch[var_key_map["movie_ids"]].to(device)
        real_user_ids = batch["real_user_id"].to(device)
        
        user_vectors = get_user_embeddings(
            two_towers=two_towers,
            user_ids=user_ids,
            ages=ages,
            genders=genders,
            jobs=jobs,
        )
        
        match = calculate_batch_hits_macro(
            user_vectors=user_vectors,
            all_movie_vectors=all_movie_vectors,
            all_movie_ids=all_movie_ids,
            liked_movie_ids=liked_movie_ids,
            max_k=max_k,
            device=device,
        )
        
        users.append(real_user_ids)
        matches.append(match)
            
    users_tensor = torch.cat(users)
    matches_tensor = torch.cat(matches)
    
    return users_tensor, matches_tensor


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


recall_report = dict[Literal["global", "warm", "cold"], float]

def split_metrics(
    user_ids: torch.Tensor,
    user_recalls: torch.Tensor,
    cold_user_ids: torch.Tensor,
) -> recall_report:
    
    cold_users_mask = torch.isin(user_ids, cold_user_ids)
    
    cold_recalls = user_recalls[cold_users_mask]
    warm_recalls = user_recalls[~cold_users_mask]
    
    return {
        "global": user_recalls.mean().item(),
        "warm": warm_recalls.mean().item(),
        "cold": cold_recalls.mean().item(),
    }


def recall_at_ks_macro(
    two_towers: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    var_key_map: dict[str, str],
    all_movie_ids: torch.Tensor,
    all_genres: torch.Tensor,
    device: torch.device,
    cold_user_ids: torch.Tensor,
    ks: list[int] = [10, 50, 100, 200, 500],
) -> dict[int, recall_report]:
    
    two_towers.eval()
    max_k = max(ks)
    
    with torch.no_grad():
        all_movie_ids = all_movie_ids.to(device)
        all_genres = all_genres.to(device)
        
        all_movie_vectors = get_movie_embeddings(
            two_towers=two_towers,
            movie_ids=all_movie_ids,
            genres=all_genres,
        )
        
        users, matches = collect_user_hits(
            two_towers=two_towers,
            dataloader=dataloader,
            var_key_map=var_key_map,
            all_movie_vectors=all_movie_vectors,
            all_movie_ids=all_movie_ids,
            max_k=max_k,
            device=device,
        )
            
    recalls_by_k = {}

    for k in ks:
        hits = matches[:, :k].sum(dim=1).float()
        user_ids, user_recalls = compute_user_recalls(users, hits)
        recalls_by_k[k] = split_metrics(user_ids, user_recalls, cold_user_ids)
    
    return recalls_by_k
