import random

import pandas as pd

import torch
from torch.utils.data import Dataset


class PositiveRatings(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        genres_cols: list[str],
        min_rating=4.0,
        is_training=False,
        user_dropout=0.0,
        trained_users: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        
        positive_df = df[df["rating"] >= min_rating].reset_index(
            drop=True,
        )
        

        self.real_user_ids = torch.tensor(
            positive_df["user_id"].values,
            dtype=torch.long,
        )

        self.user_ids = self.real_user_ids.clone()
        if not is_training:
            new_users_mask = ~torch.isin(self.user_ids, trained_users)
            self.user_ids[new_users_mask] = 0
    
        self.movie_ids = torch.tensor(
            positive_df["movie_id"].values, 
            dtype=torch.long,
        )
        self.ages = torch.tensor(
            positive_df["age"].values, 
            dtype=torch.long,
        )
        self.genders = torch.tensor(
            positive_df["gender"].values, 
            dtype=torch.long,
        )
        self.jobs = torch.tensor(
            positive_df["occupation"].values, 
            dtype=torch.long,
        )
        self.genres = torch.tensor(
            positive_df[genres_cols].values, 
            dtype=torch.float32,
        )

        self.user_dropout = user_dropout
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id = self.user_ids[idx]
        if self.is_training and random.random() < self.user_dropout:
            user_id = torch.tensor(0, dtype=torch.long)

        return {
            "user_id": user_id,
            "real_user_id": self.real_user_ids[idx],
            "movie_id": self.movie_ids[idx],
            "age": self.ages[idx],
            "gender": self.genders[idx],
            "job": self.jobs[idx],
            "genre": self.genres[idx],
        }


class Ratings(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        genres_cols: list[str],
        is_training=True,
        user_dropout=0.05,
    ) -> None:
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.movie_ids = torch.tensor(df["movie_id"].values, dtype=torch.long)
        self.ages = torch.tensor(df["age"].values, dtype=torch.long)
        self.genders = torch.tensor(df["gender"].values, dtype=torch.long)
        self.jobs = torch.tensor(df["occupation"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["normalized_rating"].values, dtype=torch.float32)
        self.genres = torch.tensor(df[genres_cols].values, dtype=torch.float32)
        self.is_training = is_training
        self.user_dropout = user_dropout

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id = self.user_ids[idx]
        if self.is_training and random.random() < self.user_dropout:
            user_id = torch.tensor(0, dtype=torch.long)
            
        return {
            "user_id": user_id,
            "movie_id": self.movie_ids[idx],
            "age": self.ages[idx],
            "gender": self.genders[idx],
            "job": self.jobs[idx],
            "genres": self.genres[idx], 
            "rating": self.ratings[idx],
        }
