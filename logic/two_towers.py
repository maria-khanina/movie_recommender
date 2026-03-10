import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
        input_dim: int, 
        hidden_dims: list[int],
        output_dim: int,
        dropout_ratio: float,
) -> nn.Sequential:
    layers = []
    curr_dim = input_dim
    
    for ind, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(curr_dim, dim))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.Dropout(p=dropout_ratio))
            
        curr_dim = dim
        
    layers.append(nn.Linear(curr_dim, output_dim))
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    def __init__(
        self,
        users_num: int,
        ages_num: int,
        genders_num: int,
        jobs_num: int,
        user_dim=16,
        age_dim=3,
        gender_dim=2,
        job_dim=4,
        hidden_dims=[64, 32],
        output_dim=16,
        dropout_ratio=0.2,
    ) -> None:
        super().__init__()
        
        self.user_embedding = nn.Embedding(
            users_num + 1, 
            user_dim,
        )
        self.age_embedding = nn.Embedding(
            ages_num + 1, 
            age_dim,
        )
        self.gender_embedding = nn.Embedding(
            genders_num + 1, 
            gender_dim,
        )
        self.job_embedding = nn.Embedding(
            jobs_num + 1, 
            job_dim,
        )
        
        input_dim = user_dim + age_dim + gender_dim + job_dim
        
        self.mlp = build_mlp(
            input_dim, 
            hidden_dims,
            output_dim,
            dropout_ratio,
        )

    def forward(
        self, 
        user_ids: torch.Tensor, 
        ages: torch.Tensor, 
        genders: torch.Tensor, 
        jobs: torch.Tensor,
    ) -> torch.Tensor:
        user_vector = self.user_embedding(user_ids)
        age_vector = self.age_embedding(ages)
        gender_vector = self.gender_embedding(genders)
        job_vector = self.job_embedding(jobs)
        
        cat_features = torch.cat(
            [
                user_vector, 
                age_vector, 
                gender_vector, 
                job_vector,
            ], 
            dim=1,
        )
        return self.mlp(cat_features)


class MovieTower(nn.Module):
    def __init__(
        self,
        movies_num: int,
        genres_num: int,
        movie_dim=16,
        genre_dim=4,
        hidden_dims=[64, 32],
        output_dim=16,
        dropout_ratio=0.1,
    ) -> None:
        super().__init__()
        
        self.movie_embedding = nn.Embedding(
            movies_num + 1, 
            movie_dim,
        )
        self.genre_projector = nn.Linear(
            genres_num, 
            genre_dim, 
            bias=False,
        )
        
        input_dim = movie_dim + genre_dim
        
        self.mlp = build_mlp(
            input_dim, 
            hidden_dims,
            output_dim,
            dropout_ratio,
        )

    def forward(
            self, 
            movie_ids: torch.Tensor, 
            genres: torch.Tensor,
    ) -> torch.Tensor:
        movie_vector = self.movie_embedding(movie_ids)
        genre_vector = self.genre_projector(genres)
        
        cat_features = torch.cat(
            [
                movie_vector, 
                genre_vector,
            ], 
            dim=1,
        )
        return self.mlp(cat_features)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_tower: UserTower,
        movie_tower: MovieTower,
        normalize_outputs=True,
    ) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower
        self.normalize_outputs = normalize_outputs

    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        ages: torch.Tensor,
        genders: torch.Tensor,
        jobs: torch.Tensor,
        genres: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        user_vectors = self.user_tower(
            user_ids, 
            ages, 
            genders, 
            jobs,
        )
        movie_vectors = self.movie_tower(
            movie_ids, 
            genres,
        )
        
        if self.normalize_outputs:
            user_vectors = F.normalize(
                user_vectors, 
                p=2, 
                dim=1,
            )
            movie_vectors = F.normalize(
                movie_vectors, 
                p=2, 
                dim=1,
            )
        
        return user_vectors, movie_vectors
