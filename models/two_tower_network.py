"""User and movie towers of the two-tower retriever.

The user tower embeds the user id, age group, gender and occupation, the movie
tower embeds the movie id and the release period and projects the genre vector;
each tower passes the concatenated embeddings through a multilayer perceptron
(linear layer, SiLU, batch normalization, dropout) to a vector of the same size.
Both vectors are normalized, so their inner product is a cosine similarity. Code
0 of every embedding means unknown: rare and unseen users share it, which is how
the model serves a user it was not trained on.
"""
from typing import NamedTuple

import torch
import torch.nn.functional

from models import two_tower_settings as settings


class UserFeatureSizes(NamedTuple):
    user_count: int
    age_count: int
    gender_count: int
    occupation_count: int


class UserInputs(NamedTuple):
    user_codes: torch.Tensor
    ages: torch.Tensor
    genders: torch.Tensor
    occupations: torch.Tensor


class MovieInputs(NamedTuple):
    movie_codes: torch.Tensor
    genres: torch.Tensor
    release_periods: torch.Tensor


class MovieFeatureSizes(NamedTuple):
    movie_count: int
    genre_count: int
    release_period_count: int


def build_perceptron(input_dimension: int) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    current_dimension = input_dimension
    for hidden_dimension in settings.HIDDEN_DIMENSIONS:
        layers.append(torch.nn.Linear(current_dimension, hidden_dimension))
        layers.append(torch.nn.SiLU())
        layers.append(torch.nn.BatchNorm1d(hidden_dimension))
        layers.append(torch.nn.Dropout(p=settings.DROPOUT_RATIO))
        current_dimension = hidden_dimension
    layers.append(torch.nn.Linear(current_dimension, settings.OUTPUT_DIMENSION))
    return torch.nn.Sequential(*layers)


class UserTower(torch.nn.Module):
    def __init__(self, sizes: UserFeatureSizes) -> None:
        super().__init__()
        self.user_embedding = torch.nn.Embedding(
            sizes.user_count + 1,
            settings.USER_ID_DIMENSION,
        )
        self.age_embedding = torch.nn.Embedding(
            sizes.age_count + 1,
            settings.AGE_DIMENSION,
        )
        self.gender_embedding = torch.nn.Embedding(
            sizes.gender_count + 1,
            settings.GENDER_DIMENSION,
        )
        self.occupation_embedding = torch.nn.Embedding(
            sizes.occupation_count + 1,
            settings.OCCUPATION_DIMENSION,
        )
        self.perceptron = build_perceptron(
            settings.USER_ID_DIMENSION
            + settings.AGE_DIMENSION
            + settings.GENDER_DIMENSION
            + settings.OCCUPATION_DIMENSION
        )

    def forward(self, inputs: UserInputs) -> torch.Tensor:
        embeddings: list[torch.Tensor] = [
            self.user_embedding(inputs.user_codes),
            self.age_embedding(inputs.ages),
            self.gender_embedding(inputs.genders),
            self.occupation_embedding(inputs.occupations),
        ]
        vectors: torch.Tensor = self.perceptron(torch.cat(embeddings, dim=1))
        return torch.nn.functional.normalize(vectors, p=2.0, dim=1)


class MovieTower(torch.nn.Module):
    def __init__(self, sizes: MovieFeatureSizes) -> None:
        super().__init__()
        self.movie_embedding = torch.nn.Embedding(
            sizes.movie_count + 1,
            settings.MOVIE_ID_DIMENSION,
        )
        self.genre_projection = torch.nn.Linear(
            sizes.genre_count,
            settings.GENRE_DIMENSION,
            bias=False,
        )
        self.release_embedding = torch.nn.Embedding(
            sizes.release_period_count + 1,
            settings.RELEASE_PERIOD_DIMENSION,
        )
        self.perceptron = build_perceptron(
            settings.MOVIE_ID_DIMENSION
            + settings.GENRE_DIMENSION
            + settings.RELEASE_PERIOD_DIMENSION
        )

    def forward(self, inputs: MovieInputs) -> torch.Tensor:
        embeddings: list[torch.Tensor] = [
            self.movie_embedding(inputs.movie_codes),
            self.genre_projection(inputs.genres),
            self.release_embedding(inputs.release_periods),
        ]
        vectors: torch.Tensor = self.perceptron(torch.cat(embeddings, dim=1))
        return torch.nn.functional.normalize(vectors, p=2.0, dim=1)
