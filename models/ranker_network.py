"""Multilayer perceptron with early crossing of user and movie features.

User id, age group, gender and occupation, and movie id, genres and release
period are embedded and concatenated before the first layer, so every layer can
combine user and movie features (early crossing), unlike the two towers, which
meet only in a final inner product. Rankers that learn on candidate lists also
receive the retriever's score of the candidate as one more input. The output is
one number per user and movie.
"""
import torch

from models import ranker_settings as settings
from models.two_tower_network import (
    MovieFeatureSizes,
    MovieInputs,
    UserFeatureSizes,
    UserInputs,
)


def _build_perceptron(input_dimension: int) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    current_dimension = input_dimension
    for hidden_dimension in settings.HIDDEN_DIMENSIONS:
        layers.append(torch.nn.Linear(current_dimension, hidden_dimension))
        layers.append(torch.nn.SiLU())
        layers.append(torch.nn.BatchNorm1d(hidden_dimension))
        layers.append(torch.nn.Dropout(p=settings.DROPOUT_RATIO))
        current_dimension = hidden_dimension
    layers.append(torch.nn.Linear(current_dimension, 1))
    return torch.nn.Sequential(*layers)


class RankerNetwork(torch.nn.Module):
    def __init__(
        self,
        user_sizes: UserFeatureSizes,
        movie_sizes: MovieFeatureSizes,
        uses_retrieval_score: bool,
    ) -> None:
        super().__init__()
        self.uses_retrieval_score = uses_retrieval_score
        self.user_embedding = torch.nn.Embedding(
            user_sizes.user_count + 1,
            settings.USER_ID_DIMENSION,
        )
        self.age_embedding = torch.nn.Embedding(
            user_sizes.age_count + 1,
            settings.AGE_DIMENSION,
        )
        self.gender_embedding = torch.nn.Embedding(
            user_sizes.gender_count + 1,
            settings.GENDER_DIMENSION,
        )
        self.occupation_embedding = torch.nn.Embedding(
            user_sizes.occupation_count + 1,
            settings.OCCUPATION_DIMENSION,
        )
        self.movie_embedding = torch.nn.Embedding(
            movie_sizes.movie_count + 1,
            settings.MOVIE_ID_DIMENSION,
        )
        self.genre_projection = torch.nn.Linear(
            movie_sizes.genre_count,
            settings.GENRE_DIMENSION,
            bias=False,
        )
        self.release_embedding = torch.nn.Embedding(
            movie_sizes.release_period_count + 1,
            settings.RELEASE_PERIOD_DIMENSION,
        )
        self.perceptron = _build_perceptron(
            settings.USER_ID_DIMENSION
            + settings.AGE_DIMENSION
            + settings.GENDER_DIMENSION
            + settings.OCCUPATION_DIMENSION
            + settings.MOVIE_ID_DIMENSION
            + settings.GENRE_DIMENSION
            + settings.RELEASE_PERIOD_DIMENSION
            + int(uses_retrieval_score)
        )

    def forward(
        self,
        users: UserInputs,
        movies: MovieInputs,
        retrieval_scores: torch.Tensor,
    ) -> torch.Tensor:
        inputs: list[torch.Tensor] = [
            self.user_embedding(users.user_codes),
            self.age_embedding(users.ages),
            self.gender_embedding(users.genders),
            self.occupation_embedding(users.occupations),
            self.movie_embedding(movies.movie_codes),
            self.genre_projection(movies.genres),
            self.release_embedding(movies.release_periods),
        ]
        if self.uses_retrieval_score:
            inputs.append(retrieval_scores.unsqueeze(1))
        outputs: torch.Tensor = self.perceptron(torch.cat(inputs, dim=1))
        return outputs.squeeze(1)
