"""User towers that read the user's rating history.

- Liked-mean tower: the user id, age group, gender and occupation embeddings,
  the mean embedding of the liked movies in the history window and the log of
  their count go through a multilayer perceptron. An empty history gives a zero
  mean and a zero count, which the network can tell apart from a real mean.
- Sequence tower (SASRec): a token per rating is the sum of the movie
  embedding, the rating embedding, the projected genres, the release period
  embedding and the position embedding; a first token carries the age group,
  gender and occupation. Transformer encoder layers with the normalization
  before each sublayer read the window, padding masked, and the output at the
  most recent rating (or at the user token, when the window is empty) is
  projected to the user vector. With concatenated demographics there is no
  user token: the demographics embedding is concatenated to every rating
  token and projected back to the token size; an empty window is read as a
  single token that holds the demographics alone. It has no user id:
  everything it knows about a user comes from the history and the
  demographics, so a new user is served as soon as there are ratings.

Both return normalized vectors, comparable with the movie tower by cosine. The
sequence tower's movie embedding can also serve as a flat movie tower, which
makes the model the original SASRec: one embedding per movie on both sides.
"""
from typing import NamedTuple

import torch
import torch.nn.functional

from evaluation.evaluation_settings import SECONDS_PER_DAY
from models import history_tower_settings as settings
from models import two_tower_settings as tower_settings
from models.rating_sequences import HistoryWindow
from models.two_tower_network import (
    MovieFeatureSizes,
    MovieInputs,
    UserFeatureSizes,
    UserInputs,
    build_perceptron,
)


class SequenceOptions(NamedTuple):
    window_length: int
    uses_positions: bool
    uses_rating_age: bool
    concatenates_demographics: bool


class LikedMeanUserTower(torch.nn.Module):
    def __init__(self, user_sizes: UserFeatureSizes, movie_count: int) -> None:
        super().__init__()
        self.user_embedding = torch.nn.Embedding(
            user_sizes.user_count + 1,
            tower_settings.USER_ID_DIMENSION,
        )
        self.age_embedding = torch.nn.Embedding(
            user_sizes.age_count + 1,
            tower_settings.AGE_DIMENSION,
        )
        self.gender_embedding = torch.nn.Embedding(
            user_sizes.gender_count + 1,
            tower_settings.GENDER_DIMENSION,
        )
        self.occupation_embedding = torch.nn.Embedding(
            user_sizes.occupation_count + 1,
            tower_settings.OCCUPATION_DIMENSION,
        )
        self.history_embedding = torch.nn.Embedding(
            movie_count + 1,
            settings.HISTORY_EMBEDDING_DIMENSION,
            padding_idx=0,
        )
        self.perceptron = build_perceptron(
            tower_settings.USER_ID_DIMENSION
            + tower_settings.AGE_DIMENSION
            + tower_settings.GENDER_DIMENSION
            + tower_settings.OCCUPATION_DIMENSION
            + settings.HISTORY_EMBEDDING_DIMENSION
            + 1
        )

    def _pool_liked(self, window: HistoryWindow) -> list[torch.Tensor]:
        weights = window.liked.float()
        counts = weights.sum(dim=1, keepdim=True)
        embeddings: torch.Tensor = self.history_embedding(window.movie_codes)
        sums = (embeddings * weights.unsqueeze(2)).sum(dim=1)
        return [sums / counts.clamp(min=1.0), torch.log1p(counts)]

    def forward(self, users: UserInputs, window: HistoryWindow) -> torch.Tensor:
        inputs: list[torch.Tensor] = [
            self.user_embedding(users.user_codes),
            self.age_embedding(users.ages),
            self.gender_embedding(users.genders),
            self.occupation_embedding(users.occupations),
            *self._pool_liked(window),
        ]
        vectors: torch.Tensor = self.perceptron(torch.cat(inputs, dim=1))
        return torch.nn.functional.normalize(vectors, p=2.0, dim=1)


class SequenceUserTower(torch.nn.Module):
    def __init__(
        self,
        user_sizes: UserFeatureSizes,
        movie_sizes: MovieFeatureSizes,
        movies: MovieInputs,
        options: SequenceOptions,
    ) -> None:
        super().__init__()
        self.movies = movies
        self.options = options
        self._build_rating_embeddings(movie_sizes)
        self._build_user_embeddings(user_sizes)
        self.encoder = _build_encoder()
        self.output_projection = torch.nn.Linear(
            settings.SEQUENCE_DIMENSION,
            settings.SEQUENCE_DIMENSION,
        )
        self.demographics_projection = torch.nn.Linear(
            2 * settings.SEQUENCE_DIMENSION,
            settings.SEQUENCE_DIMENSION,
        )

    def _build_rating_embeddings(self, movie_sizes: MovieFeatureSizes) -> None:
        dimension = settings.SEQUENCE_DIMENSION
        self.movie_embedding = torch.nn.Embedding(
            movie_sizes.movie_count + 1,
            dimension,
            padding_idx=0,
        )
        self.rating_embedding = torch.nn.Embedding(
            settings.RATING_CODE_COUNT + 1,
            dimension,
            padding_idx=0,
        )
        self.genre_projection = torch.nn.Linear(movie_sizes.genre_count, dimension)
        self.release_embedding = torch.nn.Embedding(
            movie_sizes.release_period_count + 1,
            dimension,
        )
        self.position_embedding = torch.nn.Embedding(
            self.options.window_length + 1,
            dimension,
        )
        self.age_embedding_by_bucket = torch.nn.Embedding(
            settings.RATING_AGE_BUCKET_COUNT + 1,
            dimension,
            padding_idx=0,
        )

    def _build_user_embeddings(self, user_sizes: UserFeatureSizes) -> None:
        dimension = settings.SEQUENCE_DIMENSION
        self.age_embedding = torch.nn.Embedding(user_sizes.age_count + 1, dimension)
        self.gender_embedding = torch.nn.Embedding(
            user_sizes.gender_count + 1,
            dimension,
        )
        self.occupation_embedding = torch.nn.Embedding(
            user_sizes.occupation_count + 1,
            dimension,
        )

    def _embed_ratings(self, window: HistoryWindow) -> torch.Tensor:
        movie_rows = (window.movie_codes - 1).clamp(min=0)
        tokens: torch.Tensor = (
            self.movie_embedding(window.movie_codes)
            + self.rating_embedding(window.rating_codes)
            + self.genre_projection(self.movies.genres[movie_rows])
            + self.release_embedding(self.movies.release_periods[movie_rows])
        )
        if self.options.uses_rating_age:
            tokens = tokens + self.age_embedding_by_bucket(_bucket_ages(window))
        return tokens * window.valid.unsqueeze(2)

    def _embed_user(self, users: UserInputs) -> torch.Tensor:
        user_token: torch.Tensor = (
            self.age_embedding(users.ages)
            + self.gender_embedding(users.genders)
            + self.occupation_embedding(users.occupations)
        )
        return user_token.unsqueeze(1)

    def _prepend_user_token(
        self,
        users: UserInputs,
        window: HistoryWindow,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_token = self._embed_user(users)
        tokens = torch.cat([user_token, self._embed_ratings(window)], dim=1)
        user_valid = torch.ones_like(window.valid[:, :1])
        return tokens, torch.cat([user_valid, window.valid], dim=1)

    def _concatenate_demographics(
        self,
        users: UserInputs,
        window: HistoryWindow,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratings = self._embed_ratings(window)
        demographics = self._embed_user(users).expand(-1, ratings.shape[1], -1)
        joined = torch.cat([ratings, demographics], dim=2)
        tokens: torch.Tensor = self.demographics_projection(joined)
        valid = window.valid.clone()
        valid[:, -1] = True
        return tokens, valid

    def _build_tokens(
        self,
        users: UserInputs,
        window: HistoryWindow,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.options.concatenates_demographics:
            tokens, valid = self._concatenate_demographics(users, window)
        else:
            tokens, valid = self._prepend_user_token(users, window)
        if not self.options.uses_positions:
            return tokens, valid
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return tokens + self.position_embedding(positions).unsqueeze(0), valid

    def forward(self, users: UserInputs, window: HistoryWindow) -> torch.Tensor:
        tokens, valid = self._build_tokens(users, window)
        encoded: torch.Tensor = self.encoder(tokens, src_key_padding_mask=~valid)
        latest = encoded[:, -1]
        if not self.options.concatenates_demographics:
            latest = torch.where(window.valid[:, -1:], latest, encoded[:, 0])
        vectors: torch.Tensor = self.output_projection(latest)
        return torch.nn.functional.normalize(vectors, p=2.0, dim=1)


class FlatMovieTower(torch.nn.Module):
    def __init__(self, sequence_tower: SequenceUserTower) -> None:
        super().__init__()
        self.sequence_tower = sequence_tower

    def forward(self, movies: MovieInputs) -> torch.Tensor:
        embedding = self.sequence_tower.movie_embedding
        vectors: torch.Tensor = embedding(movies.movie_codes)
        return torch.nn.functional.normalize(vectors, p=2.0, dim=1)


def _bucket_ages(window: HistoryWindow) -> torch.Tensor:
    """Return, per rating, 1 + floor(log2(1 + days before the window's latest
    rating)), capped; 0 for padding.
    """
    latest = window.timestamps[:, -1:]
    age_days = (latest - window.timestamps).clamp(min=0).float() / SECONDS_PER_DAY
    buckets = torch.floor(torch.log2(1.0 + age_days)).long() + 1
    buckets = buckets.clamp(max=settings.RATING_AGE_BUCKET_COUNT)
    return buckets.masked_fill(~window.valid, 0)


def _build_encoder() -> torch.nn.TransformerEncoder:
    layer = torch.nn.TransformerEncoderLayer(
        d_model=settings.SEQUENCE_DIMENSION,
        nhead=settings.SEQUENCE_HEAD_COUNT,
        dim_feedforward=settings.SEQUENCE_FEEDFORWARD_DIMENSION,
        dropout=settings.SEQUENCE_DROPOUT_RATIO,
        batch_first=True,
        norm_first=True,
    )
    return torch.nn.TransformerEncoder(
        layer,
        num_layers=settings.SEQUENCE_LAYER_COUNT,
        enable_nested_tensor=False,
    )
