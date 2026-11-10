"""Feature tensors of all users and catalog movies for the two-tower retriever.

Users and movies are addressed by their row in these tensors. A user's id code is
nonzero only if the user has enough liked training movies; rarer users get code 0
and are represented by their age group, gender and occupation alone, the same way
the model will see a new user. Movie codes are the catalog row plus one.
"""
from typing import NamedTuple

import numpy
import numpy.typing
import pandas
import torch

from dataset.dataset_settings import RELEASE_PERIOD_COLUMN
from dataset.movielens_tables import MovieLensTables, list_genre_columns
from evaluation import evaluation_settings as columns
from models import two_tower_settings as settings
from models.candidate_retrieval import MovieIds
from models.two_tower_network import (
    MovieFeatureSizes,
    MovieInputs,
    UserFeatureSizes,
    UserInputs,
)


class TowerFeatures(NamedTuple):
    user_index: pandas.Index
    movie_index: pandas.Index
    users: UserInputs
    movies: MovieInputs
    user_sizes: UserFeatureSizes
    movie_sizes: MovieFeatureSizes


def _code_frequent_users(
    positives: pandas.DataFrame,
    user_index: pandas.Index,
) -> numpy.typing.NDArray[numpy.int64]:
    positive_counts = positives.groupby(columns.USER_COLUMN).size()
    counts = positive_counts.reindex(user_index, fill_value=0).to_numpy()
    is_frequent = counts >= settings.MINIMUM_USER_POSITIVES
    codes = numpy.zeros(len(user_index), dtype=numpy.int64)
    codes[is_frequent] = numpy.arange(1, int(is_frequent.sum()) + 1)
    return codes


def _to_long_tensor(values: pandas.Series, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values.to_numpy(dtype=numpy.int64, copy=True), device=device)


def _build_user_inputs(
    users: pandas.DataFrame,
    user_codes: numpy.typing.NDArray[numpy.int64],
    device: torch.device,
) -> UserInputs:
    return UserInputs(
        user_codes=torch.as_tensor(user_codes, device=device),
        ages=_to_long_tensor(users["age"], device),
        genders=_to_long_tensor(users["gender"], device),
        occupations=_to_long_tensor(users["occupation"], device),
    )


def _build_movie_inputs(movies: pandas.DataFrame, device: torch.device) -> MovieInputs:
    genres = movies[list_genre_columns(movies)].to_numpy(dtype=numpy.float32)
    return MovieInputs(
        movie_codes=torch.arange(1, len(movies) + 1, device=device),
        genres=torch.as_tensor(genres, device=device),
        release_periods=_to_long_tensor(movies[RELEASE_PERIOD_COLUMN], device),
    )


def _measure_movie_sizes(movies: pandas.DataFrame) -> MovieFeatureSizes:
    return MovieFeatureSizes(
        movie_count=len(movies),
        genre_count=len(list_genre_columns(movies)),
        release_period_count=int(movies[RELEASE_PERIOD_COLUMN].max()),
    )


def build_tower_features(
    positives: pandas.DataFrame,
    tables: MovieLensTables,
    device: torch.device,
) -> TowerFeatures:
    """Return the feature tensors; a user gets an id code when the liked training
    ratings in positives are frequent enough.
    """
    user_index = pandas.Index(tables.users.index)
    user_codes = _code_frequent_users(positives, user_index)
    sizes = UserFeatureSizes(
        user_count=int(user_codes.max()),
        age_count=int(tables.users["age"].max()),
        gender_count=int(tables.users["gender"].max()),
        occupation_count=int(tables.users["occupation"].max()),
    )
    return TowerFeatures(
        user_index=user_index,
        movie_index=pandas.Index(tables.movies.index),
        users=_build_user_inputs(tables.users, user_codes, device),
        movies=_build_movie_inputs(tables.movies, device),
        user_sizes=sizes,
        movie_sizes=_measure_movie_sizes(tables.movies),
    )


def select_user_inputs(features: TowerFeatures, rows: torch.Tensor) -> UserInputs:
    return UserInputs(
        user_codes=features.users.user_codes[rows],
        ages=features.users.ages[rows],
        genders=features.users.genders[rows],
        occupations=features.users.occupations[rows],
    )


def select_movie_inputs(features: TowerFeatures, rows: torch.Tensor) -> MovieInputs:
    return MovieInputs(
        movie_codes=features.movies.movie_codes[rows],
        genres=features.movies.genres[rows],
        release_periods=features.movies.release_periods[rows],
    )


def find_rows(index: pandas.Index, ids: pandas.Series | MovieIds) -> MovieIds:
    return index.get_indexer(pandas.Index(ids)).astype(numpy.int64)
