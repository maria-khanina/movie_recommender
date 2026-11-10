import numpy
import pandas
import torch

from evaluation import evaluation_settings as settings
from evaluation.rating_split import sort_ratings_by_time
from models.rating_sequences import (
    HistoryWindow,
    RatingMarks,
    SequencedRatings,
    build_rated_sequences,
    cut_windows,
)
from models.two_tower_features import TowerFeatures, build_tower_features
from tests import test_settings
from tests.synthetic_ratings import build_synthetic_tables, generate_ratings


def _build_features(ratings: pandas.DataFrame) -> TowerFeatures:
    tables = build_synthetic_tables(ratings)
    return build_tower_features(ratings, tables, torch.device("cpu"))


def _expected_window(user_ratings: pandas.DataFrame, end: int) -> list[int]:
    ordered = sort_ratings_by_time(user_ratings)
    movies: list[int] = ordered[settings.MOVIE_COLUMN].tolist()
    start = max(0, end - test_settings.WINDOW_CHECK_LENGTH)
    return [movie + 1 for movie in movies[start:end]]


def _window_movies(codes: torch.Tensor, valid: torch.Tensor) -> list[int]:
    window_movies: list[int] = []
    for code in codes[valid]:
        window_movies.append(int(code))
    return window_movies


def _check_window(
    ratings: pandas.DataFrame,
    user_id: int,
    end: int,
    window: HistoryWindow,
) -> None:
    user_ratings = ratings[ratings[settings.USER_COLUMN] == user_id]
    expected = _expected_window(user_ratings, end)
    assert _window_movies(window.movie_codes[0], window.valid[0]) == expected
    assert bool(window.valid[0, -1]) == (len(expected) > 0)


def test_windows_hold_the_latest_ratings_before_the_end() -> None:
    ratings = generate_ratings()
    features = _build_features(ratings)
    liked = pandas.Series(numpy.ones(len(ratings), dtype=bool))
    sequenced = build_rated_sequences(
        ratings,
        RatingMarks(liked, ~liked, ratings[settings.RATING_COLUMN]),
        features,
        torch.device("cpu"),
    )
    for index in range(0, len(sequenced.user_rows), test_settings.WINDOW_CHECK_STEP):
        user_row = torch.as_tensor(sequenced.user_rows[index:index + 1])
        end = torch.as_tensor(sequenced.positions[index:index + 1])
        window = cut_windows(
            sequenced.sequences,
            user_row,
            end,
            test_settings.WINDOW_CHECK_LENGTH,
        )
        user_id = int(features.user_index[int(user_row[0])])
        _check_window(ratings, user_id, int(end[0]), window)


def test_an_empty_history_gives_an_empty_window() -> None:
    ratings = generate_ratings()
    features = _build_features(ratings)
    empty = ratings.iloc[:0]
    liked = pandas.Series(numpy.ones(0, dtype=bool))
    sequenced = build_rated_sequences(
        empty,
        RatingMarks(liked, ~liked, empty[settings.RATING_COLUMN]),
        features,
        torch.device("cpu"),
    )
    user_rows = torch.zeros(1, dtype=torch.int64)
    ends = sequenced.sequences.lengths[user_rows]
    window = cut_windows(
        sequenced.sequences,
        user_rows,
        ends,
        test_settings.WINDOW_CHECK_LENGTH,
    )
    assert not bool(window.valid.any())


def _locate_second_starts(
    sequenced: SequencedRatings,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = sequenced.sequences
    starts = sequences.starts[torch.as_tensor(sequenced.user_rows)]
    flat_rows = starts + torch.as_tensor(sequenced.positions)
    first_rows = starts + sequences.second_starts[flat_rows]
    return flat_rows, first_rows


def test_second_starts_mark_the_first_rating_of_each_second() -> None:
    ratings = generate_ratings()
    features = _build_features(ratings)
    liked = pandas.Series(numpy.ones(len(ratings), dtype=bool))
    sequenced = build_rated_sequences(
        ratings,
        RatingMarks(liked, ~liked, ratings[settings.RATING_COLUMN]),
        features,
        torch.device("cpu"),
    )
    timestamps = sequenced.sequences.timestamps
    flat_rows, first_rows = _locate_second_starts(sequenced)
    assert bool((timestamps[first_rows] == timestamps[flat_rows]).all())
    has_earlier = sequenced.sequences.second_starts[flat_rows] > 0
    earlier_times = timestamps[first_rows[has_earlier] - 1]
    assert bool((earlier_times < timestamps[first_rows[has_earlier]]).all())
