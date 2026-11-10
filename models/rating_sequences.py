"""Every user's ratings as one time-ordered sequence, and windows cut from it.

All sequences are stored back to back in flat tensors, with each user's start
and length, so a batch of windows is cut with index arithmetic on the device,
without Python loops. A window holds the last ratings strictly before a given
position, padded on the left: the most recent rating is always in the last
column. Each element keeps the movie code (catalog row plus one; 0 is padding),
the rating code given by the caller (the raw rating or a bucket of its
percentile) and whether the rating counts as liked on the user's own scale,
plus the position of the first rating given in the same second, which lets a
window end at a second boundary.
"""
from typing import NamedTuple

import numpy
import numpy.typing
import pandas
import torch

from evaluation import evaluation_settings as columns
from evaluation.rating_split import sort_ratings_by_time
from models.two_tower_features import TowerFeatures, find_rows


class RatedSequences(NamedTuple):
    movie_codes: torch.Tensor
    rating_codes: torch.Tensor
    liked: torch.Tensor
    timestamps: torch.Tensor
    second_starts: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor


class HistoryWindow(NamedTuple):
    movie_codes: torch.Tensor
    rating_codes: torch.Tensor
    liked: torch.Tensor
    timestamps: torch.Tensor
    valid: torch.Tensor


class SequencedRatings(NamedTuple):
    sequences: RatedSequences
    user_rows: numpy.typing.NDArray[numpy.int64]
    movie_rows: numpy.typing.NDArray[numpy.int64]
    positions: numpy.typing.NDArray[numpy.int64]
    liked: numpy.typing.NDArray[numpy.bool_]
    timestamps: numpy.typing.NDArray[numpy.int64]


def _order_by_user_row(
    ratings: pandas.DataFrame,
    features: TowerFeatures,
) -> tuple[pandas.DataFrame, numpy.typing.NDArray[numpy.int64]]:
    ordered = sort_ratings_by_time(ratings)
    user_rows = find_rows(features.user_index, ordered[columns.USER_COLUMN])
    order = numpy.argsort(user_rows, kind="stable")
    return ordered.iloc[order].reset_index(drop=True), user_rows[order]


def _locate_users(
    user_rows: numpy.typing.NDArray[numpy.int64],
    user_count: int,
) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    all_rows = numpy.arange(user_count, dtype=numpy.int64)
    starts = numpy.searchsorted(user_rows, all_rows, side="left")
    ends = numpy.searchsorted(user_rows, all_rows, side="right")
    return starts.astype(numpy.int64), (ends - starts).astype(numpy.int64)


def _find_second_starts(
    ordered: pandas.DataFrame,
    positions: numpy.typing.NDArray[numpy.int64],
) -> numpy.typing.NDArray[numpy.int64]:
    key_columns: list[str] = [columns.USER_COLUMN, columns.TIMESTAMP_COLUMN]
    first_positions = pandas.Series(positions).groupby(
        [ordered[name].to_numpy() for name in key_columns],
    ).transform("min")
    return first_positions.to_numpy(dtype=numpy.int64)


def build_rated_sequences(
    ratings: pandas.DataFrame,
    liked: pandas.Series,
    rating_codes: pandas.Series,
    features: TowerFeatures,
    device: torch.device,
) -> SequencedRatings:
    """Return the sequences of the given ratings and, per rating, its user row,
    movie row and position in the user's sequence; liked and rating_codes are
    aligned with ratings.
    """
    flagged = ratings.assign(
        liked=liked.to_numpy(),
        rating_code=rating_codes.to_numpy(),
    )
    ordered, user_rows = _order_by_user_row(flagged, features)
    starts, lengths = _locate_users(user_rows, len(features.user_index))
    movie_rows = find_rows(features.movie_index, ordered[columns.MOVIE_COLUMN])
    liked_flags = ordered["liked"].to_numpy(dtype=numpy.bool_)
    timestamps = ordered[columns.TIMESTAMP_COLUMN].to_numpy(dtype=numpy.int64)
    positions = numpy.arange(len(ordered), dtype=numpy.int64) - starts[user_rows]
    sequences = RatedSequences(
        movie_codes=torch.as_tensor(movie_rows + 1, device=device),
        rating_codes=torch.as_tensor(
            ordered["rating_code"].to_numpy(dtype=numpy.int64),
            device=device,
        ),
        liked=torch.as_tensor(liked_flags, device=device),
        timestamps=torch.as_tensor(timestamps, device=device),
        second_starts=torch.as_tensor(
            _find_second_starts(ordered, positions),
            device=device,
        ),
        starts=torch.as_tensor(starts, device=device),
        lengths=torch.as_tensor(lengths, device=device),
    )
    return SequencedRatings(
        sequences,
        user_rows,
        movie_rows,
        positions,
        liked_flags,
        timestamps,
    )


def _build_empty_window(valid: torch.Tensor) -> HistoryWindow:
    zeros = torch.zeros(valid.shape, dtype=torch.int64, device=valid.device)
    return HistoryWindow(
        movie_codes=zeros,
        rating_codes=zeros,
        liked=torch.zeros_like(valid),
        timestamps=zeros,
        valid=torch.zeros_like(valid),
    )


def cut_windows(
    sequences: RatedSequences,
    user_rows: torch.Tensor,
    end_positions: torch.Tensor,
    window_length: int,
) -> HistoryWindow:
    """Return, for every user row, the ratings at positions
    [end - window_length, end) of the user's sequence, padded on the left.
    """
    lengths = end_positions.clamp(min=0, max=window_length)
    columns_grid = torch.arange(window_length, device=user_rows.device)
    padding = (window_length - lengths).unsqueeze(1)
    valid = columns_grid.unsqueeze(0) >= padding
    offsets = end_positions.unsqueeze(1) - window_length + columns_grid.unsqueeze(0)
    if len(sequences.movie_codes) == 0:
        return _build_empty_window(valid)
    flat = (sequences.starts[user_rows].unsqueeze(1) + offsets).clamp(min=0)
    flat = flat.clamp(max=len(sequences.movie_codes) - 1)
    return HistoryWindow(
        movie_codes=sequences.movie_codes[flat].masked_fill(~valid, 0),
        rating_codes=sequences.rating_codes[flat].masked_fill(~valid, 0),
        liked=sequences.liked[flat] & valid,
        timestamps=sequences.timestamps[flat].masked_fill(~valid, 0),
        valid=valid,
    )
