"""Candidate lists as fixed-width arrays, their labels, and reranked lists.

Rankers work on one row per user and one column per candidate position. A list
shorter than the width is padded, and valid marks the real candidates. Labels
come from a block of held-out ratings: a candidate the user rated there gets its
percentile gain and the rated flag, any other candidate gets gain 0.
"""
from typing import NamedTuple

import numpy
import numpy.typing
import pandas

from evaluation import evaluation_settings as columns
from models.candidate_retrieval import MovieIds

_SCORE_COLUMN = "score"
_KEY_COLUMNS: list[str] = [columns.USER_COLUMN, columns.MOVIE_COLUMN]

ListValues = numpy.typing.NDArray[numpy.float32]
ListFlags = numpy.typing.NDArray[numpy.bool_]


class CandidateLists(NamedTuple):
    user_ids: MovieIds
    movie_ids: numpy.typing.NDArray[numpy.int64]
    scores: ListValues
    valid: ListFlags


class ListLabels(NamedTuple):
    gains: ListValues
    rated: ListFlags


def _locate_cells(
    candidates: pandas.DataFrame,
    user_index: pandas.Index,
) -> tuple[numpy.typing.NDArray[numpy.intp], numpy.typing.NDArray[numpy.intp]]:
    rows = user_index.get_indexer(pandas.Index(candidates[columns.USER_COLUMN]))
    positions = candidates[columns.POSITION_COLUMN].to_numpy(dtype=numpy.intp)
    return rows, positions


def build_candidate_lists(
    candidates: pandas.DataFrame,
    list_size: int,
) -> CandidateLists:
    kept = candidates[candidates[columns.POSITION_COLUMN] < list_size]
    user_index = pandas.Index(numpy.sort(kept[columns.USER_COLUMN].unique()))
    rows, positions = _locate_cells(kept, user_index)
    shape = (len(user_index), list_size)
    movie_ids = numpy.full(shape, -1, dtype=numpy.int64)
    movie_ids[rows, positions] = kept[columns.MOVIE_COLUMN].to_numpy()
    scores = numpy.zeros(shape, dtype=numpy.float32)
    scores[rows, positions] = kept[_SCORE_COLUMN].to_numpy(dtype=numpy.float32)
    return CandidateLists(
        user_ids=user_index.to_numpy(dtype=numpy.int64),
        movie_ids=movie_ids,
        scores=scores,
        valid=movie_ids >= 0,
    )


def flatten_lists(lists: CandidateLists) -> pandas.DataFrame:
    list_size = lists.movie_ids.shape[1]
    positions = numpy.arange(list_size, dtype=numpy.int64)
    return pandas.DataFrame(
        {
            columns.USER_COLUMN: numpy.repeat(lists.user_ids, list_size),
            columns.MOVIE_COLUMN: lists.movie_ids.reshape(-1),
            columns.POSITION_COLUMN: numpy.tile(positions, len(lists.user_ids)),
            _SCORE_COLUMN: lists.scores.reshape(-1),
        }
    )


def label_candidate_lists(
    lists: CandidateLists,
    labeled: pandas.DataFrame,
) -> ListLabels:
    """Return the gain and rated flag of every candidate from labeled ratings."""
    cells = flatten_lists(lists)
    label_columns: list[str] = [*_KEY_COLUMNS, columns.GAIN_COLUMN]
    joined = cells.merge(labeled[label_columns], on=_KEY_COLUMNS, how="left")
    rated = joined[columns.GAIN_COLUMN].notna().to_numpy()
    gains = joined[columns.GAIN_COLUMN].fillna(0.0).to_numpy(dtype=numpy.float32)
    shape = lists.movie_ids.shape
    return ListLabels(gains=gains.reshape(shape), rated=rated.reshape(shape))


def rerank_lists(lists: CandidateLists, new_scores: ListValues) -> pandas.DataFrame:
    """Return (user, movie, position) rows ordered by new_scores in each list."""
    masked_scores = numpy.where(lists.valid, new_scores, -numpy.inf)
    order = numpy.argsort(-masked_scores, axis=1, kind="stable")
    reordered = lists._replace(
        movie_ids=numpy.take_along_axis(lists.movie_ids, order, axis=1),
        scores=numpy.take_along_axis(masked_scores, order, axis=1),
        valid=numpy.take_along_axis(lists.valid, order, axis=1),
    )
    cells = flatten_lists(reordered)
    return cells[reordered.valid.reshape(-1)].reset_index(drop=True)
