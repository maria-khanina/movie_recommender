"""Common interface of candidate retrievers and the retrieval of unseen candidates
that every retriever shares.

A retriever is fitted on training ratings and, for a batch of users, returns the
movies with the highest scores, already sorted. Removing the movies a user has
already rated is done here, the same way for every retriever: the retriever is
asked for more movies than needed, the known ones are dropped and the list is cut
to the requested size. Users are grouped by how deep the search must go (the
requested size plus the number of known movies, rounded up to a power of two),
so a heavy rater does not make the search deeper for everyone else.
"""
from typing import NamedTuple, Protocol

import numpy
import numpy.typing
import pandas

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from evaluation.percentile_gains import attach_percentile_gains
from models import models_settings as settings

MovieIds = numpy.typing.NDArray[numpy.int64]
Scores = numpy.typing.NDArray[numpy.float32]

_SCORE_COLUMN = "score"
_MERGE_INDICATOR_COLUMN = "_merge"


class SearchResult(NamedTuple):
    movie_ids: MovieIds
    scores: Scores


class CandidateRetriever(Protocol):
    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None: ...

    def search(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
        depth: int,
    ) -> SearchResult: ...


def _choose_search_depths(
    user_ids: MovieIds,
    known: pandas.DataFrame,
    candidate_count: int,
    catalog_size: int,
) -> MovieIds:
    known_counts = known.groupby(columns.USER_COLUMN).size()
    user_known_counts = known_counts.reindex(user_ids, fill_value=0).to_numpy()
    needed = candidate_count + user_known_counts
    rounded = numpy.power(2.0, numpy.ceil(numpy.log2(needed)))
    return numpy.minimum(rounded, catalog_size).astype(numpy.int64)


def _flatten_search_result(
    user_ids: MovieIds,
    result: SearchResult,
) -> pandas.DataFrame:
    depth = result.movie_ids.shape[1]
    return pandas.DataFrame(
        {
            columns.USER_COLUMN: numpy.repeat(user_ids, depth),
            columns.MOVIE_COLUMN: result.movie_ids.reshape(-1),
            _SCORE_COLUMN: result.scores.reshape(-1),
        }
    )


def _drop_known_movies(
    found: pandas.DataFrame,
    known: pandas.DataFrame,
) -> pandas.DataFrame:
    key_columns: list[str] = [columns.USER_COLUMN, columns.MOVIE_COLUMN]
    marked = found.merge(
        known[key_columns].drop_duplicates(),
        on=key_columns,
        how="left",
        indicator=True,
        sort=False,
    )
    is_unknown = marked[_MERGE_INDICATOR_COLUMN] == "left_only"
    return marked.loc[is_unknown, [*key_columns, _SCORE_COLUMN]]


def _rank_and_cut(found: pandas.DataFrame, candidate_count: int) -> pandas.DataFrame:
    positions = found.groupby(columns.USER_COLUMN, sort=False).cumcount()
    ranked = found.assign(position=positions.to_numpy())
    ranked = ranked.rename(columns={"position": columns.POSITION_COLUMN})
    return ranked[ranked[columns.POSITION_COLUMN] < candidate_count]


def _retrieve_batch(
    retriever: CandidateRetriever,
    user_ids: MovieIds,
    depth: int,
    known: pandas.DataFrame,
) -> pandas.DataFrame:
    result = retriever.search(user_ids, known, depth)
    found = _flatten_search_result(user_ids, result)
    return _drop_known_movies(found, known)


def retrieve_unseen_candidates(
    retriever: CandidateRetriever,
    user_ids: MovieIds,
    known: pandas.DataFrame,
    candidate_count: int,
    catalog_size: int,
) -> pandas.DataFrame:
    """Return (user, movie, score, position) rows: the top candidate_count movies
    of every user among the movies the user has not rated.
    """
    depths = _choose_search_depths(user_ids, known, candidate_count, catalog_size)
    batches: list[pandas.DataFrame] = []
    for depth in numpy.unique(depths).tolist():
        depth_users = user_ids[depths == depth]
        for start in range(0, len(depth_users), settings.SEARCH_BATCH_SIZE):
            batch_users = depth_users[start:start + settings.SEARCH_BATCH_SIZE]
            batches.append(_retrieve_batch(retriever, batch_users, depth, known))
    return _rank_and_cut(pandas.concat(batches, ignore_index=True), candidate_count)


def select_positive_ratings(
    ratings: pandas.DataFrame,
    percentile_threshold: float,
) -> pandas.DataFrame:
    """Return the ratings that count as liked: those whose percentile on the
    user's own scale, among the same user's ratings in this table, reaches the
    threshold. A threshold of 0 keeps every rating.
    """
    gained = attach_percentile_gains(ratings, ratings)
    is_positive = gained[columns.PERCENTILE_COLUMN] >= percentile_threshold
    return gained[is_positive].reset_index(drop=True)
