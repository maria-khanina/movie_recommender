"""Item-to-item retriever: movies liked by the same people as the movies the user
liked.

For two movies i and j, c(i, j) is the number of training users who liked both,
and n(i) is the number who liked i. The similarity is
c(i, j) / (n(i) ** a * n(j) ** (1 - a)): a = 0.5 gives the cosine similarity,
a = 1 gives the share of i's fans who also liked j, which favors popular j. Only
the most similar neighbors of every movie are kept. A user's score for a movie is
the sum of its similarities to the movies the user liked in the history known at
query time (so a new user is served from the first ratings), and a query costs the
history length times the neighbor count, whatever the catalog size. A tiny
popularity term breaks ties, which also gives users without liked movies the
popularity ranking.

The neighbor table is built here through a dense catalog-by-catalog matrix, which
fits catalogs of tens of thousands of movies; a larger catalog needs the same
table computed in a distributed job or from approximate nearest neighbors.
"""
import numpy
import numpy.typing
import pandas
import scipy.sparse

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from models import models_settings as settings
from models.candidate_retrieval import (
    MovieIds,
    SearchResult,
    select_positive_ratings,
)

_Matrix = numpy.typing.NDArray[numpy.float64]


def _build_interactions(
    positives: pandas.DataFrame,
    user_index: pandas.Index,
    movie_index: pandas.Index,
) -> scipy.sparse.csr_array:
    user_codes = user_index.get_indexer(pandas.Index(positives[columns.USER_COLUMN]))
    movie_codes = movie_index.get_indexer(
        pandas.Index(positives[columns.MOVIE_COLUMN]),
    )
    values = numpy.ones(len(positives), dtype=numpy.float64)
    shape = (len(user_index), len(movie_index))
    return scipy.sparse.csr_array((values, (user_codes, movie_codes)), shape=shape)


def _compute_similarities(
    interactions: scipy.sparse.csr_array,
    popularity_exponent: float,
) -> _Matrix:
    co_counts = (interactions.T @ interactions).toarray()
    fan_counts = numpy.maximum(numpy.diag(co_counts), 1.0)
    row_norms = numpy.power(fan_counts, popularity_exponent)[:, numpy.newaxis]
    column_norms = numpy.power(fan_counts, 1.0 - popularity_exponent)[numpy.newaxis, :]
    similarities = co_counts / (row_norms * column_norms)
    numpy.fill_diagonal(similarities, 0.0)
    return similarities


def _keep_top_neighbors(
    similarities: _Matrix,
    requested_count: int,
) -> scipy.sparse.csr_array:
    neighbor_count = min(requested_count, similarities.shape[1] - 1)
    neighbors = numpy.argpartition(-similarities, neighbor_count, axis=1)
    kept_columns = neighbors[:, :neighbor_count]
    kept_rows = numpy.arange(similarities.shape[0], dtype=numpy.int64)[
        :, numpy.newaxis
    ]
    kept_values = similarities[kept_rows, kept_columns]
    row_codes = numpy.repeat(kept_rows, neighbor_count, axis=1)
    return scipy.sparse.csr_array(
        (kept_values.reshape(-1), (row_codes.reshape(-1), kept_columns.reshape(-1))),
        shape=similarities.shape,
    )


def _select_top(scores: _Matrix, depth: int) -> numpy.typing.NDArray[numpy.intp]:
    top_unsorted = numpy.argpartition(-scores, depth - 1, axis=1)[:, :depth]
    top_scores = numpy.take_along_axis(scores, top_unsorted, axis=1)
    order = numpy.argsort(-top_scores, axis=1, kind="stable")
    return numpy.take_along_axis(top_unsorted, order, axis=1)


class ItemNeighborsRetriever:
    def __init__(
        self,
        neighbor_count: int,
        popularity_exponent: float,
        positive_threshold: float,
    ) -> None:
        self._positive_threshold = positive_threshold
        self._neighbor_count = neighbor_count
        self._popularity_exponent = popularity_exponent
        self._movie_index = pandas.Index([], dtype=numpy.int64)
        self._neighbors = scipy.sparse.csr_array((0, 0))
        self._popularity: _Matrix = numpy.empty((1, 0))

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        positives = select_positive_ratings(training, self._positive_threshold)
        user_index = pandas.Index(positives[columns.USER_COLUMN].unique())
        self._movie_index = pandas.Index(tables.movies.index)
        interactions = _build_interactions(positives, user_index, self._movie_index)
        similarities = _compute_similarities(
            interactions,
            self._popularity_exponent,
        )
        self._neighbors = _keep_top_neighbors(similarities, self._neighbor_count)
        fan_counts = numpy.asarray(interactions.sum(axis=0), dtype=numpy.float64)
        self._popularity = fan_counts[numpy.newaxis, :] / max(fan_counts.max(), 1.0)

    def _score_users(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
    ) -> _Matrix:
        is_queried = history[columns.USER_COLUMN].isin(user_ids)
        liked = select_positive_ratings(history[is_queried], self._positive_threshold)
        user_index = pandas.Index(user_ids)
        histories = _build_interactions(liked, user_index, self._movie_index)
        scores = numpy.asarray((histories @ self._neighbors).toarray())
        return scores + settings.POPULARITY_TIE_BREAK_WEIGHT * self._popularity

    def search(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
        depth: int,
    ) -> SearchResult:
        scores = self._score_users(user_ids, history)
        top_codes = _select_top(scores, depth)
        movie_ids = self._movie_index.to_numpy(dtype=numpy.int64)[top_codes]
        top_scores = numpy.take_along_axis(scores, top_codes, axis=1)
        return SearchResult(
            movie_ids=movie_ids,
            scores=top_scores.astype(numpy.float32),
        )
