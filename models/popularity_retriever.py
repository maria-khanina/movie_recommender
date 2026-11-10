"""Popularity retriever: the same ranking of movies for every user.

Movies are ranked by how many training users liked them, a liked rating being
one whose percentile on the user's own scale reaches the threshold (0 counts
every rating). Every user receives the top of this one ranking, and the shared
retrieval step removes the movies the user has already rated, so lists differ
only by what each user has seen. It is the baseline every personalized
retriever must beat, and its query cost does not depend on the catalog size.
"""
import numpy
import pandas

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from models.candidate_retrieval import (
    MovieIds,
    Scores,
    SearchResult,
    select_positive_ratings,
)


class PopularityRetriever:
    def __init__(self, positive_threshold: float) -> None:
        self._positive_threshold = positive_threshold
        self._ranked_movie_ids: MovieIds = numpy.empty(0, dtype=numpy.int64)
        self._ranked_scores: Scores = numpy.empty(0, dtype=numpy.float32)

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        counted = select_positive_ratings(training, self._positive_threshold)
        like_counts = counted.groupby(columns.MOVIE_COLUMN).size()
        catalog_counts = like_counts.reindex(tables.movies.index, fill_value=0)
        ranking = numpy.argsort(-catalog_counts.to_numpy(), kind="stable")
        catalog_ids = tables.movies.index.to_numpy(dtype=numpy.int64)
        self._ranked_movie_ids = catalog_ids[ranking]
        catalog_scores = numpy.asarray(catalog_counts, dtype=numpy.float32)
        self._ranked_scores = catalog_scores[ranking]

    def search(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
        depth: int,
    ) -> SearchResult:
        user_count = len(user_ids)
        return SearchResult(
            movie_ids=numpy.tile(self._ranked_movie_ids[:depth], (user_count, 1)),
            scores=numpy.tile(self._ranked_scores[:depth], (user_count, 1)),
        )
