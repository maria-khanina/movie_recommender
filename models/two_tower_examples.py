"""Training examples of the two-tower retriever and the per-epoch choice of the
disliked movie that goes with each example.

An example is a (user, liked movie) pair. A disliked movie is a training rating
in the lowest part of the user's own scale (percentile gain below the bad
threshold). Every epoch, each example gets one movie as an explicit negative:
the user's disliked movies are shuffled and dealt out to the user's examples in
turn, and once every disliked movie has been dealt the maximum number of times,
the remaining examples get a uniformly random catalog movie instead. Everything
is a sort and a few array lookups, so it scales with the number of ratings.
"""
from typing import NamedTuple

import numpy
import pandas

from evaluation import evaluation_settings as columns
from evaluation.percentile_gains import attach_percentile_gains
from models import two_tower_settings as settings
from models.candidate_retrieval import MovieIds
from models.two_tower_features import TowerFeatures, find_rows


class TrainingExamples(NamedTuple):
    user_rows: MovieIds
    movie_rows: MovieIds
    disliked_user_rows: MovieIds
    disliked_movie_rows: MovieIds


def _select_disliked_ratings(training: pandas.DataFrame) -> pandas.DataFrame:
    gained = attach_percentile_gains(training, training)
    is_disliked = gained[columns.PERCENTILE_COLUMN] < columns.BAD_PERCENTILE_THRESHOLD
    return gained[is_disliked]


def build_training_examples(
    training: pandas.DataFrame,
    positives: pandas.DataFrame,
    features: TowerFeatures,
) -> TrainingExamples:
    disliked = _select_disliked_ratings(training)
    return TrainingExamples(
        user_rows=find_rows(features.user_index, positives[columns.USER_COLUMN]),
        movie_rows=find_rows(features.movie_index, positives[columns.MOVIE_COLUMN]),
        disliked_user_rows=find_rows(
            features.user_index,
            disliked[columns.USER_COLUMN],
        ),
        disliked_movie_rows=find_rows(
            features.movie_index,
            disliked[columns.MOVIE_COLUMN],
        ),
    )


def _shuffle_within_users(
    user_rows: MovieIds,
    generator: numpy.random.Generator,
) -> MovieIds:
    random_keys = generator.random(len(user_rows))
    return numpy.lexsort((random_keys, user_rows)).astype(numpy.int64)


def _rank_within_users(user_rows: MovieIds, order: MovieIds) -> MovieIds:
    sorted_users = user_rows[order]
    group_starts = numpy.searchsorted(sorted_users, sorted_users, side="left")
    ranks = numpy.empty(len(user_rows), dtype=numpy.int64)
    ranks[order] = numpy.arange(len(user_rows)) - group_starts
    return ranks


def assign_hard_negatives(
    examples: TrainingExamples,
    generator: numpy.random.Generator,
    movie_count: int,
) -> MovieIds:
    """Return one negative movie row for every training example."""
    random_movies = generator.integers(0, movie_count, size=len(examples.user_rows))
    if len(examples.disliked_movie_rows) == 0:
        return random_movies.astype(numpy.int64)
    disliked_order = _shuffle_within_users(examples.disliked_user_rows, generator)
    disliked_users = examples.disliked_user_rows[disliked_order]
    disliked_movies = examples.disliked_movie_rows[disliked_order]
    starts = numpy.searchsorted(disliked_users, examples.user_rows, side="left")
    ends = numpy.searchsorted(disliked_users, examples.user_rows, side="right")
    ranks = _rank_within_users(
        examples.user_rows,
        _shuffle_within_users(examples.user_rows, generator),
    )
    return _deal_disliked_movies(
        starts,
        ends - starts,
        ranks,
        disliked_movies,
        random_movies,
    )


def _deal_disliked_movies(
    starts: MovieIds,
    counts: MovieIds,
    ranks: MovieIds,
    disliked_movies: MovieIds,
    random_movies: MovieIds,
) -> MovieIds:
    uses_disliked = ranks < counts * settings.HARD_NEGATIVE_MAXIMUM_USES_PER_EPOCH
    offsets = ranks % numpy.maximum(counts, 1)
    positions = numpy.minimum(starts + offsets, len(disliked_movies) - 1)
    return numpy.where(uses_disliked, disliked_movies[positions], random_movies)
