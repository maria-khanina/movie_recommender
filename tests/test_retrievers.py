import numpy
import pandas

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as settings
from models.candidate_retrieval import (
    retrieve_unseen_candidates,
    select_positive_ratings,
)
from models.item_neighbors_retriever import ItemNeighborsRetriever
from models.popularity_retriever import PopularityRetriever
from tests import test_settings
from tests.synthetic_ratings import generate_ratings


def _build_tables(ratings: pandas.DataFrame) -> MovieLensTables:
    movie_index = pandas.Index(
        range(test_settings.SYNTHETIC_MOVIE_COUNT),
        name=settings.MOVIE_COLUMN,
    )
    return MovieLensTables(
        ratings=ratings,
        users=pandas.DataFrame(),
        movies=pandas.DataFrame(index=movie_index),
    )


def _collect_known_movies(ratings: pandas.DataFrame) -> dict[int, set[int]]:
    known: dict[int, set[int]] = {}
    for row in ratings.itertuples(index=False):
        user_id = int(str(getattr(row, settings.USER_COLUMN)))
        movie_id = int(str(getattr(row, settings.MOVIE_COLUMN)))
        known.setdefault(user_id, set()).add(movie_id)
    return known


def _retrieve_with_popularity(
    ratings: pandas.DataFrame,
    candidate_count: int,
) -> pandas.DataFrame:
    tables = _build_tables(ratings)
    retriever = PopularityRetriever(test_settings.POSITIVE_THRESHOLD)
    retriever.fit(ratings, tables)
    user_ids = ratings[settings.USER_COLUMN].unique().astype(numpy.int64)
    return retrieve_unseen_candidates(
        retriever,
        user_ids,
        ratings,
        candidate_count,
        test_settings.SYNTHETIC_MOVIE_COUNT,
    )


def _check_user_candidates(
    user_rows: pandas.DataFrame,
    user_known: set[int],
    candidate_count: int,
) -> None:
    unseen_count = test_settings.SYNTHETIC_MOVIE_COUNT - len(user_known)
    listed: list[int] = user_rows[settings.MOVIE_COLUMN].tolist()
    assert not user_known.intersection(listed)
    assert len(listed) == min(candidate_count, unseen_count)
    assert user_rows[settings.POSITION_COLUMN].tolist() == list(range(len(listed)))


def test_retrieval_drops_known_movies_and_fills_the_list() -> None:
    ratings = generate_ratings()
    candidate_count = test_settings.SYNTHETIC_LIST_SIZE
    candidates = _retrieve_with_popularity(ratings, candidate_count)
    known = _collect_known_movies(ratings)
    for user_id, user_rows in candidates.groupby(settings.USER_COLUMN):
        user_known = known[int(str(user_id))]
        _check_user_candidates(user_rows, user_known, candidate_count)


def _score_by_brute_force(
    positives: pandas.DataFrame,
    user_id: int,
) -> numpy.typing.NDArray[numpy.float64]:
    movie_count = test_settings.SYNTHETIC_MOVIE_COUNT
    matrix = numpy.zeros((positives[settings.USER_COLUMN].max() + 1, movie_count))
    matrix[positives[settings.USER_COLUMN], positives[settings.MOVIE_COLUMN]] = 1.0
    co_counts = matrix.T @ matrix
    fans = numpy.sqrt(numpy.maximum(numpy.diag(co_counts), 1.0))
    similarities = co_counts / numpy.outer(fans, fans)
    numpy.fill_diagonal(similarities, 0.0)
    return matrix[user_id] @ similarities


def _build_item_neighbors(ratings: pandas.DataFrame) -> ItemNeighborsRetriever:
    retriever = ItemNeighborsRetriever(
        test_settings.SYNTHETIC_MOVIE_COUNT,
        test_settings.COSINE_EXPONENT,
        test_settings.POSITIVE_THRESHOLD,
    )
    retriever.fit(ratings, _build_tables(ratings))
    return retriever


def test_item_neighbors_scores_match_brute_force_cosine() -> None:
    ratings = generate_ratings()
    movie_count = test_settings.SYNTHETIC_MOVIE_COUNT
    retriever = _build_item_neighbors(ratings)
    positives = select_positive_ratings(ratings, test_settings.POSITIVE_THRESHOLD)
    user_id = int(positives[settings.USER_COLUMN].iloc[0])
    expected = numpy.sort(_score_by_brute_force(positives, user_id))[::-1]
    query = numpy.array([user_id], dtype=numpy.int64)
    result = retriever.search(query, ratings, movie_count)
    differences = numpy.abs(result.scores[0].astype(numpy.float64) - expected)
    assert float(differences.max()) < test_settings.SCORE_TOLERANCE
