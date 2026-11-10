"""Random ratings and recommendation lists for testing the evaluation code.

Timestamps are drawn from a small range so that many ratings of one user share a
timestamp, as they do in MovieLens, and the tie-breaking rule is exercised.
"""
import numpy
import numpy.typing
import pandas

from dataset.dataset_settings import RELEASE_PERIOD_COLUMN, USER_FEATURE_COLUMNS
from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as settings
from tests import test_settings


def _generate_user_ratings(
    generator: numpy.random.Generator,
    user_id: int,
) -> pandas.DataFrame:
    rating_count = int(
        generator.integers(
            test_settings.SYNTHETIC_MINIMUM_RATINGS_PER_USER,
            test_settings.SYNTHETIC_MAXIMUM_RATINGS_PER_USER,
        )
    )
    movie_ids = generator.choice(
        test_settings.SYNTHETIC_MOVIE_COUNT,
        size=rating_count,
        replace=False,
    )
    return pandas.DataFrame(
        {
            settings.USER_COLUMN: user_id,
            settings.MOVIE_COLUMN: movie_ids,
            settings.RATING_COLUMN: generator.choice(
                test_settings.SYNTHETIC_RATING_VALUES,
                size=rating_count,
            ),
            settings.TIMESTAMP_COLUMN: generator.integers(
                0,
                test_settings.SYNTHETIC_TIMESTAMP_RANGE,
                size=rating_count,
            ),
        }
    )


def generate_ratings() -> pandas.DataFrame:
    generator = numpy.random.default_rng(test_settings.SYNTHETIC_SEED)
    user_ratings: list[pandas.DataFrame] = []
    for user_id in range(test_settings.SYNTHETIC_USER_COUNT):
        user_ratings.append(_generate_user_ratings(generator, user_id))
    return pandas.concat(user_ratings, ignore_index=True)


def generate_lists(list_size: int) -> pandas.DataFrame:
    """Return a random ranked list of movies for every synthetic user."""
    generator = numpy.random.default_rng(test_settings.SYNTHETIC_SEED)
    user_lists: list[pandas.DataFrame] = []
    for user_id in range(test_settings.SYNTHETIC_USER_COUNT):
        movie_ids = generator.choice(
            test_settings.SYNTHETIC_MOVIE_COUNT,
            size=list_size,
            replace=False,
        )
        user_lists.append(
            pandas.DataFrame(
                {
                    settings.USER_COLUMN: user_id,
                    settings.MOVIE_COLUMN: movie_ids,
                    settings.POSITION_COLUMN: range(list_size),
                }
            )
        )
    return pandas.concat(user_lists, ignore_index=True)


def _draw_codes(
    generator: numpy.random.Generator,
    count: int,
) -> numpy.typing.NDArray[numpy.int64]:
    return generator.integers(1, test_settings.FEATURE_CODE_COUNT, count)


def _build_synthetic_movies(generator: numpy.random.Generator) -> pandas.DataFrame:
    movie_count = test_settings.SYNTHETIC_MOVIE_COUNT
    movies = pandas.DataFrame(index=pandas.Index(range(movie_count)))
    movies["title"] = "untitled"
    movies[RELEASE_PERIOD_COLUMN] = _draw_codes(generator, movie_count)
    for genre in test_settings.SYNTHETIC_GENRES:
        movies[genre] = generator.integers(0, 2, movie_count).astype(float)
    return movies


def build_synthetic_tables(ratings: pandas.DataFrame) -> MovieLensTables:
    generator = numpy.random.default_rng(test_settings.SYNTHETIC_SEED)
    user_count = test_settings.SYNTHETIC_USER_COUNT
    users = pandas.DataFrame(index=pandas.Index(range(user_count)))
    for column in USER_FEATURE_COLUMNS:
        users[column] = _draw_codes(generator, user_count)
    movies = _build_synthetic_movies(generator)
    return MovieLensTables(ratings=ratings, users=users, movies=movies)
