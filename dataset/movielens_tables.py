"""Ratings, user features and movie features read from the project database in
the form the models use.

User features (gender, age group, occupation) become integer codes starting at 1,
so that code 0 stays free for an unknown value. Movie genres become a 0/1 column
per genre, and the release year, taken from the end of the title, becomes the
code of its five-year period, starting at 1. Both feature tables are indexed by
the original user and movie ids.
"""
from typing import NamedTuple

import duckdb
import numpy
import numpy.typing
import pandas

from dataset import dataset_settings as settings


class MovieLensTables(NamedTuple):
    ratings: pandas.DataFrame
    users: pandas.DataFrame
    movies: pandas.DataFrame


def _read_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: settings.TableName,
) -> pandas.DataFrame:
    return connection.execute(f"SELECT * FROM {table_name}").df()


def _encode_user_features(users: pandas.DataFrame) -> pandas.DataFrame:
    encoded = pandas.DataFrame(index=pandas.Index(users["user_id"], name="user_id"))
    for column in settings.USER_FEATURE_COLUMNS:
        codes, _ = pandas.factorize(users[column], sort=True)
        encoded[column] = codes + 1
    return encoded


def _encode_movie_genres(movies: pandas.DataFrame) -> pandas.DataFrame:
    genres = movies["genres"].str.get_dummies(sep=settings.GENRE_SEPARATOR)
    genres.index = pandas.Index(movies["movie_id"], name="movie_id")
    genres = genres.astype(float)
    genres.insert(0, "title", movies["title"].to_numpy())
    genres.insert(1, settings.RELEASE_PERIOD_COLUMN, _code_release_periods(movies))
    return genres


def _code_release_periods(
    movies: pandas.DataFrame,
) -> numpy.typing.NDArray[numpy.int64]:
    years = movies["title"].str.extract(settings.RELEASE_YEAR_PATTERN)[0]
    year_values = years.astype(int).to_numpy(dtype=numpy.int64)
    periods = (year_values - year_values.min()) // settings.RELEASE_PERIOD_YEARS
    return periods + 1


def read_movielens_tables(database_path: str) -> MovieLensTables:
    """Return the MovieLens-1M ratings and encoded feature tables."""
    with duckdb.connect(database_path, read_only=True) as connection:
        ratings = _read_table(connection, "ratings")
        users = _read_table(connection, "users")
        movies = _read_table(connection, "movies")
    return MovieLensTables(
        ratings=ratings,
        users=_encode_user_features(users),
        movies=_encode_movie_genres(movies),
    )


def list_genre_columns(movies: pandas.DataFrame) -> list[str]:
    genre_columns: list[str] = [
        str(column)
        for column in movies.columns
        if column not in settings.NON_GENRE_MOVIE_COLUMNS
    ]
    return genre_columns
