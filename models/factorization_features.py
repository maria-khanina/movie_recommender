"""Sparse feature matrix of a factorization machine, split into user and movie
features.

User features are the one-hot user id (only for users of the fitting ratings),
age group, gender and occupation; movie features are the one-hot movie id, the
genre indicators and the one-hot release period. A row of the full matrix holds
one user's and one movie's features. Because every column belongs to the user
side or to the movie side, the machine's score of a pair splits into a user-only
part, a movie-only part and the inner product of a user vector and a movie
vector, which is what retrieval needs.
"""
import numpy
import numpy.typing
import pandas
import scipy.sparse

from dataset.dataset_settings import RELEASE_PERIOD_COLUMN, USER_FEATURE_COLUMNS
from dataset.movielens_tables import MovieLensTables, list_genre_columns
from evaluation import evaluation_settings as columns
from models.candidate_retrieval import MovieIds

_Codes = numpy.typing.NDArray[numpy.int64]


def _one_hot_offsets(sizes: list[int]) -> list[int]:
    offsets: list[int] = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
    return offsets


class _SideMatrix:
    def __init__(self, codes: list[_Codes], sizes: list[int]) -> None:
        self.codes = codes
        self.offsets = _one_hot_offsets(sizes)
        self.width = sum(sizes)

    def _locate_entries(self, rows: _Codes) -> tuple[_Codes, _Codes]:
        row_parts: list[_Codes] = []
        column_parts: list[_Codes] = []
        for codes, offset in zip(self.codes, self.offsets):
            selected = codes[rows]
            is_known = selected >= 0
            row_parts.append(numpy.flatnonzero(is_known))
            column_parts.append(selected[is_known] + offset)
        return numpy.concatenate(row_parts), numpy.concatenate(column_parts)

    def build(
        self,
        rows: _Codes,
        extra: scipy.sparse.csr_array,
    ) -> scipy.sparse.csr_array:
        row_codes, column_codes = self._locate_entries(rows)
        values = numpy.ones(len(row_codes), dtype=numpy.float64)
        one_hot = scipy.sparse.csr_array(
            (values, (row_codes, column_codes)),
            shape=(len(rows), self.width),
        )
        return scipy.sparse.hstack([one_hot, extra], format="csr")


class FeatureSpace:
    def __init__(self, fitting: pandas.DataFrame, tables: MovieLensTables) -> None:
        self.user_index = pandas.Index(tables.users.index)
        self.movie_index = pandas.Index(tables.movies.index)
        fitted_users = pandas.Index(fitting[columns.USER_COLUMN].unique())
        self._users = self._build_user_side(fitted_users, tables)
        self._genres = scipy.sparse.csr_array(
            tables.movies[list_genre_columns(tables.movies)].to_numpy(dtype=float),
        )
        self._movies = self._build_movie_side(tables)
        self.user_width = self._users.width
        movie_width = self._movies.width + self._genres.shape[1]
        self.feature_count = self.user_width + movie_width

    def _build_user_side(
        self,
        fitted_users: pandas.Index,
        tables: MovieLensTables,
    ) -> _SideMatrix:
        id_codes = fitted_users.get_indexer(self.user_index).astype(numpy.int64)
        codes: list[_Codes] = [id_codes]
        sizes: list[int] = [len(fitted_users)]
        for column in USER_FEATURE_COLUMNS:
            values = tables.users[column].to_numpy(dtype=numpy.int64)
            codes.append(values)
            sizes.append(int(values.max()) + 1)
        return _SideMatrix(codes, sizes)

    def _build_movie_side(self, tables: MovieLensTables) -> _SideMatrix:
        periods = tables.movies[RELEASE_PERIOD_COLUMN].to_numpy(dtype=numpy.int64)
        codes: list[_Codes] = [
            numpy.arange(len(self.movie_index), dtype=numpy.int64),
            periods,
        ]
        sizes: list[int] = [len(self.movie_index), int(periods.max()) + 1]
        return _SideMatrix(codes, sizes)

    def encode_users(self, user_ids: MovieIds) -> scipy.sparse.csr_array:
        """Return user rows that are zero in every movie column."""
        rows = self.user_index.get_indexer(pandas.Index(user_ids)).astype(numpy.int64)
        empty_movie_part = scipy.sparse.csr_array(
            (len(rows), self.feature_count - self.user_width),
        )
        return self._users.build(rows, empty_movie_part)

    def encode_movies(self, movie_ids: MovieIds) -> scipy.sparse.csr_array:
        """Return movie rows that are zero in every user column."""
        rows = self.movie_index.get_indexer(pandas.Index(movie_ids)).astype(numpy.int64)
        movie_part = self._movies.build(rows, self._genres[rows])
        empty_user_part = scipy.sparse.csr_array((len(rows), self.user_width))
        return scipy.sparse.hstack([empty_user_part, movie_part], format="csr")

    def encode_ids(
        self,
        user_ids: MovieIds,
        movie_ids: MovieIds,
    ) -> scipy.sparse.csr_array:
        users = self.encode_users(user_ids)
        movies = self.encode_movies(movie_ids)
        return (users + movies).tocsr()

    def encode_pairs(self, ratings: pandas.DataFrame) -> scipy.sparse.csr_array:
        return self.encode_ids(
            ratings[columns.USER_COLUMN].to_numpy(dtype=numpy.int64),
            ratings[columns.MOVIE_COLUMN].to_numpy(dtype=numpy.int64),
        )
