"""Descriptive metrics of what a model recommends, independent of held-out ratings.

Novelty is the mean self-information of the listed movies, -log2 of the share of
training users who rated the movie, smoothed so that a movie absent from
training has a finite value. Coverage is the share of the catalog that appears in
at least one user's list. Neither is a quality score: they show whether a model
recommends the same popular movies to everyone or spreads over the catalog.
"""
import numpy
import pandas

from evaluation import evaluation_settings as settings


def compute_movie_popularity(
    training: pandas.DataFrame,
    catalog: pandas.Index,
) -> pandas.Series:
    """Return the smoothed share of training users who rated each catalog movie."""
    user_count = training[settings.USER_COLUMN].nunique()
    raters_per_movie = training.groupby(settings.MOVIE_COLUMN)[
        settings.USER_COLUMN
    ].nunique()
    raters_per_movie = raters_per_movie.reindex(catalog, fill_value=0)
    return (raters_per_movie + 1.0) / (user_count + 1.0)


def compute_novelty(
    lists: pandas.DataFrame,
    popularity: pandas.Series,
    list_size: int,
) -> pandas.Series:
    """Return the mean self-information of the listed movies per user."""
    top_movies = lists[lists[settings.POSITION_COLUMN] < list_size]
    listed_popularity = top_movies[settings.MOVIE_COLUMN].map(popularity)
    self_information = -numpy.log2(listed_popularity.to_numpy(dtype=float))
    information = pandas.Series(self_information, index=top_movies.index)
    return information.groupby(top_movies[settings.USER_COLUMN]).mean()


def compute_catalog_coverage(
    lists: pandas.DataFrame,
    catalog: pandas.Index,
    list_size: int,
) -> float:
    """Return the share of catalog movies listed for at least one user."""
    top_movies = lists[lists[settings.POSITION_COLUMN] < list_size]
    listed_movies = pandas.Index(top_movies[settings.MOVIE_COLUMN].unique())
    return len(listed_movies.intersection(catalog)) / len(catalog)
