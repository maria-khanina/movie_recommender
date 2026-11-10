"""Per-user quality metrics of recommendation and candidate lists.

A list is a table of (user, movie, position) rows, with position 0 at the top.
Held-out ratings are a table of (user, movie, rating, gain) rows, the gain being
the percentile of the rating on the user's own scale. Only held-out movies can
be judged: a listed movie the user never rated counts as no gain.

Every metric returns one value per evaluated user, so that the mean, confidence
intervals and paired comparisons are computed over users.
"""
from typing import NamedTuple

import numpy
import pandas

from evaluation import evaluation_settings as settings

_KEY_COLUMNS: list[str] = [settings.USER_COLUMN, settings.MOVIE_COLUMN]
_LIST_COLUMNS: list[str] = [
    settings.USER_COLUMN,
    settings.MOVIE_COLUMN,
    settings.POSITION_COLUMN,
]


class BadShares(NamedTuple):
    per_list_slot: pandas.Series
    per_judged_movie: pandas.Series


def _select_top(lists: pandas.DataFrame, list_size: int) -> pandas.DataFrame:
    is_in_top = lists[settings.POSITION_COLUMN] < list_size
    return lists.loc[is_in_top, _LIST_COLUMNS]


def _find_hits(
    lists: pandas.DataFrame,
    held_out: pandas.DataFrame,
    list_size: int,
) -> pandas.DataFrame:
    top_movies = _select_top(lists, list_size)
    labels = held_out.drop(columns=[settings.POSITION_COLUMN], errors="ignore")
    return top_movies.merge(labels, on=_KEY_COLUMNS, how="inner")


def _compute_discounts(positions: pandas.Series) -> pandas.Series:
    shifted_positions = positions.to_numpy(dtype=float) + 2.0
    return pandas.Series(1.0 / numpy.log2(shifted_positions), index=positions.index)


def _compute_ideal_dcg(held_out: pandas.DataFrame, list_size: int) -> pandas.Series:
    by_gain = held_out.sort_values(
        [settings.USER_COLUMN, settings.GAIN_COLUMN],
        ascending=[True, False],
    )
    ideal_positions = by_gain.groupby(settings.USER_COLUMN).cumcount()
    in_list = ideal_positions < list_size
    ideal_gains = by_gain[settings.GAIN_COLUMN] * _compute_discounts(ideal_positions)
    ideal_users = by_gain.loc[in_list, settings.USER_COLUMN]
    return ideal_gains[in_list].groupby(ideal_users).sum()


def compute_ndcg(
    lists: pandas.DataFrame,
    held_out: pandas.DataFrame,
    list_size: int,
) -> pandas.Series:
    """Return NDCG@list_size per user with a positive ideal DCG."""
    hits = _find_hits(lists, held_out, list_size)
    hit_gains = hits[settings.GAIN_COLUMN] * _compute_discounts(
        hits[settings.POSITION_COLUMN],
    )
    dcg = hit_gains.groupby(hits[settings.USER_COLUMN]).sum()
    ideal_dcg = _compute_ideal_dcg(held_out, list_size)
    ideal_dcg = ideal_dcg[ideal_dcg > 0]
    dcg = dcg.reindex(ideal_dcg.index, fill_value=0.0)
    return dcg / ideal_dcg


def compute_recall(
    lists: pandas.DataFrame,
    relevant: pandas.DataFrame,
    list_size: int,
) -> pandas.Series:
    """Return the share of each user's relevant movies found in the list."""
    hits = _find_hits(lists, relevant, list_size)
    hit_counts = hits.groupby(settings.USER_COLUMN).size()
    relevant_counts = relevant.groupby(settings.USER_COLUMN).size()
    hit_counts = hit_counts.reindex(relevant_counts.index, fill_value=0)
    return hit_counts / relevant_counts


def compute_gain_recall(
    lists: pandas.DataFrame,
    held_out: pandas.DataFrame,
    list_size: int,
) -> pandas.Series:
    """Return the share of each user's held-out gain found in the list, for
    users whose held-out gain is positive.
    """
    hits = _find_hits(lists, held_out, list_size)
    found_gain = hits.groupby(settings.USER_COLUMN)[settings.GAIN_COLUMN].sum()
    total_gain = held_out.groupby(settings.USER_COLUMN)[settings.GAIN_COLUMN].sum()
    total_gain = total_gain[total_gain > 0]
    found_gain = found_gain.reindex(total_gain.index, fill_value=0.0)
    return found_gain / total_gain


def select_bad(held_out: pandas.DataFrame) -> pandas.DataFrame:
    is_bad = held_out[settings.PERCENTILE_COLUMN] < settings.BAD_PERCENTILE_THRESHOLD
    return held_out[is_bad]


def compute_bad_shares(
    lists: pandas.DataFrame,
    held_out: pandas.DataFrame,
    list_size: int,
) -> BadShares:
    """Return the share of bad movies among the list slots and among the judged
    movies of the list; the second share exists only for users with at least one
    held-out movie in the list.
    """
    hits = _find_hits(lists, held_out, list_size)
    is_bad = hits[settings.PERCENTILE_COLUMN] < settings.BAD_PERCENTILE_THRESHOLD
    hit_users = hits[settings.USER_COLUMN]
    bad_counts = is_bad.groupby(hit_users).sum()
    judged_counts = hits.groupby(settings.USER_COLUMN).size()
    evaluated_users = pandas.Index(held_out[settings.USER_COLUMN].unique())
    per_slot = bad_counts.reindex(evaluated_users, fill_value=0) / list_size
    return BadShares(
        per_list_slot=per_slot,
        per_judged_movie=bad_counts / judged_counts,
    )
