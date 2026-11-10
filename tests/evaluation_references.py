"""Straightforward per-row and per-user versions of the evaluation formulas, used
as an independent reference for the vectorized implementations.
"""
import math

import pandas

from evaluation import evaluation_settings as settings


def compute_reference_percentile(
    user_reference: list[float],
    rating: float,
) -> float:
    below = sum(1 for value in user_reference if value < rating)
    equal = sum(1 for value in user_reference if value == rating)
    weighted_equal = settings.EQUAL_RATING_PERCENTILE_WEIGHT * equal
    return (below + weighted_equal) / len(user_reference)


def _sum_discounted_gains(gains: list[float]) -> float:
    total = 0.0
    for position, gain in enumerate(gains):
        total += gain / math.log2(position + 2.0)
    return total


def compute_reference_ndcg(
    listed_movies: list[int],
    held_out_gains: dict[int, float],
    list_size: int,
) -> float:
    listed_gains: list[float] = []
    for movie_id in listed_movies[:list_size]:
        listed_gains.append(held_out_gains.get(movie_id, 0.0))
    ideal_gains = sorted(held_out_gains.values(), reverse=True)[:list_size]
    return _sum_discounted_gains(listed_gains) / _sum_discounted_gains(ideal_gains)


def collect_user_lists(lists: pandas.DataFrame) -> dict[int, list[int]]:
    ordered = lists.sort_values([settings.USER_COLUMN, settings.POSITION_COLUMN])
    user_lists: dict[int, list[int]] = {}
    for user_id, user_rows in ordered.groupby(settings.USER_COLUMN):
        user_lists[int(str(user_id))] = user_rows[settings.MOVIE_COLUMN].tolist()
    return user_lists


def compute_reference_gain(percentile: float) -> float:
    above_median = max(0.0, percentile - settings.GAIN_START_PERCENTILE)
    return settings.GAIN_SCALE * above_median


def collect_user_values(
    held_out: pandas.DataFrame,
    column: str,
) -> dict[int, dict[int, float]]:
    user_values: dict[int, dict[int, float]] = {}
    for row in held_out.itertuples(index=False):
        user_id = int(str(getattr(row, settings.USER_COLUMN)))
        movie_id = int(str(getattr(row, settings.MOVIE_COLUMN)))
        value = float(str(getattr(row, column)))
        user_values.setdefault(user_id, {})[movie_id] = value
    return user_values


def collect_user_gains(held_out: pandas.DataFrame) -> dict[int, dict[int, float]]:
    return collect_user_values(held_out, settings.GAIN_COLUMN)
