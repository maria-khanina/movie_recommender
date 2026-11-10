"""Percentile and gain of a rating measured on the scale of the user who gave it.

The percentile is the mid-rank percentile of the rating among the same user's
reference ratings: the share of reference ratings strictly below it plus a fixed
weight times the share equal to it. A 3 from a user who rates almost everything
4 or 5 gets a percentile near 0, while a 3 from a strict user gets a large one.
The gain counts only liked ratings: it is 0 up to the median of the user's scale
and grows linearly to 1 at the top, 2 * max(0, percentile - 0.5), so a rating
below the user's usual level adds nothing to recall or NDCG; bad ratings are
counted separately by their percentile. The computation is a sort-merge on
(user, rating) pairs and scales to large rating tables.
"""
import numpy
import numpy.typing
import pandas

from evaluation import evaluation_settings as settings

_EQUAL_COUNT_COLUMN = "equal_count"
_CUMULATIVE_COUNT_COLUMN = "cumulative_count"
_ROW_COLUMN = "row"


def _count_reference_ratings(reference: pandas.DataFrame) -> pandas.DataFrame:
    key_columns: list[str] = [settings.USER_COLUMN, settings.RATING_COLUMN]
    rating_counts = reference.groupby(key_columns).size()
    counts = rating_counts.reset_index(name=_EQUAL_COUNT_COLUMN)
    user_counts = counts.groupby(settings.USER_COLUMN)[_EQUAL_COUNT_COLUMN]
    counts[_CUMULATIVE_COUNT_COLUMN] = user_counts.cumsum()
    counts[settings.RATING_COLUMN] = counts[settings.RATING_COLUMN].astype(float)
    return counts.sort_values(settings.RATING_COLUMN, kind="stable")


def _prepare_held_out_keys(held_out: pandas.DataFrame) -> pandas.DataFrame:
    key_columns: list[str] = [settings.USER_COLUMN, settings.RATING_COLUMN]
    keys = held_out[key_columns].copy()
    keys[_ROW_COLUMN] = range(len(keys))
    keys[settings.RATING_COLUMN] = keys[settings.RATING_COLUMN].astype(float)
    return keys.sort_values(settings.RATING_COLUMN, kind="stable")


def _count_ratings_below(
    held_out_keys: pandas.DataFrame,
    counts: pandas.DataFrame,
) -> pandas.Series:
    nearest_below = pandas.merge_asof(
        held_out_keys,
        counts,
        on=settings.RATING_COLUMN,
        by=settings.USER_COLUMN,
        allow_exact_matches=False,
        direction="backward",
    )
    below = nearest_below.set_index(_ROW_COLUMN)[_CUMULATIVE_COUNT_COLUMN]
    return below.fillna(0.0).sort_index()


def _count_equal_ratings(
    held_out_keys: pandas.DataFrame,
    counts: pandas.DataFrame,
) -> pandas.Series:
    key_columns: list[str] = [settings.USER_COLUMN, settings.RATING_COLUMN]
    equal = held_out_keys.merge(counts, on=key_columns, how="left")
    equal_counts = equal.set_index(_ROW_COLUMN)[_EQUAL_COUNT_COLUMN]
    return equal_counts.fillna(0.0).sort_index()


def attach_percentile_gains(
    held_out: pandas.DataFrame,
    reference: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return the held-out ratings with a gain column added.

    Users without reference ratings are dropped, because their scale is unknown.
    """
    counts = _count_reference_ratings(reference)
    held_out_keys = _prepare_held_out_keys(held_out)
    below = _count_ratings_below(held_out_keys, counts).to_numpy()
    equal = _count_equal_ratings(held_out_keys, counts).to_numpy()
    percentiles = _combine_percentiles(held_out, reference, below, equal)
    gained = held_out.assign(
        percentile=percentiles,
        gain=_compute_gains(percentiles),
    )
    has_scale = gained[settings.PERCENTILE_COLUMN].notna()
    return gained[has_scale].reset_index(drop=True)


def _combine_percentiles(
    held_out: pandas.DataFrame,
    reference: pandas.DataFrame,
    below: numpy.typing.NDArray[numpy.float64],
    equal: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    reference_sizes = reference.groupby(settings.USER_COLUMN).size()
    user_sizes = held_out[settings.USER_COLUMN].map(reference_sizes).to_numpy()
    weighted_equal = settings.EQUAL_RATING_PERCENTILE_WEIGHT * equal
    return (below + weighted_equal) / user_sizes


def _compute_gains(
    percentiles: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    above_median = numpy.maximum(percentiles - settings.GAIN_START_PERCENTILE, 0.0)
    return settings.GAIN_SCALE * above_median
