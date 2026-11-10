"""Mean of a per-user metric with a bootstrap confidence interval over users, and
the paired difference of two models measured on the same users.

Users, not ratings, are the resampled unit: every user contributes one value, and
the spread between users is the main source of noise in the metrics. A paired
comparison resamples the users both models were evaluated on, so the shared
difficulty of each user cancels and the interval of the difference is narrower
than the two separate intervals suggest.
"""
from typing import NamedTuple

import numpy
import numpy.typing
import pandas

from evaluation import evaluation_settings as settings

_Values = numpy.typing.NDArray[numpy.float64]


class MetricEstimate(NamedTuple):
    mean: float
    lower: float
    upper: float
    user_count: int


def _resample_means(values: _Values) -> _Values:
    generator = numpy.random.default_rng(settings.RANDOM_SEED)
    sample_indices = generator.integers(
        0,
        len(values),
        size=(settings.BOOTSTRAP_SAMPLE_COUNT, len(values)),
    )
    return values[sample_indices].mean(axis=1)


def _estimate_mean(values: _Values) -> MetricEstimate:
    resampled_means = _resample_means(values)
    tail_share = (1.0 - settings.BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = numpy.quantile(resampled_means, [tail_share, 1.0 - tail_share])
    return MetricEstimate(
        mean=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        user_count=len(values),
    )


def summarize_metric(per_user: pandas.Series) -> MetricEstimate:
    """Return the mean of a per-user metric and its bootstrap interval."""
    return _estimate_mean(per_user.dropna().to_numpy(dtype=numpy.float64))


def compare_paired(
    first: pandas.Series,
    second: pandas.Series,
) -> MetricEstimate:
    """Return the mean of first minus second over the users both were evaluated
    on, with its bootstrap interval.
    """
    shared_users = first.dropna().index.intersection(second.dropna().index)
    differences = first.loc[shared_users] - second.loc[shared_users]
    return _estimate_mean(differences.to_numpy(dtype=numpy.float64))
