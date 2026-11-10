"""Users held out of all training to check recommendations for new users.

A share of users is chosen at random before the temporal blocks are built; half
of them belong to the validation stage and half to the test stage. None of their
ratings trains any model. A held-out user plays a new member of the site who has
rated a few self-chosen movies: the user's first ratings (in time order) are the
only history, and the ratings after them are what the recommendations are
checked against. To see how quality grows with the number of ratings, the same
held-out ratings are checked with histories of several lengths. The held-out
ratings start in the first second after the longest history ends and take whole
seconds until they reach the held-out size, so they are strictly later than any
history rating. Later ratings are not used.
"""
from typing import NamedTuple

import numpy
import pandas

from evaluation import evaluation_settings as settings
from evaluation.percentile_gains import attach_percentile_gains
from evaluation.rating_split import EvaluationStage, sort_ratings_by_time


class NewUserGroups(NamedTuple):
    validation: pandas.Index
    test: pandas.Index


class NewUserRatings(NamedTuple):
    histories: dict[int, pandas.DataFrame]
    held_out: pandas.DataFrame


def choose_new_users(ratings: pandas.DataFrame) -> NewUserGroups:
    user_ids = numpy.sort(ratings[settings.USER_COLUMN].unique())
    generator = numpy.random.default_rng(settings.RANDOM_SEED)
    chosen_count = int(len(user_ids) * settings.NEW_USER_SHARE)
    chosen = generator.choice(user_ids, size=chosen_count, replace=False)
    half = chosen_count // 2
    return NewUserGroups(
        validation=pandas.Index(numpy.sort(chosen[:half])),
        test=pandas.Index(numpy.sort(chosen[half:])),
    )


def remove_new_users(
    ratings: pandas.DataFrame,
    groups: NewUserGroups,
) -> pandas.DataFrame:
    new_users = groups.validation.append(groups.test)
    is_known = ~ratings[settings.USER_COLUMN].isin(new_users)
    return ratings[is_known].reset_index(drop=True)


def _select_stage_users(
    ratings: pandas.DataFrame,
    groups: NewUserGroups,
    stage: EvaluationStage,
) -> pandas.DataFrame:
    is_validation = stage == EvaluationStage.VALIDATION
    stage_users = groups.validation if is_validation else groups.test
    is_stage_user = ratings[settings.USER_COLUMN].isin(stage_users)
    return sort_ratings_by_time(ratings[is_stage_user])


def _mark_taken_seconds(after: pandas.DataFrame) -> pandas.DataFrame:
    key_columns: list[str] = [settings.USER_COLUMN, settings.TIMESTAMP_COLUMN]
    seconds = after.groupby(key_columns).size().reset_index(name="second_size")
    taken_through = seconds.groupby(settings.USER_COLUMN)["second_size"].cumsum()
    taken_before = taken_through - seconds["second_size"]
    seconds["is_taken"] = taken_before < settings.NEW_USER_HELD_OUT_SIZE
    return after.merge(seconds, on=key_columns, how="left")


def _select_held_out(
    ordered: pandas.DataFrame,
    boundaries: pandas.Series,
) -> pandas.Series:
    times = ordered[settings.TIMESTAMP_COLUMN]
    is_after = times > ordered[settings.USER_COLUMN].map(boundaries)
    marked = _mark_taken_seconds(ordered[is_after])
    is_held_out = pandas.Series(False, index=ordered.index)
    is_held_out[is_after] = marked["is_taken"].to_numpy()
    return is_held_out


def _collect_histories(
    ordered: pandas.DataFrame,
    positions: pandas.Series,
) -> dict[int, pandas.DataFrame]:
    histories: dict[int, pandas.DataFrame] = {}
    for size in settings.NEW_USER_HISTORY_SIZES:
        histories[size] = ordered[positions < size].reset_index(drop=True)
    return histories


def select_new_user_ratings(
    ratings: pandas.DataFrame,
    groups: NewUserGroups,
    stage: EvaluationStage,
) -> NewUserRatings:
    """Return the histories by length and the gained held-out ratings of the
    stage's new users who have enough ratings after the longest history.
    """
    ordered = _select_stage_users(ratings, groups, stage)
    positions = ordered.groupby(settings.USER_COLUMN).cumcount()
    longest = max(settings.NEW_USER_HISTORY_SIZES)
    is_longest_history = positions < longest
    boundaries = ordered[is_longest_history].groupby(settings.USER_COLUMN)[
        settings.TIMESTAMP_COLUMN
    ].max()
    held_out_ratings = ordered[_select_held_out(ordered, boundaries)]
    return _assemble_new_users(ordered, positions, held_out_ratings)


def _assemble_new_users(
    ordered: pandas.DataFrame,
    positions: pandas.Series,
    held_out_ratings: pandas.DataFrame,
) -> NewUserRatings:
    held_out_sizes = held_out_ratings.groupby(settings.USER_COLUMN).size()
    complete = held_out_sizes[held_out_sizes >= settings.NEW_USER_HELD_OUT_SIZE].index
    is_complete = ordered[settings.USER_COLUMN].isin(complete)
    histories = _collect_histories(ordered[is_complete], positions[is_complete])
    longest_history = histories[max(settings.NEW_USER_HISTORY_SIZES)]
    complete_held_out = held_out_ratings[
        held_out_ratings[settings.USER_COLUMN].isin(complete)
    ]
    held_out = attach_percentile_gains(
        complete_held_out,
        pandas.concat([longest_history, complete_held_out]),
    )
    return NewUserRatings(histories=histories, held_out=held_out)
