"""Everything the validation-stage comparisons need, built once.

Known users are split into temporal blocks; models train on the history block,
the ranker block labels the lists of rankers that learn on candidates, and the
validation block is held out. The new users of the validation stage are absent
from all training: their first ratings are their only history and the next ones
are held out, checked with histories of several lengths. Known users are also
grouped by how many ratings the site knows about them at the moment of
recommending (history and ranker blocks), and by whether the validation block
starts on the same day as the known history ends: in MovieLens most users rate
in one sitting, and a model can do well on the same-day group by continuing the
sitting rather than by knowing the user's taste.
"""
from typing import NamedTuple

import pandas

from dataset.movielens_tables import MovieLensTables, read_movielens_tables
from evaluation import evaluation_settings as settings
from evaluation.new_user_split import (
    choose_new_users,
    remove_new_users,
    select_new_user_ratings,
)
from evaluation.percentile_gains import attach_percentile_gains
from evaluation.rating_split import (
    EvaluationStage,
    assign_rating_blocks,
    select_stage_ratings,
)

ALL_KNOWN_SEGMENT = "all known users"
SAME_DAY_SEGMENT = "known users, validation starts the same day as the history ends"
LATER_DAY_SEGMENT = "known users, validation starts a later day"


def name_new_user_segment(history_size: int) -> str:
    return f"new users, first {history_size} ratings as history"


class ValidationData(NamedTuple):
    tables: MovieLensTables
    training: pandas.DataFrame
    known: pandas.DataFrame
    held_out: pandas.DataFrame
    ranker_labels: pandas.DataFrame
    new_user_histories: dict[int, pandas.DataFrame]
    new_user_held_out: pandas.DataFrame
    segments: dict[str, pandas.Index]


def _describe_length_range(lower: int, upper: int) -> str:
    if upper < 0:
        return f"known users, history of at least {lower} ratings"
    return f"known users, history of {lower} to {upper - 1} ratings"


def _group_by_history_length(
    known: pandas.DataFrame,
    evaluated: pandas.Index,
) -> dict[str, pandas.Index]:
    sizes = known.groupby(settings.USER_COLUMN).size()
    sizes = sizes.reindex(evaluated, fill_value=0)
    lowers: list[int] = [0, *settings.HISTORY_LENGTH_EDGES]
    uppers: list[int] = [*settings.HISTORY_LENGTH_EDGES, -1]
    segments: dict[str, pandas.Index] = {}
    for lower, upper in zip(lowers, uppers):
        in_range = (sizes >= lower) & ((sizes < upper) | (upper < 0))
        segments[_describe_length_range(lower, upper)] = evaluated[in_range.to_numpy()]
    return segments


def _group_by_time_gap(
    known: pandas.DataFrame,
    held_out: pandas.DataFrame,
    evaluated: pandas.Index,
) -> dict[str, pandas.Index]:
    last_known = known.groupby(settings.USER_COLUMN)[
        settings.TIMESTAMP_COLUMN
    ].max()
    first_held_out = held_out.groupby(settings.USER_COLUMN)[
        settings.TIMESTAMP_COLUMN
    ].min()
    aligned_last = last_known.reindex(first_held_out.index)
    gaps = (first_held_out - aligned_last).reindex(evaluated)
    is_same_day = (gaps < settings.SECONDS_PER_DAY).to_numpy()
    segments: dict[str, pandas.Index] = {
        SAME_DAY_SEGMENT: evaluated[is_same_day],
        LATER_DAY_SEGMENT: evaluated[~is_same_day],
    }
    return segments


def _group_known_users(
    known: pandas.DataFrame,
    held_out: pandas.DataFrame,
) -> dict[str, pandas.Index]:
    evaluated = pandas.Index(held_out[settings.USER_COLUMN].unique())
    segments = _group_by_history_length(known, evaluated)
    segments[ALL_KNOWN_SEGMENT] = evaluated
    segments.update(_group_by_time_gap(known, held_out, evaluated))
    return segments


def prepare_validation_data(database_path: str) -> ValidationData:
    tables = read_movielens_tables(database_path)
    groups = choose_new_users(tables.ratings)
    blocked = assign_rating_blocks(remove_new_users(tables.ratings, groups))
    stage = select_stage_ratings(blocked, EvaluationStage.VALIDATION)
    new_users = select_new_user_ratings(
        tables.ratings,
        groups,
        EvaluationStage.VALIDATION,
    )
    known = pandas.concat([stage.training, stage.ranker], ignore_index=True)
    held_out = attach_percentile_gains(
        stage.evaluation,
        pandas.concat([stage.training, stage.evaluation]),
    )
    return ValidationData(
        tables=tables,
        training=stage.training,
        known=known,
        held_out=held_out,
        ranker_labels=attach_percentile_gains(
            stage.ranker,
            pandas.concat([stage.training, stage.ranker]),
        ),
        new_user_histories=new_users.histories,
        new_user_held_out=new_users.held_out,
        segments=_group_known_users(known, held_out),
    )


def select_segment(per_user: pandas.Series, users: pandas.Index) -> pandas.Series:
    return per_user[per_user.index.isin(users)]
