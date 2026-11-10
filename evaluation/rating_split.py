"""Per-user temporal split of ratings into history, ranker, validation and test
blocks, and the two evaluation stages built from those blocks.

Every user's ratings are ordered by time. The last ratings form the test block,
the ratings before them the validation block, and the ones before those the
ranker block; everything earlier is history. Each block takes whole seconds from
the end until it holds at least its size, so ratings given in the same second
never fall on both sides of a boundary: everything a model knows when it
recommends is strictly earlier than every rating it is checked against. A user
who cannot fill every held-out block keeps all ratings in history and is not
evaluated; there is no minimum history, so a user can be evaluated with a short
or empty history.

The validation stage trains on history, gives the ranker block to rankers that
learn on candidate lists, and evaluates on the validation block. The test stage
is the same procedure shifted by one block: it trains on history and the ranker
block, gives the validation block to the rankers and evaluates on the test block.
"""
import enum
from typing import NamedTuple

import pandas

from evaluation import evaluation_settings as settings


class RatingBlock(enum.StrEnum):
    HISTORY = "history"
    RANKER = "ranker"
    VALIDATION = "validation"
    TEST = "test"


class EvaluationStage(enum.StrEnum):
    VALIDATION = "validation"
    TEST = "test"


class StageRatings(NamedTuple):
    training: pandas.DataFrame
    ranker: pandas.DataFrame
    evaluation: pandas.DataFrame


class _StageBlocks(NamedTuple):
    training: tuple[RatingBlock, ...]
    ranker: tuple[RatingBlock, ...]
    evaluation: tuple[RatingBlock, ...]


_STAGE_BLOCKS: dict[EvaluationStage, _StageBlocks] = {
    EvaluationStage.VALIDATION: _StageBlocks(
        training=(RatingBlock.HISTORY,),
        ranker=(RatingBlock.RANKER,),
        evaluation=(RatingBlock.VALIDATION,),
    ),
    EvaluationStage.TEST: _StageBlocks(
        training=(RatingBlock.HISTORY, RatingBlock.RANKER),
        ranker=(RatingBlock.VALIDATION,),
        evaluation=(RatingBlock.TEST,),
    ),
}

_SECOND_SIZE_COLUMN = "second_size"
_INCLUSIVE_COLUMN = "ratings_from_this_second"
_LATER_COLUMN = "ratings_after_this_second"
_BLOCK_SIZES: list[tuple[RatingBlock, int]] = [
    (RatingBlock.TEST, settings.TEST_BLOCK_SIZE),
    (RatingBlock.VALIDATION, settings.VALIDATION_BLOCK_SIZE),
    (RatingBlock.RANKER, settings.RANKER_BLOCK_SIZE),
]


def _draw_tie_order(ratings: pandas.DataFrame) -> pandas.Series:
    key_columns: list[str] = [settings.USER_COLUMN, settings.MOVIE_COLUMN]
    return pandas.util.hash_pandas_object(
        ratings[key_columns],
        index=False,
        hash_key=settings.TIE_ORDER_HASH_KEY,
    )


def sort_ratings_by_time(ratings: pandas.DataFrame) -> pandas.DataFrame:
    """Return the ratings ordered by user and time. Ratings of one user in the
    same second are ordered by movie id, or, when the random tie order is on, by
    a hash of the (user, movie) pair: random-looking but reproducible.
    """
    tie_values = (
        _draw_tie_order(ratings)
        if settings.RANDOM_TIE_ORDER
        else ratings[settings.MOVIE_COLUMN]
    )
    keyed = ratings.assign(tie_order=tie_values.to_numpy())
    sorting_columns: list[str] = [
        settings.USER_COLUMN,
        settings.TIMESTAMP_COLUMN,
        settings.TIE_ORDER_COLUMN,
    ]
    sorted_ratings = keyed.sort_values(sorting_columns, kind="stable")
    return sorted_ratings.drop(columns=[settings.TIE_ORDER_COLUMN]).reset_index(
        drop=True,
    )


def _count_seconds(blocked_ratings: pandas.DataFrame) -> pandas.DataFrame:
    key_columns: list[str] = [settings.USER_COLUMN, settings.TIMESTAMP_COLUMN]
    seconds = blocked_ratings.groupby(key_columns, sort=True).size()
    seconds = seconds.reset_index(name=_SECOND_SIZE_COLUMN)
    reversed_sizes = seconds.iloc[::-1].groupby(settings.USER_COLUMN)[
        _SECOND_SIZE_COLUMN
    ]
    seconds[_INCLUSIVE_COLUMN] = reversed_sizes.cumsum().iloc[::-1]
    seconds[_LATER_COLUMN] = seconds[_INCLUSIVE_COLUMN] - seconds[_SECOND_SIZE_COLUMN]
    return seconds


def _close_block(
    seconds: pandas.DataFrame,
    is_taken: pandas.Series,
    block_size: int,
) -> pandas.Series:
    """Return which still-free seconds join the next block from the end: whole
    seconds are added until the block holds at least block_size ratings.
    """
    taken_counts = seconds[_INCLUSIVE_COLUMN].where(is_taken, 0)
    boundary = taken_counts.groupby(seconds[settings.USER_COLUMN]).transform("max")
    return ~is_taken & (seconds[_LATER_COLUMN] < boundary + block_size)


def _label_seconds(seconds: pandas.DataFrame) -> pandas.Series:
    is_taken = pandas.Series(False, index=seconds.index)
    labels = pandas.Series(str(RatingBlock.HISTORY), index=seconds.index)
    for block, block_size in _BLOCK_SIZES:
        joins = _close_block(seconds, is_taken, block_size)
        labels[joins] = str(block)
        is_taken |= joins
    return labels


def _keep_complete_users(blocked_ratings: pandas.DataFrame) -> pandas.Series:
    """Return block labels where users who cannot fill every held-out block keep
    all their ratings in history.
    """
    labels = blocked_ratings[settings.BLOCK_COLUMN]
    counts = pandas.crosstab(blocked_ratings[settings.USER_COLUMN], labels)
    complete = pandas.Series(True, index=counts.index)
    for block, block_size in _BLOCK_SIZES:
        block_counts = counts[str(block)] if str(block) in counts else 0
        complete &= block_counts >= block_size
    is_complete = blocked_ratings[settings.USER_COLUMN].map(complete)
    return labels.where(is_complete.astype(bool), str(RatingBlock.HISTORY))


def assign_rating_blocks(ratings: pandas.DataFrame) -> pandas.DataFrame:
    """Return the ratings sorted by user and time with a block column added.

    Held-out blocks are taken from the end in whole seconds, so no block
    boundary falls inside one second.
    """
    blocked_ratings = sort_ratings_by_time(ratings)
    seconds = _count_seconds(blocked_ratings)
    seconds[settings.BLOCK_COLUMN] = _label_seconds(seconds).to_numpy()
    key_columns: list[str] = [settings.USER_COLUMN, settings.TIMESTAMP_COLUMN]
    labeled = blocked_ratings.merge(
        seconds[[*key_columns, settings.BLOCK_COLUMN]],
        on=key_columns,
        how="left",
    )
    labeled[settings.BLOCK_COLUMN] = _keep_complete_users(labeled)
    return labeled


def _select_blocks(
    blocked_ratings: pandas.DataFrame,
    blocks: tuple[RatingBlock, ...],
) -> pandas.DataFrame:
    block_names: list[str] = list(blocks)
    is_selected = blocked_ratings[settings.BLOCK_COLUMN].isin(block_names)
    return blocked_ratings[is_selected].reset_index(drop=True)


def select_stage_ratings(
    blocked_ratings: pandas.DataFrame,
    stage: EvaluationStage,
) -> StageRatings:
    """Return the training, ranker and evaluation ratings of one stage."""
    stage_blocks = _STAGE_BLOCKS[stage]
    return StageRatings(
        training=_select_blocks(blocked_ratings, stage_blocks.training),
        ranker=_select_blocks(blocked_ratings, stage_blocks.ranker),
        evaluation=_select_blocks(blocked_ratings, stage_blocks.evaluation),
    )
