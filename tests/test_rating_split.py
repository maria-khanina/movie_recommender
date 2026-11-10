import pandas

from evaluation import evaluation_settings as settings
from evaluation.rating_split import (
    EvaluationStage,
    RatingBlock,
    assign_rating_blocks,
    select_stage_ratings,
)
from tests.synthetic_ratings import generate_ratings

_BLOCK_SEQUENCE: list[str] = [
    RatingBlock.HISTORY,
    RatingBlock.RANKER,
    RatingBlock.VALIDATION,
    RatingBlock.TEST,
]
_BLOCK_SIZES: dict[str, int] = {
    RatingBlock.RANKER: settings.RANKER_BLOCK_SIZE,
    RatingBlock.VALIDATION: settings.VALIDATION_BLOCK_SIZE,
    RatingBlock.TEST: settings.TEST_BLOCK_SIZE,
}


def _summarize_blocks(blocked_ratings: pandas.DataFrame) -> pandas.DataFrame:
    key_columns: list[str] = [settings.USER_COLUMN, settings.BLOCK_COLUMN]
    times = blocked_ratings.groupby(key_columns)[settings.TIMESTAMP_COLUMN]
    return times.agg(["min", "max", "size"]).reset_index()


def _check_user_blocks(user_blocks: pandas.DataFrame) -> None:
    by_block = user_blocks.set_index(settings.BLOCK_COLUMN)
    present: list[str] = [
        block for block in _BLOCK_SEQUENCE if block in by_block.index
    ]
    for earlier, later in zip(present, present[1:]):
        earlier_end = int(str(by_block.loc[earlier, "max"]))
        assert earlier_end < int(str(by_block.loc[later, "min"]))
    for block in present:
        if block in _BLOCK_SIZES:
            assert int(str(by_block.loc[block, "size"])) >= _BLOCK_SIZES[block]


def test_blocks_never_share_a_second_and_are_full() -> None:
    blocked_ratings = assign_rating_blocks(generate_ratings())
    summary = _summarize_blocks(blocked_ratings)
    for _, user_blocks in summary.groupby(settings.USER_COLUMN):
        _check_user_blocks(user_blocks)


def test_users_have_all_held_out_blocks_or_none() -> None:
    blocked_ratings = assign_rating_blocks(generate_ratings())
    blocks_per_user = blocked_ratings.groupby(settings.USER_COLUMN)[
        settings.BLOCK_COLUMN
    ].apply(set)
    held_out: set[str] = set(_BLOCK_SIZES)
    for user_blocks in blocks_per_user.tolist():
        user_held_out = held_out.intersection(user_blocks)
        assert user_held_out in (set(), held_out)


def _collect_blocks(ratings: pandas.DataFrame) -> set[str]:
    return set(ratings[settings.BLOCK_COLUMN].unique())


def test_test_stage_is_validation_stage_shifted_by_one_block() -> None:
    blocked_ratings = assign_rating_blocks(generate_ratings())
    validation = select_stage_ratings(blocked_ratings, EvaluationStage.VALIDATION)
    test = select_stage_ratings(blocked_ratings, EvaluationStage.TEST)
    assert _collect_blocks(validation.ranker) == {RatingBlock.RANKER}
    assert _collect_blocks(validation.evaluation) == {RatingBlock.VALIDATION}
    assert _collect_blocks(test.training) == {
        RatingBlock.HISTORY,
        RatingBlock.RANKER,
    }
    assert _collect_blocks(test.ranker) == {RatingBlock.VALIDATION}
    assert _collect_blocks(test.evaluation) == {RatingBlock.TEST}
