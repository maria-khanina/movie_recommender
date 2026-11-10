import pandas

from evaluation import evaluation_settings as settings
from evaluation.new_user_split import (
    choose_new_users,
    remove_new_users,
    select_new_user_ratings,
)
from evaluation.rating_split import EvaluationStage, sort_ratings_by_time
from tests.synthetic_ratings import generate_ratings


def _collect_users(ratings: pandas.DataFrame) -> set[int]:
    return set(ratings[settings.USER_COLUMN].unique().tolist())


def test_new_users_never_reach_training_ratings() -> None:
    ratings = generate_ratings()
    groups = choose_new_users(ratings)
    known = remove_new_users(ratings, groups)
    new_users: set[int] = set(groups.validation.append(groups.test).tolist())
    assert not new_users.intersection(_collect_users(known))
    assert not set(groups.validation.tolist()).intersection(groups.test.tolist())


def _select_user(ratings: pandas.DataFrame, user_id: int) -> pandas.DataFrame:
    return ratings[ratings[settings.USER_COLUMN] == user_id]


def _check_histories(
    ordered: pandas.DataFrame,
    histories: dict[int, pandas.DataFrame],
    held_out: pandas.DataFrame,
) -> None:
    movies: list[int] = ordered[settings.MOVIE_COLUMN].tolist()
    held_out_start = int(held_out[settings.TIMESTAMP_COLUMN].min())
    for size, history in histories.items():
        assert history[settings.MOVIE_COLUMN].tolist() == movies[:size]
        if len(history) > 0:
            assert int(history[settings.TIMESTAMP_COLUMN].max()) < held_out_start
    assert len(held_out) >= settings.NEW_USER_HELD_OUT_SIZE


def test_new_user_histories_precede_the_held_out_ratings() -> None:
    ratings = generate_ratings()
    groups = choose_new_users(ratings)
    split = select_new_user_ratings(ratings, groups, EvaluationStage.VALIDATION)
    assert _collect_users(split.held_out) <= set(groups.validation.tolist())
    for user_id in _collect_users(split.held_out):
        user_histories: dict[int, pandas.DataFrame] = {}
        for size, history in split.histories.items():
            user_histories[size] = _select_user(history, user_id)
        _check_histories(
            sort_ratings_by_time(_select_user(ratings, user_id)),
            user_histories,
            _select_user(split.held_out, user_id),
        )
