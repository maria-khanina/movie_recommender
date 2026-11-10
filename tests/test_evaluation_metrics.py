import math

import pandas

from evaluation import evaluation_settings as settings
from evaluation.percentile_gains import attach_percentile_gains
from evaluation.ranking_metrics import (
    BadShares,
    compute_bad_shares,
    compute_gain_recall,
    compute_ndcg,
    compute_recall,
    select_bad,
)
from evaluation.rating_split import (
    EvaluationStage,
    StageRatings,
    assign_rating_blocks,
    select_stage_ratings,
)
from tests import test_settings
from tests.evaluation_references import (
    collect_user_gains,
    collect_user_lists,
    collect_user_values,
    compute_reference_gain,
    compute_reference_ndcg,
    compute_reference_percentile,
)
from tests.synthetic_ratings import generate_lists, generate_ratings


def _build_validation_stage() -> StageRatings:
    blocked_ratings = assign_rating_blocks(generate_ratings())
    return select_stage_ratings(blocked_ratings, EvaluationStage.VALIDATION)


def _build_gained_held_out() -> pandas.DataFrame:
    stage = _build_validation_stage()
    scale = pandas.concat([stage.training, stage.evaluation])
    return attach_percentile_gains(stage.evaluation, scale)


def _collect_reference_ratings(training: pandas.DataFrame) -> dict[int, list[float]]:
    reference: dict[int, list[float]] = {}
    for row in training.itertuples(index=False):
        user_id = int(str(getattr(row, settings.USER_COLUMN)))
        rating = float(str(getattr(row, settings.RATING_COLUMN)))
        reference.setdefault(user_id, []).append(rating)
    return reference


def _is_close(first: float, second: float) -> bool:
    return math.isclose(first, second, abs_tol=test_settings.COMPARISON_TOLERANCE)


def _select_users_with_positive_gain(
    user_gains: dict[int, dict[int, float]],
) -> list[int]:
    positive_users: list[int] = []
    for user_id, gains in user_gains.items():
        if max(gains.values()) > 0:
            positive_users.append(user_id)
    return sorted(positive_users)


def _check_gained_row(
    row: tuple[object, ...],
    reference: dict[int, list[float]],
) -> None:
    user_id = int(str(getattr(row, settings.USER_COLUMN)))
    rating = float(str(getattr(row, settings.RATING_COLUMN)))
    percentile = compute_reference_percentile(reference[user_id], rating)
    stored = float(str(getattr(row, settings.PERCENTILE_COLUMN)))
    assert _is_close(stored, percentile)
    gain = float(str(getattr(row, settings.GAIN_COLUMN)))
    assert _is_close(gain, compute_reference_gain(percentile))


def test_percentile_gains_match_per_row_reference() -> None:
    stage = _build_validation_stage()
    scale = pandas.concat([stage.training, stage.evaluation])
    gained = attach_percentile_gains(stage.evaluation, scale)
    reference = _collect_reference_ratings(scale)
    assert len(gained) == len(stage.evaluation)
    for row in gained.itertuples(index=False):
        _check_gained_row(row, reference)


def _compute_expected_ndcg(
    lists: pandas.DataFrame,
    held_out: pandas.DataFrame,
    list_size: int,
) -> dict[int, float]:
    user_lists = collect_user_lists(lists)
    user_gains = collect_user_gains(held_out)
    expected: dict[int, float] = {}
    for user_id in _select_users_with_positive_gain(user_gains):
        expected[user_id] = compute_reference_ndcg(
            user_lists[user_id],
            user_gains[user_id],
            list_size,
        )
    return expected


def test_ndcg_matches_per_user_reference() -> None:
    held_out = _build_gained_held_out()
    lists = generate_lists(test_settings.SYNTHETIC_MOVIE_COUNT)
    list_size = test_settings.SYNTHETIC_LIST_SIZE
    ndcg = compute_ndcg(lists, held_out, list_size)
    expected = _compute_expected_ndcg(lists, held_out, list_size)
    assert sorted(ndcg.index.tolist()) == sorted(expected)
    for user_id, expected_ndcg in expected.items():
        assert _is_close(float(ndcg.loc[user_id]), expected_ndcg)


def test_ideal_list_has_ndcg_one() -> None:
    held_out = _build_gained_held_out()
    ideal_order = held_out.sort_values(
        [settings.USER_COLUMN, settings.GAIN_COLUMN],
        ascending=[True, False],
    )
    ideal_positions = ideal_order.groupby(settings.USER_COLUMN).cumcount()
    ideal_lists = ideal_order.assign(position=ideal_positions)
    ndcg = compute_ndcg(ideal_lists, held_out, test_settings.SYNTHETIC_LIST_SIZE)
    assert all(_is_close(value, 1.0) for value in ndcg.tolist())


def _count_listed(listed_movies: list[int], movie_ids: set[int]) -> int:
    return len(movie_ids.intersection(listed_movies))


def test_recall_matches_per_user_counts() -> None:
    held_out = _build_gained_held_out()
    lists = generate_lists(test_settings.SYNTHETIC_LIST_SIZE)
    bad = select_bad(held_out)
    recall = compute_recall(lists, bad, test_settings.SYNTHETIC_LIST_SIZE)
    user_lists = collect_user_lists(lists)
    for user_id, bad_gains in collect_user_gains(bad).items():
        bad_movies: set[int] = set(bad_gains)
        hit_count = _count_listed(user_lists[user_id], bad_movies)
        expected = hit_count / len(bad_movies)
        assert _is_close(float(recall.loc[user_id]), expected)


def _sum_listed_gain(listed_movies: list[int], gains: dict[int, float]) -> float:
    found = 0.0
    for movie_id in listed_movies:
        found += gains.get(movie_id, 0.0)
    return found


def test_gain_recall_matches_per_user_sums() -> None:
    held_out = _build_gained_held_out()
    lists = generate_lists(test_settings.SYNTHETIC_LIST_SIZE)
    recall = compute_gain_recall(lists, held_out, test_settings.SYNTHETIC_LIST_SIZE)
    user_lists = collect_user_lists(lists)
    user_gains = collect_user_gains(held_out)
    assert sorted(recall.index.tolist()) == _select_users_with_positive_gain(user_gains)
    for user_id in recall.index.tolist():
        gains = user_gains[user_id]
        expected = _sum_listed_gain(user_lists[user_id], gains) / sum(gains.values())
        assert _is_close(float(recall.loc[user_id]), expected)


def _split_user_movies(gains: dict[int, float]) -> tuple[set[int], set[int]]:
    bad_movies: set[int] = set()
    for movie_id, gain in gains.items():
        if gain < settings.BAD_PERCENTILE_THRESHOLD:
            bad_movies.add(movie_id)
    return set(gains), bad_movies


def _check_user_bad_shares(
    user_id: int,
    gains: dict[int, float],
    listed_movies: list[int],
    bad_shares: BadShares,
) -> None:
    judged_movies, bad_movies = _split_user_movies(gains)
    bad_count = _count_listed(listed_movies, bad_movies)
    judged_count = _count_listed(listed_movies, judged_movies)
    per_slot = float(bad_shares.per_list_slot.loc[user_id])
    assert _is_close(per_slot, bad_count / test_settings.SYNTHETIC_LIST_SIZE)
    assert (user_id in bad_shares.per_judged_movie.index) == (judged_count > 0)
    if judged_count > 0:
        per_judged = float(bad_shares.per_judged_movie.loc[user_id])
        assert _is_close(per_judged, bad_count / judged_count)


def test_bad_shares_match_per_user_counts() -> None:
    held_out = _build_gained_held_out()
    lists = generate_lists(test_settings.SYNTHETIC_LIST_SIZE)
    list_size = test_settings.SYNTHETIC_LIST_SIZE
    bad_shares = compute_bad_shares(lists, held_out, list_size)
    user_lists = collect_user_lists(lists)
    percentiles = collect_user_values(held_out, settings.PERCENTILE_COLUMN)
    for user_id, user_percentiles in percentiles.items():
        _check_user_bad_shares(
            user_id,
            user_percentiles,
            user_lists[user_id],
            bad_shares,
        )
