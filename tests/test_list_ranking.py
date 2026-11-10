import math

import numpy
import pandas
import torch

from evaluation import evaluation_settings as settings
from evaluation.ranking_metrics import compute_ndcg
from models.candidate_lists import (
    build_candidate_lists,
    label_candidate_lists,
    rerank_lists,
)
from models.list_losses import (
    compute_lambdarank_loss,
    compute_list_ndcg,
    compute_listnet_loss,
    compute_pairwise_loss,
)
from tests import test_settings
from tests.synthetic_ratings import generate_lists


def _build_random_lists() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(test_settings.SYNTHETIC_SEED)
    shape = (test_settings.SYNTHETIC_USER_COUNT, test_settings.SYNTHETIC_LIST_SIZE)
    gains = torch.rand(shape, generator=generator)
    rated_draws = torch.rand(shape, generator=generator)
    is_rated = rated_draws < test_settings.RATED_CANDIDATE_SHARE
    gains = gains * is_rated
    scores = torch.randn(shape, generator=generator)
    return scores, gains, torch.ones(shape, dtype=torch.bool)


def test_list_ndcg_is_one_for_scores_equal_to_gains() -> None:
    _, gains, valid = _build_random_lists()
    cutoff = test_settings.SYNTHETIC_LIST_SIZE // 2
    ndcg = compute_list_ndcg(gains, gains, valid, cutoff)
    assert math.isclose(float(ndcg), 1.0, abs_tol=test_settings.SCORE_TOLERANCE)


def test_losses_prefer_scores_that_follow_the_gains() -> None:
    scores, gains, valid = _build_random_lists()
    ideal_scores = gains * test_settings.IDEAL_SCORE_SCALE
    for loss in (compute_listnet_loss, compute_lambdarank_loss):
        ideal = loss(ideal_scores, gains, valid)
        assert float(ideal) < float(loss(scores, gains, valid))
    for uses_hinge in (False, True):
        ideal = compute_pairwise_loss(ideal_scores, gains, valid, uses_hinge)
        random = compute_pairwise_loss(scores, gains, valid, uses_hinge)
        assert float(ideal) < float(random)


def _build_gained_lists() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    candidates = generate_lists(test_settings.SYNTHETIC_LIST_SIZE)
    generator = numpy.random.default_rng(test_settings.SYNTHETIC_SEED)
    candidates["score"] = generator.random(len(candidates))
    labeled = candidates.sample(
        frac=test_settings.RATED_CANDIDATE_SHARE,
        random_state=test_settings.SYNTHETIC_SEED,
    )
    labeled = labeled.assign(gain=generator.random(len(labeled)))
    return candidates, labeled


def test_reranking_by_gains_gives_the_ideal_ndcg() -> None:
    candidates, labeled = _build_gained_lists()
    lists = build_candidate_lists(candidates, test_settings.SYNTHETIC_LIST_SIZE)
    labels = label_candidate_lists(lists, labeled)
    reranked = rerank_lists(lists, labels.gains)
    ndcg = compute_ndcg(reranked, labeled, settings.RECOMMENDATION_LIST_SIZE)
    assert len(reranked) == len(candidates)
    assert int(labels.rated.sum()) == len(labeled)
    assert all(math.isclose(value, 1.0) for value in ndcg.tolist())
