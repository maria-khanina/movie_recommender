"""Losses of rankers that learn on whole candidate lists.

Every tensor has one row per user's candidate list and one column per position;
valid marks the positions that hold a candidate, gains hold the percentile gain
of a candidate the user rated in the labeling block and 0 otherwise, and rated
marks those rated candidates.

- Selection head: binary cross-entropy of "the user rated this candidate".
- ListNet: cross-entropy between the softmax of the scores and the list's gains
  normalized to sum to one, for lists with a positive gain.
- LambdaRank: for every pair with different gains, the logistic pairwise loss
  weighted by how much NDCG would change if the two candidates swapped places
  in the current order, so errors near the top of the list count most.
- Weighted pairwise (BPR or hinge with margin 1): every pair where the better
  candidate has a higher gain, weighted by the gain difference; since the gain
  is 0 below the median of the user's scale, the better candidate of every pair
  is a liked movie.
"""
import torch
import torch.nn.functional

from models import ranker_settings as settings


def compute_selection_loss(
    scores: torch.Tensor,
    rated: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        scores,
        rated.float(),
        reduction="none",
    )
    return losses[valid].mean()


def compute_listnet_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    log_probabilities = torch.log_softmax(masked_scores, dim=1)
    log_probabilities = log_probabilities.masked_fill(~valid, 0.0)
    total_gains = gains.sum(dim=1, keepdim=True)
    has_gain = total_gains.squeeze(1) > 0
    targets = gains / total_gains.clamp(min=settings.GAIN_SUM_FLOOR)
    list_losses = -(targets * log_probabilities).sum(dim=1)
    return list_losses[has_gain].mean()


def _compute_discounts(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    order = torch.argsort(masked_scores, dim=1, descending=True)
    positions = torch.argsort(order, dim=1).float()
    return 1.0 / torch.log2(positions + 2.0)


def _compute_ideal_dcg(gains: torch.Tensor) -> torch.Tensor:
    sorted_gains = torch.sort(gains, dim=1, descending=True).values
    positions = torch.arange(gains.shape[1], device=gains.device).float()
    return (sorted_gains / torch.log2(positions + 2.0)).sum(dim=1)


def _find_ordered_pairs(gains: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    both_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    return (gains.unsqueeze(2) > gains.unsqueeze(1)) & both_valid


def compute_lambdarank_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    discounts = _compute_discounts(scores.detach(), valid)
    ideal_dcg = _compute_ideal_dcg(gains).clamp(min=settings.GAIN_SUM_FLOOR)
    gain_gaps = (gains.unsqueeze(2) - gains.unsqueeze(1)).abs()
    discount_gaps = (discounts.unsqueeze(2) - discounts.unsqueeze(1)).abs()
    swap_weights = gain_gaps * discount_gaps / ideal_dcg.view(-1, 1, 1)
    score_gaps = scores.unsqueeze(2) - scores.unsqueeze(1)
    pair_losses = torch.nn.functional.softplus(-score_gaps) * swap_weights
    is_pair = _find_ordered_pairs(gains, valid)
    return pair_losses[is_pair].sum() / len(scores)


def compute_pairwise_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    valid: torch.Tensor,
    uses_hinge: bool,
) -> torch.Tensor:
    is_pair = _find_ordered_pairs(gains, valid)
    is_pair &= gains.unsqueeze(2) > 0
    weights = (gains.unsqueeze(2) - gains.unsqueeze(1))[is_pair]
    score_gaps = (scores.unsqueeze(2) - scores.unsqueeze(1))[is_pair]
    if uses_hinge:
        pair_losses = torch.relu(1.0 - score_gaps)
    else:
        pair_losses = torch.nn.functional.softplus(-score_gaps)
    return (weights * pair_losses).sum() / weights.sum().clamp(min=1.0)


def compute_list_ndcg(
    scores: torch.Tensor,
    gains: torch.Tensor,
    valid: torch.Tensor,
    cutoff: int,
) -> torch.Tensor:
    """Return the mean NDCG@cutoff of the lists with a positive gain, when every
    list is ordered by its scores.
    """
    discounts = _compute_discounts(scores, valid)
    is_counted = discounts >= 1.0 / torch.log2(torch.tensor(cutoff + 1.0))
    dcg = (gains * discounts * is_counted).sum(dim=1)
    sorted_gains = torch.sort(gains, dim=1, descending=True).values[:, :cutoff]
    ideal_dcg = _compute_ideal_dcg(sorted_gains)
    has_gain = ideal_dcg > 0
    return (dcg[has_gain] / ideal_dcg[has_gain]).mean()
