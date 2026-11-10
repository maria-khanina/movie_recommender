"""Disliked movies of the same user as explicit negatives of the history towers.

A disliked rating is one below the bad percentile threshold on the user's own
scale. Each training example gets at most one, drawn anew every batch:
- anywhere: a random disliked rating from the user's whole training history;
- nearby: a random disliked rating among the target_window ratings that start
  at the window end, the span the example's target comes from, so it belongs
  to the same rating session as the target.
An example without a disliked rating gets its own target movie, which the loss
masks. Everything is index arithmetic on the device.
"""
import enum
from typing import NamedTuple

import torch

from models.rating_sequences import RatedSequences


class DislikedNegatives(enum.StrEnum):
    NONE = "none"
    ANYWHERE = "anywhere"
    NEARBY = "nearby"


class DislikedIndex(NamedTuple):
    flat_rows: torch.Tensor
    starts: torch.Tensor
    counts: torch.Tensor


def index_disliked(sequences: RatedSequences) -> DislikedIndex:
    """Return the flat rows of the disliked ratings, grouped by user, with each
    user's start and count among them.
    """
    user_count = len(sequences.lengths)
    users = torch.arange(user_count, device=sequences.lengths.device)
    user_of_rating = torch.repeat_interleave(users, sequences.lengths)
    flat_rows = torch.nonzero(sequences.disliked).squeeze(1)
    counts = torch.bincount(user_of_rating[flat_rows], minlength=user_count)
    return DislikedIndex(flat_rows, torch.cumsum(counts, dim=0) - counts, counts)


def draw_anywhere(
    sequences: RatedSequences,
    index: DislikedIndex,
    user_rows: torch.Tensor,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    """Return a movie row per example: a random disliked movie of its user."""
    if len(index.flat_rows) == 0:
        return target_rows
    counts = index.counts[user_rows]
    draws = (torch.rand(len(user_rows), device=user_rows.device) * counts).long()
    offsets = (index.starts[user_rows] + draws).clamp(max=len(index.flat_rows) - 1)
    picked = sequences.movie_codes[index.flat_rows[offsets]] - 1
    return torch.where(counts > 0, picked, target_rows)


def draw_nearby(
    sequences: RatedSequences,
    user_rows: torch.Tensor,
    ends: torch.Tensor,
    target_rows: torch.Tensor,
    span: int,
) -> torch.Tensor:
    """Return a movie row per example: a random disliked movie among the span
    ratings that start at the window end.
    """
    positions = ends.unsqueeze(1) + torch.arange(span, device=ends.device)
    inside = positions < sequences.lengths[user_rows].unsqueeze(1)
    flat = sequences.starts[user_rows].unsqueeze(1) + positions
    flat = flat.clamp(max=len(sequences.movie_codes) - 1)
    eligible = sequences.disliked[flat] & inside
    scores = torch.rand(eligible.shape, device=ends.device).masked_fill(~eligible, -1)
    picked = flat.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
    movie_rows = sequences.movie_codes[picked] - 1
    return torch.where(eligible.any(dim=1), movie_rows, target_rows)
