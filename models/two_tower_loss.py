"""In-batch softmax loss of the two-tower retriever with one explicit negative
per example.

Row r of a batch is a user and a liked movie. Its logits are the similarities of
the user to every liked movie of the batch, divided by the temperature; the
other rows' movies serve as negatives. Those movies enter the batch in
proportion to how often they are liked, so the log of that frequency is
subtracted from their logits (the logQ correction), otherwise popular movies are
pushed down just for being sampled often. Entries where the column movie is the
row's own movie or belongs to the same user are masked. One more column holds
the row's own explicit negative (a disliked or random movie); it is not sampled
by popularity, so it gets no correction, and adding log(weight) to its logit
counts it as if it appeared weight times in the softmax. A weight of 0 removes
the column.
"""
import math
from typing import NamedTuple

import torch
import torch.nn.functional

from models import two_tower_settings as settings


class BatchVectors(NamedTuple):
    users: torch.Tensor
    movies: torch.Tensor
    negatives: torch.Tensor


class BatchRows(NamedTuple):
    users: torch.Tensor
    movies: torch.Tensor
    negatives: torch.Tensor


def _compute_in_batch_logits(
    vectors: BatchVectors,
    rows: BatchRows,
    movie_log_frequencies: torch.Tensor,
    masked_pairs: torch.Tensor,
) -> torch.Tensor:
    logits = vectors.users @ vectors.movies.T / settings.TEMPERATURE
    logits = logits - movie_log_frequencies[rows.movies].unsqueeze(0)
    same_movie = rows.movies.unsqueeze(1) == rows.movies.unsqueeze(0)
    same_user = rows.users.unsqueeze(1) == rows.users.unsqueeze(0)
    diagonal = torch.eye(len(rows.users), dtype=torch.bool, device=logits.device)
    collisions = (same_movie | same_user | masked_pairs) & ~diagonal
    return logits.masked_fill(collisions, settings.MASKED_LOGIT)


def _compute_negative_logits(
    vectors: BatchVectors,
    rows: BatchRows,
    negative_weight: float,
) -> torch.Tensor:
    similarities = (vectors.users * vectors.negatives).sum(dim=1)
    logits = similarities / settings.TEMPERATURE + math.log(negative_weight)
    is_own_movie = rows.negatives == rows.movies
    return logits.masked_fill(is_own_movie, settings.MASKED_LOGIT).unsqueeze(1)


def compute_retrieval_loss(
    vectors: BatchVectors,
    rows: BatchRows,
    movie_log_frequencies: torch.Tensor,
    negative_weight: float,
    masked_pairs: torch.Tensor,
    example_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the weighted mean loss; masked_pairs marks (row, column) movies
    that must not count as negatives of that row, such as movies the row's user
    likes, and example_weights weighs the rows.
    """
    logits = _compute_in_batch_logits(
        vectors,
        rows,
        movie_log_frequencies,
        masked_pairs,
    )
    if negative_weight > 0:
        negative_logits = _compute_negative_logits(vectors, rows, negative_weight)
        logits = torch.cat([logits, negative_logits], dim=1)
    targets = torch.arange(len(rows.users), device=logits.device)
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    return (losses * example_weights).sum() / example_weights.sum()
