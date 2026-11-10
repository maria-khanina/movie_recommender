"""Ranker trained on whole candidate lists labeled by a later block of ratings.

The lists are the retriever's candidates for each user, retrieved by a
retriever that never saw the labeling block, so the ranker learns on the same
kind of lists it will reorder. The network sees the user and movie features and
the retriever's score, standardized on the training lists. A share of users is
held out to choose the epoch by the NDCG@10 of their lists. The loss is one of:
the selection head (will the user rate this candidate), ListNet, LambdaRank,
weighted BPR or weighted hinge.
"""
import copy
import enum
from typing import NamedTuple, Protocol, cast

import numpy
import pandas
import torch

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from models import models_settings
from models import ranker_settings as settings
from models.candidate_lists import CandidateLists, ListLabels, ListValues
from models.candidate_retrieval import select_positive_ratings
from models.early_stopping import StoppingTracker
from models.list_losses import (
    compute_lambdarank_loss,
    compute_list_ndcg,
    compute_listnet_loss,
    compute_pairwise_loss,
    compute_selection_loss,
)
from models.ranker_network import RankerNetwork
from models.two_tower_features import (
    TowerFeatures,
    build_tower_features,
    find_rows,
    select_movie_inputs,
    select_user_inputs,
)

_NetworkState = dict[str, torch.Tensor]


class ListLoss(enum.StrEnum):
    SELECTION = "selection head"
    LISTNET = "ListNet"
    LAMBDARANK = "LambdaRank"
    BPR = "weighted BPR"
    HINGE = "weighted hinge"


class _DifferentiableLoss(Protocol):
    def backward(self) -> None: ...


class _ListTensors(NamedTuple):
    user_rows: torch.Tensor
    movie_rows: torch.Tensor
    scores: torch.Tensor
    gains: torch.Tensor
    rated: torch.Tensor
    valid: torch.Tensor


def _compute_loss(
    loss: ListLoss,
    scores: torch.Tensor,
    tensors: _ListTensors,
) -> torch.Tensor:
    if loss == ListLoss.SELECTION:
        return compute_selection_loss(scores, tensors.rated, tensors.valid)
    if loss == ListLoss.LISTNET:
        return compute_listnet_loss(scores, tensors.gains, tensors.valid)
    if loss == ListLoss.LAMBDARANK:
        return compute_lambdarank_loss(scores, tensors.gains, tensors.valid)
    uses_hinge = loss == ListLoss.HINGE
    return compute_pairwise_loss(scores, tensors.gains, tensors.valid, uses_hinge)


def _select_lists(tensors: _ListTensors, rows: torch.Tensor) -> _ListTensors:
    return _ListTensors(
        user_rows=tensors.user_rows[rows],
        movie_rows=tensors.movie_rows[rows],
        scores=tensors.scores[rows],
        gains=tensors.gains[rows],
        rated=tensors.rated[rows],
        valid=tensors.valid[rows],
    )


class ListRanker:
    def __init__(self, loss: ListLoss) -> None:
        self._loss = loss
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = numpy.random.default_rng(columns.RANDOM_SEED)
        self._features: TowerFeatures | None = None
        self._network: RankerNetwork | None = None
        self._score_center = 0.0
        self._score_scale = 1.0

    def _require_model(self) -> tuple[RankerNetwork, TowerFeatures]:
        if self._network is None or self._features is None:
            raise RuntimeError("The ranker is not fitted")
        return self._network, self._features

    def _to_tensors(self, lists: CandidateLists, labels: ListLabels) -> _ListTensors:
        _, features = self._require_model()
        movie_rows = find_rows(features.movie_index, lists.movie_ids.reshape(-1))
        standardized = (lists.scores - self._score_center) / self._score_scale
        return _ListTensors(
            user_rows=self._move(find_rows(features.user_index, lists.user_ids)),
            movie_rows=self._move(movie_rows.reshape(lists.movie_ids.shape)),
            scores=self._move(standardized.astype(numpy.float32)),
            gains=self._move(labels.gains),
            rated=self._move(labels.rated),
            valid=self._move(lists.valid),
        )

    def _move(self, values: numpy.typing.NDArray[numpy.generic]) -> torch.Tensor:
        return torch.as_tensor(values, device=self._device)

    def _score_lists(self, tensors: _ListTensors) -> torch.Tensor:
        network, features = self._require_model()
        list_size = tensors.movie_rows.shape[1]
        user_rows = tensors.user_rows.repeat_interleave(list_size)
        movie_rows = tensors.movie_rows.reshape(-1).clamp(min=0)
        scores: torch.Tensor = network(
            select_user_inputs(features, user_rows),
            select_movie_inputs(features, movie_rows),
            tensors.scores.reshape(-1),
        )
        return scores.view(-1, list_size)

    def _train_epoch(
        self,
        tensors: _ListTensors,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        network, _ = self._require_model()
        network.train()
        order = self._move(self._generator.permutation(len(tensors.user_rows)))
        for rows in torch.split(order, settings.LIST_BATCH_SIZE):
            batch = _select_lists(tensors, rows)
            optimizer.zero_grad()
            loss = _compute_loss(self._loss, self._score_lists(batch), batch)
            cast(_DifferentiableLoss, loss).backward()
            optimizer.step()

    def _measure_ndcg(self, tensors: _ListTensors) -> float:
        network, _ = self._require_model()
        network.eval()
        with torch.no_grad():
            scores = self._score_lists(tensors)
            ndcg = compute_list_ndcg(
                scores,
                tensors.gains,
                tensors.valid,
                settings.LIST_STOPPING_CUTOFF,
            )
        return float(ndcg.item())

    def _split_stopping_lists(
        self,
        tensors: _ListTensors,
    ) -> tuple[_ListTensors, _ListTensors]:
        list_count = len(tensors.user_rows)
        order = self._move(self._generator.permutation(list_count))
        stopping_count = int(list_count * settings.LIST_STOPPING_USER_SHARE)
        return (
            _select_lists(tensors, order[stopping_count:]),
            _select_lists(tensors, order[:stopping_count]),
        )

    def _train_until_stopping(self, tensors: _ListTensors) -> None:
        network, _ = self._require_model()
        fitting, stopping = self._split_stopping_lists(tensors)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY,
        )
        tracker = StoppingTracker(_copy_state(network), settings.PATIENCE_IN_EPOCHS)
        for _ in range(settings.MAXIMUM_EPOCHS):
            self._train_epoch(fitting, optimizer)
            if tracker.record(self._measure_ndcg(stopping), _copy_state(network)):
                break
        network.load_state_dict(tracker.best_states)

    def _build_network(
        self,
        training: pandas.DataFrame,
        tables: MovieLensTables,
    ) -> None:
        positives = select_positive_ratings(
            training,
            models_settings.POSITIVE_PERCENTILE_THRESHOLD,
        )
        self._features = build_tower_features(positives, tables, self._device)
        self._network = RankerNetwork(
            self._features.user_sizes,
            self._features.movie_sizes,
            True,
        ).to(self._device)

    def fit(
        self,
        training: pandas.DataFrame,
        tables: MovieLensTables,
        lists: CandidateLists,
        labels: ListLabels,
    ) -> None:
        """Train on labeled candidate lists; training ratings give the features."""
        self._build_network(training, tables)
        valid_scores = lists.scores[lists.valid]
        self._score_center = float(valid_scores.mean())
        self._score_scale = max(float(valid_scores.std()), settings.GAIN_SUM_FLOOR)
        self._train_until_stopping(self._to_tensors(lists, labels))

    def score(self, lists: CandidateLists) -> ListValues:
        """Return the ranker's score of every candidate of every list."""
        network, _ = self._require_model()
        network.eval()
        empty_labels = ListLabels(
            gains=numpy.zeros(lists.scores.shape, dtype=numpy.float32),
            rated=numpy.zeros(lists.scores.shape, dtype=numpy.bool_),
        )
        tensors = self._to_tensors(lists, empty_labels)
        with torch.no_grad():
            scores = self._score_lists(tensors)
        return numpy.asarray(scores.cpu(), dtype=numpy.float32)


def _copy_state(network: RankerNetwork) -> _NetworkState:
    return copy.deepcopy(network.state_dict())
