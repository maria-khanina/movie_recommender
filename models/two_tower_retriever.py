"""Two-tower candidate retriever.

The last ratings of every user's training history are held back as a stopping
set; the towers are trained on the liked movies of the rest with the in-batch
softmax loss, and after every epoch the gain-weighted recall of the stopping set
is measured through the same retrieval path the evaluation uses (approximate
search, known movies removed). Training stops when that recall has not improved
for a number of epochs, and the best epoch's weights are kept. The stopping set
comes from the training history itself, so the validation block is not used to
choose the epoch. A share of examples have their user id hidden, which teaches
the user tower to serve users it has no id embedding for.
"""
import copy
import logging
from typing import Protocol, cast

import numpy
import pandas
import torch

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from models import two_tower_settings as settings
from models.candidate_retrieval import (
    MovieIds,
    SearchResult,
    select_positive_ratings,
)
from models.early_stopping import (
    StoppingData,
    StoppingTracker,
    measure_stopping_recall,
    split_stopping_ratings,
)
from models.two_tower_examples import (
    TrainingExamples,
    assign_hard_negatives,
    build_training_examples,
)
from models.two_tower_features import (
    TowerFeatures,
    build_tower_features,
    find_rows,
    select_movie_inputs,
    select_user_inputs,
)
from models.two_tower_loss import BatchRows, BatchVectors, compute_retrieval_loss
from models.two_tower_network import MovieTower, UserTower
from models.vector_search import MovieVectorIndex

_logger = logging.getLogger(__name__)

_TowerStates = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


class _DifferentiableLoss(Protocol):
    def backward(self) -> None: ...


class _Seeding(Protocol):
    def manual_seed(self, seed: int, /) -> torch.Generator: ...


_SEEDING = cast(_Seeding, torch)


def _compute_log_frequencies(
    positives: pandas.DataFrame,
    features: TowerFeatures,
    device: torch.device,
) -> torch.Tensor:
    counts = positives.groupby(columns.MOVIE_COLUMN).size()
    catalog_counts = counts.reindex(features.movie_index, fill_value=0).to_numpy()
    smoothed = catalog_counts.astype(numpy.float64) + 1.0
    frequencies = torch.as_tensor(smoothed / smoothed.sum(), dtype=torch.float32)
    return torch.log(frequencies).to(device)


class TwoTowerRetriever:
    def __init__(self, negative_weight: float, positive_threshold: float) -> None:
        self._negative_weight = negative_weight
        self._positive_threshold = positive_threshold
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = numpy.random.default_rng(columns.RANDOM_SEED)
        self._features: TowerFeatures | None = None
        self._user_tower: UserTower | None = None
        self._movie_tower: MovieTower | None = None
        self._index: MovieVectorIndex | None = None
        self.stopping_recall = -1.0

    def _require_towers(self) -> tuple[UserTower, MovieTower, TowerFeatures]:
        if self._user_tower is None or self._movie_tower is None:
            raise RuntimeError("The retriever is not fitted")
        if self._features is None:
            raise RuntimeError("The retriever is not fitted")
        return self._user_tower, self._movie_tower, self._features

    def _build_towers(self, features: TowerFeatures) -> torch.optim.Optimizer:
        user_tower = UserTower(features.user_sizes).to(self._device)
        movie_tower = MovieTower(features.movie_sizes)
        self._user_tower, self._movie_tower = user_tower, movie_tower.to(self._device)
        parameters: list[torch.nn.Parameter] = [
            *user_tower.parameters(),
            *self._movie_tower.parameters(),
        ]
        return torch.optim.AdamW(
            parameters,
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY,
        )

    def _hide_user_ids(self, user_codes: torch.Tensor) -> torch.Tensor:
        hidden = torch.rand(len(user_codes), device=self._device)
        is_hidden = hidden < settings.USER_DROPOUT_RATIO
        return user_codes.masked_fill(is_hidden, 0)

    def _encode_batch(self, rows: BatchRows) -> BatchVectors:
        user_tower, movie_tower, features = self._require_towers()
        user_inputs = select_user_inputs(features, rows.users)
        user_inputs = user_inputs._replace(
            user_codes=self._hide_user_ids(user_inputs.user_codes),
        )
        return BatchVectors(
            users=user_tower(user_inputs),
            movies=movie_tower(select_movie_inputs(features, rows.movies)),
            negatives=movie_tower(select_movie_inputs(features, rows.negatives)),
        )

    def _train_batch(
        self,
        rows: BatchRows,
        optimizer: torch.optim.Optimizer,
        log_frequencies: torch.Tensor,
    ) -> None:
        optimizer.zero_grad()
        vectors = self._encode_batch(rows)
        loss = compute_retrieval_loss(
            vectors,
            rows,
            log_frequencies,
            self._negative_weight,
            torch.zeros(
                (len(rows.users), len(rows.users)),
                dtype=torch.bool,
                device=self._device,
            ),
            torch.ones(len(rows.users), device=self._device),
        )
        cast(_DifferentiableLoss, loss).backward()
        optimizer.step()

    def _run_epoch(
        self,
        examples: TrainingExamples,
        optimizer: torch.optim.Optimizer,
        log_frequencies: torch.Tensor,
    ) -> None:
        user_tower, movie_tower, features = self._require_towers()
        user_tower.train()
        movie_tower.train()
        negatives = assign_hard_negatives(
            examples,
            self._generator,
            len(features.movie_index),
        )
        order = self._generator.permutation(len(examples.user_rows))
        for start in range(0, len(order), settings.BATCH_SIZE):
            batch = order[start:start + settings.BATCH_SIZE]
            rows = self._move_rows(examples, negatives, batch)
            self._train_batch(rows, optimizer, log_frequencies)

    def _move_rows(
        self,
        examples: TrainingExamples,
        negatives: MovieIds,
        batch: numpy.typing.NDArray[numpy.int64],
    ) -> BatchRows:
        return BatchRows(
            users=torch.as_tensor(examples.user_rows[batch], device=self._device),
            movies=torch.as_tensor(examples.movie_rows[batch], device=self._device),
            negatives=torch.as_tensor(negatives[batch], device=self._device),
        )

    def _rebuild_index(self) -> None:
        user_tower, movie_tower, features = self._require_towers()
        user_tower.eval()
        movie_tower.eval()
        with torch.no_grad():
            movie_vectors: torch.Tensor = movie_tower(features.movies)
        self._index = MovieVectorIndex(
            features.movie_index.to_numpy(dtype=numpy.int64),
            numpy.asarray(movie_vectors.cpu(), dtype=numpy.float32),
        )

    def _measure_stopping_recall(self, stopping: StoppingData) -> float:
        self._rebuild_index()
        return measure_stopping_recall(
            self,
            stopping,
            settings.STOPPING_LIST_SIZE,
            len(self._require_features().movie_index),
        )

    def _copy_states(self) -> _TowerStates:
        user_tower, movie_tower, _ = self._require_towers()
        return (
            copy.deepcopy(user_tower.state_dict()),
            copy.deepcopy(movie_tower.state_dict()),
        )

    def _restore_states(self, states: _TowerStates) -> None:
        user_tower, movie_tower, _ = self._require_towers()
        user_tower.load_state_dict(states[0])
        movie_tower.load_state_dict(states[1])

    def _train_until_stopping(
        self,
        examples: TrainingExamples,
        stopping: StoppingData,
        log_frequencies: torch.Tensor,
    ) -> None:
        optimizer = self._build_towers(self._require_features())
        tracker = StoppingTracker(self._copy_states(), settings.PATIENCE_IN_EPOCHS)
        for epoch_index in range(settings.MAXIMUM_EPOCHS):
            self._run_epoch(examples, optimizer, log_frequencies)
            recall = self._measure_stopping_recall(stopping)
            _logger.info(f"Epoch {epoch_index + 1}: stopping gain recall {recall:.4f}")
            if tracker.record(recall, self._copy_states()):
                break
        self._restore_states(tracker.best_states)
        self.stopping_recall = tracker.best_recall

    def _require_features(self) -> TowerFeatures:
        if self._features is None:
            raise RuntimeError("The retriever has no features")
        return self._features

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        _SEEDING.manual_seed(columns.RANDOM_SEED)
        stopping = split_stopping_ratings(
            training,
            settings.STOPPING_RATINGS_PER_USER,
        )
        positives = select_positive_ratings(stopping.fitting, self._positive_threshold)
        self._features = build_tower_features(positives, tables, self._device)
        examples = build_training_examples(stopping.fitting, positives, self._features)
        log_frequencies = _compute_log_frequencies(
            positives,
            self._features,
            self._device,
        )
        self._train_until_stopping(examples, stopping, log_frequencies)
        self._rebuild_index()

    def search(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
        depth: int,
    ) -> SearchResult:
        user_tower, _, features = self._require_towers()
        if self._index is None:
            raise RuntimeError("The retriever is not fitted")
        user_rows = torch.as_tensor(
            find_rows(features.user_index, user_ids),
            device=self._device,
        )
        with torch.no_grad():
            user_vectors: torch.Tensor = user_tower(
                select_user_inputs(features, user_rows),
            )
        return self._index.search(
            numpy.asarray(user_vectors.cpu(), dtype=numpy.float32),
            depth,
        )
