"""Regressor of the expected gain of a movie for a user, given that the user
watches and rates it.

The target of every training rating is its percentile gain on the user's own
scale, and the loss is the squared error, whose minimum is the conditional mean:
the prediction estimates E[gain | user, movie, rated]. It is trained on rated
pairs only, so it says how much a user would like a movie, not whether the user
would choose it. The last training ratings of every user are held back to choose
the epoch by their squared error.
"""
import copy
from typing import NamedTuple, Protocol, cast

import numpy
import numpy.typing
import pandas
import torch

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from evaluation.percentile_gains import attach_percentile_gains
from models import ranker_settings as settings
from models import models_settings
from models.candidate_retrieval import MovieIds, select_positive_ratings
from models.early_stopping import StoppingTracker, split_stopping_ratings
from models.ranker_network import RankerNetwork
from models.two_tower_features import (
    TowerFeatures,
    build_tower_features,
    find_rows,
    select_movie_inputs,
    select_user_inputs,
)

_NetworkState = dict[str, torch.Tensor]
Gains = numpy.typing.NDArray[numpy.float64]


class _DifferentiableLoss(Protocol):
    def backward(self) -> None: ...


class _RatedPairs(NamedTuple):
    user_rows: torch.Tensor
    movie_rows: torch.Tensor
    gains: torch.Tensor


class GainRegressor:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = numpy.random.default_rng(columns.RANDOM_SEED)
        self._features: TowerFeatures | None = None
        self._network: RankerNetwork | None = None

    def _require_model(self) -> tuple[RankerNetwork, TowerFeatures]:
        if self._network is None or self._features is None:
            raise RuntimeError("The regressor is not fitted")
        return self._network, self._features

    def _build_pairs(self, ratings: pandas.DataFrame) -> _RatedPairs:
        _, features = self._require_model()
        return _RatedPairs(
            user_rows=torch.as_tensor(
                find_rows(features.user_index, ratings[columns.USER_COLUMN]),
                device=self._device,
            ),
            movie_rows=torch.as_tensor(
                find_rows(features.movie_index, ratings[columns.MOVIE_COLUMN]),
                device=self._device,
            ),
            gains=torch.as_tensor(
                ratings[columns.GAIN_COLUMN].to_numpy(dtype=numpy.float32),
                device=self._device,
            ),
        )

    def _predict_rows(
        self,
        user_rows: torch.Tensor,
        movie_rows: torch.Tensor,
    ) -> torch.Tensor:
        network, features = self._require_model()
        empty_scores = torch.zeros(len(user_rows), device=self._device)
        predictions: torch.Tensor = network(
            select_user_inputs(features, user_rows),
            select_movie_inputs(features, movie_rows),
            empty_scores,
        )
        return predictions

    def _train_epoch(
        self,
        pairs: _RatedPairs,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        network, _ = self._require_model()
        network.train()
        order = torch.as_tensor(
            self._generator.permutation(len(pairs.gains)),
            device=self._device,
        )
        for batch in torch.split(order, settings.REGRESSION_BATCH_SIZE):
            optimizer.zero_grad()
            predictions = self._predict_rows(
                pairs.user_rows[batch],
                pairs.movie_rows[batch],
            )
            loss = torch.nn.functional.mse_loss(predictions, pairs.gains[batch])
            cast(_DifferentiableLoss, loss).backward()
            optimizer.step()

    def _measure_negative_error(self, pairs: _RatedPairs) -> float:
        network, _ = self._require_model()
        network.eval()
        with torch.no_grad():
            predictions = self._predict_rows(pairs.user_rows, pairs.movie_rows)
            error = torch.nn.functional.mse_loss(predictions, pairs.gains)
        return -float(error.item())

    def _train_until_stopping(
        self,
        fitting: _RatedPairs,
        stopping: _RatedPairs,
    ) -> None:
        network, _ = self._require_model()
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY,
        )
        tracker = StoppingTracker(_copy_state(network), settings.PATIENCE_IN_EPOCHS)
        for _ in range(settings.MAXIMUM_EPOCHS):
            self._train_epoch(fitting, optimizer)
            score = self._measure_negative_error(stopping)
            if tracker.record(score, _copy_state(network)):
                break
        network.load_state_dict(tracker.best_states)

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        stopping = split_stopping_ratings(
            training,
            settings.REGRESSION_STOPPING_RATINGS_PER_USER,
        )
        positives = select_positive_ratings(
            stopping.fitting,
            models_settings.POSITIVE_PERCENTILE_THRESHOLD,
        )
        self._features = build_tower_features(positives, tables, self._device)
        self._network = RankerNetwork(
            self._features.user_sizes,
            self._features.movie_sizes,
            False,
        ).to(self._device)
        fitting_gains = attach_percentile_gains(stopping.fitting, stopping.fitting)
        self._train_until_stopping(
            self._build_pairs(fitting_gains),
            self._build_pairs(stopping.held_out),
        )

    def predict(self, user_ids: MovieIds, movie_ids: MovieIds) -> Gains:
        """Return the expected gain of every (user, movie) pair."""
        network, features = self._require_model()
        network.eval()
        user_rows = torch.as_tensor(find_rows(features.user_index, user_ids))
        movie_rows = torch.as_tensor(find_rows(features.movie_index, movie_ids))
        with torch.no_grad():
            predictions = self._predict_rows(
                user_rows.to(self._device),
                movie_rows.to(self._device),
            )
        return numpy.asarray(predictions.cpu(), dtype=numpy.float64)


def _copy_state(network: RankerNetwork) -> _NetworkState:
    return copy.deepcopy(network.state_dict())
