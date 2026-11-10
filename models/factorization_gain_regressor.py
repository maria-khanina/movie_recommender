"""Factorization machine as the regressor of the expected gain.

Same role as the multilayer perceptron regressor: trained on rated history
pairs, with the percentile gain of each rating on the user's own scale as the
target and the squared error as the loss, so the prediction estimates
E[gain | user, movie, rated]. The features are the one-hot user and movie ids,
as in the production factorization machine. The last history ratings of every
user choose the epoch by their squared error.
"""
import numpy
import numpy.typing
import pandas
import scipy.sparse

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from evaluation.percentile_gains import attach_percentile_gains
from models import factorization_machine_settings as settings
from models.candidate_retrieval import MovieIds
from models.early_stopping import StoppingTracker, split_stopping_ratings
from models.factorization_features import FeatureSpace
from models.factorization_machine import (
    FactorizationMachine,
    MachineConfig,
    MachineParameters,
)

Gains = numpy.typing.NDArray[numpy.float64]


class _EncodedRatings:
    def __init__(self, space: FeatureSpace, gained: pandas.DataFrame) -> None:
        self.features = space.encode_pairs(gained)
        self.targets = numpy.asarray(gained[columns.GAIN_COLUMN], dtype=float)


class FactorizationGainRegressor:
    def __init__(self, config: MachineConfig) -> None:
        self._config = config
        self._generator = numpy.random.default_rng(columns.RANDOM_SEED)
        self._space: FeatureSpace | None = None
        self._machine: FactorizationMachine | None = None

    def _require_model(self) -> tuple[FeatureSpace, FactorizationMachine]:
        if self._space is None or self._machine is None:
            raise RuntimeError("The regressor is not fitted")
        return self._space, self._machine

    def _run_epoch(self, encoded: _EncodedRatings) -> None:
        _, machine = self._require_model()
        order = self._generator.permutation(len(encoded.targets))
        for start in range(0, len(order), settings.BATCH_SIZE):
            batch = order[start:start + settings.BATCH_SIZE]
            machine.update(encoded.features[batch], encoded.targets[batch])

    def _measure_negative_error(self, encoded: _EncodedRatings) -> float:
        _, machine = self._require_model()
        errors = machine.predict(encoded.features) - encoded.targets
        return -float(numpy.mean(numpy.square(errors)))

    def _restore_best(self, tracker: StoppingTracker[MachineParameters]) -> None:
        _, machine = self._require_model()
        machine.restore_parameters(tracker.best_states)

    def _train_until_stopping(
        self,
        fitting: _EncodedRatings,
        stopping: _EncodedRatings,
    ) -> None:
        _, machine = self._require_model()
        tracker = StoppingTracker(
            machine.copy_parameters(),
            settings.PATIENCE_IN_EPOCHS,
        )
        for _ in range(settings.MAXIMUM_EPOCHS):
            self._run_epoch(fitting)
            score = self._measure_negative_error(stopping)
            if tracker.record(score, machine.copy_parameters()):
                break
        self._restore_best(tracker)

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        stopping = split_stopping_ratings(
            training,
            settings.STOPPING_RATINGS_PER_USER,
        )
        self._space = FeatureSpace(stopping.fitting, tables)
        fitting_gains = attach_percentile_gains(stopping.fitting, stopping.fitting)
        fitting = _EncodedRatings(self._space, fitting_gains)
        self._machine = FactorizationMachine(
            self._space.feature_count,
            self._config,
            float(fitting.targets.mean()),
            self._generator,
        )
        stopping_ratings = _EncodedRatings(self._space, stopping.held_out)
        self._train_until_stopping(fitting, stopping_ratings)

    def predict(self, user_ids: MovieIds, movie_ids: MovieIds) -> Gains:
        """Return the expected gain of every (user, movie) pair."""
        space, machine = self._require_model()
        features: scipy.sparse.csr_array = space.encode_ids(user_ids, movie_ids)
        return machine.predict(features)
