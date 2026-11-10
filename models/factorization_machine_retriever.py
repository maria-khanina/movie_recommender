"""Factorization machine used as a candidate retriever.

The machine is trained, as in the production pipeline, on one-hot user and movie
ids to predict the rating with the squared error. With only these two features
the prediction is w0 + w_u + w_i + <v_u, v_i>. For one user, w0 + w_u is the same
for every movie and does not change the order, so the order is given by
w_i + <v_u, v_i> = <[v_u, 1], [v_i, w_i]>: an inner product of a user vector and
a movie vector. The movie vectors go into the same approximate search index as
the two-tower vectors, and a query builds the user vector from the trained
factors. A user without a trained id gets the vector [0, 1], which ranks movies
by their weight alone. The epoch is chosen by the stopping recall of the shared
early stopping.
"""
import numpy
import numpy.typing
import pandas
import scipy.sparse

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from models import factorization_machine_settings as settings
from models.candidate_retrieval import MovieIds, SearchResult
from models.early_stopping import (
    StoppingData,
    StoppingTracker,
    measure_stopping_recall,
    split_stopping_ratings,
)
from models.factorization_features import FeatureSpace
from models.factorization_machine import (
    FactorizationMachine,
    MachineConfig,
    MachineParameters,
)
from models.vector_search import MovieVectorIndex, Vectors


def build_movie_vectors(
    parameters: MachineParameters,
    movie_features: scipy.sparse.csr_array,
) -> Vectors:
    projection = numpy.asarray(movie_features @ parameters.factors)
    squared_projection = numpy.asarray(
        movie_features.power(2) @ numpy.square(parameters.factors),
    )
    pair_terms = (numpy.square(projection) - squared_projection).sum(axis=1)
    linear = numpy.asarray(movie_features @ parameters.weights)
    movie_terms = linear + settings.PAIRWISE_SCALE * pair_terms
    return numpy.hstack([projection, movie_terms[:, numpy.newaxis]]).astype(
        numpy.float32,
    )


class FactorizationMachineRetriever:
    def __init__(self, config: MachineConfig) -> None:
        self._config = config
        self._generator = numpy.random.default_rng(columns.RANDOM_SEED)
        self._space: FeatureSpace | None = None
        self._machine: FactorizationMachine | None = None
        self._index: MovieVectorIndex | None = None
        self.stopping_recall = -1.0

    def _require_model(self) -> tuple[FeatureSpace, FactorizationMachine]:
        if self._space is None or self._machine is None:
            raise RuntimeError("The retriever is not fitted")
        return self._space, self._machine

    def _rebuild_index(self) -> None:
        space, machine = self._require_model()
        self._index = MovieVectorIndex(
            space.movie_index.to_numpy(dtype=numpy.int64),
            build_movie_vectors(
                machine.parameters,
                space.encode_movies(space.movie_index.to_numpy(dtype=numpy.int64)),
            ),
        )

    def _run_epoch(
        self,
        features: scipy.sparse.csr_array,
        targets: numpy.typing.NDArray[numpy.float64],
    ) -> None:
        _, machine = self._require_model()
        order = self._generator.permutation(len(targets))
        for start in range(0, len(order), settings.BATCH_SIZE):
            batch = order[start:start + settings.BATCH_SIZE]
            machine.update(features[batch], targets[batch])

    def _measure_recall(self, stopping: StoppingData) -> float:
        space, _ = self._require_model()
        self._rebuild_index()
        return measure_stopping_recall(
            self,
            stopping,
            settings.STOPPING_LIST_SIZE,
            len(space.movie_index),
        )

    def _encode_fitting(
        self,
        stopping: StoppingData,
    ) -> tuple[scipy.sparse.csr_array, numpy.typing.NDArray[numpy.float64]]:
        space, _ = self._require_model()
        features = space.encode_pairs(stopping.fitting)
        targets = numpy.asarray(stopping.fitting[columns.RATING_COLUMN], dtype=float)
        return features, targets

    def _restore_best(self, tracker: StoppingTracker[MachineParameters]) -> None:
        _, machine = self._require_model()
        machine.restore_parameters(tracker.best_states)
        self.stopping_recall = tracker.best_recall

    def _train_until_stopping(self, stopping: StoppingData) -> None:
        _, machine = self._require_model()
        features, targets = self._encode_fitting(stopping)
        tracker = StoppingTracker(
            machine.copy_parameters(),
            settings.PATIENCE_IN_EPOCHS,
        )
        for _ in range(settings.MAXIMUM_EPOCHS):
            self._run_epoch(features, targets)
            recall = self._measure_recall(stopping)
            if tracker.record(recall, machine.copy_parameters()):
                break
        self._restore_best(tracker)

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        stopping = split_stopping_ratings(
            training,
            settings.STOPPING_RATINGS_PER_USER,
        )
        self._space = FeatureSpace(stopping.fitting, tables)
        mean_rating = float(stopping.fitting[columns.RATING_COLUMN].mean())
        self._machine = FactorizationMachine(
            self._space.feature_count,
            self._config,
            mean_rating,
            self._generator,
        )
        self._train_until_stopping(stopping)
        self._rebuild_index()

    def _build_user_vectors(self, user_ids: MovieIds) -> Vectors:
        space, machine = self._require_model()
        factors = numpy.asarray(
            space.encode_users(user_ids) @ machine.parameters.factors,
        )
        ones = numpy.ones((len(user_ids), 1))
        return numpy.hstack([factors, ones]).astype(numpy.float32)

    def search(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
        depth: int,
    ) -> SearchResult:
        if self._index is None:
            raise RuntimeError("The retriever is not fitted")
        return self._index.search(self._build_user_vectors(user_ids), depth)
