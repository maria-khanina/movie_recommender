import numpy
import numpy.typing

from evaluation import evaluation_settings as settings
from models.factorization_features import FeatureSpace
from models.factorization_machine import FactorizationMachine, MachineConfig
from models.factorization_machine_retriever import build_movie_vectors
from tests import test_settings
from tests.synthetic_ratings import build_synthetic_tables, generate_ratings


def _build_machine(space: FeatureSpace) -> FactorizationMachine:
    config = MachineConfig(test_settings.GRADIENT_CHECK_FACTOR_COUNT, 0.0, 0.0, 0.0)
    generator = numpy.random.default_rng(test_settings.SYNTHETIC_SEED)
    machine = FactorizationMachine(space.feature_count, config, 0.0, generator)
    machine.parameters.weights[:] = generator.normal(size=space.feature_count)
    return machine


def _build_user_vector(
    space: FeatureSpace,
    machine: FactorizationMachine,
    user_ids: numpy.typing.NDArray[numpy.int64],
) -> numpy.typing.NDArray[numpy.float64]:
    user_factors = space.encode_users(user_ids) @ machine.parameters.factors
    return numpy.append(numpy.asarray(user_factors), 1.0)


def test_retrieval_vectors_reproduce_the_machine_order() -> None:
    ratings = generate_ratings()
    space = FeatureSpace(ratings, build_synthetic_tables(ratings))
    machine = _build_machine(space)
    movie_ids = space.movie_index.to_numpy(dtype=numpy.int64)
    movie_vectors = build_movie_vectors(
        machine.parameters,
        space.encode_movies(movie_ids),
    )
    user_ids = numpy.full(len(movie_ids), ratings[settings.USER_COLUMN].iloc[0])
    full_scores = machine.predict(space.encode_ids(user_ids, movie_ids))
    user_vector = _build_user_vector(space, machine, user_ids[:1])
    spread = numpy.ptp(full_scores - movie_vectors.astype(numpy.float64) @ user_vector)
    assert float(spread) < test_settings.SCORE_TOLERANCE
