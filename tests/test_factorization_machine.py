import numpy
import scipy.sparse

from models.factorization_machine import FactorizationMachine, MachineConfig
from tests import test_settings


def _build_machine_and_data() -> tuple[
    FactorizationMachine,
    scipy.sparse.csr_array,
    numpy.typing.NDArray[numpy.float64],
]:
    generator = numpy.random.default_rng(test_settings.SYNTHETIC_SEED)
    config = MachineConfig(
        factor_count=test_settings.GRADIENT_CHECK_FACTOR_COUNT,
        learning_rate=test_settings.GRADIENT_CHECK_LEARNING_RATE,
        weights_penalty=test_settings.GRADIENT_CHECK_PENALTY,
        factors_penalty=test_settings.GRADIENT_CHECK_PENALTY,
    )
    feature_count = test_settings.GRADIENT_CHECK_FEATURE_COUNT
    machine = FactorizationMachine(feature_count, config, 0.0, generator)
    machine.parameters.weights[:] = generator.normal(size=feature_count)
    dense = generator.random((test_settings.GRADIENT_CHECK_ROW_COUNT, feature_count))
    targets = generator.normal(size=test_settings.GRADIENT_CHECK_ROW_COUNT)
    return machine, scipy.sparse.csr_array(dense), targets


def _compute_loss(
    machine: FactorizationMachine,
    features: scipy.sparse.csr_array,
    targets: numpy.typing.NDArray[numpy.float64],
) -> float:
    errors = machine.predict(features) - targets
    penalty = machine.config.weights_penalty * numpy.square(
        machine.parameters.weights,
    ).sum() + machine.config.factors_penalty * numpy.square(
        machine.parameters.factors,
    ).sum()
    return float(numpy.mean(numpy.square(errors)) + penalty)


def _measure_shifted_loss(
    machine: FactorizationMachine,
    data: tuple[scipy.sparse.csr_array, numpy.typing.NDArray[numpy.float64]],
    parameter: numpy.typing.NDArray[numpy.float64],
    position: tuple[int, ...],
) -> float:
    step = test_settings.GRADIENT_CHECK_STEP
    original = float(parameter[position])
    parameter[position] = original + step
    loss_above = _compute_loss(machine, data[0], data[1])
    parameter[position] = original - step
    loss_below = _compute_loss(machine, data[0], data[1])
    parameter[position] = original
    return (loss_above - loss_below) / (2.0 * step)


def _estimate_numeric_gradient(
    machine: FactorizationMachine,
    features: scipy.sparse.csr_array,
    targets: numpy.typing.NDArray[numpy.float64],
    parameter: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    gradient = numpy.zeros_like(parameter)
    for position in numpy.ndindex(parameter.shape):
        gradient[position] = _measure_shifted_loss(
            machine,
            (features, targets),
            parameter,
            position,
        )
    return gradient


def test_analytic_gradients_match_numeric_gradients() -> None:
    machine, features, targets = _build_machine_and_data()
    analytic = machine.compute_gradients(features, targets)
    for parameter, analytic_gradient in zip(machine.parameters, analytic):
        numeric = _estimate_numeric_gradient(machine, features, targets, parameter)
        difference = numpy.abs(numeric - analytic_gradient).max()
        assert float(difference) < test_settings.GRADIENT_CHECK_TOLERANCE


def test_updates_reduce_training_loss() -> None:
    machine, features, targets = _build_machine_and_data()
    initial_loss = _compute_loss(machine, features, targets)
    for _ in range(test_settings.GRADIENT_CHECK_UPDATE_COUNT):
        machine.update(features, targets)
    assert _compute_loss(machine, features, targets) < initial_loss
