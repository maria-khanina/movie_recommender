"""Second-order factorization machine trained with mini-batch gradient descent
and the Adam update.

For a sparse feature vector x the prediction is
w0 + sum_k w_k x_k + sum_{k<l} <v_k, v_l> x_k x_l, where the pairwise sum is
computed as 0.5 * sum_f ((x V)_f ** 2 - (x ** 2) (V ** 2)_f). The loss is the
mean squared error with L2 penalties on the linear weights and the factors. Adam
keeps exponential averages of each gradient and of its square, so the step does
not shrink forever the way the accumulated squares of AdaGrad make it shrink.
"""
from typing import NamedTuple

import numpy
import numpy.typing
import scipy.sparse

from models import factorization_machine_settings as settings

Vector = numpy.typing.NDArray[numpy.float64]
Matrix = numpy.typing.NDArray[numpy.float64]


class MachineConfig(NamedTuple):
    factor_count: int
    learning_rate: float
    weights_penalty: float
    factors_penalty: float


class MachineParameters(NamedTuple):
    bias: Vector
    weights: Vector
    factors: Matrix


class _MomentEstimates(NamedTuple):
    first: MachineParameters
    second: MachineParameters


def _zeros_like(parameters: MachineParameters) -> MachineParameters:
    return MachineParameters(
        bias=numpy.zeros_like(parameters.bias),
        weights=numpy.zeros_like(parameters.weights),
        factors=numpy.zeros_like(parameters.factors),
    )


def _adam_step(
    parameter: Matrix,
    gradient: Matrix,
    moments: tuple[Matrix, Matrix],
    step_index: int,
    learning_rate: float,
) -> None:
    first, second = moments
    first *= settings.ADAM_FIRST_MOMENT_DECAY
    first += (1.0 - settings.ADAM_FIRST_MOMENT_DECAY) * gradient
    second *= settings.ADAM_SECOND_MOMENT_DECAY
    second += (1.0 - settings.ADAM_SECOND_MOMENT_DECAY) * numpy.square(gradient)
    first_corrected = first / (1.0 - settings.ADAM_FIRST_MOMENT_DECAY**step_index)
    second_corrected = second / (1.0 - settings.ADAM_SECOND_MOMENT_DECAY**step_index)
    denominator = numpy.sqrt(second_corrected) + settings.ADAM_EPSILON
    parameter -= learning_rate * first_corrected / denominator


class FactorizationMachine:
    def __init__(
        self,
        feature_count: int,
        config: MachineConfig,
        target_mean: float,
        generator: numpy.random.Generator,
    ) -> None:
        self.config = config
        factor_count = config.factor_count
        scale = numpy.sqrt(2.0 / (feature_count + factor_count))
        self.parameters = MachineParameters(
            bias=numpy.array([target_mean]),
            weights=numpy.zeros(feature_count),
            factors=generator.normal(scale=scale, size=(feature_count, factor_count)),
        )
        self._moments = _MomentEstimates(
            first=_zeros_like(self.parameters),
            second=_zeros_like(self.parameters),
        )
        self._step_index = 0

    def _predict_with_projection(
        self,
        features: scipy.sparse.csr_array,
    ) -> tuple[Vector, Matrix]:
        projection = numpy.asarray(features @ self.parameters.factors)
        squared_projection = numpy.asarray(
            features.power(2) @ numpy.square(self.parameters.factors),
        )
        pair_terms = numpy.square(projection) - squared_projection
        pairwise = settings.PAIRWISE_SCALE * pair_terms.sum(axis=1)
        linear = numpy.asarray(features @ self.parameters.weights)
        return self.parameters.bias[0] + linear + pairwise, projection

    def predict(self, features: scipy.sparse.csr_array) -> Vector:
        return self._predict_with_projection(features)[0]

    def compute_gradients(
        self,
        features: scipy.sparse.csr_array,
        targets: Vector,
    ) -> MachineParameters:
        predictions, projection = self._predict_with_projection(features)
        loss_slopes = 2.0 * (predictions - targets) / len(targets)
        weights_penalty = 2.0 * self.config.weights_penalty * self.parameters.weights
        factors_penalty = 2.0 * self.config.factors_penalty * self.parameters.factors
        transposed = features.T
        squared_slopes = numpy.asarray(features.power(2).T @ loss_slopes)
        weighted_projection = loss_slopes[:, None] * projection
        factor_gradient = numpy.asarray(transposed @ weighted_projection)
        return MachineParameters(
            bias=numpy.array([loss_slopes.sum()]),
            weights=numpy.asarray(transposed @ loss_slopes) + weights_penalty,
            factors=factor_gradient
            - squared_slopes[:, None] * self.parameters.factors
            + factors_penalty,
        )

    def update(self, features: scipy.sparse.csr_array, targets: Vector) -> None:
        """Take one Adam step on a mini-batch."""
        gradients = self.compute_gradients(features, targets)
        self._step_index += 1
        for parameter, gradient, first, second in zip(
            self.parameters,
            gradients,
            self._moments.first,
            self._moments.second,
        ):
            _adam_step(
                parameter,
                gradient,
                (first, second),
                self._step_index,
                self.config.learning_rate,
            )

    def copy_parameters(self) -> MachineParameters:
        return MachineParameters(
            bias=self.parameters.bias.copy(),
            weights=self.parameters.weights.copy(),
            factors=self.parameters.factors.copy(),
        )

    def restore_parameters(self, parameters: MachineParameters) -> None:
        self.parameters.bias[:] = parameters.bias
        self.parameters.weights[:] = parameters.weights
        self.parameters.factors[:] = parameters.factors
