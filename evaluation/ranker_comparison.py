"""Comparison of rankers on the validation stage.

The retriever named in the settings is fitted on the history block and builds
three sets of candidate lists: for the ranker block of known users (excluding
movies rated in history), for the validation block of known users (excluding
movies rated in history or in the ranker block) and for new users (excluding
their first ratings). Rankers that learn on candidate lists are trained on the
ranker-block lists with ranker-block labels; the gain regressors (multilayer
perceptron and factorization machine) are trained on rated history pairs. Every
ranker reorders the same validation lists, the top ten is scored by segment,
and every ranker is compared with the retriever's own order on the same users.
The test block is never read here.
"""
from collections.abc import Callable
import os
import pathlib
from typing import NamedTuple, Protocol

import numpy
import numpy.typing
import pandas

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as settings
from evaluation import ranker_comparison_settings as comparison_settings
from evaluation.ranker_report import (
    OrderMetrics,
    RankerReport,
    format_report,
    measure_order,
    summarize_segments,
)
from evaluation.retriever_factories import list_retriever_factories
from evaluation.validation_data import ValidationData, prepare_validation_data
from models.candidate_lists import (
    CandidateLists,
    ListLabels,
    ListValues,
    build_candidate_lists,
    label_candidate_lists,
    rerank_lists,
)
from models.candidate_retrieval import (
    CandidateRetriever,
    MovieIds,
    retrieve_unseen_candidates,
)
from models.factorization_gain_regressor import FactorizationGainRegressor
from models.factorization_machine import MachineConfig
from models.gain_regressor import GainRegressor
from models.list_ranker import ListLoss, ListRanker
from models.ranker_settings import MINIMUM_EXPECTED_GAIN, MINIMUM_PROBABILITY

ListScorer = Callable[[CandidateLists], ListValues]


class GainPredictor(Protocol):
    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None: ...

    def predict(
        self,
        user_ids: MovieIds,
        movie_ids: MovieIds,
    ) -> numpy.typing.NDArray[numpy.float64]: ...


class PreparedLists(NamedTuple):
    ranker_lists: CandidateLists
    ranker_labels: ListLabels
    known_lists: CandidateLists
    new_lists: dict[int, CandidateLists]


class _Evaluation(NamedTuple):
    data: ValidationData
    prepared: PreparedLists
    known_reference: OrderMetrics
    new_references: dict[int, OrderMetrics]


def _retrieve_lists(
    retriever: CandidateRetriever,
    labeled: pandas.DataFrame,
    history: pandas.DataFrame,
    catalog_size: int,
) -> CandidateLists:
    user_ids = labeled[settings.USER_COLUMN].unique().astype(numpy.int64)
    list_size = comparison_settings.CANDIDATE_LIST_SIZE
    candidates = retrieve_unseen_candidates(
        retriever,
        user_ids,
        history,
        list_size,
        catalog_size,
    )
    return build_candidate_lists(candidates, list_size)


def prepare_lists(data: ValidationData) -> PreparedLists:
    factory = list_retriever_factories()[comparison_settings.RETRIEVER_NAME]
    retriever = factory()
    retriever.fit(data.training, data.tables)
    size = len(data.tables.movies)
    ranker_lists = _retrieve_lists(retriever, data.ranker_labels, data.training, size)
    return PreparedLists(
        ranker_lists=ranker_lists,
        ranker_labels=label_candidate_lists(ranker_lists, data.ranker_labels),
        known_lists=_retrieve_lists(retriever, data.held_out, data.known, size),
        new_lists=_retrieve_new_user_lists(retriever, data, size),
    )


def _retrieve_new_user_lists(
    retriever: CandidateRetriever,
    data: ValidationData,
    catalog_size: int,
) -> dict[int, CandidateLists]:
    lists: dict[int, CandidateLists] = {}
    for history_size, history in data.new_user_histories.items():
        lists[history_size] = _retrieve_lists(
            retriever,
            data.new_user_held_out,
            history,
            catalog_size,
        )
    return lists


def _measure(
    lists: CandidateLists,
    scores: ListValues,
    held_out: pandas.DataFrame,
    data: ValidationData,
) -> OrderMetrics:
    return measure_order(rerank_lists(lists, scores), held_out, data)


def evaluate_order(
    name: str,
    scorer: ListScorer,
    evaluation: _Evaluation,
) -> RankerReport:
    data, prepared = evaluation.data, evaluation.prepared
    known = _measure(
        prepared.known_lists,
        scorer(prepared.known_lists),
        data.held_out,
        data,
    )
    new: dict[int, tuple[OrderMetrics, OrderMetrics]] = {}
    for history_size, lists in prepared.new_lists.items():
        metrics = _measure(lists, scorer(lists), data.new_user_held_out, data)
        new[history_size] = (metrics, evaluation.new_references[history_size])
    segments = summarize_segments((known, evaluation.known_reference), new, data)
    print(f"Evaluated {name}", flush=True)
    return RankerReport(name, segments, known.novelty, known.coverage)


def predict_log_gains(regressor: GainPredictor, lists: CandidateLists) -> ListValues:
    users = numpy.repeat(lists.user_ids, lists.movie_ids.shape[1])
    movies = numpy.maximum(lists.movie_ids.reshape(-1), 0)
    gains = regressor.predict(users, movies).reshape(lists.movie_ids.shape)
    return numpy.log(numpy.maximum(gains, MINIMUM_EXPECTED_GAIN)).astype(numpy.float32)


def _compute_log_probabilities(logits: ListValues) -> ListValues:
    probabilities = 1.0 / (1.0 + numpy.exp(-logits.astype(numpy.float64)))
    clipped = numpy.maximum(probabilities, MINIMUM_PROBABILITY)
    return numpy.log(clipped).astype(numpy.float32)


def bind_composition(regressor: GainPredictor, weight: float) -> ListScorer:
    def score(lists: CandidateLists) -> ListValues:
        log_gains = predict_log_gains(regressor, lists)
        return (log_gains + weight * lists.scores).astype(numpy.float32)
    return score


def bind_selection_composition(
    head: ListRanker,
    regressor: GainPredictor,
    weight: float,
) -> ListScorer:
    def score(lists: CandidateLists) -> ListValues:
        log_selection = _compute_log_probabilities(head.score(lists))
        log_gains = predict_log_gains(regressor, lists)
        return (log_selection + weight * log_gains).astype(numpy.float32)
    return score


def _keep_retriever_order(lists: CandidateLists) -> ListValues:
    return lists.scores


def list_gain_regressors() -> dict[str, GainPredictor]:
    factorization_config = MachineConfig(
        factor_count=comparison_settings.FACTORIZATION_FACTOR_COUNT,
        learning_rate=comparison_settings.FACTORIZATION_LEARNING_RATE,
        weights_penalty=comparison_settings.FACTORIZATION_PENALTY,
        factors_penalty=comparison_settings.FACTORIZATION_PENALTY,
    )
    return {
        "MLP": GainRegressor(),
        "FM": FactorizationGainRegressor(factorization_config),
    }


def _fit_list_ranker(loss: ListLoss, evaluation: _Evaluation) -> ListRanker:
    data, prepared = evaluation.data, evaluation.prepared
    ranker = ListRanker(loss)
    ranker.fit(
        data.training,
        data.tables,
        prepared.ranker_lists,
        prepared.ranker_labels,
    )
    return ranker


def _list_regressor_scorers(
    name: str,
    regressor: GainPredictor,
    head: ListRanker,
) -> dict[str, ListScorer]:
    scorers: dict[str, ListScorer] = {}
    for weight in comparison_settings.RETRIEVAL_SCORE_WEIGHTS:
        label = f"A ({name}): log expected gain + {weight} * retriever score"
        scorers[label] = bind_composition(regressor, weight)
    for weight in comparison_settings.EXPECTED_GAIN_WEIGHTS:
        label = (
            f"B1 ({name}): log selection probability "
            f"+ {weight} * log expected gain"
        )
        scorers[label] = bind_selection_composition(head, regressor, weight)
    return scorers


def _evaluate_regressor(
    name: str,
    regressor: GainPredictor,
    head: ListRanker,
    evaluation: _Evaluation,
) -> list[RankerReport]:
    regressor.fit(evaluation.data.training, evaluation.data.tables)
    reports: list[RankerReport] = []
    for label, scorer in _list_regressor_scorers(name, regressor, head).items():
        reports.append(evaluate_order(label, scorer, evaluation))
    return reports


def _prepare_evaluation(data: ValidationData) -> _Evaluation:
    prepared = prepare_lists(data)
    known_scores = _keep_retriever_order(prepared.known_lists)
    new_references: dict[int, OrderMetrics] = {}
    for history_size, lists in prepared.new_lists.items():
        new_references[history_size] = _measure(
            lists,
            _keep_retriever_order(lists),
            data.new_user_held_out,
            data,
        )
    return _Evaluation(
        data=data,
        prepared=prepared,
        known_reference=_measure(
            prepared.known_lists,
            known_scores,
            data.held_out,
            data,
        ),
        new_references=new_references,
    )


def _run_all_rankers(data: ValidationData) -> list[RankerReport]:
    evaluation = _prepare_evaluation(data)
    reports: list[RankerReport] = [
        evaluate_order("retriever order", _keep_retriever_order, evaluation),
    ]
    head = _fit_list_ranker(ListLoss.SELECTION, evaluation)
    for name, regressor in list_gain_regressors().items():
        reports.extend(_evaluate_regressor(name, regressor, head, evaluation))
    for loss in (ListLoss.LISTNET, ListLoss.LAMBDARANK, ListLoss.BPR, ListLoss.HINGE):
        ranker = _fit_list_ranker(loss, evaluation)
        reports.append(evaluate_order(f"B2: {loss}", ranker.score, evaluation))
    return reports


def main() -> None:
    data = prepare_validation_data(
        os.environ[comparison_settings.DATABASE_PATH_VARIABLE],
    )
    reports = _run_all_rankers(data)
    report_directory = pathlib.Path(comparison_settings.REPORTS_DIRECTORY)
    report_directory.mkdir(exist_ok=True)
    report_path = report_directory / comparison_settings.RANKER_REPORT_FILE_NAME
    report_path.write_text(format_report(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
