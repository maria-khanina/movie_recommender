"""The retriever variants compared on the validation stage, by name.

Every retriever that learns from liked ratings is tried with each percentile
threshold of a liked rating; popularity is also tried counting every rating. The
factorization machine learns from all ratings with the squared error, so it has
no threshold.
"""
from collections.abc import Callable

from evaluation import retriever_comparison_settings as comparison_settings
from models.candidate_retrieval import CandidateRetriever
from models.disliked_negatives import DislikedNegatives
from models.factorization_machine import MachineConfig
from models.factorization_machine_retriever import FactorizationMachineRetriever
from models.history_tower_retriever import (
    HistoryTowerConfig,
    HistoryTowerRetriever,
    UserEncoder,
)
from models.item_neighbors_retriever import ItemNeighborsRetriever
from models.popularity_retriever import PopularityRetriever
from models.rating_sequences import RatingToken
from models.two_tower_retriever import TwoTowerRetriever

RetrieverFactory = Callable[[], CandidateRetriever]


def bind_popularity(threshold: float) -> RetrieverFactory:
    def build_retriever() -> CandidateRetriever:
        return PopularityRetriever(threshold)
    return build_retriever


def bind_item_neighbors(
    neighbor_count: int,
    exponent: float,
    threshold: float,
) -> RetrieverFactory:
    def build_retriever() -> CandidateRetriever:
        return ItemNeighborsRetriever(neighbor_count, exponent, threshold)
    return build_retriever


def bind_factorization_machine(config: MachineConfig) -> RetrieverFactory:
    def build_retriever() -> CandidateRetriever:
        return FactorizationMachineRetriever(config)
    return build_retriever


def bind_two_towers(negative_weight: float, threshold: float) -> RetrieverFactory:
    def build_retriever() -> CandidateRetriever:
        return TwoTowerRetriever(negative_weight, threshold)
    return build_retriever


def _list_popularity_factories() -> dict[str, RetrieverFactory]:
    factories: dict[str, RetrieverFactory] = {
        "popularity, all ratings": bind_popularity(
            comparison_settings.ALL_RATINGS_THRESHOLD,
        ),
    }
    for threshold in comparison_settings.POSITIVE_PERCENTILE_THRESHOLDS:
        factories[f"popularity, liked from {threshold}"] = bind_popularity(threshold)
    return factories


def _list_item_neighbor_factories(threshold: float) -> dict[str, RetrieverFactory]:
    factories: dict[str, RetrieverFactory] = {}
    for neighbor_count in comparison_settings.ITEM_NEIGHBOR_COUNTS:
        for exponent in comparison_settings.ITEM_POPULARITY_EXPONENTS:
            name = (
                f"item neighbors, {neighbor_count} neighbors, "
                f"exponent {exponent}, liked from {threshold}"
            )
            factories[name] = bind_item_neighbors(neighbor_count, exponent, threshold)
    return factories


def _list_factorization_factories() -> dict[str, RetrieverFactory]:
    factories: dict[str, RetrieverFactory] = {}
    for factor_count in comparison_settings.FACTORIZATION_FACTOR_COUNTS:
        config = MachineConfig(
            factor_count=factor_count,
            learning_rate=comparison_settings.FACTORIZATION_LEARNING_RATE,
            weights_penalty=comparison_settings.FACTORIZATION_PENALTY,
            factors_penalty=comparison_settings.FACTORIZATION_PENALTY,
        )
        name = f"factorization machine, {factor_count} factors"
        factories[name] = bind_factorization_machine(config)
    return factories


def _list_two_tower_factories(threshold: float) -> dict[str, RetrieverFactory]:
    factories: dict[str, RetrieverFactory] = {}
    for weight in comparison_settings.TWO_TOWER_NEGATIVE_WEIGHTS:
        name = f"two towers, disliked weight {weight}, liked from {threshold}"
        factories[name] = bind_two_towers(weight, threshold)
    return factories


def bind_history_towers(config: HistoryTowerConfig) -> RetrieverFactory:
    def build_retriever() -> CandidateRetriever:
        return HistoryTowerRetriever(config)
    return build_retriever


def _describe_window(target_window: int) -> str:
    if target_window == comparison_settings.ALL_LATER_TARGETS:
        return "all later liked movies"
    return f"liked movies within {target_window} ratings"


def _add_history_factory(
    factories: dict[str, RetrieverFactory],
    encoder: UserEncoder,
    flat_movie_tower: bool,
    target_window: int,
) -> None:
    config = HistoryTowerConfig(
        user_encoder=encoder,
        flat_movie_tower=flat_movie_tower,
        target_window=target_window,
        positive_threshold=comparison_settings.HISTORY_POSITIVE_THRESHOLD,
        window_length=comparison_settings.HISTORY_WINDOW_LENGTH,
        uses_positions=True,
        uses_rating_age=False,
        masks_future_liked=True,
        hidden_id_share=comparison_settings.HISTORY_HIDDEN_ID_SHARE,
        user_weight_exponent=0.0,
        recency_half_life_days=0.0,
        cuts_between_seconds=False,
        rating_token=RatingToken.RAW,
        concatenates_demographics=False,
        disliked_negatives=DislikedNegatives.NONE,
        disliked_weight=0.0,
    )
    movie_side = "flat movie embeddings" if flat_movie_tower else "movie tower"
    name = f"{encoder} user tower, {movie_side}, {_describe_window(target_window)}"
    factories[name] = bind_history_towers(config)


def list_history_factories() -> dict[str, RetrieverFactory]:
    factories: dict[str, RetrieverFactory] = {}
    for window in comparison_settings.LIKED_MEAN_TARGET_WINDOWS:
        _add_history_factory(factories, UserEncoder.LIKED_MEAN, False, window)
    for flat_movie_tower in (False, True):
        for window in comparison_settings.SEQUENCE_TARGET_WINDOWS:
            _add_history_factory(
                factories,
                UserEncoder.SEQUENCE,
                flat_movie_tower,
                window,
            )
    return factories


def list_retriever_factories() -> dict[str, RetrieverFactory]:
    factories = _list_popularity_factories()
    for threshold in comparison_settings.POSITIVE_PERCENTILE_THRESHOLDS:
        factories.update(_list_item_neighbor_factories(threshold))
    factories.update(_list_factorization_factories())
    for threshold in comparison_settings.POSITIVE_PERCENTILE_THRESHOLDS:
        factories.update(_list_two_tower_factories(threshold))
    factories.update(list_history_factories())
    return factories
