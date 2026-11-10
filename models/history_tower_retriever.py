"""Two-tower retrievers whose user tower reads the rating history.

Three variants share this training procedure:
- the liked-mean user tower with the feature movie tower,
- the sequence (SASRec) user tower with the feature movie tower,
- the sequence user tower with a flat movie tower that reuses the sequence
  tower's movie embedding: the original SASRec layout.

A training example is a liked rating. Its history window ends strictly before
it, and how far before is drawn at random: up to target_window ratings earlier
(1 means right before the target), or anywhere before it when target_window is
0. So one model learns from the next liked movie, the next few liked movies or
all later liked movies. With cuts_between_seconds, a window end that falls
inside a second moves back to the start of that second, so, as at query time,
every window rating is strictly earlier than the target. Other movies of the
batch are the negatives, with the logQ correction; movies the user likes
anywhere in the training history are never negatives of that user's rows.
In the liked-mean variant, a share of examples with at least the site's
minimum history hide the user id, which teaches the tower to serve a new user
from the history alone; the sequence tower has no user id at all. At query
time the window is the end of the history the site knows, so new ratings are
used without retraining.
"""
import copy
import enum
from typing import NamedTuple, Protocol, cast

import numpy
import pandas
import torch

from dataset.movielens_tables import MovieLensTables
from evaluation import evaluation_settings as columns
from evaluation.percentile_gains import attach_percentile_gains
from models import history_tower_settings as settings
from models import two_tower_settings as tower_settings
from models.candidate_retrieval import MovieIds, SearchResult
from models.early_stopping import (
    StoppingData,
    StoppingTracker,
    measure_stopping_recall,
    split_stopping_ratings,
)
from models.history_encoders import (
    FlatMovieTower,
    LikedMeanUserTower,
    SequenceOptions,
    SequenceUserTower,
)
from models.rating_sequences import (
    HistoryWindow,
    RatedSequences,
    SequencedRatings,
    build_rated_sequences,
    cut_windows,
)
from models.two_tower_features import (
    TowerFeatures,
    build_tower_features,
    find_rows,
    select_movie_inputs,
    select_user_inputs,
)
from models.two_tower_loss import BatchRows, BatchVectors, compute_retrieval_loss
from models.two_tower_network import MovieTower
from models.vector_search import MovieVectorIndex

_States = dict[str, torch.Tensor]


class UserEncoder(enum.StrEnum):
    LIKED_MEAN = "liked mean"
    SEQUENCE = "sequence"


class HistoryTowerConfig(NamedTuple):
    user_encoder: UserEncoder
    flat_movie_tower: bool
    target_window: int
    positive_threshold: float
    window_length: int
    uses_positions: bool
    uses_rating_age: bool
    masks_future_liked: bool
    hidden_id_share: float
    user_weight_exponent: float
    recency_half_life_days: float
    cuts_between_seconds: bool


class _DifferentiableLoss(Protocol):
    def backward(self) -> None: ...


class _Seeding(Protocol):
    def manual_seed(self, seed: int, /) -> torch.Generator: ...


_SEEDING = cast(_Seeding, torch)


class _Examples(NamedTuple):
    user_rows: torch.Tensor
    movie_rows: torch.Tensor
    positions: torch.Tensor
    liked_keys: torch.Tensor
    log_frequencies: torch.Tensor
    weights: torch.Tensor


def _flag_liked(ratings: pandas.DataFrame, threshold: float) -> pandas.Series:
    gained = attach_percentile_gains(ratings, ratings)
    return gained[columns.PERCENTILE_COLUMN] >= threshold


def _count_log_frequencies(
    sequenced: SequencedRatings,
    movie_count: int,
    device: torch.device,
) -> torch.Tensor:
    liked_movies = sequenced.movie_rows[sequenced.liked]
    counts = numpy.bincount(liked_movies, minlength=movie_count) + 1.0
    frequencies = torch.as_tensor(counts / counts.sum(), dtype=torch.float32)
    return torch.log(frequencies).to(device)


def _weigh_examples(
    sequenced: SequencedRatings,
    config: HistoryTowerConfig,
) -> numpy.typing.NDArray[numpy.float64]:
    """Return a weight per liked rating: users with many examples are damped by
    (example count) ** -exponent, and old ratings by a half-life in days.
    """
    liked_users = sequenced.user_rows[sequenced.liked]
    counts = numpy.bincount(liked_users).astype(numpy.float64)
    weights = numpy.power(counts[liked_users], -config.user_weight_exponent)
    if config.recency_half_life_days > 0:
        times = sequenced.timestamps[sequenced.liked]
        age_days = (times.max() - times) / columns.SECONDS_PER_DAY
        weights *= numpy.power(settings.HALF, age_days / config.recency_half_life_days)
    return weights / weights.mean()


def _build_examples(
    sequenced: SequencedRatings,
    config: HistoryTowerConfig,
    movie_count: int,
    device: torch.device,
) -> _Examples:
    liked = sequenced.liked
    keys = sequenced.user_rows[liked] * movie_count + sequenced.movie_rows[liked]
    return _Examples(
        user_rows=torch.as_tensor(sequenced.user_rows[liked], device=device),
        movie_rows=torch.as_tensor(sequenced.movie_rows[liked], device=device),
        positions=torch.as_tensor(sequenced.positions[liked], device=device),
        liked_keys=torch.as_tensor(numpy.unique(keys), device=device),
        log_frequencies=_count_log_frequencies(sequenced, movie_count, device),
        weights=torch.as_tensor(
            _weigh_examples(sequenced, config),
            dtype=torch.float32,
            device=device,
        ),
    )


class HistoryTowerRetriever:
    def __init__(self, config: HistoryTowerConfig) -> None:
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = numpy.random.default_rng(columns.RANDOM_SEED)
        self._features: TowerFeatures | None = None
        self._user_tower: torch.nn.Module | None = None
        self._movie_tower: torch.nn.Module | None = None
        self._index: MovieVectorIndex | None = None
        self.stopping_recall = -1.0

    def _require_model(
        self,
    ) -> tuple[torch.nn.Module, torch.nn.Module, TowerFeatures]:
        if self._user_tower is None or self._movie_tower is None:
            raise RuntimeError("The retriever is not fitted")
        if self._features is None:
            raise RuntimeError("The retriever is not fitted")
        return self._user_tower, self._movie_tower, self._features

    def _build_user_tower(self, features: TowerFeatures) -> torch.nn.Module:
        if self._config.user_encoder == UserEncoder.LIKED_MEAN:
            return LikedMeanUserTower(features.user_sizes, len(features.movie_index))
        options = SequenceOptions(
            window_length=self._config.window_length,
            uses_positions=self._config.uses_positions,
            uses_rating_age=self._config.uses_rating_age,
        )
        return SequenceUserTower(
            features.user_sizes,
            features.movie_sizes,
            features.movies,
            options,
        )

    def _build_towers(self, features: TowerFeatures) -> torch.optim.Optimizer:
        user_tower = self._build_user_tower(features).to(self._device)
        if self._config.flat_movie_tower and isinstance(user_tower, SequenceUserTower):
            movie_tower: torch.nn.Module = FlatMovieTower(user_tower)
        else:
            movie_tower = MovieTower(features.movie_sizes).to(self._device)
        self._user_tower, self._movie_tower = user_tower, movie_tower
        unique_parameters: dict[int, torch.nn.Parameter] = {}
        for parameter in [*user_tower.parameters(), *movie_tower.parameters()]:
            unique_parameters[id(parameter)] = parameter
        return torch.optim.AdamW(
            list(unique_parameters.values()),
            lr=tower_settings.LEARNING_RATE,
            weight_decay=tower_settings.WEIGHT_DECAY,
        )

    def _draw_window_ends(
        self,
        sequences: RatedSequences,
        user_rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        window = self._config.target_window
        if window == settings.ALL_REMAINING_TARGETS:
            limits = positions + 1
        else:
            limits = torch.full_like(positions, window)
        draws = torch.rand(len(positions), device=self._device)
        gaps = (draws * limits.float()).long()
        ends = (positions - gaps).clamp(min=0)
        if not self._config.cuts_between_seconds:
            return ends
        return sequences.second_starts[sequences.starts[user_rows] + ends]

    def _hide_user_ids(
        self,
        user_codes: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        draws = torch.rand(len(user_codes), device=self._device)
        return user_codes.masked_fill(draws < self._config.hidden_id_share, 0)

    def _encode_users(
        self,
        user_rows: torch.Tensor,
        window: HistoryWindow,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        user_tower, _, features = self._require_model()
        inputs = select_user_inputs(features, user_rows)
        if user_tower.training:
            inputs = inputs._replace(
                user_codes=self._hide_user_ids(inputs.user_codes, ends),
            )
        vectors: torch.Tensor = user_tower(inputs, window)
        return vectors

    def _encode_movies(self, movie_rows: torch.Tensor) -> torch.Tensor:
        _, movie_tower, features = self._require_model()
        vectors: torch.Tensor = movie_tower(select_movie_inputs(features, movie_rows))
        return vectors

    def _mask_liked_pairs(
        self,
        examples: _Examples,
        user_rows: torch.Tensor,
        movie_rows: torch.Tensor,
    ) -> torch.Tensor:
        _, _, features = self._require_model()
        movie_count = len(features.movie_index)
        keys = user_rows.unsqueeze(1) * movie_count + movie_rows.unsqueeze(0)
        is_liked = torch.isin(keys, examples.liked_keys)
        return is_liked & self._config.masks_future_liked

    def _compute_batch_loss(
        self,
        examples: _Examples,
        sequences: SequencedRatings,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        user_rows, movie_rows = examples.user_rows[batch], examples.movie_rows[batch]
        ends = self._draw_window_ends(
            sequences.sequences,
            user_rows,
            examples.positions[batch],
        )
        window = cut_windows(
            sequences.sequences,
            user_rows,
            ends,
            self._config.window_length,
        )
        movie_vectors = self._encode_movies(movie_rows)
        vectors = BatchVectors(
            users=self._encode_users(user_rows, window, ends),
            movies=movie_vectors,
            negatives=movie_vectors,
        )
        rows = BatchRows(users=user_rows, movies=movie_rows, negatives=movie_rows)
        return compute_retrieval_loss(
            vectors,
            rows,
            examples.log_frequencies,
            0.0,
            self._mask_liked_pairs(examples, user_rows, movie_rows),
            examples.weights[batch],
        )

    def _run_epoch(
        self,
        examples: _Examples,
        sequences: SequencedRatings,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        user_tower, movie_tower, _ = self._require_model()
        user_tower.train()
        movie_tower.train()
        order = torch.randperm(len(examples.user_rows), device=self._device)
        for batch in torch.split(order, tower_settings.BATCH_SIZE):
            optimizer.zero_grad()
            loss = self._compute_batch_loss(examples, sequences, batch)
            cast(_DifferentiableLoss, loss).backward()
            optimizer.step()

    def _rebuild_index(self) -> None:
        user_tower, movie_tower, features = self._require_model()
        user_tower.eval()
        movie_tower.eval()
        with torch.no_grad():
            movie_vectors: torch.Tensor = movie_tower(features.movies)
        self._index = MovieVectorIndex(
            features.movie_index.to_numpy(dtype=numpy.int64),
            numpy.asarray(movie_vectors.cpu(), dtype=numpy.float32),
        )

    def _copy_states(self) -> tuple[_States, _States]:
        user_tower, movie_tower, _ = self._require_model()
        return (
            copy.deepcopy(user_tower.state_dict()),
            copy.deepcopy(movie_tower.state_dict()),
        )

    def _measure_recall(self, stopping: StoppingData) -> float:
        _, _, features = self._require_model()
        self._rebuild_index()
        return measure_stopping_recall(
            self,
            stopping,
            tower_settings.STOPPING_LIST_SIZE,
            len(features.movie_index),
        )

    def _train_until_stopping(
        self,
        examples: _Examples,
        sequences: SequencedRatings,
        stopping: StoppingData,
    ) -> None:
        optimizer = self._build_towers(self._require_model_features())
        patience = tower_settings.PATIENCE_IN_EPOCHS
        tracker = StoppingTracker(self._copy_states(), patience)
        for _ in range(tower_settings.MAXIMUM_EPOCHS):
            self._run_epoch(examples, sequences, optimizer)
            if tracker.record(self._measure_recall(stopping), self._copy_states()):
                break
        self._restore(tracker)

    def _restore(self, tracker: StoppingTracker[tuple[_States, _States]]) -> None:
        user_tower, movie_tower, _ = self._require_model()
        user_tower.load_state_dict(tracker.best_states[0])
        movie_tower.load_state_dict(tracker.best_states[1])
        self.stopping_recall = tracker.best_recall

    def _require_model_features(self) -> TowerFeatures:
        if self._features is None:
            raise RuntimeError("The retriever has no features")
        return self._features

    def fit(self, training: pandas.DataFrame, tables: MovieLensTables) -> None:
        _SEEDING.manual_seed(columns.RANDOM_SEED)
        stopping = split_stopping_ratings(
            training,
            tower_settings.STOPPING_RATINGS_PER_USER,
        )
        liked = _flag_liked(stopping.fitting, self._config.positive_threshold)
        features = build_tower_features(stopping.fitting[liked], tables, self._device)
        self._features = features
        sequenced = build_rated_sequences(
            stopping.fitting,
            liked,
            features,
            self._device,
        )
        examples = _build_examples(
            sequenced,
            self._config,
            len(features.movie_index),
            self._device,
        )
        self._train_until_stopping(examples, sequenced, stopping)
        self._rebuild_index()

    def _sequence_query_history(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
    ) -> SequencedRatings:
        _, _, features = self._require_model()
        queried = history[history[columns.USER_COLUMN].isin(user_ids)]
        liked = _flag_liked(queried, self._config.positive_threshold)
        return build_rated_sequences(queried, liked, features, self._device)

    def _encode_query_users(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
    ) -> numpy.typing.NDArray[numpy.float32]:
        _, _, features = self._require_model()
        sequenced = self._sequence_query_history(user_ids, history)
        rows = find_rows(features.user_index, user_ids)
        user_rows = torch.as_tensor(rows, device=self._device)
        ends = sequenced.sequences.lengths[user_rows]
        window = cut_windows(
            sequenced.sequences,
            user_rows,
            ends,
            self._config.window_length,
        )
        with torch.no_grad():
            vectors = self._encode_users(user_rows, window, ends)
        return numpy.asarray(vectors.cpu(), dtype=numpy.float32)

    def search(
        self,
        user_ids: MovieIds,
        history: pandas.DataFrame,
        depth: int,
    ) -> SearchResult:
        if self._index is None:
            raise RuntimeError("The retriever is not fitted")
        return self._index.search(self._encode_query_users(user_ids, history), depth)
