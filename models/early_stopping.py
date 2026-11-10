"""Early stopping shared by the trained retrievers.

The last ratings of every user's training history are held back as a stopping
set and the model is trained on the rest. After every epoch the gain-weighted
recall of the stopping set is measured through the same retrieval path the
evaluation uses, and training stops when it has not improved for a number of
epochs; the best epoch's parameters are restored. The stopping set comes from
the training history, so the validation block never chooses the epoch.
"""
from typing import NamedTuple

import numpy
import pandas

from evaluation import evaluation_settings as columns
from evaluation.percentile_gains import attach_percentile_gains
from evaluation.ranking_metrics import compute_gain_recall
from evaluation.rating_split import sort_ratings_by_time
from models.candidate_retrieval import (
    CandidateRetriever,
    MovieIds,
    retrieve_unseen_candidates,
)


class StoppingData(NamedTuple):
    fitting: pandas.DataFrame
    held_out: pandas.DataFrame
    user_ids: MovieIds


def split_stopping_ratings(
    training: pandas.DataFrame,
    ratings_per_user: int,
) -> StoppingData:
    ordered = sort_ratings_by_time(training)
    positions_from_end = ordered.groupby(columns.USER_COLUMN).cumcount(ascending=False)
    is_stopping = positions_from_end < ratings_per_user
    fitting = ordered[~is_stopping].reset_index(drop=True)
    held_out = attach_percentile_gains(ordered[is_stopping], fitting)
    user_ids = held_out[columns.USER_COLUMN].unique().astype(numpy.int64)
    return StoppingData(fitting=fitting, held_out=held_out, user_ids=user_ids)


def measure_stopping_recall(
    retriever: CandidateRetriever,
    stopping: StoppingData,
    list_size: int,
    catalog_size: int,
) -> float:
    candidates = retrieve_unseen_candidates(
        retriever,
        stopping.user_ids,
        stopping.fitting,
        list_size,
        catalog_size,
    )
    recall = compute_gain_recall(candidates, stopping.held_out, list_size)
    return float(recall.mean())


class StoppingTracker[State]:
    def __init__(self, states: State, patience: int) -> None:
        self.best_recall = -1.0
        self.best_states = states
        self.stale_epochs = 0
        self._patience = patience

    def record(self, recall: float, states: State) -> bool:
        """Remember the states if the recall is the best so far and return
        whether training should stop.
        """
        if recall > self.best_recall:
            self.best_recall, self.best_states, self.stale_epochs = recall, states, 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self._patience
