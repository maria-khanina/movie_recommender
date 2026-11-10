"""Comparison of candidate retrievers on the validation stage.

Every retriever is fitted on the history block of known users. Known users are
served with the ratings the site knows at that moment (history and ranker
blocks) as their history, and those movies are excluded from the candidates;
new users are served with their first ratings only. The candidates are scored on
the held-out ratings: the gain-weighted recall at each list size and the share
of held-out bad movies that got into the list, for known users with a long
history, with a short history, all known users, and new users. Fit and search
times are reported too. The tables are written to
reports/retriever_validation.md. The test block is never read here.
"""
import os
import pathlib
import time
from typing import NamedTuple

import numpy
import pandas

from evaluation import evaluation_settings as settings
from evaluation import retriever_comparison_settings as comparison_settings
from evaluation.metric_summary import MetricEstimate, summarize_metric
from evaluation.ranking_metrics import compute_gain_recall, compute_recall, select_bad
from evaluation.retriever_factories import RetrieverFactory, list_retriever_factories
from evaluation.validation_data import (
    ValidationData,
    name_new_user_segment,
    prepare_validation_data,
    select_segment,
)
from models.candidate_retrieval import CandidateRetriever, retrieve_unseen_candidates


class SegmentScores(NamedTuple):
    gain_recalls: list[MetricEstimate]
    bad_recalls: list[MetricEstimate]


class RetrieverReport(NamedTuple):
    name: str
    segments: dict[str, SegmentScores]
    fit_seconds: float
    search_seconds_per_thousand_users: float


class _PerUserScores(NamedTuple):
    gain_recalls: list[pandas.Series]
    bad_recalls: list[pandas.Series]


def _retrieve(
    retriever: CandidateRetriever,
    held_out: pandas.DataFrame,
    history: pandas.DataFrame,
    catalog_size: int,
) -> pandas.DataFrame:
    user_ids = held_out[settings.USER_COLUMN].unique().astype(numpy.int64)
    return retrieve_unseen_candidates(
        retriever,
        user_ids,
        history,
        max(settings.CANDIDATE_LIST_SIZES),
        catalog_size,
    )


def _score_per_user(
    candidates: pandas.DataFrame,
    held_out: pandas.DataFrame,
) -> _PerUserScores:
    bad = select_bad(held_out)
    scores = _PerUserScores(gain_recalls=[], bad_recalls=[])
    for list_size in settings.CANDIDATE_LIST_SIZES:
        scores.gain_recalls.append(
            compute_gain_recall(candidates, held_out, list_size),
        )
        scores.bad_recalls.append(compute_recall(candidates, bad, list_size))
    return scores


def _summarize(scores: _PerUserScores, users: pandas.Index) -> SegmentScores:
    gain_recalls: list[MetricEstimate] = []
    bad_recalls: list[MetricEstimate] = []
    for gain_recall, bad_recall in zip(scores.gain_recalls, scores.bad_recalls):
        gain_recalls.append(summarize_metric(select_segment(gain_recall, users)))
        bad_recalls.append(summarize_metric(select_segment(bad_recall, users)))
    return SegmentScores(gain_recalls=gain_recalls, bad_recalls=bad_recalls)


def _summarize_segments(
    known_scores: _PerUserScores,
    new_scores: dict[int, _PerUserScores],
    data: ValidationData,
) -> dict[str, SegmentScores]:
    segments: dict[str, SegmentScores] = {}
    for name, users in data.segments.items():
        segments[name] = _summarize(known_scores, users)
    new_users = pandas.Index(data.new_user_held_out[settings.USER_COLUMN].unique())
    for history_size, scores in new_scores.items():
        segments[name_new_user_segment(history_size)] = _summarize(scores, new_users)
    return segments


def _score_new_users(
    retriever: CandidateRetriever,
    data: ValidationData,
) -> dict[int, _PerUserScores]:
    catalog_size = len(data.tables.movies)
    scores: dict[int, _PerUserScores] = {}
    for history_size, history in data.new_user_histories.items():
        candidates = _retrieve(retriever, data.new_user_held_out, history, catalog_size)
        scores[history_size] = _score_per_user(candidates, data.new_user_held_out)
    return scores


def _score_known_users(
    retriever: CandidateRetriever,
    data: ValidationData,
) -> tuple[_PerUserScores, float]:
    catalog_size = len(data.tables.movies)
    started = time.perf_counter()
    known = _retrieve(retriever, data.held_out, data.known, catalog_size)
    search_seconds = time.perf_counter() - started
    scale = comparison_settings.SECONDS_PER_THOUSAND_USERS_SCALE
    known_user_count = data.held_out[settings.USER_COLUMN].nunique()
    per_thousand = search_seconds * scale / known_user_count
    return _score_per_user(known, data.held_out), per_thousand


def evaluate_retriever(
    name: str,
    factory: RetrieverFactory,
    data: ValidationData,
) -> RetrieverReport:
    retriever = factory()
    started = time.perf_counter()
    retriever.fit(data.training, data.tables)
    fit_seconds = time.perf_counter() - started
    known_scores, search_seconds = _score_known_users(retriever, data)
    new_scores = _score_new_users(retriever, data)
    return RetrieverReport(
        name=name,
        segments=_summarize_segments(known_scores, new_scores, data),
        fit_seconds=fit_seconds,
        search_seconds_per_thousand_users=search_seconds,
    )


def _format_estimate(estimate: MetricEstimate) -> str:
    decimals = comparison_settings.REPORT_DECIMALS
    return (
        f"{estimate.mean:.{decimals}f} "
        f"({estimate.lower:.{decimals}f}–{estimate.upper:.{decimals}f})"
    )


def format_report_row(report: RetrieverReport, segment: str) -> str:
    scores = report.segments[segment]
    cells: list[str] = [report.name]
    for estimate in scores.gain_recalls:
        cells.append(_format_estimate(estimate))
    for estimate in scores.bad_recalls:
        cells.append(f"{estimate.mean:.{comparison_settings.REPORT_DECIMALS}f}")
    cells.append(f"{report.fit_seconds:.1f}")
    cells.append(f"{report.search_seconds_per_thousand_users:.2f}")
    return "| " + " | ".join(cells) + " |"


def _format_header() -> list[str]:
    sizes = settings.CANDIDATE_LIST_SIZES
    header_cells: list[str] = ["Retriever"]
    header_cells.extend(f"Gain recall@{size}" for size in sizes)
    header_cells.extend(f"Bad recall@{size}" for size in sizes)
    header_cells.extend(["Fit, s", "Search, s per 1000 users"])
    return [
        "| " + " | ".join(header_cells) + " |",
        "|" + "---|" * len(header_cells),
    ]


def format_report(reports: list[RetrieverReport]) -> str:
    lines: list[str] = []
    for segment in reports[0].segments:
        users = reports[0].segments[segment].gain_recalls[0].user_count
        lines.extend(["", f"## {segment} ({users} users)", ""])
        lines.extend(_format_header())
        for report in reports:
            lines.append(format_report_row(report, segment))
    return "\n".join(lines).lstrip() + "\n"


def main() -> None:
    data = prepare_validation_data(
        os.environ[comparison_settings.DATABASE_PATH_VARIABLE],
    )
    report_directory = pathlib.Path(comparison_settings.REPORTS_DIRECTORY)
    report_directory.mkdir(exist_ok=True)
    report_path = report_directory / comparison_settings.RETRIEVER_REPORT_FILE_NAME
    reports: list[RetrieverReport] = []
    for name, factory in list_retriever_factories().items():
        reports.append(evaluate_retriever(name, factory, data))
        report_path.write_text(format_report(reports), encoding="utf-8")
        print(f"Evaluated {name}", flush=True)


if __name__ == "__main__":
    main()
