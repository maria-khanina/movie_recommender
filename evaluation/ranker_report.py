"""Metrics of one reordering of the validation candidate lists, their paired
differences from a reference order, and the report tables.

Every metric is kept per user, so a ranker is compared with the reference on the
same users: the paired difference removes the shared difficulty of each user and
gives a narrower interval than the two separate intervals. The share of bad
movies among judged movies exists only for users with a held-out movie in the
top ten, so its paired difference uses the users where both orders have one.
Known users are reported by history length and together; new users separately.
"""
from typing import NamedTuple

import pandas

from evaluation import evaluation_settings as settings
from evaluation import ranker_comparison_settings as comparison_settings
from evaluation.catalog_metrics import (
    compute_catalog_coverage,
    compute_movie_popularity,
    compute_novelty,
)
from evaluation.metric_summary import MetricEstimate, compare_paired, summarize_metric
from evaluation.ranking_metrics import compute_bad_shares, compute_ndcg
from evaluation.validation_data import (
    ValidationData,
    name_new_user_segment,
    select_segment,
)


class OrderMetrics(NamedTuple):
    ndcg: pandas.Series
    bad_per_slot: pandas.Series
    bad_per_judged: pandas.Series
    novelty: float
    coverage: float


class SegmentSummary(NamedTuple):
    ndcg: MetricEstimate
    bad_per_slot: MetricEstimate
    bad_per_judged: MetricEstimate
    differences: list[MetricEstimate]


class RankerReport(NamedTuple):
    name: str
    segments: dict[str, SegmentSummary]
    novelty: float
    coverage: float


def measure_order(
    reranked: pandas.DataFrame,
    held_out: pandas.DataFrame,
    data: ValidationData,
) -> OrderMetrics:
    list_size = settings.RECOMMENDATION_LIST_SIZE
    bad_shares = compute_bad_shares(reranked, held_out, list_size)
    catalog = pandas.Index(data.tables.movies.index)
    popularity = compute_movie_popularity(data.training, catalog)
    return OrderMetrics(
        ndcg=compute_ndcg(reranked, held_out, list_size),
        bad_per_slot=bad_shares.per_list_slot,
        bad_per_judged=bad_shares.per_judged_movie,
        novelty=float(compute_novelty(reranked, popularity, list_size).mean()),
        coverage=compute_catalog_coverage(reranked, catalog, list_size),
    )


def _select_metrics(metrics: OrderMetrics, users: pandas.Index) -> list[pandas.Series]:
    return [
        select_segment(metrics.ndcg, users),
        select_segment(metrics.bad_per_slot, users),
        select_segment(metrics.bad_per_judged, users),
    ]


def summarize_segment(
    metrics: OrderMetrics,
    reference: OrderMetrics,
    users: pandas.Index,
) -> SegmentSummary:
    selected = _select_metrics(metrics, users)
    reference_selected = _select_metrics(reference, users)
    differences: list[MetricEstimate] = []
    for values, reference_values in zip(selected, reference_selected):
        differences.append(compare_paired(values, reference_values))
    return SegmentSummary(
        ndcg=summarize_metric(selected[0]),
        bad_per_slot=summarize_metric(selected[1]),
        bad_per_judged=summarize_metric(selected[2]),
        differences=differences,
    )


def summarize_segments(
    known: tuple[OrderMetrics, OrderMetrics],
    new: dict[int, tuple[OrderMetrics, OrderMetrics]],
    data: ValidationData,
) -> dict[str, SegmentSummary]:
    """Return the summaries by segment; each pair is (ranker, reference), and
    new users have one pair per history length.
    """
    segments: dict[str, SegmentSummary] = {}
    for name, users in data.segments.items():
        segments[name] = summarize_segment(known[0], known[1], users)
    new_users = pandas.Index(data.new_user_held_out[settings.USER_COLUMN].unique())
    for history_size, pair in new.items():
        segment = name_new_user_segment(history_size)
        segments[segment] = summarize_segment(pair[0], pair[1], new_users)
    return segments


def _format_estimate(estimate: MetricEstimate) -> str:
    decimals = comparison_settings.REPORT_DECIMALS
    return (
        f"{estimate.mean:.{decimals}f} "
        f"({estimate.lower:.{decimals}f}–{estimate.upper:.{decimals}f})"
    )


def format_report_row(report: RankerReport, segment: str) -> str:
    summary = report.segments[segment]
    cells: list[str] = [
        report.name,
        _format_estimate(summary.ndcg),
        _format_estimate(summary.differences[0]),
        _format_estimate(summary.bad_per_slot),
        _format_estimate(summary.differences[1]),
        f"{_format_estimate(summary.bad_per_judged)}, "
        f"{summary.bad_per_judged.user_count} users",
        _format_estimate(summary.differences[2]),
        f"{report.novelty:.2f}",
        f"{report.coverage:.3f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _format_header() -> list[str]:
    header_cells: list[str] = [
        "Ranker",
        "NDCG@10",
        "Minus retriever order",
        "Bad share of 10 slots",
        "Minus retriever order",
        "Bad share of judged movies",
        "Minus retriever order",
        "Novelty (known users)",
        "Coverage (known users)",
    ]
    return [
        "| " + " | ".join(header_cells) + " |",
        "|" + "---|" * len(header_cells),
    ]


def format_report(reports: list[RankerReport]) -> str:
    lines: list[str] = []
    for segment, summary in reports[0].segments.items():
        lines.extend(["", f"## {segment} ({summary.ndcg.user_count} users)", ""])
        lines.extend(_format_header())
        for report in reports:
            lines.append(format_report_row(report, segment))
    return "\n".join(lines).lstrip() + "\n"
