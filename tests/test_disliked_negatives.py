import torch

from models.disliked_negatives import draw_anywhere, draw_nearby, index_disliked
from models.rating_sequences import RatedSequences
from tests import test_settings


def _build_sequences() -> RatedSequences:
    disliked = torch.tensor(test_settings.DISLIKED_FLAGS)
    lengths = torch.tensor(test_settings.DISLIKED_SEQUENCE_LENGTHS)
    starts = torch.cumsum(lengths, dim=0) - lengths
    user_of_rating = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    positions = torch.arange(len(disliked)) - starts[user_of_rating]
    return RatedSequences(
        movie_codes=torch.arange(1, len(disliked) + 1),
        rating_codes=torch.ones(len(disliked), dtype=torch.int64),
        liked=~disliked,
        disliked=disliked,
        timestamps=torch.arange(len(disliked)),
        second_starts=positions,
        starts=starts,
        lengths=lengths,
    )


def _repeat(values: list[int]) -> torch.Tensor:
    return torch.tensor(values).repeat(test_settings.DISLIKED_DRAW_COUNT)


def _collect(drawn: torch.Tensor, first: int, step: int) -> set[int]:
    collected: set[int] = set()
    for value in drawn[first::step]:
        collected.add(int(value))
    return collected


def _disliked_rows(user_row: int) -> set[int]:
    sequences = _build_sequences()
    start = int(sequences.starts[user_row])
    end = start + int(sequences.lengths[user_row])
    found = torch.nonzero(sequences.disliked[start:end]).squeeze(1) + start
    return _collect(found, 0, 1)


def test_nearby_draws_a_disliked_movie_of_the_span_or_the_target() -> None:
    marker = test_settings.NO_DISLIKED_MARKER
    span = test_settings.DISLIKED_SPAN
    cases: list[int] = [0, 0, 1]
    step = len(cases)
    ends = _repeat([span, 0, 0])
    targets = _repeat([marker] * step)
    drawn = draw_nearby(_build_sequences(), _repeat(cases), ends, targets, span)
    assert _collect(drawn, 0, step) == {marker}
    assert _collect(drawn, 1, step) == {min(_disliked_rows(0))}
    assert _collect(drawn, 2, step) == {marker}


def test_anywhere_draws_only_disliked_movies_of_the_same_user() -> None:
    sequences = _build_sequences()
    marker = test_settings.NO_DISLIKED_MARKER
    user_rows = _repeat([0, 1])
    index = index_disliked(sequences)
    drawn = draw_anywhere(sequences, index, user_rows, _repeat([marker] * 2))
    assert _collect(drawn, 0, 2) == _disliked_rows(0)
    assert _collect(drawn, 1, 2) == {marker}
