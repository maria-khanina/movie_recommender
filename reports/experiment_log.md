# Experiment log

Every validation run, in order, with what was run and on which data. Reports in
`reports/` are overwritten by each new run; this log and the copies in
`reports/runs/` are not. The test block has not been opened in any run.

Data: MovieLens-1M (6,040 users, 3,883 movies, 1,000,209 ratings).

## Split A (runs 1–3)

Per user, ratings ordered by time, ties within a second broken by movie id.
Blocks from the end: test 5, validation 5, ranker 5, history the rest; users
with fewer than 20 ratings keep everything in history (none in ML-1M). In the
validation stage models fit on the history block, candidates exclude movies
rated in the history and ranker blocks, and metrics use the validation block.
Gains: percentile of a rating among the same user's history ratings.

**Problem found later:** a block boundary can fall inside one second, so the end
of the known history and the start of the validation block can be ratings given
at the same moment, ordered by movie id.

### Run 1 — retrievers, positives = rating ≥ 4 (2026-11-05)

Gain-weighted recall of held-out movies at 100 / 200 / 500 candidates (95%
bootstrap interval over users for @100) and the share of held-out bad movies
(percentile < 0.2) that got into the candidates. Retrievers know only the
history block at query time. Two towers: the architecture of the original
notebook (user id, age, gender, occupation / movie id, genres), in-batch softmax
with logQ, temperature 0.05; "disliked weight" is the weight of one extra
negative per example, a movie the user rated low.

| Retriever | Recall@100 | @200 | @500 | Bad@100 | Bad@200 | Bad@500 | Fit, s |
|---|---|---|---|---|---|---|---|
| popularity, ratings ≥ 4 | 0.247 (0.239–0.253) | 0.368 | 0.592 | 0.134 | 0.220 | 0.426 | 0.0 |
| popularity, all ratings | 0.247 (0.239–0.253) | 0.373 | 0.599 | 0.155 | 0.255 | 0.500 | 0.0 |
| item neighbors, 50, exponent 0.5 | 0.326 (0.319–0.334) | 0.472 | 0.677 | 0.158 | 0.272 | 0.469 | 0.7 |
| item neighbors, 50, exponent 0.75 | 0.312 | 0.449 | 0.631 | 0.146 | 0.250 | 0.440 | 0.7 |
| item neighbors, 50, exponent 1.0 | 0.297 | 0.403 | 0.599 | 0.144 | 0.231 | 0.428 | 0.7 |
| item neighbors, 100, exponent 0.5 | 0.314 | 0.466 | 0.696 | 0.152 | 0.272 | 0.492 | 0.8 |
| item neighbors, 100, exponent 0.75 | 0.294 | 0.451 | 0.656 | 0.146 | 0.256 | 0.455 | 0.8 |
| item neighbors, 100, exponent 1.0 | 0.286 | 0.432 | 0.612 | 0.143 | 0.243 | 0.432 | 0.7 |
| item neighbors, 200, exponent 0.5 | 0.297 | 0.446 | 0.697 | 0.147 | 0.255 | 0.495 | 0.8 |
| item neighbors, 200, exponent 0.75 | 0.283 | 0.428 | 0.678 | 0.142 | 0.244 | 0.481 | 0.8 |
| item neighbors, 200, exponent 1.0 | 0.277 | 0.419 | 0.647 | 0.140 | 0.238 | 0.454 | 0.8 |
| FM (user and movie ids, MSE on ratings), 32 factors, penalty 0 | 0.190 | 0.282 | 0.426 | 0.070 | 0.122 | 0.237 | 56.5 |
| FM, 32 factors, penalty 1e-4 | 0.191 | 0.278 | 0.413 | 0.078 | 0.138 | 0.246 | 83.6 |
| FM, 64 factors, penalty 0 | 0.172 | 0.263 | 0.432 | 0.046 | 0.082 | 0.181 | 186.7 |
| FM, 64 factors, penalty 1e-4 | 0.195 | 0.281 | 0.420 | 0.075 | 0.130 | 0.233 | 208.0 |
| two towers, disliked weight 0 | 0.341 (0.334–0.349) | 0.499 | 0.730 | 0.189 | 0.314 | 0.553 | 269.5 |
| two towers, disliked weight 1 | 0.342 (0.334–0.350) | 0.499 | 0.730 | 0.190 | 0.314 | 0.553 | 236.4 |
| two towers, disliked weight 10 | 0.347 (0.339–0.355) | 0.502 | 0.736 | 0.200 | 0.325 | 0.559 | 356.7 |
| two towers, disliked weight 100 | 0.352 (0.344–0.360) | 0.510 | 0.738 | 0.196 | 0.320 | 0.560 | 260.0 |

### Run 2 — rankers on two-tower candidates (2026-11-06)

Retriever: two towers, disliked weight 100, 100 candidates. Rankers reorder the
validation lists; NDCG@10 with percentile gains, share of bad movies among the
10 slots and among judged movies (users with a held-out movie in the top 10).
Regressor of the expected gain ĝ: the MLP of the original ranker notebook, MSE
on percentile gains. List rankers and the selection head are not fully
deterministic on GPU.

| Ranker | NDCG@10 | Bad of 10 slots | Bad of judged (users) |
|---|---|---|---|
| retriever order | 0.0475 (0.0447–0.0503) | 0.0021 | 0.078 (1324) |
| A: log ĝ + 0 · retriever score | 0.0353 | 0.0008 | 0.038 (975) |
| A: log ĝ + 1 · score | 0.0389 | 0.0009 | 0.040 (1030) |
| A: log ĝ + 3 · score | 0.0448 | 0.0009 | 0.037 (1143) |
| A: log ĝ + 10 · score | 0.0488 | 0.0013 | 0.050 (1290) |
| A: log ĝ + 20 · score | 0.0494 (0.0465–0.0524) | 0.0015 | 0.058 (1330) |
| A: log ĝ + 30 · score | 0.0490 | 0.0016 | 0.059 (1333) |
| A: log ĝ + 50 · score | 0.0491 | 0.0017 | 0.062 (1344) |
| B1: log selection + 0.5 · log ĝ | 0.0467 | 0.0019 | 0.071 (1287) |
| B1: log selection + 1 · log ĝ | 0.0458 | 0.0016 | 0.065 (1221) |
| B1: log selection + 2 · log ĝ | 0.0441 | 0.0012 | 0.050 (1156) |
| B2: ListNet | 0.0423 | 0.0030 | 0.116 (1379) |
| B2: LambdaRank | 0.0444 | 0.0029 | 0.109 (1344) |
| B2: weighted BPR | 0.0485 | 0.0027 | 0.098 (1444) |
| B2: weighted hinge | 0.0485 | 0.0028 | 0.103 (1382) |

### Run 3 — rankers with paired differences, MLP and FM regressors (2026-11-06)

Same retriever and lists as run 2, rerun, with the FM (ids only) as a second ĝ
regressor and paired differences from the retriever order. Full table:
`reports/runs/2026-11-06_rankers_paired_mlp_fm.md`. Main lines:
- A (MLP), weight 10: NDCG +0.0019 (+0.0001…+0.0037); bad of 10 slots −0.0008
  (−0.0011…−0.0005); bad of judged −0.0062 (−0.0121…−0.0004).
- A (MLP), weight 20: NDCG +0.0016 (+0.0004…+0.0028); bad of 10 slots −0.0006.
- A (FM), weight 10: NDCG +0.0019 (+0.0000…+0.0037); bad of 10 slots −0.0006.
- B2 ListNet and LambdaRank raise both bad shares significantly; between runs 2
  and 3 ListNet's NDCG moved from 0.0423 to 0.0480.

## Split B (runs 4–5)

Split A plus: 5% of users (302) are held out of all training as new users, half
for the validation stage and half for the test stage; their first 15 ratings are
their history and the next 5 are held out. Known users are reported by the
number of ratings known at query time (history and ranker blocks): at least 15,
fewer than 15, all. Positives of every retriever: a rating whose percentile
among the same user's ratings reaches the threshold. Retrievers receive the
history known at query time (history and ranker blocks for known users, the
first 15 ratings for new users).

### Run 4 — retrievers, percentile positives, segments, new users (2026-11-09)

Full tables: `reports/runs/2026-11-09_retrievers_percentile_positives_segments.md`.
Recall@100 / @500 for all known users; new users after the semicolon:
- item neighbors, 50, exponent 0.5, liked from 0.5: 0.356 / 0.692; 0.488 / 0.745
- two towers, disliked weight 0, liked from 0.5: 0.338 / 0.728; 0.367 / 0.717
- two towers, disliked weight 100, liked from 0.5: 0.329 / 0.716; 0.360 / 0.714
- two towers, disliked weight 0, liked from 0.7: 0.305 / 0.668; 0.369 / 0.716
- popularity, liked from 0.5: 0.247 / 0.593; 0.382 / 0.708
- FM, 32 factors (ids only): 0.195 / 0.411; 0.323 / 0.564

Threshold 0.7 leaves 201 users without positives: their most common rating is
5, whose mid-rank percentile is below 0.7.

### Run 5 — history-based retrievers (2026-11-09; partial copy taken while running)

Added: release period (five-year bins) in the movie tower and in the MLP ranker;
FM with demographics, genres and release period. New models, all with positives
from 0.5 and future liked movies masked as negatives:
- two towers with the mean of liked movies among the last 50 ratings (plus the
  log of their count) in the user tower;
- the sequence (SASRec-style) user tower, a token per rating = movie + rating +
  genres + release period + position, a demographics token, 2 pre-LN encoder
  layers, no user id; with the feature movie tower;
- the same sequence user tower with flat movie embeddings shared with its input.

Target window: the history ends right before the target (1), up to 5 ratings
before, or anywhere before (all later liked movies). Partial table:
`reports/runs/2026-11-09_history_retrievers_partial.md`. Recall@100 / @500, all
known users:
- two towers by id, with release period: 0.336 / 0.733 (bad@100 0.182)
- FM with side features, 32 factors: 0.133 / 0.366 (bad@100 0.037)
- liked-mean towers, window 1: 0.462 / 0.785 (bad@100 0.322)
- liked-mean towers, all later: 0.357 / 0.733 (bad@100 0.188)
- sequence tower + movie tower, window 1: 0.633 / 0.866 (bad@100 0.527)
- sequence tower + movie tower, window 5: 0.637 / 0.875 (bad@100 0.506)

**Not trusted.** The sequence tower nearly doubles recall while the share of bad
movies in its candidates rises to about 0.5: it predicts what the user rates
next, liked or not. The likely cause is split A's block boundaries inside one
second: the last known ratings and the first validation ratings are often from
the same questionnaire page, ordered by movie id. Window-1 models gain the most,
which fits this explanation. To be rerun on a split whose block boundaries never
fall inside a second.

### Run 5, complete (2026-11-09)

Full table: `reports/runs/2026-11-09_history_retrievers_complete.md`. Recall@100 /
@500 for all known users (bad@100); new users after the semicolon:
- liked-mean towers, window 1: 0.462 / 0.785 (0.322); 0.606 / 0.860
- liked-mean towers, all later: 0.357 / 0.733 (0.188); 0.492 / 0.803
- sequence tower + movie tower, window 1: 0.633 / 0.866 (0.527); 0.764 / 0.918
- sequence tower + movie tower, window 5: 0.637 / 0.875 (0.506); 0.731 / 0.929
- sequence tower + movie tower, all later: 0.364 / 0.724 (0.177); 0.498 / 0.800
- sequence tower + flat movie embeddings, window 1: 0.646 / 0.872 (0.544); 0.760 / 0.943
- sequence tower + flat movie embeddings, window 5: 0.633 / 0.876 (0.499); 0.754 / 0.933
- sequence tower + flat movie embeddings, all later: 0.364 / 0.725 (0.166); 0.473 / 0.807

Only models trained to predict the next one to five ratings jump; trained on all
later liked movies, the same towers land next to the id-only towers (0.364
against 0.336) with a normal share of bad movies. This fits the explanation that
the jump comes from the validation block continuing the same questionnaire
session as the end of the known history, not from taste.

### Run 6 — tie order check of the sequence tower (2026-11-10)

Split B, sequence user tower with the feature movie tower, window 1, liked from
0.5, trained twice: ratings of one second ordered by movie id, and in a
reproducible random order (hash of user and movie). Recall@100 of all known
users, split by whether the validation block starts in the same second as the
last known rating (47.6% of users) or in a later second (bad@100 in brackets):

| Tie order | All users | Same second | Later second |
|---|---|---|---|
| movie id | 0.635 (0.626–0.644), bad 0.524 | 0.729 (0.718–0.740), bad 0.633 | 0.550 (0.538–0.563), bad 0.430 |
| random | 0.651 (0.641–0.660), bad 0.546 | 0.744 (0.732–0.755), bad 0.643 | 0.567 (0.554–0.579), bad 0.462 |

The movie-id order is not the cause: a random order does not lower recall.
Sharing a second with the validation block adds about 0.18 of recall, so split
A's boundaries inside one second do leak. But users whose validation starts in
a later second still reach 0.55 with a high share of bad movies, so most of the
jump remains: the model predicts what the user rates next in the same sitting,
liked or not. The split with whole-second boundaries (split C) removes the first
effect; its same-day / later-day segments measure the second.

## Split C (run 7 on)

Split B with these changes:
- Every block boundary falls between seconds: a block takes whole seconds from
  the end until it holds at least its size, so all known ratings are strictly
  earlier than every held-out rating (tested). A user who cannot fill the three
  held-out blocks keeps everything in history; there is no minimum history.
- New users: the held-out ratings start in the first second after their 15th
  rating and take whole seconds; each is checked with histories of their first
  0, 1, 3, 5, 10 and 15 ratings.
- Gain = 2 · max(0, percentile − 0.5): ratings at or below the median of the
  user's scale add nothing; bad movies (percentile < 0.2) are counted
  separately. The user's scale for labels is the history plus the held-out block.
  Recall is defined only for users with a positive held-out gain (about 10% of
  users have none in the validation block).
- Known users are grouped by known history length (0–4, 5–14, 15–49, 50+; the
  first group is empty in the validation stage, because the known history always
  includes the 5-rating ranker block) and by whether the validation block starts
  the same day as the known history ends (4,898 users) or a later day (235).
- Positives from 0.5 only; no extra disliked negative; id hidden in 5% of
  examples at any history length.

### Run 7 — stage 1 retrievers on split C (2026-11-10)

First 14 models (the run then failed on new users with an empty history in the
history models; fixed and the history models rerun separately). Full tables:
`reports/runs/2026-11-10_stage1_retrievers_split_c.md`. Recall@100 / @500
(bad@100):

| Retriever | All known (5,133) | Same day (4,898) | Later day (235) | New, 0 ratings | New, 5 | New, 15 |
|---|---|---|---|---|---|---|
| item neighbors, 50, exponent 0.5 | 0.413 / 0.756 (0.176) | 0.416 | 0.363 | 0.392 | 0.302 | 0.506 |
| item neighbors, 100, exponent 0.5 | 0.397 / 0.770 (0.173) | 0.399 | 0.351 | 0.392 | 0.322 | 0.509 |
| item neighbors, 50, exponent 1.0 | 0.368 / 0.665 (0.148) | 0.368 | 0.371 | 0.392 | 0.395 | 0.458 |
| two towers by id | 0.395 / 0.785 (0.178) | 0.397 | 0.348 | 0.370 | 0.370 | 0.378 |
| popularity, liked from 0.5 | 0.295 / 0.656 (0.139) | 0.292 | 0.344 | 0.392 | 0.395 | 0.403 |
| popularity, all ratings | 0.284 / 0.644 (0.162) | 0.282 | 0.317 | 0.379 | 0.380 | 0.380 |
| FM with side features, 64 factors | 0.200 / 0.472 (0.049) | 0.197 | 0.255 | 0.277 | 0.277 | 0.281 |
| FM with side features, 32 factors | 0.190 / 0.462 (0.047) | 0.186 | 0.262 | 0.258 | 0.258 | 0.261 |

With no history, item neighbors fall back to popularity, as designed. With 5
ratings, item neighbors with exponent 0.5 are worse than popularity (0.30
against 0.40); with 15 they are clearly better (0.51). On the later-day users
the gap between models narrows and popularity is close to the best.

### Run 8 — stage 1 history retrievers on split C (2026-11-10)

The eight history models that failed in run 7, rerun after the empty-window fix.
Window 50, positions on, mask of future liked movies on, id hidden in 5% of
examples, no user or recency weights. Full tables:
`reports/runs/2026-11-10_stage1_history_retrievers_split_c.md`. Recall@100 /
@500 (bad@100); run 7 rows for comparison:

| Retriever | All known (5,133) | Same day (4,898) | Later day (235) | New, 0 | New, 5 | New, 10 | New, 15 |
|---|---|---|---|---|---|---|---|
| sequence, movie tower, within 5 | 0.649 / 0.894 (0.486) | 0.661 | 0.398 | 0.372 | 0.394 | 0.593 | 0.745 |
| sequence, flat movies, within 5 | 0.647 / 0.893 (0.478) | 0.661 | 0.367 | 0.307 | 0.341 | 0.559 | 0.737 |
| sequence, flat movies, within 1 | 0.637 / 0.876 (0.506) | 0.650 | 0.358 | 0.290 | 0.332 | 0.520 | 0.739 |
| sequence, movie tower, within 1 | 0.627 / 0.873 (0.490) | 0.639 | 0.373 | 0.354 | 0.362 | 0.527 | 0.738 |
| liked mean, movie tower, within 1 | 0.496 / 0.808 (0.299) | 0.504 | 0.330 | 0.344 | 0.358 | 0.479 | 0.630 |
| sequence, flat movies, all later | 0.427 / 0.786 (0.180) | 0.430 | 0.346 | 0.404 | 0.392 | 0.494 | 0.515 |
| sequence, movie tower, all later | 0.425 / 0.783 (0.185) | 0.428 | 0.355 | 0.403 | 0.421 | 0.447 | 0.510 |
| liked mean, movie tower, all later | 0.412 / 0.789 (0.183) | 0.415 | 0.364 | 0.369 | 0.382 | 0.449 | 0.499 |
| (run 7) item neighbors, 50, exponent 0.5 | 0.413 / 0.756 (0.176) | 0.416 | 0.363 | 0.392 | 0.302 | — | 0.506 |
| (run 7) popularity, liked from 0.5 | 0.295 / 0.656 (0.139) | 0.292 | 0.344 | 0.392 | 0.395 | — | 0.403 |

Short targets (the liked movies among the next 1 or 5 ratings) give the large
jump, but mostly on users whose validation block starts the same day: there the
model learns which movies a person rates next in the same sitting, good or bad,
which is why bad@100 is 0.49 against 0.18 for item neighbors. On the 235
later-day users the best model (sequence, movie tower, within 5) is 0.398
against 0.363 for item neighbors and 0.344 for popularity: still the best, but
with overlapping intervals. Targets of all later liked movies lose the session
signal and land at the item-neighbor level. For new users with up to 5 ratings
no model beats popularity; from 10 ratings the short-target sequence towers lead
clearly. The movie tower with features is better than flat movie embeddings for
new users with short histories and equal on known users. The high bad recall
means the ranker after this retriever must push bad movies down; that is checked
in stage 4. Stage 2 varies the training of "sequence, movie tower, within 5".

### Run 9 — stage 2 training variants of SASRec, stopped (2026-11-10)

Variants of "sequence user tower, movie tower, within 5" (window 50, positions
on, training cut anywhere). Stopped after 4 of 10 variants, because the base
for training variants is to be chosen between SASRec and the transformer without
positions first, and the training cut is to be fixed. Partial tables:
`reports/runs/2026-11-10_stage2_history_variants_split_c.md`. Recall@100:

| Variant | All known | Later day | New, 5 | New, 15 |
|---|---|---|---|---|
| base, trained again | 0.651 (bad 0.487) | 0.398 | 0.400 | 0.743 |
| user weight count^-0.5 | 0.650 (bad 0.485) | 0.400 | 0.377 | 0.737 |
| user weight count^-1 | 0.594 (bad 0.415) | 0.356 | 0.341 | 0.691 |
| recency half-life 365 days | 0.647 (bad 0.477) | 0.389 | 0.382 | 0.742 |

Training the base again gives 0.651 against 0.649 in run 8: differences below
about 0.01 are noise. No variant beats the base; count^-1 hurts.

Found while planning: the training window end is drawn by rating count, so it
often falls inside a second, and the window then holds ratings given in the same
second as the target. At query time that never happens. In ML-1M 77% of ratings
share their second with another rating of the same user (2.1 ratings per
second on average). Added cuts_between_seconds: a window end inside a second
moves back to the start of that second; the window length stays the same.
