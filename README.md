# Movie Recommender

A two-stage movie recommender on MovieLens-1M (6,040 users, 3,883 movies,
1,000,209 ratings): a retrieval model picks a few hundred candidates out of the
whole catalog, and a ranker orders them into the final top ten. Every model is
chosen on a validation stage built to match what the site sees in production;
the test stage is opened once, for the final report.

The selected models are trained by an Airflow DAG, which checks the new model
against a baseline before it replaces the production model, and are served by a
FastAPI web application. Everything runs in Docker.

## What the system recommends

A recommendation is good when the user will choose the movie *and* like it.
"Liked" is defined on each user's own scale: a rating's percentile among that
user's ratings (mid-rank for ties). A 4 from a user who gives everything 5 is a
weak rating; a 4 from a strict user is a strong one. Every model learns from
these user-relative labels, and every metric uses them:

- gain of a held-out movie = 2 · max(0, percentile − 0.5): movies at or below
  the user's median add nothing, the user's favorite adds 1;
- bad movie = percentile below 0.2; how many of them get into the
  recommendations is reported separately, because recommending a movie the user
  will dislike is worse than missing one they would like.

## Evaluation design

- **Temporal split per user.** Each user's ratings are ordered by time. The
  latest block is the test, the one before it the validation, the one before
  that a ranker-training block, and the rest is history. Every block holds whole
  seconds of ratings, so every rating the models know is strictly earlier than
  every rating they are checked on.
- **New users.** 5% of users are held out of all training. Each of them is
  served from their first 0, 1, 3, 5, 10 and 15 ratings, and checked on the
  ratings that follow: a curve of quality against the number of ratings a new
  member has given.
- **Segments.** Known users are reported by history length and by whether the
  checked ratings start on the same day as the known history ends or on a later
  day. The later-day users show how well a model knows a person's taste rather
  than the rest of a rating session.
- **Metrics.** Retrieval: gain-weighted recall at 100, 200 and 500 candidates
  and the share of held-out bad movies that got into the candidates. Ranking:
  NDCG@10, the share of bad movies in the ten slots, novelty and catalog
  coverage.
- **Uncertainty.** 95% bootstrap intervals over users; models are compared by
  paired differences on the same users, which removes the shared difficulty of
  each user. A model is retrained with another seed to measure training noise
  (about 0.01 of recall), and smaller differences are not acted on.

## Retrieval models

| Model | Idea |
|---|---|
| Popularity | most liked movies; the baseline for every user |
| Item neighbors | movies liked by the same people as the user's liked movies; similarity c(i, j) / (n(i)^a · n(j)^(1−a)), from cosine (a = 0.5) to the conditional share (a = 1) |
| Factorization machine | user and movie ids with demographics, genres and release period; its prediction is split into a user vector and a movie vector for vector search |
| Two towers | user tower (id, age, gender, occupation) and movie tower (id, genres, release period), in-batch softmax with the logQ popularity correction |
| Liked-mean towers | the user tower also reads the mean embedding of the liked movies among the last ratings |
| SASRec user tower | a transformer over the last 50 ratings (movie, rating, genres, release period, position) and a demographics token; no user id, so a new user is served from the first ratings without retraining |
| Transformer over a set | the same tower without position embeddings: it sees the window as a set of ratings |

All embedding models search the catalog with a FAISS HNSW index. The history
models are trained on (window, target) pairs cut from each user's history: the
window ends before a liked movie, and the targets are the liked movies that
follow it, within the next k ratings.

### Results on validation

Gain-weighted recall@100 (and @500 for all known users); the bad column is the
share of held-out bad movies among the 100 candidates. 5,133 known users, of
whom 235 continue on a later day; 146 new users.

| Model | All known | Bad | Later day | New, 0 ratings | New, 5 | New, 15 |
|---|---|---|---|---|---|---|
| Popularity | 0.295 / 0.656 | 0.139 | 0.344 | 0.392 | 0.395 | 0.403 |
| Factorization machine | 0.200 / 0.472 | 0.049 | 0.255 | 0.277 | 0.277 | 0.281 |
| Item neighbors, cosine | 0.413 / 0.756 | 0.176 | 0.363 | 0.392 | 0.302 | 0.506 |
| Two towers | 0.395 / 0.785 | 0.178 | 0.348 | 0.370 | 0.370 | 0.378 |
| Liked-mean towers | 0.496 / 0.808 | 0.299 | 0.330 | 0.344 | 0.358 | 0.630 |
| Transformer over a set | 0.621 / 0.883 | 0.443 | 0.372 | 0.379 | 0.442 | 0.722 |
| **SASRec user tower** | **0.654 / 0.895** | 0.483 | **0.414** | 0.365 | 0.396 | **0.738** |

What the experiments showed:

- **The target horizon matters more than the architecture.** The same SASRec
  tower trained on the liked movies among the next 1 or 5 ratings reaches 0.63
  and 0.65; trained on all later liked movies, 0.43, the level of item
  neighbors.
- **Much of the gain is the rest of the rating session.** For users who keep
  rating on the same day SASRec reaches 0.67, for those who come back on a
  later day 0.41. It also brings in more bad movies: it learns which movies a
  person rates next, and in the same sitting people rate both movies they like
  and movies they do not. Pushing those down is the ranker's job.
- **Order in the window helps.** Removing the position embeddings costs about
  0.03 of recall for known users, with no difference for new users.
- **Movie features help new users.** A movie tower with genres and release
  period, instead of the SASRec movie embedding alone, gives the same recall to
  known users and more to new users with few ratings (0.37 against 0.31 with no
  ratings).
- **New users.** With up to five ratings no model clearly beats popularity;
  from ten ratings the sequence towers lead, and with fifteen SASRec finds
  0.74 against 0.51 for item neighbors.
- **Weighting did not help.** Down-weighting heavy users (count^−0.5 and
  count^−1) and decaying old ratings (half-life of a year) did not raise recall;
  count^−1 lowered it by 0.06.
- **The factorization machine ranks poorly as a retriever.** It is trained to
  predict the rating, so it surfaces rarely watched, highly rated movies; it has
  the lowest share of bad movies, which makes it more useful as a ranker
  feature than as a retriever.

## Ranking

The ranker reorders the retriever's candidates. A movie deserves a slot when the
user will choose it and like it, so most variants multiply two estimates: the
probability that the user will rate the movie, and ĝ, a regressor of the user's
gain from the movie if they rate it. The gain regressor is a multilayer
perceptron with early crossing of user and movie features, a factorization
machine, or a transformer over the feature fields (AutoInt). Variants compared
on the same candidates and users:

- **Retriever probability × gain:** log ĝ + α · retriever score. The
  retriever's softmax score already estimates which movies the user will rate,
  so no extra model is needed; α is chosen on validation.
- **Separate selection model × gain:** log p + β · log ĝ, where p is the
  probability that the user rates the candidate, predicted by a separate head
  trained on the ranker block. Unlike the retriever, it sees user and movie
  features together from its first layer, at the cost of its own training data.
- **List-wise rankers** trained on the ranker block to order the candidate list
  directly: ListNet, LambdaRank, gain-weighted BPR and a hinge loss.

NDCG@10, the share of bad movies in the ten slots and catalog coverage are
reported against the retriever's own order.

## Test results

The test stage is opened once, for the selected retriever and ranker; its
results are reported here.

## Pipeline and serving

- `airflow/dags/recommender_dag.py`: loads the current ratings, trains the
  selected models, evaluates them against a baseline, and stores the model only
  if it passes the quality check.
- `web_app/`: a FastAPI application with a web page: search movies, rate them,
  and get recommendations that use the new ratings at once, because the user
  tower reads the rating history at query time.

## Repository

```
dataset/      MovieLens download and import into DuckDB
evaluation/   temporal split, new users, gains, metrics, retriever and ranker comparisons
models/       retrievers, rankers, vector search
airflow/      training DAG
web_app/      FastAPI application and front end
tests/        unit tests, including reference implementations of every metric
reports/      validation reports and the log of every run
```

## Running

```
python -m dataset.movielens_loader            # DATA_DIR, DB_PATH
python -m evaluation.retriever_comparison     # DB_PATH; writes reports/retriever_validation.md
python -m evaluation.ranker_comparison        # DB_PATH; writes reports/ranker_validation.md
python -m pytest tests
docker compose up                             # Airflow and the web application
```

The code is type-checked with pyright in strict mode and checked with
pycodestyle.

Stack: Python, PyTorch, FAISS, NumPy, pandas, SciPy, DuckDB, Airflow, FastAPI,
Docker.
