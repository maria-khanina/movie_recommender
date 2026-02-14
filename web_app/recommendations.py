import os

import duckdb
import joblib
import numpy as np
import pandas as pd


DATA_DIR = os.environ["DATA_DIR"]
PRODUCTION_MODEL_PATH = os.path.join(DATA_DIR, os.environ["PRODUCTION_MODEL"])


with duckdb.connect(os.environ["DB_PATH"], read_only=True) as db_conn:
    all_movies = db_conn.execute(
        "SELECT movie_id, title FROM movies"
    ).fetchdf()

all_movies.set_index("movie_id")


def extract_unseen_movies(user_id: int) -> np.array:
    with duckdb.connect(os.environ["DB_PATH"], read_only=True) as db_conn:
        seen_movies = db_conn.execute(
            "SELECT movie_id "
            "FROM ratings WHERE user_id=?",
            [user_id],
        ).fetchdf()["movie_id"].to_numpy()

    unseen_ids = np.setdiff1d(all_movies.index, seen_movies)

    return unseen_ids


def recommend(user_id: int, recommend_num: int = 5):
    unseen_movie_ids = extract_unseen_movies(user_id)
    target_df = pd.DataFrame({
        "user_id": np.full(unseen_movie_ids.shape[0], user_id),
        "movie_id": unseen_movie_ids,
    })
    model = joblib.load(PRODUCTION_MODEL_PATH)
    df_transformed = model["fitted_transformer"].transform(target_df)
    predictions = model["fitted_estimator"].predict(df_transformed)
    top_indices = np.argsort(predictions)[-recommend_num:][::-1]
    recommended_ids = unseen_movie_ids[top_indices]
    return all_movies.loc[recommended_ids, "title"]
