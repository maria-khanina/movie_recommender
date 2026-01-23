"""
Title: Recommender DAG

Author: Masha Khanina
Reviewer: Nir Landerer

Description: Airflow that loads relevant MovieLens files,
makes a join dataframe from them, preprocesses data, trains and saves
model, compares model with current saved best, and makes dataframe of
movie recommendations for each user of MovieLens.
"""
from datetime import datetime
import os

from airflow.sdk import dag, task
import duckdb
import joblib
import numpy as np
import pandas as pd

from logic.loader import load_table, load_raw_files
from logic.recommender import extract_features, split, \
            build_predictor, recommender_scorer, GENRES


DATASETS_PATH = "/opt/airflow/data/splitted_datasets.joblib"
FILES_DIR = "/opt/airflow/data/files"
FILE_SEPARATORS = {"u.user": "|", "u.item": "|", "u.data": "\t"}
MODELS_DIR = "/opt/airflow/data/models"
TOP_NUM = 5


def _extract_df():
    """Return joined dataframe of users, movies, and ratings."""
    db_path = os.environ["DB_PATH"]
    with duckdb.connect(db_path, read_only=True) as db_conn:
        return db_conn.execute("""
            SELECT user_id, movie_id, rating, movies.*, users.*
            FROM ratings 
                JOIN movies USING (movie_id) 
                JOIN users USING (user_id)
        """).df()


def _save_model(model_name: str, estimator):
    today = datetime.now().date()
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_name = f"{model_name}_{today}.joblib"
    model_path = os.path.join(MODELS_DIR, file_name)
    joblib.dump(estimator, model_path)
    return model_path


def _make_models_tables(db_conn: duckdb.DuckDBPyConnection):
    all_table = "all_models"
    best_table = "best_model"
    db_conn.execute(
        f"CREATE TABLE IF NOT EXISTS {all_table} ("
        "model_path VARCHAR(200) PRIMARY KEY, " \
        "description VARCHAR(300), " \
        "test_score DOUBLE" \
        ")"
    )
    db_conn.execute(
        f"CREATE TABLE IF NOT EXISTS {best_table} ("
        "model_path VARCHAR(200), " \
        "description VARCHAR(300)," \
        "test_score DOUBLE" \
        ")"
    )
    return all_table, best_table


def _update_table(
        db_conn: duckdb.DuckDBPyConnection,
        table_name: str,
        model_path: str,
        description: str,
        score: float,
):
    db_conn.execute(
        f"INSERT INTO {table_name} VALUES (?, ?, ?)",
        [model_path, description, score]
    )


def _compare_models(
        best_model, recent_model, recent_path,
        X_validate, y_validate, test_score,
        db_conn, best_table,
) -> bool:
    best_model_val_score = recommender_scorer(
        best_model, X_validate, y_validate,
    )
    recent_model_val_score = recommender_scorer(
        recent_model, X_validate, y_validate,
    )
    if best_model_val_score < recent_model_val_score:
        db_conn.execute("DELETE FROM best_model")
        _update_table(
            db_conn, best_table, recent_path, 
            recent_model["description"], test_score, 
        )         
        return True
    return False


def _generate_new_pairs():
    """Make pairs (user, movie) for all movies user didn't rate."""
    X_train, X_validate, X_test, _, _, _, = joblib.load(DATASETS_PATH)
    X_all = pd.concat([X_train, X_validate, X_test])
    movies = X_all[
        ["release_year", "release_month"] + GENRES
    ].groupby(level="movie_id").head(1)

    users = X_all[
        ["gender", "occupation", "age"]
    ].groupby(level="user_id").head(1)

    users.reset_index(level="movie_id", drop=True, inplace=True)
    movies.reset_index(level="user_id", drop=True, inplace=True)

    pairs = movies.join(users, how="cross")
    pairs.index = pd.MultiIndex.from_product(
        [movies.index, users.index],
    )
    return pairs.drop(X_all.index, errors='ignore')


def _make_recommendations(new_pairs: pd.DataFrame):
    top = new_pairs[["predicted_rating"]].sort_values(
        by=["user_id", "predicted_rating"],
        ascending=[True, False],
    ).groupby(level="user_id").head(TOP_NUM)
    top.reset_index(drop=False, inplace=True)

    with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
        db_conn.execute("DROP TABLE IF EXISTS recommendations")
        db_conn.execute(
            "CREATE TABLE recommendations AS SELECT " \
            "user_id, movie_id, title " \
            "FROM top JOIN movies USING (movie_id)"
        )


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    """Define recommender system tasks and their dependencies."""
    @task
    def load_files():
        """Load u.user, u.data, u.item to local storage."""
        os.makedirs(FILES_DIR, exist_ok=True)
        load_raw_files(FILES_DIR)

    @task
    def convert_to_table(file_name: str):
        """
            Create SQL table with processed csv file.
            Fits for huge files.
        """
        load_table(
            os.path.join(FILES_DIR, file_name), FILE_SEPARATORS[file_name],
        )

    @task
    def prepare_data():
        """
            Make training, validation, and test datasets.
        """
        ratings_df = _extract_df()
        features, ratings = extract_features(ratings_df)
        splitted = split(features, ratings)
        joblib.dump(splitted, DATASETS_PATH)
    
    @task
    def build_fit_save_model():
        """
            Preprocess data, train model, and save it locally.
        """
        X_train, _, _, y_train, _, _ = joblib.load(DATASETS_PATH)
        model = build_predictor()
        searcher = model["estimator"]
        searcher.fit(X_train, y_train)
        model_path = _save_model(
            model["name"], searcher.best_estimator_,
        )

        return {
            "path": model_path,
            "description": model["description"]
        }
    
    @task
    def evaluate_model(recent_model) -> bool:
        """
            Compare recent model with current best, add model metadata
            to all_models table. Return True if best model was updated
            and False otherwise.
        """
        _, X_validate, X_test, _, \
            y_validate, y_test = joblib.load(DATASETS_PATH)
        recent_path = recent_model["path"]
        test_score = recommender_scorer(
            joblib.load(recent_path), X_test, y_test,
        )
        with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
            all_table, best_table = _make_models_tables(db_conn)
            _update_table(
                db_conn, all_table, recent_path, 
                recent_model["description"], test_score,
            )
            best_record = db_conn.execute(
                f"SELECT model_path FROM {best_table}"
            ).fetchone()
            try:
                best_model = joblib.load(best_record[0])
            except TypeError:
                _update_table(
                    db_conn, best_table, recent_path, 
                    recent_model["description"], test_score, 
                )
                return True
            else:
                if _compare_models(
                    best_model, recent_model, recent_path,
                    X_validate, y_validate, test_score,
                    db_conn, best_table, 
                ):
                    return True
        return False

    @task
    def prepare_recommendations(need_update: bool):
        """
            Save up-to-date recommendations of 5 movies for each user.
        """
        if not need_update:
            return None
        with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
            best_record = db_conn.execute(
                "SELECT model_path FROM best_model"
            ).fetchone()
        best_model = joblib.load(best_record[0])

        new_pairs = _generate_new_pairs()
        new_pairs["predicted_rating"] = best_model.predict(new_pairs)
        _make_recommendations(new_pairs)


    file_loading = load_files()
    make_users = convert_to_table("u.user")
    make_movies = convert_to_table("u.item")
    make_ratings = convert_to_table("u.data")
    file_loading >> [make_movies, make_ratings, make_users]

    splitting = prepare_data()
    training = build_fit_save_model()
    [make_movies, make_ratings, make_users] >> splitting >> training

    evaluation = evaluate_model(training)
    prepare_recommendations(evaluation)

recommender()
