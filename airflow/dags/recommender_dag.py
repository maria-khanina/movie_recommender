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
from typing import Literal

from airflow.sdk import dag, task
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from logic.loader import load_table, load_raw_files
from logic.recommender_fm import extract_features, split, \
            build_fit_estimator, mean_lowest_scorer

supported_datasets = Literal["100k", "1m"]
supported_models = Literal["KNN", "FM"]

CURR_MODEL: supported_models = "FM"
DATASET: supported_datasets = "1m"
DATASETS_PATH = "/opt/airflow/data/splitted_datasets.joblib"
FILES_DIR = "/opt/airflow/data/files"
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


def _save_estimator(model_name: str, estimator: BaseEstimator):
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
        X_val, y_val, test_score,
        db_conn, best_table,
):
    best_model_val_score = mean_lowest_scorer(
        best_model, X_val, y_val, TOP_NUM
    )
    recent_model_val_score = mean_lowest_scorer(
        recent_model, X_val, y_val, TOP_NUM
    )
    if best_model_val_score < recent_model_val_score:
        db_conn.execute("DELETE FROM best_model")
        _update_table(
            db_conn, best_table, recent_path, 
            recent_model["description"], test_score, 
        )


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    """Define recommender system tasks and their dependencies."""
    @task
    def load_files():
        """Load users, movies, ratings to local storage."""
        load_raw_files(FILES_DIR)

    @task
    def convert_to_table(file_name: str):
        """
            Create SQL table with processed csv file.
            Fits for huge files.
        """
        load_table(
            os.path.join(FILES_DIR, file_name),
            DATASET,
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
    def build_fit_save_estimator():
        """
            Preprocess data, train model, and save it locally.
        """
        X_train, X_val, _, y_train, y_val, _ = joblib.load(DATASETS_PATH)
        model = build_fit_estimator(X_train, X_val, y_train, y_val)
        model_path = _save_estimator(
            model["name"], 
            model["estimator"],
        )

        return {
            "path": model_path,
            "description": model["description"],
        }

    @task
    def evaluate_model(
        recent_model: dict[Literal["path", "description"], str],
    ) -> bool:
        """
            Compare recent model with current best, add model metadata
            to all_models table. Return True if best model was updated
            and False otherwise.
        """
        _, X_val, X_test, _, \
            y_val, y_test = joblib.load(DATASETS_PATH)
        recent_path = recent_model["path"]
        test_score = mean_lowest_scorer(
            joblib.load(recent_path), X_test, y_test, TOP_NUM,
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
                    X_val, y_val, test_score,
                    db_conn, best_table, 
                ):
                    return True
        return False

    file_loading = load_files()
    make_users = convert_to_table("u.user")
    make_movies = convert_to_table("u.item")
    make_ratings = convert_to_table("u.data")
    file_loading >> [make_movies, make_ratings, make_users]

    splitting = prepare_data()
    training = build_fit_save_estimator()
    [make_movies, make_ratings, make_users] >> splitting >> training

    evaluation = evaluate_model(training)

recommender()
