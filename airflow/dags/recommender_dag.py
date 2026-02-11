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

from logic.recommender_fm import split, find_best_estimator


supported_datasets = Literal["100k", "1m"]
supported_models = Literal["KNN", "FM"]

CURR_MODEL: supported_models = "FM"
DATASET: supported_datasets = "1m"
DATA_DIR = "/opt/airflow/data"
TOP_NUM = 5


def _extract_df():
    """Return joined dataframe of users, movies, and ratings."""
    db_path = os.environ["DB_PATH"]
    with duckdb.connect(db_path, read_only=True) as db_conn:
        return db_conn.execute("""
            SELECT ratings.*, movies.*, users.*
            FROM ratings
                JOIN movies USING (movie_id) 
                JOIN users USING (user_id)
        """).df()


def _save_estimator(model_name: str, estimator):
    models_dir = os.path.join(DATA_DIR, "models")
    today = datetime.now().date()
    os.makedirs(models_dir, exist_ok=True)
    file_name = f"{model_name}_{today}.joblib"
    model_path = os.path.join(models_dir, file_name)
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


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    @task
    def prepare_data():
        """
            Make training, validation, and test datasets.
        """
        ratings_df = _extract_df()
        features, ratings = ratings_df.drop(columns="rating"), ratings_df["rating"]
        train_all, test_all = split(features, ratings)
        train_path = os.path.join(DATA_DIR, "train_set.joblib")
        test_path = os.path.join(DATA_DIR, "test_set.joblib")
        joblib.dump(train_all, train_path)
        joblib.dump(test_all, test_path)
        return train_path, test_path

    @task
    def build_fit_save_estimator(train_path: str):
        """
            Preprocess data, train model, and save it locally.
        """
        train_sets = joblib.load(train_path)
        model = find_best_estimator(train_sets)
        model_path = _save_estimator(
            model["name"],
            model["fitted_estimator"],
        )

        return {
            "path": model_path,
            "description": model["description"],
            "val_score": model["val_score"],
        }

    @task
    def evaluate_model(
        recent_model_info: dict[Literal["path", "description", "val_score"], str],
        test_path: str,
    ) -> bool:
        """
            Compare recent model with current best, add model metadata
            to all_models table. Return True if best model was updated
            and False otherwise.
        """
        X_test, y_test = joblib.load(test_path)
        recent_path = recent_model_info["path"]
        recent_model = joblib.load(recent_path)
    
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
            else:
                _compare_models(
                    best_model, recent_model, recent_path,
                    X_val, y_val, test_score,
                    db_conn, best_table, 
                )

    file_loading = load_files()
    make_users = convert_to_table("users.csv")
    make_movies = convert_to_table("movies.csv")
    make_ratings = convert_to_table("ratings.csv")
    file_loading >> [make_movies, make_ratings, make_users]

    splitting = prepare_data()
    training = build_fit_save_estimator()
    [make_movies, make_ratings, make_users] >> splitting >> training

    evaluate_model(training)

recommender()
