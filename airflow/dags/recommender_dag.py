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


def _extract_df():
    """Return joined dataframe of users, movies, and ratings."""
    db_path = os.environ["DB_PATH"]
    with duckdb.connect(db_path, read_only=True) as db_conn:
        return db_conn.execute("""
            SELECT ratings.user_id, ratings.movie_id, ratings.rating,
            movies.*, users.* 
            FROM ratings 
                JOIN movies USING (movie_id) 
                JOIN users USING (user_id)
        """).df()


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


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    """Define recommender system tasks and their dependencies."""
    @task
    def load_files():
        """Load u.user, u.data, u.item to local storage."""
        dir = "/opt/airflow/data/files"
        os.makedirs(dir, exist_ok=True)
        load_raw_files(dir)
        return dir

    @task
    def load_data(from_dir: str):
        """
            Create SQL database with processed 
            u.data, u.item, u.user csv files.
        """
        file_separators = {"u.user": "|", "u.item": "|", "u.data": "\t"}
        for file_name, separator in file_separators.items():
            load_table(os.path.join(from_dir, file_name), separator)

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
        today = datetime.now().date()
        models_dir = "/opt/airflow/data/models"
        os.makedirs(models_dir, exist_ok=True)
        file_name = f"{model["name"]}_{today}.joblib"
        model_path = os.path.join(models_dir, file_name)
        joblib.dump(searcher.best_estimator_, model_path)

        return {
            "path": model_path,
            "description": model["description"]
        }
    
    @task
    def evaluate_model(recent_model):
        """
            Compare recent model with current best,
            add model metadata to all_models table.
        """
        _, X_validate, X_test, _, \
            y_validate, y_test = joblib.load(DATASETS_PATH)
        recent_path = recent_model["path"]
        recent_predictor = joblib.load(recent_path)
        test_score = recommender_scorer(
            recent_predictor, 
            X_test, 
            y_test,
        )
        with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
            db_conn.execute(
                "CREATE TABLE IF NOT EXISTS all_models ("
                "model_path VARCHAR(200) PRIMARY KEY, " \
                "description VARCHAR(300), " \
                "test_score DOUBLE" \
                ")"
            )
            db_conn.execute(
                "CREATE TABLE IF NOT EXISTS best_model ("
                "model_path VARCHAR(200), " \
                "description VARCHAR(300)," \
                "test_score DOUBLE" \
                ")"
            )
            db_conn.execute(
                "INSERT INTO all_models VALUES (?, ?, ?)",
                [
                    recent_path,
                    recent_model["description"],
                    test_score,
                ]
            )

            best_record = db_conn.execute(
                "SELECT model_path FROM best_model"
            ).fetchone()
        try:
            best_model = joblib.load(best_record[0])
        except TypeError:
            with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
                db_conn.execute(
                    "INSERT INTO best_model VALUES (?, ?, ?)",
                    [
                        recent_path,
                        recent_model["description"],
                        test_score,
                    ]
                )
                return True
        else:
            best_model_val_score = recommender_scorer(
                best_model, 
                X_validate,
                y_validate,
            )
            recent_model_val_score = recommender_scorer(
                recent_model,
                X_validate,
                y_validate,
            )
            if best_model_val_score < recent_model_val_score:
                with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
                    db_conn.execute(
                        "DELETE FROM best_model;"
                        "INSERT INTO best_model VALUES (?, ?, ?)",
                        [
                            recent_path,
                            recent_model["description"],
                            test_score,
                        ]
                    )
                    return True
        return False

    @task
    def prepare_recommendations(need_update):
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
        top_5 = new_pairs[["predicted_rating"]].sort_values(
            by=["user_id", "predicted_rating"],
            ascending=[True, False],
        ).groupby(level="user_id").head(5)
        top_5.reset_index(drop=False, inplace=True)

        with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
            db_conn.execute("DROP TABLE IF EXISTS recommendations")
            db_conn.execute(
                "CREATE TABLE recommendations AS SELECT " \
                "user_id, movie_id, title " \
                "FROM top_5 JOIN movies USING (movie_id)"
            )

    file_loading = load_files()
    database_loading = load_data(file_loading)
    splitting = prepare_data()
    training = build_fit_save_model()
    database_loading >> splitting >> training
    evaluation = evaluate_model(training)
    prepare_recommendations(evaluation)

recommender()
