from datetime import datetime
import os

from airflow.decorators import dag, task
import duckdb
import joblib
import numpy as np
import pandas as pd

from logic.loader import load_table
from logic.recommender import extract_df, select_features, split, \
            build_predictor, recommender_scorer, GENRES


DATASETS_PATH = "/opt/airflow/data/splitted_datasets.joblib"


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    @task
    def load_data():
        dir_path = "/opt/airflow/data/raw_files"
        file_separators = {"u.user": "|", "u.item": "|", "u.data": "\t"}
        for file_name, separator in file_separators.items():
            load_table(dir_path + "/" + file_name, separator)

    @task
    def prepare_data():
        ratings_df = extract_df()
        features, ratings = select_features(ratings_df)
        splitted = split(features, ratings)
        joblib.dump(splitted, DATASETS_PATH)
    
    @task
    def build_fit_save_model():
        X_train, _, _, y_train, _, _ = joblib.load(DATASETS_PATH)
        model = build_predictor()
        model["estimator"].fit(X_train, y_train)
        today = datetime.now().date()
        models_dir = "/opt/airflow/data/models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = models_dir + f"/{model["name"]}_{today}.joblib"
        joblib.dump(model.best_estimator_, model_path)

        return {
            "model_path": model_path,
        }
    
    @task
    def evaluate_model(recent_model):
        _, X_validate, X_test, _, \
            y_validate, y_test = joblib.load(DATASETS_PATH)
        recent_predictor = joblib.load(recent_model["model_path"])
        with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
            db_conn.execute(
                "CREATE TABLE IF NOT EXISTS all_models "
                "(model_path VARCHAR(200), score DOUBLE)"
            )
            db_conn.execute(
                "CREATE TABLE IF NOT EXISTS best_model "
                "(model_path VARCHAR(200), test_score DOUBLE)"
            )
            db_conn.execute(
                "INSERT INTO all_models VALUES (?, ?)",
                [recent_model["model_path"], recent_model["score"]]
            )

            best_record = db_conn.execute(
                "SELECT model_path FROM best_model"
            ).fetchone()
        try:
            best_model = joblib.load(best_record[0])
        except TypeError:
            test_score = recommender_scorer(
                recent_predictor, 
                X_test, 
                y_test,
            )
            with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
                db_conn.execute(
                    "INSERT INTO best_model VALUES (?, ?)",
                    [recent_model["model_path"], test_score]
                )
                return True
        else:
            best_model_score = recommender_scorer(
                best_model, 
                X_validate,
                y_validate,
            )
            recent_model_score = recommender_scorer(
                recent_model,
                X_validate,
                y_validate,
            )
            if best_model_score < recent_model_score:
                test_score = recommender_scorer(
                    recent_predictor, 
                    X_test, 
                    y_test,
                )
                with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
                    db_conn.execute(
                        "DELETE FROM best_model;"
                        "INSERT INTO best_model VALUES (?, ?)",
                        [recent_model["model_path"], test_score]
                    )
                    return True
        return False

    @task
    def prepare_recommendations(to_update):
        if not to_update:
            return None
        with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
            best_record = db_conn.execute(
                "SELECT model_path FROM best_model"
            ).fetchone()
        best_model = joblib.load(best_record[0])

        X_train, X_validate, X_test, _, _, _, = joblib.load(DATASETS_PATH)
        X = pd.concat([X_train, X_validate, X_test])
        movies = X[
            ["release_year", "release_month"] + GENRES
        ].groupby(level="movie_id").head(1)
        users = X[
            ["gender", "occupation", "age"]
        ].groupby(level="user_id").head(1)
        users.reset_index(level="movie_id", drop=True, inplace=True)
        movies.reset_index(level="user_id", drop=True, inplace=True)
        pairs = movies.join(users, how="cross")
        pairs.index = pd.MultiIndex.from_product(
            [movies.index, users.index],
        )
        new_pairs = pairs.drop(X.index, errors='ignore')

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
    loading = load_data()
    splitting = prepare_data()
    training = build_fit_save_model()
    loading >> splitting >> training
    evaluation = evaluate_model(training)
    prepare_recommendations(evaluation)

recommender()
