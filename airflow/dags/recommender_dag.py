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
from typing import Literal, TypedDict

from airflow.sdk import dag, task
import duckdb
import joblib
import pandas as pd

from logic.recommender_fm import split, fit_best_estimator, \
                count_baseline_score, FittedModel


DATA_DIR = "/opt/airflow/data"
MODEL_TABLE = "ready_models"

RECOMMEND_NUM = 5
MAX_TEST_RATIO = 0.1
METRIC = "ndcg"


class ModelQualityError(Exception):
    pass


class ModelMetadata(TypedDict):
    path: str
    description: str
    val_score: float


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


def _save_model(model: FittedModel):
    models_dir = os.path.join(DATA_DIR, "models")
    today = datetime.now().date()
    os.makedirs(models_dir, exist_ok=True)
    file_name = f"{model["name"]}_{today}.joblib"
    model_path = os.path.join(models_dir, file_name)
    joblib.dump(model, model_path)
    return model_path


def _make_models_table():
    with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
        db_conn.execute(
            f"CREATE TABLE IF NOT EXISTS {MODEL_TABLE} ("
            "model_path VARCHAR(200) PRIMARY KEY, " \
            "description VARCHAR(300), " \
            "test_score DOUBLE" \
            ")"
        )


def _update_table(
        model_path: str,
        description: str,
        score: float,
):
    with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
        db_conn.execute(
            f"INSERT INTO {MODEL_TABLE} VALUES (?, ?, ?)",
            [model_path, description, score]
        )


def _calculate_scores(
        test_path: str,
        model_metadata: ModelMetadata,
):
    X_test, y_test = joblib.load(test_path)
    model_path = model_metadata["path"]
    model_info: FittedModel = joblib.load(model_path)
    X_test_encoded = model_info["fitted_transformer"].transform(X_test)
    test_score = model_info["fitted_estimator"].score(
        X_test_encoded,
        y_test,
        X_test["user_id"],
    )
    val_score = model_info["val_score"]
    return val_score, test_score, model_path, model_info["description"]


def _validate_quality(
          val_score: float,
          test_score: float,
          model_path: str,
          model_description: str,
          baseline_score: float,
):
    if test_score <= baseline_score:
        raise ModelQualityError(
            f"Model score {test_score}, baseline {baseline_score}"
        )
    elif val_score > 1.5 * test_score:
        raise ModelQualityError(
            f"Model test score {test_score}, validation score {val_score}"
        )
    else:
        _make_models_table()
        _update_table(
            model_path,
            model_description,
            test_score,
        )


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    @task(multiple_outputs=True)
    def prepare_data():
        """
            Make training, validation, and test datasets.
        """
        ratings_df = _extract_df()
        features = ratings_df.drop(columns="rating")
        ratings = ratings_df["rating"]
        train_all, test_all = split(
            features, 
            ratings, 
            test_max_per_var=2 * RECOMMEND_NUM,
            max_test_ratio=MAX_TEST_RATIO,
        )
        train_path = os.path.join(DATA_DIR, "train_set.joblib")
        test_path = os.path.join(DATA_DIR, "test_set.joblib")
        joblib.dump(train_all, train_path)
        joblib.dump(test_all, test_path)
        return {
            "train": train_path,
            "test": test_path,
        }

    @task
    def build_fit_save_estimator(train_path: str):
        """
            Preprocess data, train model, and save it locally.
        """
        train_sets = joblib.load(train_path, mmap_mode='r')
        model = fit_best_estimator(
            *train_sets,
            recommend_num=RECOMMEND_NUM,
            metric=METRIC,
        )
        model_path = _save_model(model)

        return {
            "path": model_path,
            "description": model["description"],
            "val_score": model["val_score"],
        }
    
    @task
    def prepare_baseline(train_path: str, test_path: str):
        train_sets = joblib.load(train_path, mmap_mode='r')
        X_test, y_test = joblib.load(test_path)
        X_train_all = pd.concat(train_sets[:3])
        y_train_all = pd.concat(train_sets[3:])
        return count_baseline_score(
            X_train_all,
            X_test, 
            y_train_all, 
            y_test,
            RECOMMEND_NUM,
        )

    @task
    def evaluate_model(
        model_metadata: ModelMetadata,
        baseline_score: float,
        test_path: str,
    ):
        """
            
        """
        results = _calculate_scores(test_path, model_metadata)
        _validate_quality(*results, baseline_score)
        

    datasets = prepare_data()
    train_path = datasets["train"]
    test_path = datasets["test"]
    model_metadata = build_fit_save_estimator(train_path)
    baseline_score = prepare_baseline(train_path, test_path) 
    evaluate_model(model_metadata, baseline_score, test_path)


recommender()
