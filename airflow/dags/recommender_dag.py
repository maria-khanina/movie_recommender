"""
Title: Recommender DAG

Author: Masha Khanina
Reviewer: Shani Gendelis

Description: Airflow that takes up-to date dataset of users rating movies,
splits to train and test, fits and tunes Factorization Machine, compares
based model to a baseline, fits the model on the whole dataset, and saves
fitted model in storage.

Infinity Labs R&D AI #5.
"""
from datetime import datetime
import os
from typing import Literal, TypedDict

from airflow.sdk import dag, task
import duckdb
import joblib
import numpy as np
import pandas as pd

from logic.recommender_fm import split, find_best_model, fit_model_final, \
                count_baseline_score, FittedModel, ModelConfig


DATA_DIR = os.environ["DATA_DIR"]
PRODUCTION_MODEL_PATH = os.path.join(DATA_DIR, os.environ["PRODUCTION_MODEL"])
MODELS_TABLE = "ready_models"

RECOMMEND_NUM = 5
MAX_TEST_RATIO = 0.1
METRIC = "ndcg"


class ModelQualityError(Exception):
    pass


class ModelMetadata(TypedDict):
    path: str
    config: ModelConfig
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
    """Download model to the storage."""
    models_dir = os.path.join(DATA_DIR, "models")
    time = datetime.today()
    os.makedirs(models_dir, exist_ok=True)
    file_name = f"{model["name"]}_{time}.joblib"
    model_path = os.path.join(models_dir, file_name)
    joblib.dump(model, model_path)
    return model_path


def _make_models_table():
    """Create table for recording fitted models."""
    with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
        db_conn.execute(
            f"CREATE TABLE IF NOT EXISTS {MODELS_TABLE} ("
            "model_path VARCHAR(200) PRIMARY KEY, " \
            "config VARCHAR(300), " \
            "test_score DOUBLE" \
            ")"
        )


def _update_table(
        model_path: str,
        config: ModelConfig,
        test_score: float,
):
    """Add information about recent model to the database."""
    with duckdb.connect(os.environ["DB_PATH"]) as db_conn:
        db_conn.execute(
            f"INSERT INTO {MODELS_TABLE} VALUES (?, ?, ?)",
            [model_path, str(config), test_score]
        )


def _calculate_scores(
        test_path: str,
        model_metadata: ModelMetadata,
):
    """Evaluate model scores on test dataset."""
    X_test, y_test = joblib.load(test_path)
    model_path = model_metadata["path"]
    fitted_model: FittedModel = joblib.load(model_path)
    X_test_encoded = fitted_model["fitted_transformer"].transform(X_test)
    test_score = fitted_model["fitted_estimator"].score(
        X_test_encoded,
        y_test,
        X_test["user_id"],
    )
    val_score = fitted_model["val_score"]
    return val_score, test_score, model_path, fitted_model["config"]


def _validate_quality(
        val_score: float,
        test_score: float,
        model_path: str,
        model_config: str,
        baseline_score: float,
):
    """
        Compare test score to validation and baseline scores,
        add passed model to all models table.
    """
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
            model_config,
            test_score,
        )


@dag(start_date=datetime(2026, 1, 15), schedule='@weekly')
def recommender():
    """
        DAG for refitting and retuning Factorization Machine on updated
        MovieLens dataset.
    """
    @task(multiple_outputs=True)
    def prepare_data() -> dict[Literal["train", "test"], str]:
        """Make training, validation, and test datasets."""
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
    def build_fit_save_estimator(train_path: str) -> ModelMetadata:
        """Preprocess data, train model, and save it locally."""
        train_sets = joblib.load(train_path, mmap_mode='r')
        model = find_best_model(
            *train_sets,
            recommend_num=RECOMMEND_NUM,
            metric=METRIC,
        )
        model_path = _save_model(model)

        return ModelConfig(
            path=model_path, 
            config=model["config"], 
            val_score=model["val_score"]
        )

    @task
    def prepare_baseline(train_path: str, test_path: str):
        """
            Count score of movie's average training rating as prediction
            for user's grade of the movie.
        """
        train_sets = joblib.load(train_path, mmap_mode='r')
        X_test, y_test = joblib.load(test_path)
        X_train_val = pd.concat(train_sets[:2])
        y_train_val = pd.concat(train_sets[3:5])
        return count_baseline_score(
            X_train_val,
            X_test, 
            y_train_val,
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
            Validate that test scores of the fitted model pass
            sanity check: they are better than baseline scores and not too
            much worse than validation scores.
        """
        results = _calculate_scores(test_path, model_metadata)
        _validate_quality(*results, baseline_score)

    @task
    def retrain_final_model(
            model_metadata: ModelMetadata,
            train_path: str,
            test_path: str,
    ):
        """
            Retrain model that passed sanity check on 
            the whole existing dataset.
        """
        final_config = model_metadata["config"]
        train_sets = joblib.load(train_path, mmap_mode='r')
        X_test, y_test = joblib.load(test_path)
        X_all = pd.concat((*train_sets[:3], X_test))
        y_all = pd.concat((*train_sets[3:], y_test))
        final_model = fit_model_final(X_all, y_all, final_config)
        joblib.dump(final_model, PRODUCTION_MODEL_PATH)


    datasets = prepare_data()
    train_path = datasets["train"]
    test_path = datasets["test"]
    model_metadata = build_fit_save_estimator(train_path)
    baseline_score = prepare_baseline(train_path, test_path) 
    evaluation = evaluate_model(model_metadata, baseline_score, test_path)
    retraining = retrain_final_model(model_metadata, train_path, test_path)
    evaluation >> retraining


recommender()
