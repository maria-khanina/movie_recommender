"""
Title: Recommender

Author: Masha Khanina
Reviewer: Nir Landerer

Description: Module with specific implementations of functions used in
recommender_dag module. Includes extracting features, splitting dataset,
building model, and defining way to score the model.
"""
import os
from typing import TypedDict

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
                                 StandardScaler

from logic.loader import GENRES


GENRES.remove("unknown")


def extract_features(
        ratings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
        Return a dataframe with extracted features and a series with 
        target ratings.
    """
    ratings_df.set_index(keys=["movie_id", "user_id"], inplace=True)
    ratings_df["release_date"] = pd.to_datetime(ratings_df["release_date"])
    ratings_df["release_year"] = ratings_df["release_date"].dt.year
    ratings_df["release_month"] = ratings_df["release_date"].dt.month
    ratings_df = ratings_df[ratings_df["unknown"] == 0]
    features = [
        "release_year", "release_month", "gender", 
        "occupation", "age",
    ] + GENRES
    return ratings_df[features], ratings_df["rating"]


def split(
        features_df: pd.DataFrame, 
        ratings: pd.Series,
):
    """Split datasets on train, validate, and test."""
    X_train, X_test, ratings_train, ratings_test = train_test_split(
        features_df,
        ratings,
        train_size=0.8,
        random_state=42,
    )

    X_validate, X_test, ratings_validate, ratings_test = train_test_split(
        X_test,
        ratings_test,
        train_size=0.5,
        random_state=42,
    )

    return (
        X_train, X_validate, X_test,
        ratings_train, ratings_validate, ratings_test,
    )


class ModelInfo(TypedDict):
    name: str
    description: str
    estimator: GridSearchCV


def build_predictor() -> ModelInfo:
    """
        Make ratings predictor from GridSearchCV which estimator
        contains all needed preprocessing.
    """
    year_scaler = Pipeline(
        steps=[
            ("log", FunctionTransformer(
                func=np.log, 
                inverse_func=np.exp,
                feature_names_out="one-to-one",
            )),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("genre", "passthrough", GENRES),
            ("year_scaler", year_scaler, ["release_year"]),
            ("month_encoder", OneHotEncoder(drop="first"), ["release_month"]),
            ("gender_encoder", OneHotEncoder(drop="first"), ["gender"]),
            ("occuparion_encoder", OneHotEncoder(drop=None), ["occupation"]),
            ("age_scaler", StandardScaler(), ["age"]),
        ],
        verbose_feature_names_out=False,

        transformer_weights={
            "genre": 1,
            "year_scaler": 0.3,
            "gender_encoder": 0.3,
            "month_encoder": 0.7,
            "occupation_encoder": 1,
            "age_scaler": 0.3,
        }
    )

    predictor_pipe = Pipeline(
        steps=[
            ("preprocesor", preprocessor),
            ("regressor", KNeighborsRegressor()),
        ]
    )

    params = {
        "regressor__n_neighbors": [3, 7, 10, 30, 50],
        "regressor__metric": ["minkowski", "cosine"],
    }

    params_searcher = GridSearchCV(
        estimator=predictor_pipe,
        param_grid=params,
        scoring=recommender_scorer,
        n_jobs=1,
        cv=5,
        verbose=3,
    )

    return {
        "name": "KNN",
        "description": 
            "KNN in space with next features:"
            "user's age, gender, occupation, movie's genres, "
            "log of release year and release month.",
        "estimator": params_searcher,
    }


def _get_predicted_top(true_ratings, predicted_ratings):
    """
        Return dataframe with 5 movies for each user with highest
        predicted ratings from all actually rated movies.
    """
    df = pd.DataFrame(
        {
            "predicted_rating": predicted_ratings,
            "real_rating": true_ratings,
        },
        index=true_ratings.index,
    )
    df.sort_values(
        by=["user_id", "predicted_rating"],
        ascending=[True, False],
        inplace=True,
    )
    return df.groupby(level="user_id").head(5)


def _score_predictions(true_ratings, predicted_ratings):
    """
        Return lowest rating average user will see in 
        recommendations based on predicted_ratings and only from
        movies that appear in true_ratings index.
    """
    return _get_predicted_top(
        true_ratings, predicted_ratings
    ).groupby(level="user_id")["real_rating"].min().mean()


def recommender_scorer(estimator, X, y):
    """
        Score estimator according to the worst movie
        average user will receive in recommendations made based
        on estimator's predictions of ratings.
    """
    predicted = estimator.predict(X)
    y = pd.Series(y, index=X.index)
    return _score_predictions(y, predicted)
