import os

import duckdb
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
                                 StandardScaler

from logic.loader import GENRES

GENRES.remove("unknown")


def extract_df():
    db_path = os.environ["DB_PATH"]
    with duckdb.connect(db_path, read_only=True) as db_conn:
        return db_conn.execute("""
            SELECT ratings.user_id, ratings.movie_id, ratings.rating,
            movies.*, users.* 
            FROM ratings 
                JOIN movies USING (movie_id) 
                JOIN users USING (user_id)
        """).df()


def select_features(ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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


def split(features_df: pd.DataFrame, ratings: pd.Series):
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


def build_predictor():
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
        "estimator": params_searcher,
    }


def make_split(X_train, X_validate, y_train, y_validate):
    X_train_all = pd.concat([X_train, X_validate])
    y_train_all = pd.concat([y_train, y_validate])
    test_fold = np.full(X_train_all.shape[0], -1, dtype=np.int8)
    test_fold[X_train.shape[0]:] = 0


def _get_predicted_top(true_ratings, predicted_ratings):
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
    return _get_predicted_top(
        true_ratings, predicted_ratings
    ).groupby(level="user_id")["real_rating"].min().mean()


def recommender_scorer(estimator, X, y):
    predicted = estimator.predict(X)
    y = pd.Series(y, index=X.index)
    return _score_predictions(y, predicted)
