"""
Title: Recommender

Author: Masha Khanina
Reviewer: Nir Landerer

Description: Module with specific implementations of functions used in
recommender_dag module. Includes extracting features, splitting dataset,
building model, and defining way to score the model.
"""
from typing import TypedDict

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# from logic.factorization_machine import FactorizationMachine
from factorization_machine import FactorizationMachine


def extract_features(
        ratings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    return ratings_df[["user_id", "movie_id"]], ratings_df["rating"]


def per_var_split_mask(
        df_sorted: pd.DataFrame,
        var: str,
        test_ratio=0.01,
        test_min_per_var=10,
):
    per_var_count = df_sorted[var].map(df_sorted[var].value_counts())
    per_var_ranks = df_sorted.groupby(var).cumcount(ascending=False)
    per_var_test_size = np.maximum(test_min_per_var, test_ratio * per_var_count)
    test_mask = per_var_ranks < per_var_test_size
    
    return test_mask


def per_var_temporal_split(
        X: pd.DataFrame,
        y: ArrayLike,
        var: str,
        time_col: str,
        test_ratio=0.01,
        test_min_per_var=10,
):
    sort_indices = X.sort_values(by=[var, time_col]).index
    X_sorted = X.loc[sort_indices]
    y_sorted = pd.Series(y, index=X.index).loc[sort_indices]
    test_mask = per_var_split_mask(X_sorted, var, test_ratio, test_min_per_var)
    return X_sorted[~test_mask], X_sorted[test_mask], \
        y_sorted[~test_mask], y_sorted[test_mask]


def split(
        features_df: pd.DataFrame, 
        ratings: pd.Series,
):
    """Split datasets on train, validate, and test."""
    X_all_train, X_test, y_all_train, y_test = per_var_temporal_split(
        features_df,
        ratings,
        var="user_id",
        time_col="timestamp",
        test_ratio=0.01,
        test_min_per_var=10,
    )

    X_train, X_val, y_train, y_val = per_var_temporal_split(
        X_all_train,
        y_all_train,
        var="user_id",
        time_col="timestamp",
        test_ratio=0.01,
        test_min_per_var=10,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


class ModelInfo(TypedDict):
    name: str
    description: str
    estimator: BaseEstimator


def build_fit_estimator(
        X_train: ArrayLike, 
        X_val: ArrayLike, 
        y_train: ArrayLike, 
        y_val: ArrayLike,
) -> ModelInfo:
    """
        Make and fit ratings estimator.
    """
    all_users = np.sort(X_train["user_id"].unique())
    all_movies = np.sort(X_train["movie_id"].unique())

    transformer = ColumnTransformer(
        transformers=[
            (
                "encode_id",
                OneHotEncoder(
                    categories=[all_users, all_movies],
                    handle_unknown="ignore",
                ),
                ["user_id", "movie_id"],
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("transformer", transformer),
            ("estimator", FactorizationMachine(max_epoch=100)),
        ]
    )
    
    test_fold = np.concatenate([
        -1 * np.ones(len(X_train)),
        0 * np.ones(len(X_val))
    ])

    split = PredefinedSplit(test_fold)

    searcher = GridSearchCV(
        estimator=model,
        param_grid={
            "estimator__factors_num": [30],
            "estimator__weights_penalty": [0.0001, 0.001],
            "estimator__factors_penalty": [0.0001, 0.001],
            "estimator__batch_size": [256, 512, 1024],
        },
        verbose=100,
        cv=split,
    )

    X_train_all = pd.concat((X_train, X_val))
    y_train_all = pd.concat((y_train, y_val))
    searcher.fit(X_train_all, y_train_all)

    return {
        "name": "Factorization Machine",
        "description": 
            "Factorization machine with one-hot encoded (user_id, movie_id)"
            " as features",
        "estimator": searcher.best_estimator_,
    }


def get_predicted_top(
        users: pd.Series,
        true_ratings: pd.Series, 
        predicted_ratings: ArrayLike,
        recommended_num: int,
):
    """
        Return dataframe with 5 movies for each user with highest
        predicted ratings from all actually rated movies.

        ids -- index consisting of pairs (user_id, movie_id) in right order.
    """
    df = pd.DataFrame(
        {   
            "user_id": users,
            "predicted_rating": predicted_ratings,
            "real_rating": true_ratings,
        },
    )
    df.sort_values(
        by=["user_id", "predicted_rating"],
        ascending=[True, False],
        inplace=True,
    )
    return df.groupby("user_id").head(recommended_num)


def mean_lowest_rating(
        users: pd.Series,
        true_ratings: pd.Series,
        predicted_ratings: ArrayLike,
        recommended_num=5,
):
    """
        Return lowest rating average user will see in 
        recommendations based on predicted_ratings and only from
        movies that appear in true_ratings index.
    """
    return get_predicted_top(
        users,
        true_ratings,
        predicted_ratings,
        recommended_num,
    ).groupby("user_id")["real_rating"].min().mean()


def mean_lowest_scorer(
        estimator: BaseEstimator, 
        X: pd.DataFrame, 
        y: ArrayLike,
        recommended_num=5,
):
    """
        Score estimator according to the worst movie
        average user will receive in recommendations made based
        on estimator's predictions of ratings.
    """
    predicted = estimator.predict(X)
    return mean_lowest_rating(X["user_id"], y, predicted, recommended_num)
