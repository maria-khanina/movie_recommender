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
        test_min_per_var=5,
):
    """Split datasets on train, validate, and test."""
    X_all_train, X_test, y_all_train, y_test = per_var_temporal_split(
        features_df,
        ratings,
        var="user_id",
        time_col="timestamp",
        test_ratio=0.01,
        test_min_per_var=test_min_per_var,
    )

    X_train, X_outer_val, y_train, y_outer_val = per_var_temporal_split(
        X_all_train,
        y_all_train,
        var="user_id",
        time_col="timestamp",
        test_ratio=0.01,
        test_min_per_var=test_min_per_var,
    )


    X_train, X_val, y_train, y_val = per_var_temporal_split(
        X_train,
        y_train,
        var="user_id",
        time_col="timestamp",
        test_ratio=0.01,
        test_min_per_var=test_min_per_var,
    )

    train_all = X_train, X_val, X_outer_val, y_train, y_val, y_outer_val
    test_all = X_test, y_test

    return train_all, test_all


class ModelInfo(TypedDict):
    name: str
    description: str
    fitted_estimator: FactorizationMachine
    val_score: float


def find_best_estimator(
        X_train: ArrayLike,
        X_val: ArrayLike,
        X_outer_val: ArrayLike,
        y_train: ArrayLike, 
        y_val: ArrayLike,
        y_outer_val: ArrayLike,
        recommend_num=5,
        metric="ndcg",
) -> ModelInfo:
    """
        Make and fit ratings estimator.
    """
    factors_num_range = find_search_range(X_train)
    transformer = build_transformer(X_train)

    best_factors_num, val_score, last_epoch = find_best_params(
        *transform_data(transformer, X_train, X_val, X_outer_val),
        y_train, y_val, y_outer_val,
        X_val["user_id"], X_outer_val["user_id"],
        factors_num_range,
        recommend_num,
        metric,
    )

    X_all = pd.concat((X_train, X_val, X_outer_val))
    y_all = pd.concat((y_train, y_val, y_outer_val))
    best_model = FactorizationMachine(factors_num=best_factors_num)
    best_model.max_epoch = last_epoch
    best_model.fit(X_all, y_all)

    return {
        "name": "Factorization Machine",
        "description": 
            "Factorization machine with one-hot encoded (user_id, movie_id)"
            " as features",
        "fitted_estimator": best_model,
        "val_score": val_score, 
    }


def find_search_range(X_train):
    records_per_entity = pd.concat(
        X_train.groupby("user_id").size(), 
        X_train.groupby("movie_id").size()
    )
    median_per_entity = records_per_entity.median()
    return np.arange(median_per_entity / 5, median_per_entity / 3, 10)


def build_transformer(X_train):
    all_users = np.sort(X_train["user_id"].unique())
    all_movies = np.sort(X_train["movie_id"].unique())

    transformer = ColumnTransformer(
        transformers=[
            (
                "encode_id",
                OneHotEncoder(
                    categories=[all_users, all_movies],
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
                ["user_id", "movie_id"],
            ),
        ],
        remainder="drop",
    )

    return transformer


def transform_data(transformer, X_train, X_val, X_outer_val):
    X_train_transformed = transformer.fit_transform(X_train)
    X_val_transformed = transformer.transform(X_val)
    X_outer_val_transformed = transformer.transform(X_outer_val)
    return X_train_transformed, X_val_transformed, X_outer_val_transformed


def find_best_params(
        X_train_transformed, X_val_transformed, X_outer_val_transformed,
        y_train, y_val, y_outer_val,
        users_val, users_outer_val,
        factors_num_range: list[int],
        recommend_num: int, 
        metric: str
):
    best_score = -np.inf
    best_factors_num = 0
    last_epoch = 1000

    for factors_num in factors_num_range:
        model = FactorizationMachine(
            factors_num=factors_num,
            recommend_num=recommend_num, 
            metric=metric,
        )
        model.fit(
            X_train_transformed, 
            y_train, 
            eval_set=(X_val_transformed, y_val),
            users=users_val,
        )
        model_score = model.score(
            X_outer_val_transformed, 
            y_outer_val, 
            users_outer_val, 
        )
        if model_score > best_score:
            best_score = model_score
            best_factors_num = factors_num
            last_epoch = model.stop_epoch

    return best_factors_num, best_score, last_epoch
