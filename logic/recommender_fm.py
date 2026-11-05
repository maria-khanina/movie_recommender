"""
Title: Recommender

Author: Masha Khanina
Reviewer: Shani Gendelis

Description: Module with specific implementations of functions used in
recommender_dag module. Includes splitting dataset,
building model, and finding best configuration.

Infinity Labs R&D AI #5
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
import scipy.sparse as sp

from factorization_machine import FactorizationMachine, ndcg_p_score


class ModelConfig(TypedDict):
    factors_num: int
    min_frequency: int
    last_epoch: int


class FittedModel(TypedDict):
    name: str
    config: ModelConfig
    fitted_estimator: FactorizationMachine
    fitted_transformer: ColumnTransformer
    val_score: float


def per_var_split_mask(
        df_sorted: pd.DataFrame,
        var: str,
        max_test_ratio=0.1,
        test_max_per_var=10,
):
    """
        Prepare splitting for given dataframe such as each
        unique value of var has max_test_ratio ratio of records
        in test set capped at test_max_per_var.
    """
    per_var_count = df_sorted[var].map(df_sorted[var].value_counts())
    per_var_ranks = df_sorted.groupby(var).cumcount(ascending=False)
    per_var_test_size = np.minimum(
        test_max_per_var, 
        max_test_ratio * per_var_count
    )
    test_mask = per_var_ranks < per_var_test_size

    return test_mask


def per_var_temporal_split(
        X: pd.DataFrame,
        y: ArrayLike,
        var="user_id",
        time_col="timestamp",
        max_test_ratio=0.1,
        test_max_per_var=10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split X and y to train and test such as for each value of var
    all test records have greater time_col values than train records,
    and each value of var receives max_test_ratio of records in test,
    capped at test_max_per_var.
    """
    sort_indices = X.sort_values(by=[var, time_col]).index
    X_sorted = X.loc[sort_indices]
    y_sorted = pd.Series(y, index=X.index).loc[sort_indices]
    test_mask = per_var_split_mask(
        X_sorted, var, max_test_ratio, test_max_per_var,
    )
    return X_sorted[~test_mask], X_sorted[test_mask], \
          y_sorted[~test_mask], y_sorted[test_mask]


def split(
        features_df: pd.DataFrame, 
        ratings: pd.Series,
        test_max_per_var=10,
        max_test_ratio=0.1,
):
    """Split datasets on train, validate, and test."""
    X_all_train, X_test, y_all_train, y_test = per_var_temporal_split(
        features_df,
        ratings,
        var="user_id",
        time_col="timestamp",
        max_test_ratio=max_test_ratio,
        test_max_per_var=test_max_per_var,
    )

    X_train, X_out, y_train, y_out = per_var_temporal_split(
        X_all_train,
        y_all_train,
        var="user_id",
        time_col="timestamp",
        max_test_ratio=max_test_ratio,
        test_max_per_var=test_max_per_var,
    )


    X_train, X_val, y_train, y_val = per_var_temporal_split(
        X_train,
        y_train,
        var="user_id",
        time_col="timestamp",
        max_test_ratio=max_test_ratio,
        test_max_per_var=test_max_per_var,
    )

    train_all = X_train, X_val, X_out, y_train, y_val, y_out
    test_all = X_test, y_test

    return train_all, test_all


def build_transformer(
        X_train: pd.DataFrame, 
        min_frequency: int,
) -> ColumnTransformer:
    """
        Make OneHotEncoder of user_id and movie_id based on id
        seen in X_train. Don't make a column for objects appearing
        less than min_frequency times in train.
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
                    sparse_output=True,
                    min_frequency=min_frequency,
                ),
                ["user_id", "movie_id"],
            ),
        ],
        remainder="drop",
    )

    return transformer


def transform_data(
        transformer: ColumnTransformer, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        X_out: pd.DataFrame,
) -> tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix]:
    """"
        Apply transformer to given datasets: 
        fit on train, transform on all.
    """
    X_train_encoded = transformer.fit_transform(X_train)
    X_val_encoded = transformer.transform(X_val)
    X_out_encoded = transformer.transform(X_out)
    return X_train_encoded, X_val_encoded, X_out_encoded


def find_factors_search_range(X_train: pd.DataFrame) -> np.array:
    """
        Define what factors_num to check as hyperparameter depending
        on how many records objects (movies and users) have on average.
    """
    records_per_entity = pd.concat((
        X_train.groupby("user_id").size(), 
        X_train.groupby("movie_id").size()
    ))
    median_per_entity = records_per_entity.median()
    search_start = int(median_per_entity // 5)
    search_end = int(median_per_entity // 2)
    return np.arange(search_start, search_end, 8)


def find_best_fm_config(
        X_train_encoded, X_val_encoded, X_out_encoded,
        y_train, y_val, y_out,
        users_val, users_out,
        factors_num_range: list[int],
        recommend_num: int, 
        metric: str
) -> tuple[int, int, float, FactorizationMachine]:
    """
        Find best hyperparameters for Factorization Machine:
        factors num, amount of epochs to iterate through.
    """
    best_score = -np.inf
    best_factors_num = 0
    last_epoch = 1000

    for factors_num in factors_num_range:
        estimator = FactorizationMachine(
            factors_num=factors_num,
            recommend_num=recommend_num, 
            metric=metric,
            verbose=True,
        )
        estimator.fit(
            X_train_encoded, 
            y_train, 
            eval_set=(X_val_encoded, y_val),
            users=users_val,
        )
        val_score = estimator.score(
            X_out_encoded,
            y_out,
            users_out, 
        )
        if val_score > best_score:
            best_score = val_score
            best_factors_num = factors_num
            last_epoch = estimator.stop_epoch
            best_estimator = estimator

    return best_factors_num, last_epoch, best_score, best_estimator


def find_best_config(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_out: pd.DataFrame,
        y_train: ArrayLike,
        y_val: ArrayLike,
        y_out: ArrayLike,
        factors_num_range: list[int],
        recommend_num: int,
        metric: str,
        min_freq_options: list[int],
) -> tuple[float, ModelConfig, ColumnTransformer, FactorizationMachine]:
    """
        Find best configutation for the model:
        min_frequency for OneHotEncoder, hyperparameters for FM.
    """
    best_score = -np.inf
    best_config = {
        "min_frequency": -1,
        "factors_num": -1,
        "last_epoch": -1,
    }

    for freq in min_freq_options:
        current_transformer = build_transformer(X_train, min_frequency=freq)
        X_train_encoded, X_val_encoded, X_out_encoded = transform_data(
            current_transformer, X_train, X_val, X_out
        )

        best_factors, last_epoch, val_score, estimator = find_best_fm_config(
            X_train_encoded, X_val_encoded, X_out_encoded,
            y_train, y_val, y_out,
            X_val["user_id"], X_out["user_id"],
            factors_num_range,
            recommend_num,
            metric,
        )

        if val_score > best_score:
            transformer = current_transformer
            best_estimator = estimator
            best_score = val_score
            best_config = ModelConfig(
                min_frequency=freq,
                factors_num=best_factors,
                last_epoch=last_epoch,
            )
    
    return best_score, best_config, transformer, best_estimator


def fit_model_final(
        X: pd.DataFrame, 
        y: ArrayLike,
        config: ModelConfig,
):
    """
        Fit model with given config on given X, y.
    """
    transformer = build_transformer(
        X,
        min_frequency=config["min_frequency"],
    )
    X_transformed = transformer.fit_transform(X)

    estimator = FactorizationMachine(
        factors_num=config["factors_num"],
        verbose=True,
    )
    estimator.max_epoch = config["last_epoch"]
    estimator.fit(X_transformed, y)

    return {
        "fitted_transformer": transformer,
        "fitted_estimator": estimator,
    }  


def find_best_model(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_out: pd.DataFrame,
        y_train: ArrayLike,
        y_val: ArrayLike,
        y_out: ArrayLike,
        recommend_num=5,
        metric="ndcg",
        min_freq_options=[1, 5, 10],
) -> FittedModel:
    """
        Find best condfiguration for the OneHotEncoder + FM model
        and fit best model on train + val datasets.
    """
    factors_num_range = find_factors_search_range(X_train)
    val_score, best_config, transformer, best_estimator = find_best_config(
        X_train, X_val, X_out, y_train, y_val, y_out,
        factors_num_range, recommend_num, metric, min_freq_options,
    )

    return FittedModel(
        name="Factorization_Machine",
        config=best_config,
        fitted_estimator=best_estimator,
        fitted_transformer=transformer,
        val_score=val_score,
    )


def count_average_ratings(
        X_train: pd.DataFrame,
        y_train: ArrayLike, 
) -> pd.DataFrame:
    """
        Return average seen rating for each movie.
    """
    df_train = pd.concat(
        (X_train, y_train), axis="columns"
    )
    avg_train_ratings = df_train.groupby("movie_id", as_index=False)["rating"].mean()
    avg_train_ratings.columns = ["movie_id", "avg_train_rating"]
    return avg_train_ratings


def count_baseline_score(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: ArrayLike,
        y_test: ArrayLike,
        recommend_num: int,
) -> float:
    """
        Calculate ndcg score with p = recommend_num for model
        that predicts user's grade of a movie as average rating of
        the movie seen on training dataset.
    """
    avg_rating_total = np.mean(y_train)
    avg_train_ratings = count_average_ratings(X_train, y_train)
    df_test = pd.merge(
        left=X_test,
        right=avg_train_ratings, 
        on="movie_id", 
        how="left",
    )
    y_pred = df_test["avg_train_rating"].fillna(avg_rating_total)
    return ndcg_p_score(y_test, y_pred, df_test["user_id"], recommend_num)
