import copy
from collections import defaultdict

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import ndcg_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def build_mlp(
        input_dim: int, 
        hidden_dims: list[int],
        output_dim: int,
        dropout_ratio: float,
) -> nn.Sequential:
    layers = []
    curr_dim = input_dim
    
    for ind, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(curr_dim, dim))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(dim))

        if ind == 0:
            layers.append(nn.Dropout(p=dropout_ratio))
            
        curr_dim = dim
        
    layers.append(nn.Linear(curr_dim, output_dim))
    return nn.Sequential(*layers)


def move_tensors(
        var_key_map: dict[str, str],
        batch: dict[str, torch.Tensor],
        device: torch.device,
) -> dict[str, torch.Tensor]:
    feature_tensors: dict[str, torch.Tensor] = {}
    for model_var, batch_key in var_key_map.items():
        feature_tensors[model_var] = batch[batch_key].to(device)
    return feature_tensors


def per_var_split_mask(
        df_sorted: pd.DataFrame,
        var: str,
        max_test_ratio=0.1,
        test_max_per_var=8,
        test_min_per_var=3,
):
    """
        Prepare splitting for given dataframe such as each unique value of var 
        has max_test_ratio ratio of records in test set capped at test_max_per_var.
    """
    per_var_count = df_sorted[var].map(df_sorted[var].value_counts())
    per_var_ranks = df_sorted.groupby(var).cumcount(ascending=False)
    per_var_test_size = np.minimum(test_max_per_var, max_test_ratio * per_var_count)
    per_var_test_size = np.maximum(per_var_test_size, test_min_per_var)
    test_mask = per_var_ranks < per_var_test_size

    return test_mask


def per_var_temporal_split(
        X: pd.DataFrame,
        var="user_id",
        time_col="timestamp",
        max_test_ratio=0.1,
        test_max_per_var=8,
        test_min_per_var=3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
        Split X and y to train and test such as for each value of var
        all test records have greater time_col values than train records,
        and each value of var receives max_test_ratio of records in test,
        capped at test_max_per_var.
    """
    sort_indices = X.sort_values(by=[var, time_col]).index
    X_sorted = X.loc[sort_indices]
    test_mask = per_var_split_mask(
        X_sorted, 
        var, 
        max_test_ratio, 
        test_max_per_var,
        test_min_per_var,
    )
    return X_sorted[~test_mask], X_sorted[test_mask]


def train_val_split(
    df: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "timestamp",
    cold_user_ratio: float = 0.05,
    max_test_ratio: float = 0.1,
    test_max_per_var: int = 8,
    test_min_per_var: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    all_users = df[user_col].unique()
    rng = np.random.default_rng(seed=random_state)
    
    cold_users_num = int(len(all_users) * cold_user_ratio)
    cold_user_ids = rng.choice(
        all_users,
        size=cold_users_num,
        replace=False,
    )
    
    cold_mask = df[user_col].isin(cold_user_ids)
    
    cold_df = df[cold_mask].copy()
    warm_df = df[~cold_mask].copy()
    
    train_warm, val_warm = per_var_temporal_split(
        X=warm_df,
        var=user_col,
        time_col=time_col,
        max_test_ratio=max_test_ratio,
        test_max_per_var=test_max_per_var,
        test_min_per_var=test_min_per_var,
    )
    
    val_combined = pd.concat(
        [val_warm, cold_df], 
        ignore_index=True,
    )
    
    return train_warm, val_combined
