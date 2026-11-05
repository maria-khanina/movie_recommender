import copy
from typing import Callable

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn.functional as F

from two_towers import TwoTowerModel
from recall_at_k import recall_at_ks_macro


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


def per_var_split(
        X: pd.DataFrame,
        var="user_id",
        time_col="timestamp",
        temporal=False,
        max_test_ratio=0.1,
        test_max_per_var=8,
        test_min_per_var=3,
        random_state=42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
        Split X and y to train and test such as for each value of var
        all test records have greater time_col values than train records,
        and each value of var receives max_test_ratio of records in test,
        capped at test_max_per_var.
    """
    if temporal:
        sort_indices = X.sort_values(by=[var, time_col]).index
        X_sorted = X.loc[sort_indices]
    else:
        X_shuffled = X.sample(frac=1.0, random_state=42)
        sort_indices = X_shuffled.sort_values(by=[var]).index
        X_sorted = X.loc[sort_indices]

    test_mask = per_var_split_mask(
        X_sorted, 
        var, 
        max_test_ratio, 
        test_max_per_var,
        test_min_per_var,
    )
    return X_sorted[~test_mask], X_sorted[test_mask]


def split_cold_warm(
        df: pd.DataFrame,
        user_col: str,
        cold_user_ratio: float,
        random_state: int,
):
    all_users = df[user_col].unique()
    rng = np.random.default_rng(seed=random_state)

    cold_users_num = int(len(all_users) * cold_user_ratio)
    cold_user_ids = rng.choice(
        all_users,
        size=cold_users_num,
        replace=False,
    )

    cold_mask = df[user_col].isin(cold_user_ids)

    return df[cold_mask].copy(), df[~cold_mask].copy()


def train_val_split(
    df: pd.DataFrame,
    user_col="user_id",
    time_col="timestamp",
    temporal=False,
    cold_user_ratio=0.0,
    max_test_ratio=0.1,
    test_max_per_var=8,
    test_min_per_var=3,
    random_state=42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    cold_df, warm_df = split_cold_warm(
        df,
        user_col,
        cold_user_ratio,
        random_state,
    )
    
    train_warm, val_warm = per_var_split(
        X=warm_df,
        var=user_col,
        time_col=time_col,
        temporal=temporal,
        max_test_ratio=max_test_ratio,
        test_max_per_var=test_max_per_var,
        test_min_per_var=test_min_per_var,
    )

    val_combined = pd.concat(
        [val_warm, cold_df], 
        ignore_index=True,
    )

    return train_warm, val_combined


def encode_genres(
        df: pd.DataFrame,
        movie_col="movie_id",
        genre_col="genres",
) -> tuple[pd.DataFrame]:

    unique_movies = df[[movie_col, genre_col]].drop_duplicates()
    genres_encoded = unique_movies[genre_col].str.get_dummies(sep="|")
    genres_cols = genres_encoded.columns.to_list()
    genres_num = len(genres_cols)

    movies_with_genres = pd.concat(
        [unique_movies[[movie_col]], genres_encoded], 
        axis=1,
    )

    df = df.merge(
        movies_with_genres, 
        on=movie_col, 
        how="left",
    )

    return df, genres_num


def prepare_log_freq(
        df: pd.DataFrame,
        max_movie: int,
        movie_col="movie_id",
) -> torch.Tensor:
    movie_counts = df[movie_col].value_counts().sort_index()
    popularity = torch.ones(
        max_movie + 1, 
        dtype=torch.float32,
    )

    popularity[movie_counts.index.values] += torch.tensor(
        movie_counts.values, 
        dtype=torch.float32,
    )

    frequencies = popularity / popularity.sum()
    return torch.log(frequencies)


def get_unique_movies(
    df: pd.DataFrame,
    genres_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    
    unique_movies_df = df.drop_duplicates(
        subset=["movie_id"],
    ).sort_values(
        by="movie_id",
    ).reset_index(
        drop=True,
    )
    
    movie_ids = torch.tensor(
        unique_movies_df["movie_id"].values,
        dtype=torch.long,
    )
    
    movie_genres = torch.tensor(
        unique_movies_df[genres_cols].values,
        dtype=torch.float32,
    )
    
    return movie_ids, movie_genres


def get_unique_users(
    df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    unique_users_df = df.drop_duplicates(
        subset=["user_id"],
    ).sort_values(
        by="user_id",
    ).reset_index(
        drop=True,
    )
    
    user_ids = torch.tensor(
        unique_users_df["user_id"].values,
        dtype=torch.long,
    )
    
    ages = torch.tensor(
        unique_users_df["age"].values,
        dtype=torch.long,
    )
    
    genders = torch.tensor(
        unique_users_df["gender"].values,
        dtype=torch.long,
    )
    
    jobs = torch.tensor(
        unique_users_df["occupation"].values,
        dtype=torch.long,
    )
    
    return user_ids, ages, genders, jobs


def _run_retrieval_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    var_key_map: dict[str, str],
    device: torch.device,
    is_train: bool,
    embedding_loss: Callable,
    optimizer: optim.Optimizer | None = None,
    **loss_kwargs,
) -> float:
    if is_train:
        model.train()
    else:
        model.eval()
    
    epoch_loss = 0.0
    dataset_len = 0
    
    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            tensors = move_tensors(var_key_map, batch, device)

            user_vectors, movie_vectors = model(**tensors)
            
            loss = embedding_loss(
                user_vectors,
                movie_vectors,
                tensors["user_ids"],
                tensors["movie_ids"],
                **loss_kwargs,
            )
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            batch_size = user_vectors.size(0)
            epoch_loss += loss.item() * batch_size
            dataset_len += batch_size
            
    return epoch_loss / dataset_len


def train_retriever_validate(
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    all_movie_ids: torch.Tensor,
    all_genres: torch.Tensor,
    cold_user_ids: torch.Tensor,
    var_key_map: dict[str, str],
    device: torch.device,
    embedding_loss: Callable,
    epochs: int,
    learning_rate: float,
    patience: int,
    tol: float,
    weight_decay: float,
    recall_k: int,
    **loss_kwargs,
) -> None:

    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    best_val_recall = -1
    best_val_loss = np.inf
    no_improve_count = 0
    best_weights = None
    best_epoch = -1
    
    print("=== PHASE 1: Training with Validation ===")
    for epoch in range(epochs):
        avg_train_loss = _run_retrieval_epoch(
            model=model,
            dataloader=train_loader,
            var_key_map=var_key_map,
            device=device,
            is_train=True,
            embedding_loss=embedding_loss,
            optimizer=optimizer,
            **loss_kwargs,
        )
        
        avg_val_loss = _run_retrieval_epoch(
            model=model,
            dataloader=val_loader,
            var_key_map=var_key_map,
            device=device,
            is_train=False,
            embedding_loss=embedding_loss,
            optimizer=optimizer,
            **loss_kwargs,
        )

        val_recall = recall_at_ks_macro(
            two_towers=model,
            dataloader=val_loader,
            var_key_map=var_key_map,
            all_movie_ids=all_movie_ids,
            all_genres=all_genres,
            device=device,
            cold_user_ids=cold_user_ids,
            ks=[recall_k],
        )[recall_k]["global"]
        
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}. ",
            f"Val recall@{recall_k}-macro: {val_recall:.4f}"
        )
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_weights = copy.deepcopy(
                model.state_dict(),
            )

            best_epoch = epoch + 1

        if avg_val_loss < (best_val_loss - tol):
            best_val_loss = avg_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"No improvement for {no_improve_count} epochs")
        if no_improve_count >= patience:
            break
    
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(
            f"Best weights restored. Best validation recall@{recall_k}-macro "
            f"is {best_val_recall:.4f} achieved at epoch {best_epoch}."
        )
    
    return best_weights, best_epoch


def tune_retriever(
    model: torch.nn.Module,
    combined_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    var_key_map: dict[str, str],
    embedding_loss: Callable,
    epochs: int,
    tol: float,
    patience=5,
    **loss_kwargs,
) -> dict:
    
    print("=== PHASE 2: Absorbing Validation Set ===")
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * 0.1

    best_loss = np.inf
    no_improve_count = 0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        train_loss = _run_retrieval_epoch(
            model=model,
            dataloader=combined_loader,
            var_key_map=var_key_map,
            device=device,
            is_train=True,
            embedding_loss=embedding_loss,
            optimizer=optimizer,
            **loss_kwargs,
        )
        
        print(f"Epoch {epoch+1}, train Loss: {train_loss:.4f}")
        
        if train_loss < best_loss - tol:
            best_loss = train_loss
            no_improve_count = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    return best_weights


def train_retriever(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    all_movie_ids: torch.Tensor,
    all_genres: torch.Tensor,
    cold_user_ids: torch.Tensor,
    var_key_map: dict[str, str],
    embedding_loss: Callable,
    epochs=10_000,
    learning_rate=0.001,
    patience=10,
    tuning_patience=5,
    tol=1e-4,
    weight_decay=1e-4,
    device_name="cpu",
    recall_k=100,
    **loss_kwargs,
) -> None:
    
    device = torch.device(device_name)
    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    best_weights, best_epoch = train_retriever_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        all_movie_ids=all_movie_ids,
        all_genres=all_genres,
        cold_user_ids=cold_user_ids,
        var_key_map=var_key_map,
        device=device,
        embedding_loss=embedding_loss,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        tol=tol,
        recall_k=recall_k,
        **loss_kwargs,
    )
    
    model.load_state_dict(best_weights)

    val_ratio = len(val_loader.dataset) / len(train_loader.dataset)
    tune_epochs = int(val_ratio * best_epoch)

    combined_dataset = ConcatDataset(
        [train_loader.dataset, val_loader.dataset]
    )
    combined_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True, 
        num_workers=train_loader.num_workers,
    )

    final_weights = tune_retriever(
        model=model,
        combined_loader=combined_loader,
        optimizer=optimizer,
        device=device,
        var_key_map=var_key_map,
        embedding_loss=embedding_loss,
        epochs=tune_epochs,
        tol=tol,
        patience=tuning_patience,
        **loss_kwargs,
    )

    model.load_state_dict(final_weights)
