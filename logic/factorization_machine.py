"""
Title: Factorization Machine

Author: Masha Khanina
Reviewer: Shani Gendelis

Description: Implementation of Factorization Machine specialized for movies
recommendations, includes cutom metrics to evaluate recommender quality.

Infinity Labs R&D AI #5.
"""

from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import check_X_y, check_array
from scipy.sparse import csr_array, issparse, spmatrix, sparray, vstack


dataset = ArrayLike | spmatrix | sparray
array_func = Callable[[ArrayLike, ArrayLike], ArrayLike]


def make_results_df(
        true_ratings: pd.Series, 
        predicted_ratings: ArrayLike,
        users: pd.Series,
):
    return pd.DataFrame(
        {
            "user_id": np.asarray(users),
            "predicted_rating": np.asarray(predicted_ratings),
            "real_rating": np.asarray(true_ratings),
        },
    )

def get_predicted_top(
        true_ratings: pd.Series, 
        predicted_ratings: ArrayLike,
        users: pd.Series,
        recommend_num,
):
    """
        Return dataframe with recommend_num movies for each user
        with highest predicted ratings from movies rated in true_ratings.
    """
    results_df = make_results_df(true_ratings, predicted_ratings, users)
    results_df.sort_values(
        by=["user_id", "predicted_rating"],
        ascending=[True, False],
        inplace=True,
    )
    return results_df.groupby("user_id").head(recommend_num)


def mean_lowest_rating(
        true_ratings: pd.Series,
        predicted_ratings: ArrayLike,
        users: pd.Series,
        recommend_num,
):
    """
        Return lowest rating average user will see in 
        recommendations based on predicted_ratings.
    """
    return get_predicted_top(
        true_ratings,
        predicted_ratings,
        users,
        recommend_num,
    ).groupby("user_id")["real_rating"].min().mean()


def calculate_dcg(
        results_df: pd.DataFrame,
        gain: pd.Series, 
        by: str, 
        p: int,
):
    """Return dcg_p score if column by is used for ranking."""
    rank = results_df.groupby("user_id")[by].\
        rank(method="first", ascending=False)
    discount = np.log(rank + 1)
    all_dcgs = gain / discount
    return all_dcgs * (rank <= p)


def ndcg_p_score(y_true, y_pred, users, p=5):
    """
        Return users' mean NDCG_p score for movies rated using y_pred
        with real ratings y_true
    """
    results_df = make_results_df(y_true, y_pred, users)

    gain = 2**results_df["real_rating"] - 1
    results_df["dcg"] = calculate_dcg(results_df, gain, "predicted_rating", p)
    results_df["idcg"] = calculate_dcg(results_df, gain, "real_rating", p)

    dcgs = results_df.groupby("user_id")[["dcg", "idcg"]].sum()
    ndcg_per_user = dcgs["dcg"] / (dcgs["idcg"] + 1e-8)

    return ndcg_per_user.mean()


def _process_X_y(X, y):
    X, y = check_X_y(X, y, accept_sparse=["csr"])
    if issparse(X):
        X = csr_array(X)
    return X, y


def _prepare_split_data(X, y, eval_set):
    X, y = _process_X_y(X, y)
    X_eval, y_eval = X, y
    if eval_set is not None:
        X_eval, y_eval = _process_X_y(eval_set[0], eval_set[1])
    return X, X_eval, y, y_eval


class FactorizationMachine:
    def __init__(
            self, 
            factors_num: int = 1, 
            recommend_num=5,
            max_epoch=1000,
            tol=1e-4,
            learn_rate=0.01,
            loss: str | Callable[[ArrayLike, ArrayLike], float] = "mse",
            dloss_dpred: array_func | None = None,
            metric: Literal["neg_rmse", "mean_lowest_per_user", "ndcg"] = "ndcg",
            weights_penalty=0,
            factors_penalty=0,
            batch_size=1000,
            patience=10,
            random_state=None,
            verbose=False,
    ):
        """"
            Class implementing factorization machine mostly for 
            recommending movies to users based on movielens dataset.
        """
        self._init_loss(loss, dloss_dpred)
        self.metric = metric
        self._factors_num = factors_num
        self.recommend_num = recommend_num
        self.max_epoch = max_epoch
        self.stop_epoch = max_epoch
        self.tol = tol
        self.learn_rate = learn_rate
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self._weights_penalty = weights_penalty
        self._factors_penalty = factors_penalty
        self.verbose = verbose
        self.fitted_ = False

    def _init_loss(
            self, 
            loss: str | Callable[[ArrayLike, ArrayLike], float], 
            dloss_dpred: array_func | None,
    ):
        """
            Identify model's loss function and derivative
            based on given parameters.
        """
        if loss == "mse":
            self.loss = lambda y, y_pred: np.mean((y - y_pred) ** 2)
            self.dloss_dpred = lambda y, y_pred: 2 * (y_pred - y) / y.shape[0]
        else:
            self.loss = loss
            self.dloss_dpred = dloss_dpred

    def _init_params(
            self, 
            dim: int, 
            y: ArrayLike,
            rng: np.random.Generator,
    ) -> None:
        """
            Initialize model's parameters:
            constant, linear weights, factors.
        """
        self._w0 = np.mean(y)
        self._weights = np.zeros(shape=(dim,), dtype=float)

        init_std = np.sqrt(2.0 / (dim + self._factors_num))
        self._factors = rng.normal(
            scale=init_std, 
            size=(dim, self._factors_num),
        )
        self._no_improvement_count = 0
        self._best_score = -np.inf

        self._squared_weight_grads = np.zeros(dim)
        self._squared_factor_grads = np.zeros((dim, self._factors_num))
        self._squared_w0_grads = 0.0

    def _run_epoch(
            self, 
            X: dataset, 
            y: ArrayLike, 
            n_samples: int,
            rng: np.random.Generator,
    ) -> None:
        """Run one epoch of model's fitting process."""
        permutation = rng.permutation(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = permutation[i : i + self.batch_size]
            self._batch_update(X[batch_indices], y[batch_indices])

    def _update_coef(self, coef, grad, squared_grads):
        """
            Move given coef in the direction opposite to
            its gradient with adapted learning rate.
        """
        squared_grads += np.square(grad)
        curr_learn_rate = self.learn_rate / np.sqrt(squared_grads + 1e-8)
        coef -= curr_learn_rate * grad
        return coef, squared_grads

    def _batch_update(self, X_batch, y_batch):
        """
            Update model's parameters based on batch gradient descent.
        """
        y_pred_batch, X_dot_factors = self._predict(X_batch)
        dloss_dpred = self.dloss_dpred(y_batch, y_pred_batch)

        w0_grad = np.sum(dloss_dpred)
        self._w0, self._squared_w0_grads = self._update_coef(
            self._w0, w0_grad, self._squared_w0_grads,
        )

        weights_grad = X_batch.T @ dloss_dpred \
                    + 2 * self._weights_penalty * self._weights
        self._weights, self._squared_weight_grads = self._update_coef(
            self._weights, weights_grad, self._squared_weight_grads,
        )

        factor_coef_0 = X_batch.T @ \
            (dloss_dpred[:, np.newaxis] * X_dot_factors)
        factor_coef_1 = -(X_batch.T ** 2) @ dloss_dpred
        factors_grad = factor_coef_0 \
                        + factor_coef_1[:, np.newaxis] * self._factors \
                        + 2 * self._factors_penalty * self._factors

        self._factors, self._squared_factor_grads = self._update_coef(
            self._factors, factors_grad, self._squared_factor_grads,
        )

    def _is_converged(self, X_eval, y_eval, users=None):
        """
            Check if object's score didn't improve on more than tol for
            patience amount of times in a row on separated evaluation set.
        """
        curr_score = self.score(X_eval, y_eval, users)

        if curr_score > self._best_score + self.tol:
            self._best_score = curr_score
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            return True
        else:
            return False

    def fit(
            self,
            X: dataset,
            y: ArrayLike,
            eval_set: tuple[ArrayLike, ArrayLike] | None = None,
            users=None,
    ):
        """
            Iteratively update model parameters until object's score
            on eval_set stops improving. If eval_set is None, 
            update model's parameters for self.stop_epoch epochs.
        """
        X_train, X_eval, y_train, y_eval = _prepare_split_data(X, y, eval_set)
        n_samples, dim = X.shape
        rng = np.random.default_rng(seed=self.random_state)

        self._init_params(dim, y_train, rng)
        best_epoch = self.stop_epoch

        for epoch in range(self.stop_epoch):
            self._run_epoch(X_train, y_train, n_samples, rng)
            if eval_set is not None:
                is_converged = self._is_converged(X_eval, y_eval, users)
                if is_converged:
                    best_epoch = epoch
                    break

            if self.verbose == True and epoch % 20 == 0:
                print(
                    f"Epoch {epoch}, factors num {self._factors_num}, \n"
                    f"factors abs mean: {np.mean(np.abs(self._factors))}, \n"
                    f"weights abs mean: {np.mean(np.abs(self._weights))}, \n"
                    f"intercept {self._w0},"
                    f"best validation {self.metric} score {self._best_score}"
                )

        if eval_set is not None:
            X_all = vstack([X_train, X_eval])
            y_all = np.concatenate([y_train, y_eval])
            self.stop_epoch = best_epoch
            return self.fit(X_all, y_all, None, None)

        self.fitted_ = True
        return self

    def _predict(self, X) -> tuple[dataset, dataset]:
        """
            Return prediction for given dataset and X @ self._factors
            for future gradient calculation.
        """
        X_dot_factors = X @ self._factors
        X_dot_factors_squares = X ** 2 @ (self._factors ** 2)
        interactions = np.sum(X_dot_factors ** 2 - X_dot_factors_squares, axis=1) / 2
        linear_part = X @ self._weights
        return self._w0 + linear_part + interactions, X_dot_factors

    def predict(self, X) -> dataset:
        """Predict target variable values for given features dataset."""
        X = check_array(X, accept_sparse=["csr"])
        if issparse(X):
            X = csr_array(X)
        return self._predict(X)[0]

    def score(self, X, y, users=None) -> float:
        """
            Evaluate object's score [ndcg, lowest_mean_rating, or rmse]
            assigned to it at instantiation.
        """
        y_pred = self.predict(X)
        if self.metric == "neg_rmse":
            return -root_mean_squared_error(y, y_pred)
        if self.metric == "mean_lowest_per_user":
            return mean_lowest_rating(
                y, y_pred, users=users, recommend_num=self.recommend_num
            )
        if self.metric == "ndcg":
            return ndcg_p_score(y, y_pred, users, p=self.recommend_num)
        else:
            raise NotImplementedError
