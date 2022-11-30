import pickle
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class Solution:
    def __init__(self,
                 n_estimators: int = 100,
                 lr: float = 0.5,
                 ndcg_top_k: int = 10,
                 subsample: float = 0.6,
                 colsample_bytree: float = 0.9,
                 max_depth: int = 5,
                 min_samples_leaf: int = 8):
        self._prepare_data()
        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample_rate = subsample
        self.colsample_bytree_rate = colsample_bytree
        self.trees = []
        self.features_ids = []
        self.best_ndcg = 0.0

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train, X_test, y_test, self.query_ids_test) = self._get_data()
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_train = torch.FloatTensor(y_train).unsqueeze(dim=1)
        self.ys_test = torch.FloatTensor(y_test).unsqueeze(dim=1)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()
        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)
        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)
        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray, inp_query_ids: np.ndarray) -> np.ndarray:
        transformed = []
        for ids in np.unique(inp_query_ids):
            data = inp_feat_array[inp_query_ids == ids]
            transformed.append(StandardScaler().fit_transform(data))
        return np.concatenate(transformed, axis=0)

    def _train_one_tree(self, cur_tree_idx: int, train_preds: torch.FloatTensor) -> Tuple[
        DecisionTreeRegressor, np.ndarray]:
        self.set_seed(cur_tree_idx)
        lambdas = torch.zeros(train_preds.shape)
        for query in np.unique(self.query_ids_train):
            query_mask = self.query_ids_train == query
            if self.ys_train[query_mask].sum() != 0:
                lambdas[query_mask] = self._compute_lambdas(self.ys_train[query_mask], train_preds[query_mask].float())
        samples_number, features_number = self.X_train.shape
        rng = np.random.default_rng(seed=cur_tree_idx)
        samples_idx = rng.permutation(samples_number)[:int(samples_number * self.subsample_rate)]
        features_idx = rng.permutation(features_number)[:int(features_number * self.colsample_bytree_rate)]
        X_train = self.X_train[samples_idx][:, features_idx]
        y_train = -lambdas[samples_idx]
        model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=cur_tree_idx,
        )
        model.fit(X_train, y_train)
        return model, features_idx

    def _compute_lambdas(self, ys_true: torch.FloatTensor, ys_pred: torch.FloatTensor) -> torch.FloatTensor:
        ideal_dcg_k = self._dcg_k(ys_true, ys_true, ndcg_top_k=self.ndcg_top_k)
        n = 1.0 / ideal_dcg_k
        _, indices = torch.sort(ys_true, descending=True, dim=0)
        indices += 1
        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((ys_pred - ys_pred.t()))
            rel_diff = ys_true - ys_true.t()
            pos_pairs = (rel_diff > 0).type(torch.float32)
            neg_pairs = (rel_diff < 0).type(torch.float32)
            s_ij = pos_pairs - neg_pairs
            gain_diff = torch.pow(2.0, ys_true) - torch.pow(2.0, ys_true.t())
            decay_diff = (1.0 / torch.log2(indices + 1.0)) - (1.0 / torch.log2(indices.t() + 1.0))
            delta_ndcg = torch.abs(n * gain_diff * decay_diff)
            lambda_update = (0.5 * (1 - s_ij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)
        return lambda_update

    def _calc_data_ndcg(self, queries_list: np.ndarray, true_labels: torch.FloatTensor,
                        preds: torch.FloatTensor) -> float:
        ndcgs = []
        for query in np.unique(queries_list):
            query_mask = queries_list == query
            if self.ys_train[query_mask].sum() != 0:
                ndcgs.append(self._ndcg_k(true_labels[query_mask], preds[query_mask], self.ndcg_top_k))
            else:
                ndcgs.append(0.0)
        return np.mean(ndcgs)

    def fit(self) -> None:
        self.set_seed(500)
        ndcg_test_scors = []
        train_preds = torch.zeros(self.ys_train.shape, dtype=torch.float32)
        test_preds = torch.zeros(self.ys_test.shape, dtype=torch.float32)
        for n_estimator in np.arange(self.n_estimators):
            model, features_idx = self._train_one_tree(n_estimator, train_preds)
            self.features_ids.append(features_idx)
            self.trees.append(model)
            train_preds += self.lr * model.predict(self.X_train[:, features_idx]).reshape(-1, 1)
            test_preds += self.lr * model.predict(self.X_test[:, features_idx]).reshape(-1, 1)
            ndcg_test_scors.append(self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds))
            if ndcg_test_scors[-1] > self.best_ndcg:
                self.best_ndcg = ndcg_test_scors[-1]
        best_score_tree_number = np.argmax(ndcg_test_scors) + 1
        self.trees = self.trees[:best_score_tree_number]
        self.features_ids = self.features_ids[:best_score_tree_number]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        pred = torch.zeros(data.shape[0], 1)
        for model, features_idx in list(zip(self.trees, self.features_ids)):
            pred += self.lr * model.predict(data[:, features_idx]).reshape(-1, 1)
        return pred

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int,
                gain_scheme: str = 'exp2') -> float:
        dcg_pred = self._dcg_k(ys_true, ys_pred, ndcg_top_k, gain_scheme)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k, gain_scheme)
        return dcg_pred / ideal_dcg

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int, gain_scheme: str = 'exp2') -> float:
        indices = ys_pred.argsort(descending=True, dim=0)[:ndcg_top_k]
        gain = self._compute_gain(ys_true[indices].reshape(-1, 1), gain_scheme=gain_scheme).to(torch.float64)
        discount = torch.log2(torch.arange(2, gain.size()[0] + 2, dtype=torch.float64)).unsqueeze(axis=1)
        return (gain / discount).sum().item()

    def _compute_gain(self, ys: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == 'const':
            return ys
        elif gain_scheme == 'exp2':
            return torch.pow(2, ys) - 1
        else:
            raise ValueError('`gain_scheme` takes value from ["const", "exp2"]')

    def save_model(self, path: str):
        state = {
            'trees': self.trees,
            'features_ids': self.features_ids,
            'lr': self.lr,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.trees = state['trees']
        self.features_ids = state['features_ids']
        self.lr = state['lr']

    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

