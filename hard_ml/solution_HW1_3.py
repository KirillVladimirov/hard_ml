import math
import random
import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


# class ListNet(torch.nn.Module):
#     def __init__(self, num_input_features: int, hidden_dim: int):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(num_input_features, self.hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.hidden_dim, 1),
#         )
#
#     def forward(self, input_1: torch.Tensor) -> torch.Tensor:
#         logits = self.model(input_1)
#         return logits


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self,
                 n_epochs: int = 5,
                 listnet_hidden_dim: int = 30,
                 lr: float = 0.001,
                 ndcg_top_k: int = 10
                 ):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()
        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)
        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)
        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        transformed = []
        for ids in np.unique(inp_query_ids):
            data = inp_feat_array[inp_query_ids == ids]
            transformed.append(StandardScaler().fit_transform(data))
        return np.concatenate(transformed, axis=0)

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        self.set_seed(42)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def fit(self) -> List[float]:
        nDCGs = []
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            nDCGs.append(self._eval_test_set())
            print("epoch:", epoch, "nDCG:", nDCGs[-1])
        return nDCGs

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i))

    def _train_one_epoch(self) -> None:
        self.model.train()
        ids_list = torch.unique(torch.IntTensor(self.query_ids_train))[
            torch.randperm(torch.unique(torch.IntTensor(self.query_ids_train)).shape[0])]
        for ids in ids_list:
            batch_X = self.X_train[self.query_ids_train == ids.item()]
            batch_ys = self.ys_train.view(-1, 1)[self.query_ids_train == ids.item()]
            self.optimizer.zero_grad()
            if len(batch_X) > 0:
                batch_pred = self.model(batch_X)
                batch_loss = self._calc_loss(batch_ys, batch_pred)
                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            ids_list = torch.unique(torch.IntTensor(self.query_ids_test))
            for ids in ids_list:
                batch_X = self.X_test[self.query_ids_test == ids.item()]
                batch_ys = self.ys_test[self.query_ids_test == ids.item()]
                valid_pred = self.model(batch_X)
                valid_pred = torch.flatten(valid_pred)
                ndcg_score = self._ndcg_k(batch_ys, valid_pred, self.ndcg_top_k)
                # print(f"nDCG: {ndcg_score:.4f}")
                ndcgs.append(ndcg_score)
            return np.mean(ndcgs)

    def _compute_gain(self, y_value: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == 'const':
            return y_value
        elif gain_scheme == 'exp2':
            return 2 ** y_value - 1
        else:
            raise ValueError('`gain_scheme` takes value from ["const", "exp2"]')

    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        gain = self._compute_gain(ys_true[indices[:ndcg_top_k]]).to(torch.float64)
        discount = torch.log2(torch.arange(2, gain.size()[0] + 2, dtype=torch.float64))
        return (gain / discount).sum().item()

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        dcg_pred = self._dcg(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg(ys_true, ys_true, ndcg_top_k)
        return dcg_pred / ideal_dcg