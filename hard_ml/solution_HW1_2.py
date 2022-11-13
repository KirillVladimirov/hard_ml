from math import log2
import torch
from torch import Tensor


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    """ Функция для расчёта количества неправильно упорядоченных пар (корректное упорядочивание — от наибольшего
        значения в ys_true к наименьшему) или переставленных пар. Не забудьте, что одну и ту же пару
        не нужно учитывать дважды.
    :param ys_true:
    :param ys_pred: содержит в себе уникальные значения без повторений
    :return:
    """
    _, ys_true_indices = torch.sort(ys_true, descending=True)
    unique_pairs = torch.combinations(ys_true_indices, 2)
    sorted_in_pair = ys_pred[unique_pairs[:, 0]] < ys_pred[unique_pairs[:, 1]]
    return sorted_in_pair.sum().item()


def compute_gain(y_value: float, gain_scheme: str) -> float:
    """ compute_gain — вспомогательная функция для расчёта DCG и NDCG, рассчитывающая показатель Gain
        по значению релевантности.
    :param y_value:
    :param gain_scheme: ["const", "exp2"], указание схемы начисления Gain
    :return:
    """
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1
    else:
        raise ValueError('`gain_scheme` takes value from ["const", "exp2"]')


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # допишите ваш код здесь
    pass


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    pass


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    pass


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    # допишите ваш код здесь
    pass


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass
