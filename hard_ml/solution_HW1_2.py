from math import log2
import torch
from torch import Tensor


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    """ Функция для расчёта количества неправильно упорядоченных пар (корректное упорядочивание — от наибольшего
        значения в ys_true к наименьшему) или переставленных пар. Не забудьте, что одну и ту же пару
        не нужно учитывать дважды.
    :param ys_true: могут содержаться одинаковые значения
    :param ys_pred: содержит в себе уникальные значения без повторений
    :return:
    """
    _, ys_true_indices = torch.sort(ys_true, descending=True)
    unique_pairs = torch.combinations(ys_true_indices, 2)
    # ys_true могут содержаться одинаковые значения; это означает, что документы одинаково релевантны запросу.
    # И в этом случае нет разницы, какой из документов окажется ниже/выше в выдаче.
    attention_mask = ys_true[unique_pairs[:, 0]] != ys_true[unique_pairs[:, 1]]
    sorted_in_pair = ys_pred[unique_pairs[:, 0]] < ys_pred[unique_pairs[:, 1]]
    result = attention_mask & sorted_in_pair
    return result.sum().item()


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
    _, indices = torch.sort(ys_pred, descending=True)
    gain = compute_gain(ys_true[indices], gain_scheme).to(torch.float64)
    discount = torch.log2(torch.arange(2, gain.size()[0] + 2, dtype=torch.float64))
    return (gain / discount).sum().item()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_pred = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return dcg_pred / ideal_dcg


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # Если среди лейблов ys_true нет ни одного релевантного документа (единицы), то необходимо вернуть -1
    if (ys_true == 1).sum().item() == 0:
        return -1
    _, indices = torch.sort(ys_pred, descending=True)
    relevants = (ys_true[indices][:k] == 1).sum().item()
    # k может быть больше количества элементов во входных тензорах.
    # максимум функции в единице должен быть достижим при любом ys_true
    retrived = min(k, (ys_true == 1).sum().item())
    return float(relevants / retrived)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, indices = torch.sort(ys_pred, descending=True)
    rank = 1 + (ys_true[indices] == 1).nonzero(as_tuple=True)[0].item()
    return 1 / rank


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    _, ys_pred_indices = torch.sort(ys_pred, descending=True)
    p_rels = ys_true[ys_pred_indices]
    p_looks = torch.zeros(len(ys_true))
    p_looks[0] = 1
    p_found = p_looks[0] * p_rels[0]
    for i in range(1, len(ys_true)):
        p_looks[i] = p_looks[i - 1] * (1 - p_rels[i - 1]) * (1 - p_break)
        p_found += p_looks[i] * p_rels[i]
    return p_found


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if ys_true.sum().item() == 0:
        return -1
    _, ys_pred_indices = torch.sort(ys_pred, descending=True)
    relevance = ys_true[ys_pred_indices].sum().item()
    ys_sorted = ys_true[ys_pred_indices]
    ap = 0
    for i in range(len(ys_true)):
        if ys_sorted[i] == 1:
            rank = i + 1
            retrieved = ys_sorted[:rank].sum().item()
            ap += retrieved / rank
    return float(ap / relevance)
