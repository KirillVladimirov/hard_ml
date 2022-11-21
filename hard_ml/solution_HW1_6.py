from typing import Callable, Dict, List
import numpy as np
from queue import PriorityQueue


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(pointA - documents, axis=1)
    return np.expand_dims(distance, axis=1)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    graph = {}
    for i in range(data.shape[0]):
        sort_indices = np.argsort(dist_f(data[i], data), axis=0)
        candidates_for_choice_short = sort_indices[1:num_candidates_for_choice_short + 1]
        candidates_for_choice_long = sort_indices[-num_candidates_for_choice_long:]
        edges_short = np.random.choice(candidates_for_choice_short.flatten(), num_edges_short, replace=False)
        choice_long = np.random.choice(candidates_for_choice_long.flatten(), num_edges_long, replace=False)
        edges = np.concatenate([choice_long, edges_short]).tolist()
        graph[i] = edges
    return graph


def nsw(query_point: np.ndarray, all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10,
        num_start_points: int = 5,
        dist_f: Callable = distance
        ) -> np.ndarray:
    result_queue = PriorityQueue()
    result_visited_set = set()
    visited_set = set()
    entry_node = np.random.choice(range(len(graph_edges)), 1, replace=False)
    min_dist = dist_f(query_point, all_documents[entry_node])
    candidate_queue = PriorityQueue()
    candidate_queue.put((min_dist.item(), entry_node.item()))
    while not candidate_queue.empty():
        min_dist, root_node = candidate_queue.get()
        if root_node not in visited_set:
            visited_set.add(root_node)
        else:
            continue
        candidates = np.array(graph_edges[root_node])
        candidate_distance = dist_f(query_point, all_documents[candidates])
        candidates = np.array(candidates)
        if np.sum((candidate_distance < min_dist).squeeze()) == 0:
            if root_node not in result_visited_set:
                result_visited_set.add(root_node)
                result_queue.put((min_dist, root_node))
            continue
        candidat_pairs = list(zip(
            candidate_distance[(candidate_distance < min_dist).squeeze()],
            candidates[(candidate_distance < min_dist).squeeze()]
        ))
        for dist, item in candidat_pairs:
            candidate_queue.put((dist.item(), item))

    return np.array([i for _, i in result_queue.queue[:search_k]])


#
# def nsw(query_point: np.ndarray, all_documents: np.ndarray,
#         graph_edges: Dict[int, List[int]],
#         search_k: int = 10,
#         num_start_points: int = 5,
#         dist_f: Callable = distance
#         ) -> np.ndarray:
#     result_queue = PriorityQueue()
#     result_visited_set = set()
#     visited_set = set()
#     for _ in range(num_start_points):
#         entry_node = np.random.choice(range(len(graph_edges)), 1, replace=False)
#         min_dist = dist_f(query_point, all_documents[entry_node])
#         candidate_queue = PriorityQueue()
#         candidate_queue.put((min_dist.item(), entry_node.item()))
#         while not candidate_queue.empty():
#             min_dist, root_node = candidate_queue.get()
#             if root_node not in visited_set:
#                 visited_set.add(root_node)
#             else:
#                 continue
#             candidates = np.array(graph_edges[root_node])
#             candidate_distance = dist_f(query_point, all_documents[candidates])
#             candidates = np.array(candidates)
#             if np.sum((candidate_distance < min_dist).squeeze()) == 0:
#                 if root_node not in result_visited_set:
#                     result_visited_set.add(root_node)
#                     result_queue.put((min_dist, root_node))
#                 continue
#             candidat_pairs = list(zip(
#                 candidate_distance[(candidate_distance < min_dist).squeeze()],
#                 candidates[(candidate_distance < min_dist).squeeze()]
#             ))
#             for dist, item in candidat_pairs:
#                 candidate_queue.put((dist.item(), item))
#
#     return np.array([i for _, i in result_queue.queue[:search_k]])
