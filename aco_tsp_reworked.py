from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from RegExService import getData

RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

SOLUTIONS = {
    "E-n22-k4.txt": ([[18, 21, 19, 16, 13], [17, 20, 22, 15], [14, 12, 5, 4, 9, 11], [10, 8, 6, 3, 2, 7]], 375),
    "E-n33-k4.txt":
        ([[1, 15, 26, 27, 16, 28, 29], [30, 14, 31], [3, 5, 6, 10, 18, 19, 22, 21, 20, 23, 24, 25, 17, 13],
          [2, 12, 11, 32, 8, 9, 7, 4]], 835)
}


@dataclass
class Config:
    FILE_NAME = "E-n33-k4.txt"
    NUM_ANTS = 22
    ANT_CAPACITY = 6000
    NUM_ITERATIONS = 100
    DEPOT_ID = 1

    ALPHA = 2  # pheromone importance
    BETA = 5  # inverse distance heuristic importance
    RHO = 0.2  # pheromone evaporation coefficient
    Q = 80

    USE_2_OPT_STRATEGY = True
    USE_CANDIDATE_LIST_STRATEGY = False
    CANDIDATE_LIST_SIZE = NUM_ANTS // 3


class Graph:
    def __init__(self, graph_data, demand, config: Config):
        # {city_id : {city_id: value} }
        self.cfg = config
        self.adjacency_map = self.create_adjacency_map(graph_data)
        self.pheromone_map = self.create_pheromone_map(graph_data.keys())
        if self.cfg.USE_CANDIDATE_LIST_STRATEGY:
            self.candidate_list = self.create_candidate_list()

        self.demand_map = demand

    @staticmethod
    def create_adjacency_map(graph_data: dict) -> Dict[int, Dict[int, float]]:
        adjacency_map = {}

        nodes = list(sorted(graph_data.keys()))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                distance = Graph.get_euclidean_distance(graph_data[node_1], graph_data[node_2])
                adjacency_map.setdefault(node_1, {})
                adjacency_map.setdefault(node_2, {})
                adjacency_map[node_1][node_2] = distance
                adjacency_map[node_2][node_1] = distance
        return adjacency_map

    @staticmethod
    def create_pheromone_map(nodes: list) -> Dict[int, Dict[int, float]]:
        pheromone_map = {}
        nodes = list(sorted(nodes))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                # pheromone_init = np.random.uniform(1, 10)
                pheromone_init = 1
                pheromone_map.setdefault(node_1, {})
                pheromone_map.setdefault(node_2, {})
                pheromone_map[node_1][node_2] = pheromone_init
                pheromone_map[node_2][node_1] = pheromone_init
        return pheromone_map

    def update_pheromone_map(self, solutions: list):
        # avg_cost = reduce(lambda x, y: x + y, (solution.cost for solution in solutions)) / len(solutions)

        # apply evaporation to all pheromones
        nodes = list(sorted(self.pheromone_map.keys()))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                new_value = round((1 - self.cfg.RHO) * self.pheromone_map[node_1][node_2], 2)  # * Q / avg_cost
                self.pheromone_map[node_1][node_2] = new_value
                self.pheromone_map[node_2][node_1] = new_value

        for solution in solutions:
            pheromone_increase = 1 / solution.cost
            for route in solution.routes:
                edges = [(route[index], route[index + 1]) for index in range(0, len(route) - 1)]
                for edge in edges:
                    self.pheromone_map[edge[0]][edge[1]] += pheromone_increase
                    self.pheromone_map[edge[1]][edge[0]] += pheromone_increase

    @staticmethod
    def get_euclidean_distance(p1, p2) -> float:
        return np.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def create_candidate_list(self):
        candidate_list = {}

        for node, distances in self.pheromone_map.items():
            neighbours = [adjacency[0] for adjacency in sorted(distances.items(), key=lambda item: item[1])]
            candidate_list[node] = neighbours[:self.cfg.CANDIDATE_LIST_SIZE]
        return candidate_list


class Ant:
    def __init__(self, graph: Graph, capacity, config: Config):
        self.graph = graph
        self.starting_ant_capacity = capacity
        self.cfg = config
        self.reset_state()

    def get_available_cities(self):
        allowed_by_capacity = [city for city in self.cities_left if self.capacity >= self.graph.demand_map[city]]
        return allowed_by_capacity

    def select_first_city(self):
        available_cities = self.get_available_cities()
        return np.random.choice(available_cities)

    def select_next_city(self, current_city):
        available_cities = self.get_available_cities()

        if not available_cities:
            return None

        if self.cfg.USE_CANDIDATE_LIST_STRATEGY:
            available_nearest_neighbours = [city for city in available_cities if
                                            city in self.graph.candidate_list[current_city]]
            if available_nearest_neighbours:
                available_cities = available_nearest_neighbours

        scores = [pow(self.graph.pheromone_map[current_city][city], self.cfg.ALPHA) *
                  pow(1 / self.graph.adjacency_map[current_city][city], self.cfg.BETA)
                  for city in available_cities]
        denominator = sum(scores)
        probabilities = [score / denominator for score in scores]

        next_city = np.random.choice(available_cities, p=probabilities)

        return next_city

    def move_to_city(self, current_city, next_city):
        self.routes[-1].append(next_city)
        if next_city != self.cfg.DEPOT_ID:
            self.cities_left.remove(next_city)
        self.capacity -= self.graph.demand_map[next_city]
        self.total_path_cost += self.graph.adjacency_map[current_city][next_city]

    def start_new_route(self):
        self.capacity = self.starting_ant_capacity
        self.routes.append([self.cfg.DEPOT_ID])

        # randomly select first city
        first_city = self.select_first_city()
        self.move_to_city(self.cfg.DEPOT_ID, first_city)

    def find_solution(self):
        self.start_new_route()

        while self.cities_left:
            current_city = self.routes[-1][-1]
            next_city = self.select_next_city(current_city)
            if next_city is None:
                self.move_to_city(current_city, self.cfg.DEPOT_ID)
                self.start_new_route()
            else:
                self.move_to_city(current_city, next_city)

        # always end at depot
        self.move_to_city(self.routes[-1][-1], self.cfg.DEPOT_ID)

        return Solution(self.routes,
                        self.total_path_cost)

    def reset_state(self):
        self.capacity = self.starting_ant_capacity
        self.cities_left = set(self.graph.adjacency_map.keys())
        self.cities_left.remove(self.cfg.DEPOT_ID)
        self.routes = []
        self.total_path_cost = 0


class Solution:
    def __init__(self, routes, cost):
        self.routes = routes
        self.cost = cost


def get_route_cost(route, graph: Graph):
    total_cost = 0

    for i in range(0, len(route) - 1):
        total_cost += round(graph.adjacency_map[route[i]][route[i + 1]], 5)
    return total_cost


def get_route_cost_opt(route, graph: Graph, DEPOT_ID):
    # assumes route middle without starting and ending depot
    # add depot transport cost
    depot_costs = round(graph.adjacency_map[DEPOT_ID][route[0]], 5) + round(graph.adjacency_map[route[-1]][DEPOT_ID], 5)

    return depot_costs + get_route_cost(route, graph)


def two_opt(route, i, j) -> List[int]:
    """
    Perform two opt swap
    >>> two_opt([1,2,3,4,5,6], 1, 3)
    [1, 4, 3, 2, 5, 6]
    """
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def get_better_two_opt_swap(route, graph, DEPOT_ID) -> Optional[List[int]]:
    num_eligible_nodes_to_swap = len(route)
    route_cost = get_route_cost_opt(route, graph, DEPOT_ID)
    for i in range(0, num_eligible_nodes_to_swap - 1):
        for k in range(i + 1, num_eligible_nodes_to_swap):
            new_route = two_opt(route, i, k)
            new_cost = get_route_cost_opt(new_route, graph, DEPOT_ID)
            if new_cost < route_cost:
                return new_route
    return None


def get_optimal_route_intraswap(route, graph, DEPOT_ID):
    best_route = route

    while True:
        improved_route = get_better_two_opt_swap(best_route, graph, DEPOT_ID)
        if improved_route:
            best_route = improved_route
        else:
            break
    return best_route


def apply_two_opt(initial_solution, graph, DEPOT_ID):
    best_routes = [
        [DEPOT_ID] + get_optimal_route_intraswap(route[1:-1], graph, DEPOT_ID) + [DEPOT_ID]
        for route in initial_solution.routes
    ]

    return Solution(best_routes,
                    sum([get_route_cost(route, graph) for route in best_routes]))


def run_aco(cfg: Config, verbose: bool = True) -> Solution:
    capacity, graph_data, demand, optimal_value = getData("./" + cfg.FILE_NAME)
    graph = Graph(graph_data, demand, cfg)
    ants = [Ant(graph, capacity, cfg) for i in range(0, cfg.NUM_ANTS)]

    best_solution = None

    for i in range(1, cfg.NUM_ITERATIONS + 1):
        for ant in ants:
            ant.reset_state()
        solutions = []
        for ant in ants:
            solutions.append(ant.find_solution())

        candidate_best_solution = min(solutions, key=lambda solution: solution.cost)
        if cfg.USE_2_OPT_STRATEGY:
            candidate_best_solution = apply_two_opt(candidate_best_solution, graph, cfg.DEPOT_ID)
        if not best_solution or candidate_best_solution.cost < best_solution.cost:
            best_solution = candidate_best_solution

        if verbose:
            print("Best solution in iteration {}/{} = {}".format(i, cfg.NUM_ITERATIONS, best_solution.cost))
        graph.update_pheromone_map(solutions)
    if verbose:
        print("---")
        print("Final best solution:")
        print(best_solution.cost)
        print(best_solution.routes)
        print("Optimal solution: ")
        print(SOLUTIONS[cfg.FILE_NAME])
    return best_solution


if __name__ == '__main__':
    config = Config()
    run_aco(config)
