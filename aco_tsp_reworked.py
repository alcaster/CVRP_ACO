from time import sleep

import numpy as np
from RegExService import getData
from functools import reduce

RANDOM_SEED = 7

FILE_NAME = "E-n33-k4.txt"
NUM_ANTS = 22
ANT_CAPACITY = 6000
NUM_ITERATIONS = 1000
DEPOT_ID = 1

SOLUTIONS = {
    "E-n22-k4.txt": ([[18, 21, 19, 16, 13], [17, 20, 22, 15], [14, 12, 5, 4, 9, 11], [10, 8, 6, 3, 2, 7]], 375),
    "E-n33-k4.txt":
        ([[1, 15, 26, 27, 16, 28, 29], [30, 14, 31], [3, 5, 6, 10, 18, 19, 22, 21, 20, 23, 24, 25, 17, 13],
          [2, 12, 11, 32, 8, 9, 7, 4]], 835)
}

ALPHA = 2  # pheromone importance
BETA = 5  # inverse distance heuristic importance
RHO = 0.2  # pheromone evaporation coefficient
Q = 80

USE_2_OPT_STRATEGY = True
USE_CANDIDATE_LIST_STRATEGY = False
CANDIDATE_LIST_SIZE = NUM_ANTS // 3

np.random.seed(RANDOM_SEED)

"""
Assumptions:
1. Graph is complete, there is possibility to reach every city from any city.
2. Demand of each city is <= max capacity of an ant.
3. Solution quality corresponds to total sum of weights of edges which were used in the path. 
4. Every city is traversed only once, except depot. 
5. Depot is always visited at the start, end, when there is no capacity for any other move. It is also possible to return to depot at will.    
"""

"""
1. Zawsze opcja powrotu do depotu z probability jak do kazdego innego miasta
2. Jedyna metryka to totalna odległość
"""


class Graph:
    def __init__(self, graph_data, demand):
        # {city_id : {city_id: value} }
        self.init_adjacency_map(graph_data)
        self.init_pheromone_map(graph_data.keys())
        if USE_CANDIDATE_LIST_STRATEGY:
            self.init_candidate_list()

        self.demand_map = demand

    def init_adjacency_map(self, graph_data: dict):
        self.adjacency_map = {}

        nodes = list(sorted(graph_data.keys()))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                distance = self.get_euclidian_distance(graph_data[node_1], graph_data[node_2])
                self.adjacency_map.setdefault(node_1, {})
                self.adjacency_map.setdefault(node_2, {})
                self.adjacency_map[node_1][node_2] = distance
                self.adjacency_map[node_2][node_1] = distance

    def init_pheromone_map(self, nodes: list):
        self.pheromone_map = {}
        nodes = list(sorted(nodes))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                # pheromone_init = np.random.uniform(1, 10)
                pheromone_init = 1
                self.pheromone_map.setdefault(node_1, {})
                self.pheromone_map.setdefault(node_2, {})
                self.pheromone_map[node_1][node_2] = pheromone_init
                self.pheromone_map[node_2][node_1] = pheromone_init

    def update_pheromone_map(self, solutions: list):
        # avg_cost = reduce(lambda x, y: x + y, (solution.cost for solution in solutions)) / len(solutions)

        # apply evaporation to all pheromones
        nodes = list(sorted(self.pheromone_map.keys()))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                new_value = round((1 - RHO) * self.pheromone_map[node_1][node_2], 2)  # * Q / avg_cost
                self.pheromone_map[node_1][node_2] = new_value
                self.pheromone_map[node_2][node_1] = new_value

        for solution in solutions:
            pheromone_increase = 1 / solution.cost
            for route in solution.routes:
                edges = [(route[index], route[index + 1]) for index in range(0, len(route) - 1)]
                for edge in edges:
                    self.pheromone_map[edge[0]][edge[1]] += pheromone_increase
                    self.pheromone_map[edge[1]][edge[0]] += pheromone_increase

    def get_euclidian_distance(self, p1, p2):
        return np.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def init_candidate_list(self):
        self.candidate_list = {}

        for node, distances in self.pheromone_map.items():
            neighbours = [adjacency[0] for adjacency in sorted(distances.items(), key=lambda item: item[1])]
            self.candidate_list[node] = neighbours[:CANDIDATE_LIST_SIZE]


class Ant:
    def __init__(self, graph: Graph, capacity):
        self.graph = graph
        self.starting_ant_capacity = capacity
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

        if USE_CANDIDATE_LIST_STRATEGY:
            available_nearest_neighbours = [city for city in available_cities if
                                            city in self.graph.candidate_list[current_city]]
            if available_nearest_neighbours:
                available_cities = available_nearest_neighbours

        scores = [pow(self.graph.pheromone_map[current_city][city], ALPHA) *
                  pow(1 / self.graph.adjacency_map[current_city][city], BETA)
                  for city in available_cities]
        denominator = sum(scores)
        probabilities = [score / denominator for score in scores]

        next_city = np.random.choice(available_cities, p=probabilities)

        return next_city

    def move_to_city(self, current_city, next_city):
        self.routes[-1].append(next_city)
        if next_city != DEPOT_ID:
            self.cities_left.remove(next_city)
        self.capacity -= self.graph.demand_map[next_city]
        self.total_path_cost += self.graph.adjacency_map[current_city][next_city]

    def start_new_route(self):
        self.capacity = self.starting_ant_capacity
        self.routes.append([DEPOT_ID])

        # randomly select first city
        first_city = self.select_first_city()
        self.move_to_city(DEPOT_ID, first_city)

    def find_solution(self):
        self.start_new_route()

        while self.cities_left:
            current_city = self.routes[-1][-1]
            next_city = self.select_next_city(current_city)
            if next_city is None:
                self.move_to_city(current_city, DEPOT_ID)
                self.start_new_route()
            else:
                self.move_to_city(current_city, next_city)

        # always end at depot
        self.move_to_city(self.routes[-1][-1], DEPOT_ID)

        return Solution(self.routes,
                        self.total_path_cost)

    def reset_state(self):
        self.capacity = self.starting_ant_capacity
        self.cities_left = set(self.graph.adjacency_map.keys())
        self.cities_left.remove(DEPOT_ID)
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


def get_route_cost_opt(route, graph: Graph):
    # assumes route middle without starting and ending depot
    # add depot transport cost
    depot_costs = round(graph.adjacency_map[DEPOT_ID][route[0]], 5) + round(graph.adjacency_map[route[-1]][DEPOT_ID], 5)

    return depot_costs + get_route_cost(route, graph)


def two_opt_swap(route, i, k):
    new_route = []
    new_route.extend(route[:i])
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k + 1:])
    return new_route


def get_better_two_opt_swap(route, graph):
    num_eligible_nodes_to_swap = len(route)
    route_cost = get_route_cost_opt(route, graph)
    for i in range(0, num_eligible_nodes_to_swap - 1):
        for k in range(i + 1, num_eligible_nodes_to_swap):
            new_route = two_opt_swap(route, i, k)
            new_cost = get_route_cost_opt(new_route, graph)
            if new_cost < route_cost:
                return new_route
    return None


def get_optimal_route_intraswap(route, graph):
    best_route = route

    while True:
        improved_route = get_better_two_opt_swap(best_route, graph)
        if improved_route is None:
            break
        else:
            best_route = improved_route

    return best_route


def apply_two_opt(initial_solution, graph):
    best_routes = []

    for route in initial_solution.routes:
        # don't swap mandatory depots
        best_routes.append(get_optimal_route_intraswap(route[1:-1], graph))

    # apply back depot positions
    for route in best_routes:
        route.insert(0, DEPOT_ID)
        route.append(DEPOT_ID)

    return Solution(best_routes,
                    sum([get_route_cost(route, graph) for route in best_routes]))


def run_aco():
    capacity, graph_data, demand, optimal_value = getData("./" + FILE_NAME)
    graph = Graph(graph_data, demand)
    ants = [Ant(graph, capacity) for i in range(0, NUM_ANTS)]

    best_solution = None

    for i in range(1, NUM_ITERATIONS + 1):
        for ant in ants:
            ant.reset_state()
        solutions = []
        for ant in ants:
            solutions.append(ant.find_solution())

        candidate_best_solution = min(solutions, key=lambda solution: solution.cost)
        if USE_2_OPT_STRATEGY:
            candidate_best_solution = apply_two_opt(candidate_best_solution, graph)
        if not best_solution or candidate_best_solution.cost < best_solution.cost:
            best_solution = candidate_best_solution

        print("Best solution in iteration {}/{} = {}".format(i, NUM_ITERATIONS, best_solution.cost))
        graph.update_pheromone_map(solutions)

    print("---")
    print("Final best solution:")
    print(best_solution.cost)
    print(best_solution.routes)
    print("Optimal solution: ")
    print(SOLUTIONS[FILE_NAME])
