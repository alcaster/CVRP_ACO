import numpy as np
from RegExService import getData

RANDOM_SEED = 7
ANT_CAPACITY = 6000
NUM_ANTS = 22
NUM_ITERATIONS = 10000
DEPOT_ID = 1

ALPHA = 4  # pheromone importance
BETA = 5  # inverse distance heuristic importance
RHO = 0.2  # pheromone evaporation coefficient
Q = 80

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

        self.demand_map = demand

    def init_adjacency_map(self, graph_data: dict):
        self.adjacency_map = {}

        nodes = list(graph_data.keys())
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                distance = self.get_euclidian_distance(graph_data[node_1], graph_data[node_2])
                self.adjacency_map.setdefault(node_1, {})
                self.adjacency_map.setdefault(node_2, {})
                self.adjacency_map[node_1][node_2] = distance
                self.adjacency_map[node_2][node_1] = distance

    def init_pheromone_map(self, nodes: list):
        self.pheromone_map = {}
        nodes = list(nodes)
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                # pheromone_init = np.random.uniform(1, 10)
                pheromone_init = 1
                self.pheromone_map.setdefault(node_1, {})
                self.pheromone_map.setdefault(node_2, {})
                self.pheromone_map[node_1][node_2] = pheromone_init
                self.pheromone_map[node_2][node_1] = pheromone_init

    def update_pheromone_map(self, solutions: list):
        # apply evaporation to all pheromones
        nodes = list(self.pheromone_map.keys())
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                new_value = (1 - RHO) * self.pheromone_map[node_1][node_2]
                self.pheromone_map[node_1][node_2] = new_value
                self.pheromone_map[node_2][node_1] = new_value

        for solution in solutions:
            pheromone_increase = Q / solution.cost
            for route in solution.routes:
                edges = [(route[index], route[index + 1]) for index in range(0, len(route) - 1)]
                for edge in edges:
                    self.pheromone_map[edge[0]][edge[1]] += pheromone_increase
                    self.pheromone_map[edge[1]][edge[0]] += pheromone_increase

    def get_euclidian_distance(self, p1, p2):
        return np.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))


class Ant:
    def __init__(self, graph: Graph, capacity):
        self.graph = graph
        self.starting_ant_capacity = capacity
        self.reset_state()

    def get_available_cities(self, current_city):
        allowed_by_capacity = [city for city in self.cities_left if self.capacity >= self.graph.demand_map[city]]
        if current_city != DEPOT_ID:
            allowed_by_capacity.append(DEPOT_ID)
        return allowed_by_capacity

    def select_first_city(self):
        available_cities = self.get_available_cities(DEPOT_ID)
        return np.random.choice(available_cities)

    def select_next_city(self, current_city):
        available_cities = self.get_available_cities(current_city)

        if not available_cities:
            return None

        normalizer = sum([pow(self.graph.pheromone_map[current_city][city], ALPHA) *
                          pow(self.graph.adjacency_map[current_city][city], -BETA)
                          for city in available_cities])

        next_city_distribution = [pow(self.graph.pheromone_map[current_city][city], ALPHA) *
                                  pow(self.graph.adjacency_map[current_city][city], -BETA) / normalizer
                                  for city in available_cities]


        next_city = np.random.choice(available_cities, p=next_city_distribution)

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
            self.move_to_city(current_city, next_city)
            if next_city == DEPOT_ID:
                self.start_new_route()

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

    # def generate_routes(self):
    #     indices = [index for index, el in enumerate(self.route) if el == 1]
    #     routes = [(self.route[indices[i]: indices[i + 1] + 1]) for i in range(0, len(indices) - 1)]
    #     return routes


class Solution:
    def __init__(self, routes, cost):
        self.routes = routes
        self.cost = cost


def run_aco():
    capacity, graph_data, demand, optimal_value = getData("./E-n22-k4.txt")
    graph = Graph(graph_data, demand)
    ants = [Ant(graph, capacity) for i in range(0, NUM_ANTS)]

    best_solution = None

    for i in range(1, NUM_ITERATIONS + 1):
        for ant in ants:
            ant.reset_state()
        solutions = []
        for ant in ants:
            solutions.append(ant.find_solution())

        candidate_best_solution = max(solutions, key=lambda solution: solution.cost)
        print(candidate_best_solution.cost, candidate_best_solution.routes)
        if not best_solution or candidate_best_solution.cost < best_solution.cost:
            best_solution = candidate_best_solution

        print("Best solution in iteration {}/{} = {}".format(i, NUM_ITERATIONS, best_solution.cost))
        graph.update_pheromone_map(solutions)
