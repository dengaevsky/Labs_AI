import numpy as np
from numpy.random import choice as np_choice


class AntColony(object):
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1.0, beta=1.0):
        # Квадратна матриця відстаней. Діагональ вважається np.inf.
        self.distances = distances

        # Кількість мурах, що запускаються за ітерацію
        self.n_ants = n_ants

        # Кількість кращих мурах, які відкладають феромон
        self.n_best = n_best

        # Кількість ітерацій
        self.n_iterations = n_iterations

        # Швидкість розпаду феромону
        self.decay = decay

        # Eкспонента на феромоні, вища альфа надає феромону більшої ваги
        self.alpha = alpha

        # Eкспонента на дистанції, вища бета надає дистанції більшої ваги.
        self.beta = beta

        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_indexes = range(len(distances))

    # Запуск алгоритму
    def run(self, start=0):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)

        for i in range(self.n_iterations):
            all_paths = self.calc_all_paths(start)
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])

            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path

            self.pheromone = self.pheromone * self.decay

        return all_time_shortest_path

    # Ініціалізація значеннями феромонів
    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])

        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    # Обрахунок довжини шляху
    def calc_path(self, path):
        total_dist = 0

        for element in path:
            total_dist += self.distances[element]
        return total_dist

    # Обрахунок довжини всіх шляхів
    def calc_all_paths(self, start):
        all_paths = []

        for i in range(self.n_ants):
            path = self.gen_path(start)
            all_paths.append((path, self.calc_path(path)))
        return all_paths

    # Переміщення до наступного пункту переміщення
    def gen_path(self, start):
        path = []

        visited = set()
        visited.add(start)

        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)

        path.append((prev, start))
        return path

    # Обрання наступного пункту переміщення
    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        normalized_row = row / row.sum()

        move = np_choice(self.all_indexes, 1, p=normalized_row)[0]
        return move
