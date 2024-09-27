import numpy as np
import pandas as pd


class SingleNumberGenerator:
    """
    Class for generating a single number within a given range.

    Args:
        l (int): Lower bound of the range.
        r (int): Upper bound of the range.
    """

    def __init__(self, l: int, r: int):
        super().__init__()
        self.l = l
        self.r = r

    """
    Get a single random number within the given range [l, r].

    Returns:
        int: Random number within the given range [l, r].
    """

    def get_number(self) -> int:
        return np.random.randint(self.l, self.r + 1)


class ArrayGenerator:
    """
    Class for generating an array of random numbers within a given range.

    Args:
        l (int): Lower bound of the range.
        r (int): Upper bound of the range.
    """

    def __init__(self, l: int, r: int):
        super().__init__()
        self.l = l
        self.r = r

    @property
    def dimensions(self) -> list[int]:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: list[int]):
        self._dimensions = value

        self.num_numbers = np.prod(value)

    @property
    def sum(self) -> int:
        return self._sum

    @sum.setter
    def sum(self, value: int):
        self._sum = value

        if value is not None and (value < self.l * self.num_numbers or value > self.r * self.num_numbers):
            raise Exception("Invalid sum")

    """
    Get an array of random numbers within the given range [l, r].

    Returns:
        list: Array of random numbers within the given range [l, r].
    """

    def get_array(self) -> list:
        result = []

        # Inspired from https://github.com/brkdnmz/inzva-testcase-generator/blob/master/util.py, because it looks cool ngl
        if self.sum is not None:
            normalized_sum = self.sum - self.l * self.num_numbers
            diff = [0] + sorted(np.random.randint(0, normalized_sum,
                                self.num_numbers - 1)) + [normalized_sum]
            result = [diff[i+1] - diff[i] +
                      self.l for i in range(self.num_numbers)]
        else:
            result = np.random.randint(self.l, self.r + 1, self.num_numbers)

        result = np.reshape(result, self.dimensions)

        return result


class GraphGenerator:
    def __init__(self, node_count: int, edge_count: int, weight: list[int] = None, weight_sum: int = None):
        super().__init__()

        self._node_count = node_count
        self._edge_count = edge_count

        self.weight = weight  # [l, r]

        self.weighted = weight != None
        self.weight_sum = weight_sum

        self._directed = False

        self._cyclic = False
        self._self_loops = False
        self._multi_edges = False
        self._tree = False

        self.graph = [{} for _ in range(node_count)]

    @property
    def tree(self) -> bool:
        return self._tree

    @tree.setter
    def tree(self, value: bool):
        self._tree = value

        if value:
            self._cyclic = False
            self._multi_edges = False

            self.edge_count = self.node_count - 1

    @property
    def cyclic(self) -> bool:
        return self._cyclic

    @cyclic.setter
    def cyclic(self, value: bool):
        self._cyclic = value

        if value:
            self._tree = False
        else:
            self._self_loops = False
            self._directed = True

    @property
    def self_loops(self) -> bool:
        return self._self_loops

    @self_loops.setter
    def self_loops(self, value: bool):
        self._self_loops = value

        if value:
            self._cyclic = True

    @property
    def multi_edges(self) -> bool:
        return self._multi_edges

    @multi_edges.setter
    def multi_edges(self, value: bool):
        self._multi_edges = value

        if value:
            self._cyclic = True

    @property
    def directed(self) -> bool:
        return self._directed

    @directed.setter
    def directed(self, value: bool):
        self._directed = value

    @property
    def weight_sum(self) -> int:
        return self._weight_sum

    @property
    def node_count(self) -> int:
        return self._node_count

    @node_count.setter
    def node_count(self, value: int):
        self._node_count = value

        # Arrange the graph
        if value < len(self.graph):
            self.graph = self.graph[:value]
        elif value > len(self.graph):
            self.graph += [{} for _ in range(value - len(self.graph))]

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @edge_count.setter
    def edge_count(self, value: int):
        self._edge_count = value

    @weight_sum.setter
    def weight_sum(self, value: int):
        self._weight_sum = value

        if value is not None and (value < self.l * self.edge_count or value > self.r * self.edge_count):
            raise Exception("Invalid weight sum")

    def check_cycle(self) -> bool:
        visited = [False] * self.node_count
        stack = []

        stack.append(0)
        visited[0] = True

        while stack:
            u = stack.pop()

            for v in self.graph[u]:
                if not visited[v]:
                    stack.append(v)
                    visited[v] = True
                else:
                    return True

    def build_graph(self):
        current_sum = None if self.weight_sum == None else 0

        for _ in range(self.edge_count):
            u, v = np.random.randint(0, self.node_count, 2)
            w = (
                np.random.randint(self.weight[0], self.weight[1])
                if self.weight != None and len(self.weight) == 2
                else 1
            )

            if not self.self_loops and u == v:
                while u == v:
                    v = np.random.randint(0, self.node_count)

            if not self.multi_edges and v in self.graph[u]:
                continue

            self.graph[u][v] = w

            if current_sum != None:
                current_sum += w

            if not self.directed:
                self.graph[v][u] = w

        # TODO: Check graph in a more optimized way
        """ if not self._cyclic:
            if self.check_cycle():
                self.build_graph() """

        if self.weight_sum != None:
            while current_sum != self.weight_sum:
                diff = self.weight_sum - current_sum

                u, v = np.random.randint(0, self.node_count, 2)

                if not self.multi_edges and v in self.graph[u]:
                    continue

                if diff > 0:
                    w = np.random.randint(self.weight[0], self.weight[1])

                    current_sum -= self.graph[u][v]

                    self.graph[u][v] = min(
                        diff, self.weight[1] - self.graph[u][v])

                    current_sum += self.graph[u][v]
                else:
                    w = np.random.randint(self.weight[0], self.weight[1])

                    current_sum -= self.graph[u][v]

                    self.graph[u][v] = max(
                        diff, self.weight[0] - self.graph[u][v])

                    current_sum += self.graph[u][v]

    def get_graph(self, mode: str) -> list:
        if mode == "adj_matrix":
            return self.get_matrix()
        elif mode == "edge_list":
            return self.get_edge_list()
        else:
            raise Exception("Invalid mode")

    def get_matrix(self) -> list:
        if self.multi_edges:
            raise Exception("Cannot generate matrix for multi-edges")

        adj_matrix = np.zeros((self.node_count, self.node_count)).tolist()

        if not self.weighted:
            for u in range(self.node_count):
                for v in self.graph[u]:
                    adj_matrix[u][v] = 1
        else:
            for u in range(self.node_count):
                for v in self.graph[u]:
                    adj_matrix[u][v] = self.graph[u][v]

        return adj_matrix

    def get_edge_list(self) -> list:
        edge_list = []

        for u in range(self.node_count):
            for v in self.graph[u]:
                if not self.weighted:
                    edge_list.append((u, v))
                else:
                    edge_list.append((u, v, self.graph[u][v]))

        return edge_list


class StringGenerator:
    # TODO
    pass
