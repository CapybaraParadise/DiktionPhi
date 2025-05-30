
Документация по структурам графа
Класс Graph
Обеспечивает структуру графа (ориентированного или неориентированного).

Методы:
add_node(node_id, attrs=None) — добавляет узел с указанным идентификатором и необязательными атрибутами (словарь).
Ошибка, если узел уже существует.

add_edge(src_id, dst_id, attrs=None) — добавляет ребро от узла src_id к dst_id. Если узлы отсутствуют — создаются автоматически.
Для ориентированного графа ребро одностороннее, для неориентированного — двустороннее.

node_ids() — возвращает итератор по всем ID узлов.

node(node_id) — возвращает объект Node по ID.
Ошибка, если узел не существует.

__contains__(node_id) — проверка наличия узла, например if node_id in graph.

__iter__() — итератор по объектам Node в графе.

__len__() — количество узлов в графе.

to_dot(label_attr="label", weight_attr="weight") — возвращает строку в формате Graphviz DOT для визуализации графа.

Атрибуты:
type — тип графа (GraphType.DIRECTED или GraphType.UNDIRECTED).

Класс Node
Представляет узел графа с атрибутами и исходящими рёбрами.

Методы:
__getitem__(key) / __setitem__(key, val) — доступ к атрибутам узла как к словарю, например node["color"] = "red".

connect_to(dest, attrs=None) — создаёт ребро из текущего узла к dest. Можно передать атрибуты ребра.

to(dest) — возвращает объект Edge между текущим узлом и dest.
Ошибка, если ребро отсутствует.

is_edge_to(dest) — возвращает True, если существует ребро к dest.

Свойства:
neighbor_ids — итератор по ID соседей.

neighbor_nodes — итератор по объектам-соседям.

out_degree — количество исходящих рёбер.

Атрибуты:
id — идентификатор узла.

graph — ссылка на объект Graph, которому принадлежит узел.

Класс Edge
Представляет ребро между двумя узлами с атрибутами.

Методы:
__getitem__(key) / __setitem__(key, val) — доступ к атрибутам ребра как к словарю, например edge["weight"] = 5.

Атрибуты:
src — узел-источник (Node).

dest — узел-назначение (Node).

_attrs — словарь атрибутов ребра (вес, цвет и т.д.).

Класс GraphType (Enum)
GraphType.DIRECTED — ориентированный граф.
GraphType.UNDIRECTED — неориентированный граф.
Алгоритмы для графов
1. Рекурсивный DFS на Python

def dfs(graph: Graph, start: Node, visited:set =None, result: list =None) -> List[Hashable]:
    if visited is None:
        visited = set()
        result = []

    seznam = [start]

    while seznam:
        node = seznam.pop()
        if node.id in visited:
            continue
        visited.add(node.id)
        result.append(node.id)
        for neighbor in start.neighbor_nodes:
            if neighbor.id not in visited:
                dfs(graph, neighbor, visited=visited, result=result)

    return result


print(dfs(g, g.node(1)))
2. Обход в ширину (BFS)

from collections import deque

def bfs(graph: Graph, start: Node) -> List[Hashable]:
    visited = set()
    result = []
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node.id in visited:
            continue
        visited.add(node.id)
        result.append(node.id)
        for neighbor in node.neighbor_nodes:
            if neighbor.id not in visited:
                queue.append(neighbor)
    return result
3. Поиск кратчайшего пути в невзвешенном графе (BFS для пути)

from collections import deque

def shortest_path(graph: Graph, start: Node, goal: Node) -> Optional[List[Hashable]]:
    visited = set()
    queue = deque([(start, [start.id])])

    while queue:
        node, path = queue.popleft()
        if node.id == goal.id:
            return path
        if node.id in visited:
            continue
        visited.add(node.id)
        for neighbor in node.neighbor_nodes:
            if neighbor.id not in visited:
                queue.append((neighbor, path + [neighbor.id]))
    return None
4. Топологическая сортировка (для неориентированного ациклического графа)

def topological_sort(graph: Graph) -> List[Hashable]:
    in_degree = {node.id: 0 for node in graph}
    for node in graph:
        for neighbor in node.neighbor_nodes:
            in_degree[neighbor.id] += 1

    zero_in_degree = [node_id for node_id, deg in in_degree.items() if deg == 0]
    result = []

    while zero_in_degree:
        current = zero_in_degree.pop()
        result.append(current)
        current_node = graph.node(current)
        for neighbor in current_node.neighbor_nodes:
            in_degree[neighbor.id] -= 1
            if in_degree[neighbor.id] == 0:
                zero_in_degree.append(neighbor.id)

    if len(result) != len(in_degree):
        raise ValueError("Граф содержит цикл, топологическая сортировка невозможна")
    return result
5. Проверка наличия цикла в ориентированном графе (DFS)

def has_cycle(graph: Graph) -> bool:
    visited = set()
    rec_stack = set()

    def visit(node: Node) -> bool:
        if node.id in rec_stack:
            return True
        if node.id in visited:
            return False

        visited.add(node.id)
        rec_stack.add(node.id)
        for neighbor in node.neighbor_nodes:
            if visit(neighbor):
                return True
        rec_stack.remove(node.id)
        return False

    for node in graph:
        if visit(node):
            return True
    return False
6. Поиск компонент связности (для неориентированного графа)

def connected_components(graph: Graph) -> List[List[Hashable]]:
    visited = set()
    components = []

    def dfs(node: Node, comp: List[Hashable]):
        visited.add(node.id)
        comp.append(node.id)
        for neighbor in node.neighbor_nodes:
            if neighbor.id not in visited:
                dfs(neighbor, comp)

    for node in graph:
        if node.id not in visited:
            comp = []
            dfs(node, comp)
            components.append(comp)

    return components
7. Алгоритм Дейкстры для поиска кратчайшего пути в взвешенном графе


from typing import Dict, Hashable
import heapq

def dijkstra(graph: Graph, start: Node) -> Dict[Hashable, float]:
    """
    Алгоритм Дейкстры для поиска кратчайших путей в графе с неотрицательными весами.
    :param graph: объект Graph
    :param start: стартовая вершина Node
    :return: словарь {node_id: минимальное расстояние от start}
    """
    dist = {node_id: float('inf') for node_id in graph.node_ids()}
    dist[start.id] = 0

    # Очередь с приоритетом: элементы (расстояние, node_id)
    queue = [(0, start.id)]

    while queue:
        current_dist, current_id = heapq.heappop(queue)
        if current_dist > dist[current_id]:
            continue  # Мы уже нашли лучший путь к current_id

        current_node = graph.node(current_id)

        for neighbor in current_node.neighbor_nodes:
            edge = current_node.to(neighbor)
            weight = edge._attrs.get('weight', 1)  # вес ребра, по умолчанию 1
            new_dist = current_dist + weight

            if new_dist < dist[neighbor.id]:
                dist[neighbor.id] = new_dist
                heapq.heappush(queue, (new_dist, neighbor.id))

    return dist
print(dijkstra(g, g.node(1)))
8. Функции для вычисления количества выходящих и входящих ребер

from typing import List, Hashable

def nodes_with_out_degree(graph: Graph, degree: int) -> List[Hashable]:
    """
    Возвращает список ID вершин, у которых ровно degree исходящих рёбер.
    """
    result = []
    for node in graph:
        if node.out_degree == degree:
            result.append(node.id)
    return result


def nodes_with_in_degree(graph: Graph, degree: int) -> List[Hashable]:
    """
    Возвращает список ID вершин, у которых ровно degree входящих рёбер.
    """
    in_degree_counts = {node_id: 0 for node_id in graph.node_ids()}
    # Подсчитываем входящую степень для каждого узла
    for node in graph:
        for neighbor_id in node.neighbor_ids:
            in_degree_counts[neighbor_id] += 1
    # Формируем список с нужной входящей степенью
    return [node_id for node_id, count in in_degree_counts.items() if count == degree]
9. Проверка наличия цикла в неориентированном графе (DFS)

from typing import Set, Optional

def has_cycle_undirected(graph: Graph) -> bool:
    """
    Проверяет, есть ли цикл в неориентированном графе.
    """
    visited = set()

    def dfs(node: Node, parent: Optional[Node]) -> bool:
        visited.add(node.id)
        for neighbor in node.neighbor_nodes:
            if neighbor.id not in visited:
                if dfs(neighbor, node):
                    return True
            elif parent is None or neighbor.id != parent.id:
                # если сосед уже посещен и это не родитель — найден цикл
                return True
        return False

    for node in graph:
        if node.id not in visited:
            if dfs(node, None):
                return True
    return False
10. Проверка графов на изоморфность

from itertools import permutations

def are_isomorphic(graph1: Graph, graph2: Graph) -> bool:
    """
    Проверяет, являются ли два неориентированных графа изоморфными.
    Работает только для небольших графов (до 6-7 узлов) из-за комбинаторики.
    """
    nodes1 = list(graph1.node_ids())
    nodes2 = list(graph2.node_ids())

    if len(nodes1) != len(nodes2):
        return False
    if len(list(graph1.edges)) != len(list(graph2.edges)):
        return False

    # Перебираем все возможные соответствия между вершинами
    for perm in permutations(nodes2):
        mapping = dict(zip(nodes1, perm))
        match = True

        for node in nodes1:
            neighbors1 = sorted(n.id for n in graph1.node(node).neighbor_nodes)
            neighbors2 = sorted(mapping[n.id] for n in graph2.node(mapping[node]).neighbor_nodes if n.id in mapping.values())

            mapped_neighbors1 = sorted(mapping[n] for n in neighbors1 if n in mapping)
            if mapped_neighbors1 != neighbors2:
                match = False
                break

        if match:
            return True
    return False
11. Проверка является ли граф деревом

  from typing import Set

def is_tree(graph: Graph) -> bool:
    def dfs(node: Node, parent: Node, visited: Set[Hashable]) -> bool:
        visited.add(node.id)
        for neighbor in node.neighbor_nodes:
            if neighbor.id == parent.id:
                continue
            if neighbor.id in visited:
                return False  # найден цикл
            if not dfs(neighbor, node, visited):
                return False
        return True

    node_ids = list(graph.node_ids())
    if not node_ids:
        return True  # пустой граф можно считать деревом

    start_node = graph.node(node_ids[0])
    visited = set()

    if not dfs(start_node, start_node, visited):
        return False  # граф содержит цикл

    return len(visited) == len(node_ids)  # проверяем связность

12. Красивая печать неориентированного графа

class GraphFormatError(ValueError):
    """
    Исключение, возникающее при неверном формате графа или попытке использовать алгоритм
    на неподходящем типе графа (например, ориентированный вместо неориентированного).
    """
    def __init__(self, message="Неверный формат графа или неподходящий тип графа"):
        super().__init__(message)

def print_graph_tree_style(graph: Graph):
    """
    Печатает неориентированный граф в виде дерева, начиная с первой вершины.
    Работает корректно, если граф — дерево.
    """
    if graph.type != GraphType.UNDIRECTED:
        raise ValueError("Функция работает только с неориентированными графами")

    visited = set()
    nodes = list(graph)
    if not nodes:
        print("(пустой граф)")
        return

    def dfs_print(node: Node, prefix: str = '', is_last: bool = True):
        connector = '└─ ' if is_last else '├─ '
        print(prefix + connector + str(node.id))
        visited.add(node.id)

        neighbors = [n for n in node.neighbor_nodes if n.id not in visited]
        count = len(neighbors)
        for i, neighbor in enumerate(neighbors):
            is_last_child = (i == count - 1)
            extension = '   ' if is_last else '│  '
            dfs_print(neighbor, prefix + extension, is_last_child)

    # Начинаем с первого узла
    dfs_print(nodes[0])

13. Красивый вывод матрицы смежности
def adjacency_matrix(graph: Graph, directed: bool = True):
    nodes = sorted(graph.node_ids())
    index_map = {node_id: idx for idx, node_id in enumerate(nodes)}
    size = len(nodes)
    matrix = [['0' for _ in range(size)] for _ in range(size)]

    for node_id in nodes:
        node = graph.node(node_id)
        for neighbor in node.neighbor_nodes:
            i = index_map[node_id]
            j = index_map[neighbor.id]
            matrix[i][j] = '1'
            if not directed:
                matrix[j][i] = '1'

    # Печать заголовка
    header = "     " + "  ".join(str(n) for n in nodes)
    print(header)
    print("    " + "–––" * size)

    for i, row in enumerate(matrix):
        line = f"{nodes[i]:>3} | " + "  ".join(row)
        print(line)
14. Функция для поиска подстроки в строке

def find_substring(text: str, pattern: str) -> int:
    """
    Ищет подстроку `pattern` в строке `text`.
    Возвращает индекс начала подстроки или -1, если не найдена.
    """
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            return i
    return -1
15. Функция для проверки, является ли строка палиндромом

def is_palindrome(s: str) -> bool:
    s = s.lower()  # приводим к нижнему регистру, чтобы не учитывать регистр
    s = ''.join(c for c in s if c.isalnum())  # убираем все, кроме букв и цифр
    return s == s[::-1]
