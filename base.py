import numpy as np
import random
from collections import defaultdict
import heapq
from typing import List, Tuple, Set, Dict
from tqdm import tqdm

class BasicUTRP:
    """
    Базовая имплементация алгоритма для Urban Transit Routing Problem
    """
    
    def __init__(self, weight_matrix: np.ndarray, demand_matrix: np.ndarray, 
                 min_vertices: int, max_vertices: int, num_routes: int):
        """
        Инициализация UTRP
        
        Args:
            weight_matrix: Матрица весов (время перемещения между остановками)
            demand_matrix: Матрица спроса между остановками
            min_vertices: Минимальное количество остановок в маршруте
            max_vertices: Максимальное количество остановок в маршруте
            num_routes: Количество маршрутов в сети
        """
        self.W = weight_matrix
        self.D = demand_matrix
        self.n = len(weight_matrix)
        self.m_min = min_vertices
        self.m_max = max_vertices
        self.r = num_routes
        
    def dijkstra(self, graph: np.ndarray, start: int) -> List[float]:
        """Алгоритм Дейкстры для поиска кратчайших путей"""
        n = len(graph)
        dist = [float('inf')] * n
        dist[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            if current_dist > dist[u]:
                continue
                
            for v in range(n):
                if graph[u][v] < float('inf'):
                    new_dist = current_dist + graph[u][v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
        return dist
    
    def yen_ksp(self, source: int, target: int, k: int = 5) -> List[List[int]]:
        """
        Упрощенная версия алгоритма Йена для K кратчайших путей
        """
        paths = []
        
        dist = self.dijkstra(self.W, source)
        if dist[target] == float('inf'):
            return paths

        path = self._reconstruct_path(source, target, dist)
        if path:
            paths.append(path)

        for _ in range(k-1):
            if len(paths) == 0:
                break
            last_path = paths[-1]

            if len(last_path) > 2:
                idx = random.randint(0, len(last_path)-2)
                temp_graph = self.W.copy()
                temp_graph[last_path[idx]][last_path[idx+1]] = float('inf')
                
                alt_dist = self.dijkstra(temp_graph, source)
                if alt_dist[target] < float('inf'):
                    alt_path = self._reconstruct_path(source, target, alt_dist)
                    if alt_path and alt_path not in paths:
                        paths.append(alt_path)
        
        return paths[:k]
    
    def _reconstruct_path(self, source: int, target: int, dist: List[float]) -> List[int]:
        """Восстановление пути из расстояний Дейкстры"""
        path = [target]
        current = target
        
        while current != source:
            found = False
            for prev in range(self.n):
                if (self.W[prev][current] < float('inf') and 
                    abs(dist[current] - dist[prev] - self.W[prev][current]) < 1e-6):
                    path.append(prev)
                    current = prev
                    found = True
                    break
            if not found:
                return []
        
        return path[::-1]
    
    def generate_initial_solutions(self, k: int = 5, num_solutions: int = 10) -> List[List[List[int]]]:
        """
        Генерация начальных решений методом K кратчайших путей
        """
        solutions = []
        
        for _ in tqdm(range(num_solutions)):
            candidate_routes = []
            all_ksp = []
            
            # Генерируем K кратчайших путей для всех пар
            for i in range(self.n):
                for j in tqdm(range(i+1, self.n)):
                    paths = self.yen_ksp(i, j, k)
                    all_ksp.extend(paths)
            print(f"Сгенерировано {len(all_ksp)} кандидатов маршрутов")
            
            # Фильтруем пути по длине
            feasible_paths = [p for p in all_ksp if self.m_min <= len(p) <= self.m_max]
            
            # Выбираем маршруты для покрытия всех вершин
            selected_routes = []
            covered_vertices = set()
            
            while len(selected_routes) < self.r and feasible_paths:
                # Выбираем маршрут, покрывающий максимум непокрытых вершин
                best_route = None
                max_new_vertices = -1
                
                for route in feasible_paths:
                    new_vertices = len(set(route) - covered_vertices)
                    if new_vertices > max_new_vertices:
                        max_new_vertices = new_vertices
                        best_route = route
                
                if best_route and max_new_vertices > 0:
                    selected_routes.append(best_route)
                    covered_vertices.update(best_route)
                    feasible_paths = [r for r in feasible_paths if r != best_route]
                else:
                    # Если не можем покрыть новые вершины, добавляем случайный маршрут
                    if feasible_paths:
                        random_route = random.choice(feasible_paths)
                        selected_routes.append(random_route)
                        covered_vertices.update(random_route)
                        feasible_paths.remove(random_route)
            
            # Если не хватает маршрутов, добавляем случайные
            while len(selected_routes) < self.r and len(selected_routes) > 0:
                # Дублируем и модифицируем существующий маршрут
                base_route = random.choice(selected_routes).copy()
                if len(base_route) < self.m_max:
                    # Добавляем случайную вершину
                    neighbors = []
                    for v in range(self.n):
                        if (v not in base_route and 
                            self.W[base_route[-1]][v] < float('inf')):
                            neighbors.append(v)
                    if neighbors:
                        base_route.append(random.choice(neighbors))
                selected_routes.append(base_route)
            
            if len(selected_routes) == self.r:
                solutions.append(selected_routes)
        
        return solutions
    
    def calculate_att(self, route_set: List[List[int]]) -> float:
        """
        Расчет Average Travel Time (ATT)
        """
        total_travel_time = 0
        total_demand = np.sum(self.D)
        
        if total_demand == 0:
            return float('inf')
        
        # Строим граф доступности по маршрутам
        accessibility_graph = np.full((self.n, self.n), float('inf'))
        np.fill_diagonal(accessibility_graph, 0)
        
        for route in route_set:
            for i in range(len(route)):
                for j in range(i, len(route)):
                    time = self._calculate_route_time(route[i:j+1])
                    if time < accessibility_graph[route[i]][route[j]]:
                        accessibility_graph[route[i]][route[j]] = time
                    if time < accessibility_graph[route[j]][route[i]]:
                        accessibility_graph[route[j]][route[i]] = time
        
        # Учитываем пересадки (упрощенно)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if (accessibility_graph[i][k] + accessibility_graph[k][j] < 
                        accessibility_graph[i][j]):
                        accessibility_graph[i][j] = (accessibility_graph[i][k] + 
                                                   accessibility_graph[k][j] + 5)  # +5 мин за пересадку
        
        # Считаем общее время путешествий
        for i in range(self.n):
            for j in range(self.n):
                if accessibility_graph[i][j] < float('inf'):
                    total_travel_time += self.D[i][j] * accessibility_graph[i][j]
        
        return total_travel_time / total_demand
    
    def calculate_trt(self, route_set: List[List[int]]) -> float:
        """
        Расчет Total Route Time (TRT)
        """
        total_time = 0
        for route in route_set:
            total_time += self._calculate_route_time(route)
        return total_time
    
    def _calculate_route_time(self, route: List[int]) -> float:
        """Расчет времени прохождения маршрута"""
        time = 0
        for i in range(len(route)-1):
            time += self.W[route[i]][route[i+1]]
        return time
    
    def is_feasible(self, route_set: List[List[int]]) -> bool:
        """
        Проверка feasibility решения
        """
        # Проверка количества маршрутов
        if len(route_set) != self.r:
            return False
        
        # Проверка длины маршрутов
        for route in route_set:
            if not (self.m_min <= len(route) <= self.m_max):
                return False
        
        # Проверка связности (упрощенно)
        all_vertices = set()
        for route in route_set:
            all_vertices.update(route)
        
        # Должны быть покрыты все вершины или большинство
        if len(all_vertices) < self.n * 0.8:  # 80% покрытие как минимум
            return False
            
        return True
    
    def add_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """
        Оператор добавления вершины (Add Vertex)
        """
        new_route_set = [route.copy() for route in route_set]
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) >= self.m_max:
            return new_route_set
        
        # Пробуем добавить вершину в начало или конец
        terminal_positions = [0, -1]
        random.shuffle(terminal_positions)
        
        for pos in terminal_positions:
            current_vertex = route[0] if pos == 0 else route[-1]
            neighbors = []
            
            for v in range(self.n):
                if (v not in route and 
                    self.W[current_vertex][v] < float('inf')):
                    neighbors.append(v)
            
            if neighbors:
                new_vertex = random.choice(neighbors)
                if pos == 0:
                    route.insert(0, new_vertex)
                else:
                    route.append(new_vertex)
                break
        
        return new_route_set
    
    def delete_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """
        Оператор удаления вершины (Delete Vertex)
        """
        new_route_set = [route.copy() for route in route_set]
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) <= self.m_min:
            return new_route_set
        
        # Удаляем конечную вершину
        if random.random() < 0.5 and len(route) > 1:
            route.pop(0)  # удаляем из начала
        else:
            route.pop()   # удаляем из конца
        
        return new_route_set
    
    def cost_based_grow(self, route_set: List[List[int]]) -> List[List[int]]:
        """
        Cost-based grow оператор (упрощенная версия)
        """
        new_route_set = [route.copy() for route in route_set]
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) >= self.m_max:
            return new_route_set
        
        # Оцениваем кандидатов для добавления
        candidates = []
        terminal_positions = [(route[0], 'start'), (route[-1], 'end')]
        
        for terminal_vertex, position in terminal_positions:
            for candidate in range(self.n):
                if (candidate not in route and 
                    self.W[terminal_vertex][candidate] < float('inf')):
                    
                    # Упрощенная оценка выгоды (основанная на спросе)
                    benefit = 0
                    for v in route:
                        benefit += self.D[candidate][v] + self.D[v][candidate]
                    
                    cost = self.W[terminal_vertex][candidate]
                    if cost > 0:
                        score = benefit / cost
                        candidates.append((score, candidate, position))
        
        if candidates:
            # Выбираем кандидата с вероятностью, пропорциональной score
            candidates.sort(reverse=True)
            total_score = sum(score for score, _, _ in candidates)
            if total_score > 0:
                rand_val = random.random() * total_score
                cumulative = 0
                for score, candidate, position in candidates:
                    cumulative += score
                    if cumulative >= rand_val:
                        if position == 'start':
                            route.insert(0, candidate)
                        else:
                            route.append(candidate)
                        break
        
        return new_route_set
    
    def basic_search(self, num_iterations: int = 1000) -> Tuple[List[List[int]], float, float]:
        """
        Базовый поисковый алгоритм
        """
        print("Генерация начальных решений...")
        initial_solutions = self.generate_initial_solutions(num_solutions=5)
        if not initial_solutions:
            raise ValueError("Не удалось сгенерировать начальные решения")
        
        current_solution = initial_solutions[0]
        best_solution = current_solution.copy()
        best_att = self.calculate_att(current_solution)
        best_trt = self.calculate_trt(current_solution)
        
        operators = [
            self.add_vertex_operator,
            self.delete_vertex_operator, 
            self.cost_based_grow
        ]

        print("Начало поискового процесса...")
        for iteration in tqdm(range(num_iterations)):
            # Выбираем случайный оператор
            operator = random.choice(operators)
            new_solution = operator(current_solution)
            
            if self.is_feasible(new_solution):
                new_att = self.calculate_att(new_solution)
                new_trt = self.calculate_trt(new_solution)
                
                # Простое правило принятия (можно улучшить)
                if new_att < best_att and new_trt < best_trt * 1.2:  # Допускаем увеличение TRT на 20%
                    current_solution = new_solution
                    best_solution = new_solution.copy()
                    best_att = new_att
                    best_trt = new_trt
                    print(f"Iteration {iteration}: ATT={new_att:.2f}, TRT={new_trt:.2f}")
            
            # Иногда сбрасываем на лучшее решение
            if iteration % 100 == 0:
                current_solution = best_solution.copy()
        
        return best_solution, best_att, best_trt

# Пример использования
def create_sample_problem(n):
    """Создание тестовой задачи"""
    W = np.full((n, n), float('inf'))
    np.fill_diagonal(W, 0)
    
    # Создаем случайные связи
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.3:  # 30% вероятность связи
                W[i][j] = W[j][i] = random.uniform(1, 10)
    
    # Матрица спроса
    D = np.random.randint(0, 100, (n, n))
    np.fill_diagonal(D, 0)
    
    return W, D

if __name__ == "__main__":
    # Создаем тестовую задачу
    W, D = create_sample_problem()
    
    # Инициализируем решатель
    utrp_solver = BasicUTRP(
        weight_matrix=W,
        demand_matrix=D,
        min_vertices=3,
        max_vertices=8,
        num_routes=4
    )
    
    # Запускаем поиск
    print("Запуск базового алгоритма UTRP...")
    best_solution, best_att, best_trt = utrp_solver.basic_search(num_iterations=500)
    
    print(f"\nЛучшее решение:")
    print(f"ATT: {best_att:.2f} мин")
    print(f"TRT: {best_trt:.2f} мин")
    print(f"Маршруты: {best_solution}")