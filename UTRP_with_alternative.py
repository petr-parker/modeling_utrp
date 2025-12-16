import numpy as np
import random
from collections import defaultdict
import heapq
from typing import List, Tuple, Set, Dict, Any
import copy

class EnhancedUTRP:
    """
    Полная имплементация алгоритма UTRP со всеми операторами и гиперэвристиками
    """
    
    def __init__(self, weight_matrix: np.ndarray, demand_matrix: np.ndarray, alt_matrix: np.ndarray,
                 min_vertices: int, max_vertices: int, num_routes: int):
        self.W = weight_matrix
        self.D = demand_matrix
        self.A = alt_matrix
        self.n = len(weight_matrix)
        self.m_min = min_vertices
        self.m_max = max_vertices
        self.r = num_routes
        
        # Инициализация всех операторов
        self.operators = {
            'a': self.add_vertex_operator,
            'b': self.delete_vertex_operator,
            'c': self.add_inside_vertex_operator,
            'd': self.delete_inside_vertex_operator,
            'e': self.invert_vertices_operator,
            'f': self.exchange_routes_operator,
            'g': self.replace_vertex_operator,
            'h': self.donate_vertex_operator,
            'i': self.cost_based_grow_operator,
            'j': self.cost_based_trim_operator
        }
        
        # Параметры гиперэвристик
        self.llh_scores = {name: 10 for name in self.operators.keys()}
        self.llh_applications = {name: 0 for name in self.operators.keys()}
        self.llh_successes = {name: 0 for name in self.operators.keys()}
    
    def yens_k_shortest_paths(self, source: int, target: int, K: int) -> List[List[int]]:
        """
        Реализация алгоритма Йена для нахождения K кратчайших путей
        """
        if source == target:
            return [[source]]
        
        # Создаем копию матрицы весов для безопасной модификации
        W_copy = self.W.copy()
        
        # A хранит найденные K кратчайших путей
        A = []
        # B хранит потенциальные кратчайшие пути
        B = []
        
        # Находим первый кратчайший путь с помощью Дейкстры
        first_path = self._reconstruct_path_static(W_copy, source, target)
        if first_path and self._is_route_connected(first_path):
            A.append(first_path)
        else:
            return []
        
        for k in range(1, K):
            if not A:
                break
                
            prev_path = A[-1]
            
            # Для каждого spur node в предыдущем пути
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[:i+1]
                
                # Создаем временную копию графа
                W_temp = W_copy.copy()
                
                # Удаляем ребра, которые являются частью предыдущих путей с тем же root path
                for path in A:
                    if len(path) > i and path[:i+1] == root_path:
                        if i + 1 < len(path):
                            u, v = path[i], path[i+1]
                            W_temp[u][v] = float('inf')
                
                # Удаляем узлы из root path (кроме spur node)
                for node in root_path[:-1]:
                    W_temp[node, :] = float('inf')
                    W_temp[:, node] = float('inf')
                    W_temp[node, node] = 0
                
                # Находим spur path от spur_node до target
                spur_path = self._reconstruct_path_static(W_temp, spur_node, target)
                
                if spur_path:
                    # Комбинируем root path и spur path
                    total_path = root_path[:-1] + spur_path
                    
                    # Проверяем, что путь связен в оригинальном графе
                    if self._is_route_connected(total_path):
                        # Добавляем путь в B, если его там еще нет
                        path_cost = self._calculate_path_cost(total_path)
                        if (path_cost < float('inf') and 
                            total_path not in [p for c, p in B] and 
                            total_path not in A):
                            B.append((path_cost, total_path))
            
            if not B:
                break
            
            # Сортируем B и добавляем кратчайший путь в A
            B.sort(key=lambda x: x[0])
            cost, shortest_path = B.pop(0)
            A.append(shortest_path)
        
        return A
    
    def _reconstruct_path_static(self, weight_matrix: np.ndarray, 
                                 source: int, target: int) -> List[int]:
        """
        Восстановление пути от source до target используя Дейкстру
        (статическая версия, не модифицирует self.W)
        """
        n = len(weight_matrix)
        distances = [float('inf')] * n
        distances[source] = 0
        parent = [-1] * n
        pq = [(0, source)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if u == target:
                break
                
            if current_dist > distances[u]:
                continue
            
            for v in range(n):
                if weight_matrix[u][v] < float('inf'):
                    new_dist = current_dist + weight_matrix[u][v]
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        parent[v] = u
                        heapq.heappush(pq, (new_dist, v))
        
        if distances[target] == float('inf'):
            return []
        
        # Восстанавливаем путь
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    
    def _calculate_path_cost(self, path: List[int]) -> float:
        """Расчет стоимости пути"""
        if len(path) < 2:
            return 0
        
        cost = 0
        for i in range(len(path) - 1):
            weight = self.W[path[i]][path[i+1]]
            if weight >= float('inf'):
                return float('inf')
            cost += weight
        return cost
    
    def _calculate_path_demand(self, path: List[int]) -> float:
        """Расчет обслуживаемого спроса для пути"""
        demand = 0
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                demand += self.D[path[i]][path[j]] + self.D[path[j]][path[i]]
        return demand
    
    def generate_initial_solutions(self, k: int = 5, num_solutions: int = 10, l: int = 10) -> List[List[List[int]]]:
        """
        Генерация начальных решений используя K кратчайших путей (алгоритм Йена)
        
        Args:
            k: количество кратчайших путей для каждой пары вершин
            num_solutions: количество решений для генерации
            l: количество source-target пар для сохранения
        """
        solutions = []
        
        # Генерируем K кратчайших путей между всеми парами вершин
        all_paths = []
        
        print(f"Генерация {k} кратчайших путей для всех пар вершин...")
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                if i == j:
                    continue
                
                # Находим K кратчайших путей от i до j
                paths = self.yens_k_shortest_paths(i, j, k)
                
                for path in paths:
                    path_length = len(path)
                    # Фильтруем по ограничениям длины маршрута
                    if self.m_min <= path_length <= self.m_max:
                        path_cost = self._calculate_path_cost(path)
                        path_demand = self._calculate_path_demand(path)
                        
                        all_paths.append({
                            'path': path,
                            'source': i,
                            'target': j,
                            'length': path_length,
                            'cost': path_cost,
                            'demand': path_demand
                        })
        
        if not all_paths:
            print("Не удалось сгенерировать пути, используем упрощенную генерацию")
            return self._generate_simple_solutions(num_solutions)
        
        # Сортируем по длине маршрута (по убыванию), затем по спросу (по возрастанию)
        all_paths.sort(key=lambda x: (-x['length'], x['demand']))
        
        # Группируем по source-target парам
        source_target_groups = defaultdict(list)
        for path_info in all_paths:
            key = (path_info['source'], path_info['target'])
            source_target_groups[key].append(path_info)
        
        # Выбираем top l пар
        top_pairs = []
        for key, paths in source_target_groups.items():
            if paths:
                # Берем лучший путь из каждой группы для ранжирования
                best_path = paths[0]
                top_pairs.append((key, paths, best_path))
        
        # Сортируем пары по тем же критериям
        top_pairs.sort(key=lambda x: (-x[2]['length'], x[2]['demand']))
        top_pairs = top_pairs[:l]
        
        # Создаем candidate route set из выбранных путей
        candidate_routes = []
        for key, paths, _ in top_pairs:
            # Берем первый путь из каждой выбранной пары
            if paths:
                candidate_routes.append(paths[0]['path'])
        
        print(f"Сгенерировано {len(candidate_routes)} кандидатов в маршруты")
        
        # Генерируем num_solutions решений
        for sol_idx in range(num_solutions):
            solution = []
            used_paths = set()
            
            # Случайно выбираем маршруты из кандидатов
            available_routes = [r for r in candidate_routes]
            random.shuffle(available_routes)
            
            for route in available_routes[:self.r]:
                if tuple(route) not in used_paths:
                    solution.append(route)
                    used_paths.add(tuple(route))

                if len(solution) >= self.r:
                    break
            
            # Если не хватает маршрутов, добавляем случайные из всех путей
            while len(solution) < self.r and all_paths:
                random_path_info = random.choice(all_paths)
                path = random_path_info['path']
                
                if tuple(path) not in used_paths:
                    # Проверяем, что маршрут связен
                    if self._is_route_connected(path):
                        solution.append(path)
                        used_paths.add(tuple(path))
            
            # Если все еще не хватает, генерируем новые
            attempts = 0
            while len(solution) < self.r and attempts < 20:
                attempts += 1
                new_route = self._generate_connected_route(solution)
                if (tuple(new_route) not in used_paths and 
                    len(new_route) >= self.m_min and
                    self._is_route_connected(new_route)):
                    solution.append(new_route)
                    used_paths.add(tuple(new_route))
            
            # Проверяем и исправляем покрытие всех вершин
            if solution and len(solution) >= self.r:
                solution = self._ensure_full_coverage(solution)
                solutions.append(solution[:self.r])
        
        return solutions if solutions else self._generate_simple_solutions(num_solutions)
    
    def _is_route_connected(self, route: List[int]) -> bool:
        """Проверка связности маршрута"""
        if len(route) < 2:
            return True
        
        for i in range(len(route) - 1):
            if self.W[route[i]][route[i+1]] >= float('inf'):
                return False
        return True
    
    def _generate_simple_solutions(self, num_solutions: int) -> List[List[List[int]]]:
        """Упрощенная генерация решений (fallback)"""
        solutions = []
        
        for _ in range(num_solutions):
            solution = []
            attempts = 0
            
            while len(solution) < self.r and attempts < 50:
                attempts += 1
                route = self._generate_connected_route(solution)
                
                if (len(route) >= self.m_min and 
                    self._is_route_connected(route)):
                    solution.append(route)
            
            if len(solution) >= self.r:
                # Гарантируем полное покрытие
                solution = self._ensure_full_coverage(solution)
                solutions.append(solution[:self.r])
        
        return solutions
    
    def calculate_att(self, route_set: List[List[int]]) -> float:
        total_travel_time = 0
        total_demand = np.sum(self.D)
        
        if total_demand == 0:
            return float('inf')
        
        # Строим граф доступности ТОЛЬКО для автобусов
        transit_graph = self.build_accessibility_graph(route_set)
        
        # Коэффициент "неудобства" альтернативы (например, такси в 2 раза дороже времени автобуса)
        # Если 1.0 — то чистое сравнение времени.
        alt_penalty_factor = 1.5 
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                
                # Время на автобусе (может быть inf)
                t_transit = transit_graph[i][j]
                
                # Время на альтернативе (из твоей новой матрицы)
                t_alt = self.A[i][j] * alt_penalty_factor
                
                # Пассажир выбирает лучшее
                actual_time = min(t_transit, t_alt)
                
                # Если даже альтернатива недоступна (inf), то это проблема
                if actual_time < float('inf'):
                    total_travel_time += self.D[i][j] * actual_time
                else:
                    # Штраф за полную недоступность (unserved demand)
                    # Можно использовать очень большое число
                    total_travel_time += self.D[i][j] * 10000 
                    
        return total_travel_time / total_demand

    def build_accessibility_graph(self, route_set: List[List[int]]) -> np.ndarray:
        """Построение графа доступности с проверкой на inf"""
        graph = np.full((self.n, self.n), float('inf'))
        np.fill_diagonal(graph, 0)
        
        # Прямые соединения в маршрутах
        for route in route_set:
            for i in range(len(route)):
                for j in range(i, len(route)):
                    # Проверяем, что подпуть связен
                    subpath_connected = True
                    time = 0
                    for k in range(i, j):
                        weight = self.W[route[k]][route[k+1]]
                        if weight >= float('inf'):
                            subpath_connected = False
                            break
                        time += weight
                    
                    if subpath_connected and time < graph[route[i]][route[j]]:
                        graph[route[i]][route[j]] = time
                        graph[route[j]][route[i]] = time
        
        # Учитываем пересадки (только между связными вершинами)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if (graph[i][k] < float('inf') and 
                        graph[k][j] < float('inf') and 
                        graph[i][k] + graph[k][j] + 5 < graph[i][j]):
                        graph[i][j] = graph[i][k] + graph[k][j] + 5
        
        return graph
    
    def calculate_trt(self, route_set: List[List[int]]) -> float:
        """Расчет Total Route Time (TRT) с проверкой на разрывы"""
        total_time = 0
        for route in route_set:
            route_time = 0
            for i in range(len(route)-1):
                weight = self.W[route[i]][route[i+1]]
                if weight >= float('inf'):
                    # Если есть разрыв в маршруте, возвращаем inf
                    return float('inf')
                route_time += weight
            total_time += route_time
        return total_time
    
    def _calculate_route_time(self, route: List[int]) -> float:
        """Расчет времени прохождения маршрута с проверкой связности"""
        if len(route) < 2:
            return 0
        
        time = 0
        for i in range(len(route)-1):
            weight = self.W[route[i]][route[i+1]]
            if weight >= float('inf'):
                return float('inf')
            time += weight
        return time

    def is_feasible(self, route_set: List[List[int]]) -> bool:
        """Проверка feasibility с проверкой связности маршрутов и полного покрытия"""
        if len(route_set) != self.r:
            return False
        
        # Проверяем каждый маршрут на связность
        for route in route_set:
            if not (self.m_min <= len(route) <= self.m_max):
                return False
            
            # Проверяем, что все последовательные вершины соединены
            for i in range(len(route)-1):
                if self.W[route[i]][route[i+1]] >= float('inf'):
                    return False
        
        # Проверяем покрытие всех вершин (ОБЯЗАТЕЛЬНО)
        all_vertices = set()
        for route in route_set:
            all_vertices.update(route)
        
        # Все вершины должны быть покрыты
        if len(all_vertices) < self.n:
            return False
        
        # Проверяем связность всей сети
        if not self.is_connected(route_set):
            return False
            
        return True

    def repair_solution(self, route_set: List[List[int]]) -> List[List[int]]:
        """Ремонт решения: удаление разрывов в маршрутах"""
        repaired = copy.deepcopy(route_set)
        
        for route_idx in range(len(repaired)):
            route = repaired[route_idx]
            i = 0
            max_iterations = len(route) * 2  # Защита от бесконечного цикла
            iteration_count = 0
            
            while i < len(route) - 1 and iteration_count < max_iterations:
                iteration_count += 1
                
                # Проверка на превышение максимальной длины
                if len(route) > self.m_max:
                    # Обрезаем маршрут до допустимой длины
                    route = route[:self.m_max]
                    repaired[route_idx] = route
                    break
                
                if i >= len(route) - 1:
                    break
                    
                if self.W[route[i]][route[i+1]] >= float('inf'):
                    # Найден разрыв - пытаемся найти обход
                    found_bypass = False
                    
                    # Ищем промежуточную вершину для соединения
                    if len(route) < self.m_max:  # Только если можем добавить вершину
                        for k in range(self.n):
                            if (k not in route and 
                                self.W[route[i]][k] < float('inf') and 
                                self.W[k][route[i+1]] < float('inf')):
                                route.insert(i+1, k)
                                found_bypass = True
                                break
                    
                    if found_bypass:
                        # Проверяем новое соединение и двигаемся дальше
                        i += 1
                    else:
                        # Если не нашли обход, удаляем проблемный сегмент
                        if i == 0 and len(route) > self.m_min:
                            route.pop(0)
                            # i остается 0
                        elif i == len(route) - 2 and len(route) > self.m_min:
                            route.pop()
                            break  # Достигли конца
                        else:
                            # Разделяем маршрут на два в точке разрыва
                            new_route1 = route[:i+1]
                            new_route2 = route[i+1:]
                            if (self.m_min <= len(new_route1) <= self.m_max and 
                                self.m_min <= len(new_route2) <= self.m_max):
                                repaired[route_idx] = new_route1
                                repaired.append(new_route2)
                                break
                            else:
                                # Если разделение невозможно, удаляем вершину
                                if len(route) > self.m_min:
                                    del route[i+1]
                                else:
                                    # Маршрут слишком короткий, пропускаем
                                    break
                else:
                    i += 1
            
            # Обновляем маршрут в случае изменений
            repaired[route_idx] = route
        
        # Удаляем пустые или слишком короткие маршруты
        repaired = [route for route in repaired if len(route) >= self.m_min]
        
        # Добавляем новые маршруты если нужно
        attempts = 0
        while len(repaired) < self.r and attempts < 10:
            attempts += 1
            new_route = self._generate_connected_route(repaired)
            if len(new_route) >= self.m_min:
                repaired.append(new_route)
        
        # ВАЖНО: Добавляем недостающие вершины
        repaired = self._ensure_full_coverage(repaired)
        
        return repaired[:self.r]
    
    def _ensure_full_coverage(self, route_set: List[List[int]]) -> List[List[int]]:
        """Гарантирует, что все вершины покрыты хотя бы одним маршрутом"""
        covered_vertices = set()
        for route in route_set:
            covered_vertices.update(route)
        
        missing_vertices = set(range(self.n)) - covered_vertices
        
        if not missing_vertices:
            return route_set
        
        # Добавляем недостающие вершины в существующие маршруты
        for vertex in missing_vertices:
            best_route_idx = -1
            best_position = -1
            min_cost_increase = float('inf')
            
            # Ищем лучшее место для вставки вершины
            for route_idx, route in enumerate(route_set):
                if len(route) >= self.m_max:
                    continue
                
                # Пробуем вставить в начало
                if self.W[vertex][route[0]] < float('inf'):
                    cost = self.W[vertex][route[0]]
                    if cost < min_cost_increase:
                        min_cost_increase = cost
                        best_route_idx = route_idx
                        best_position = 0
                
                # Пробуем вставить в конец
                if self.W[route[-1]][vertex] < float('inf'):
                    cost = self.W[route[-1]][vertex]
                    if cost < min_cost_increase:
                        min_cost_increase = cost
                        best_route_idx = route_idx
                        best_position = len(route)
                
                # Пробуем вставить между существующими вершинами
                for i in range(len(route) - 1):
                    if (self.W[route[i]][vertex] < float('inf') and
                        self.W[vertex][route[i+1]] < float('inf')):
                        old_cost = self.W[route[i]][route[i+1]]
                        new_cost = self.W[route[i]][vertex] + self.W[vertex][route[i+1]]
                        cost_increase = new_cost - old_cost
                        
                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_route_idx = route_idx
                            best_position = i + 1
            
            # Вставляем вершину в лучшую позицию
            if best_route_idx != -1:
                route_set[best_route_idx].insert(best_position, vertex)
            else:
                # Если не нашли подходящее место, создаем новый маршрут
                # Ищем ближайшую покрытую вершину
                min_dist = float('inf')
                nearest = None
                for v in covered_vertices:
                    if self.W[vertex][v] < min_dist:
                        min_dist = self.W[vertex][v]
                        nearest = v
                
                if nearest is not None and len(route_set) < self.r:
                    # Создаем короткий маршрут с этой вершиной
                    new_route = [vertex, nearest]
                    if self.m_min <= len(new_route) <= self.m_max:
                        route_set.append(new_route)
                        covered_vertices.add(vertex)
        
        return route_set

    def _generate_connected_route(self, existing_routes: List[List[int]]) -> List[int]:
        """Генерация связного маршрута"""
        # Используем существующие вершины для лучшего соединения
        all_vertices = set()
        for route in existing_routes:
            all_vertices.update(route)
        
        if not all_vertices:
            start = random.randint(0, self.n-1)
        else:
            start = random.choice(list(all_vertices))
        
        route = [start]
        current_vertex = start
        max_attempts = self.m_max * 2  # Защита от бесконечного цикла
        attempts = 0
        
        while len(route) < self.m_min and attempts < max_attempts:
            attempts += 1
            
            # Ищем только связных соседей
            neighbors = []
            for v in range(self.n):
                if (v not in route and 
                    self.W[current_vertex][v] < float('inf')):
                    neighbors.append(v)
            
            if not neighbors:
                # Если нет доступных соседей, начинаем с новой вершины
                available = [v for v in range(self.n) if v not in route]
                if not available:
                    break
                # Пробуем найти связную вершину
                found = False
                for v in available:
                    if self.W[current_vertex][v] < float('inf'):
                        route.append(v)
                        current_vertex = v
                        found = True
                        break
                if not found:
                    break
            else:
                next_vertex = random.choice(neighbors)
                route.append(next_vertex)
                current_vertex = next_vertex
        
        return route

    # Обновите операторы чтобы использовать repair
    def apply_operator_with_repair(self, operator_func, route_set: List[List[int]]) -> List[List[int]]:
        """Применение оператора с последующим ремонтом если нужно"""
        try:
            new_solution = operator_func(route_set)
            
            # Проверяем на разрывы
            if not self.is_feasible(new_solution):
                new_solution = self.repair_solution(new_solution)
            
            # Если после ремонта все еще не feasible, возвращаем оригинал
            if not self.is_feasible(new_solution):
                return route_set
                
            return new_solution
        except Exception as e:
            # В случае ошибки возвращаем оригинальное решение
            print(f"Ошибка в apply_operator_with_repair: {e}")
            return route_set

    def is_connected(self, route_set: List[List[int]]) -> bool:
        """Проверка связности графа маршрутов"""
        if not route_set:
            return False
            
        # Собираем все вершины, покрытые маршрутами
        all_vertices = set()
        for route in route_set:
            all_vertices.update(route)
        
        if not all_vertices:
            return False
        
        # Строим граф связей
        graph = np.zeros((self.n, self.n))
        for route in route_set:
            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                graph[u][v] = 1
                graph[v][u] = 1
        
        # BFS для проверки связности покрытых вершин
        visited = set()
        start = next(iter(all_vertices))  # Берем любую вершину из покрытых
        queue = [start]
        visited.add(start)
        max_iterations = len(all_vertices) * 2  # Защита от бесконечного цикла
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            u = queue.pop(0)
            
            for v in all_vertices:
                if graph[u][v] > 0 and v not in visited:
                    visited.add(v)
                    queue.append(v)
        
        # Проверяем, что все покрытые вершины достижимы
        return len(visited) >= len(all_vertices) * 0.9  # Допускаем 10% изолированных
    
    # ========== ОПЕРАТОРЫ МОДИФИКАЦИИ ==========
    
    def add_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """a) Add vertex - добавление вершины в конец маршрута"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) >= self.m_max:
            return new_route_set
        
        terminal_vertex = route[-1]
        neighbors = []
        for v in range(self.n):
            if v not in route and self.W[terminal_vertex][v] < float('inf'):
                neighbors.append(v)
        
        if neighbors:
            new_vertex = random.choice(neighbors)
            route.append(new_vertex)
        
        return new_route_set
    
    def delete_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """b) Delete vertex - удаление вершины с конца маршрута"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) <= self.m_min:
            return new_route_set
        
        route.pop()
        return new_route_set
    
    def add_inside_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """c) Add inside vertex - добавление вершины внутрь маршрута"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) >= self.m_max:
            return new_route_set
        
        if len(route) < 2:
            return new_route_set
        
        # Выбираем случайное ребро для вставки
        edge_idx = random.randint(0, len(route)-2)
        u, v = route[edge_idx], route[edge_idx+1]
        
        # Ищем вершину для вставки
        candidates = []
        for w in range(self.n):
            if (w not in route and 
                self.W[u][w] < float('inf') and 
                self.W[w][v] < float('inf')):
                candidates.append(w)
        
        if candidates:
            new_vertex = random.choice(candidates)
            route.insert(edge_idx+1, new_vertex)
        
        return new_route_set
    
    def delete_inside_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """d) Delete inside vertex - удаление внутренней вершины"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) <= self.m_min:
            return new_route_set
        
        if len(route) <= 2:
            return new_route_set
        
        # Удаляем случайную внутреннюю вершину
        idx = random.randint(1, len(route)-2)
        del route[idx]
        
        return new_route_set
    
    def invert_vertices_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """e) Invert vertices - инвертирование части маршрута"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) < 3:
            return new_route_set
        
        start = random.randint(0, len(route)-2)
        end = random.randint(start+1, len(route)-1)
        
        # Инвертируем сегмент
        route[start:end+1] = reversed(route[start:end+1])
        
        return new_route_set
    
    def exchange_routes_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """f) Exchange routes - обмен сегментами между маршрутами"""
        new_route_set = copy.deepcopy(route_set)
        
        if len(new_route_set) < 2:
            return new_route_set
        
        # Выбираем два маршрута
        idx1, idx2 = random.sample(range(len(new_route_set)), 2)
        route1, route2 = new_route_set[idx1], new_route_set[idx2]
        
        # Находим общие вершины
        common_vertices = set(route1) & set(route2)
        if not common_vertices:
            return new_route_set
        
        common_vertex = random.choice(list(common_vertices))
        
        # Находим позиции общей вершины
        pos1 = route1.index(common_vertex)
        pos2 = route2.index(common_vertex)
        
        # Обмениваем сегменты
        new_route1 = route1[:pos1] + route2[pos2:]
        new_route2 = route2[:pos2] + route1[pos1:]
        
        # Проверяем feasibility новых маршрутов
        if (self.m_min <= len(new_route1) <= self.m_max and 
            self.m_min <= len(new_route2) <= self.m_max):
            new_route_set[idx1] = new_route1
            new_route_set[idx2] = new_route2
        
        return new_route_set
    
    def replace_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """g) Replace vertex - замена вершины в маршруте"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) < 1:
            return new_route_set
        
        # Выбираем вершину для замены
        replace_idx = random.randint(0, len(route)-1)
        old_vertex = route[replace_idx]
        
        # Ищем замену
        candidates = []
        for v in range(self.n):
            if v not in route:
                # Проверяем связность с соседями
                left_ok = (replace_idx == 0 or 
                          self.W[route[replace_idx-1]][v] < float('inf'))
                right_ok = (replace_idx == len(route)-1 or 
                           self.W[v][route[replace_idx+1]] < float('inf'))
                
                if left_ok and right_ok:
                    candidates.append(v)
        
        if candidates:
            new_vertex = random.choice(candidates)
            route[replace_idx] = new_vertex
        
        return new_route_set
    
    def donate_vertex_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """h) Donate vertex - передача вершины другому маршруту"""
        new_route_set = copy.deepcopy(route_set)
        
        if len(new_route_set) < 2:
            return new_route_set
        
        # Выбираем маршрут-донор и маршрут-реципиент
        donor_idx, recipient_idx = random.sample(range(len(new_route_set)), 2)
        donor_route = new_route_set[donor_idx]
        recipient_route = new_route_set[recipient_idx]
        
        if len(donor_route) <= self.m_min or len(recipient_route) >= self.m_max:
            return new_route_set
        
        # Выбираем вершину для передачи
        vertex_idx = random.randint(0, len(donor_route)-1)
        vertex = donor_route[vertex_idx]
        
        if vertex in recipient_route:
            return new_route_set
        
        # Удаляем из донора
        del donor_route[vertex_idx]
        
        # Добавляем в реципиент
        if len(recipient_route) == 0:
            recipient_route.append(vertex)
        else:
            insert_pos = random.randint(0, len(recipient_route))
            recipient_route.insert(insert_pos, vertex)
        
        return new_route_set
    
    def cost_based_grow_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """i) Cost-based grow - умное добавление вершины на основе спроса"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) >= self.m_max:
            return new_route_set
        
        candidates = []
        
        # Оцениваем кандидатов для начала и конца маршрута
        for position, terminal_vertex in [('start', route[0]), ('end', route[-1])]:
            for candidate in range(self.n):
                if candidate not in route and self.W[terminal_vertex][candidate] < float('inf'):
                    # Расчет выгоды на основе спроса
                    benefit = 0
                    for v in route:
                        benefit += self.D[candidate][v] + self.D[v][candidate]
                    
                    cost = self.W[terminal_vertex][candidate]
                    if cost > 0:
                        score = benefit / cost
                        candidates.append((score, candidate, position))
        
        if candidates:
            # Выбор на основе вероятностей, пропорциональных score
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
    
    def cost_based_trim_operator(self, route_set: List[List[int]]) -> List[List[int]]:
        """j) Cost-based trim - умное удаление вершины на основе спроса"""
        new_route_set = copy.deepcopy(route_set)
        route_idx = random.randint(0, len(new_route_set)-1)
        route = new_route_set[route_idx]
        
        if len(route) <= self.m_min:
            return new_route_set
        
        candidates = []
        current_demand = self.calculate_route_demand(route)
        
        # Оцениваем удаление первой и последней вершины
        for position, vertex_idx in [('first', 0), ('last', -1)]:
            test_route = route.copy()
            removed_vertex = test_route[vertex_idx]
            del test_route[vertex_idx]
            
            if self.is_route_feasible(test_route):
                new_demand = self.calculate_route_demand(test_route)
                demand_loss = current_demand - new_demand
                
                if vertex_idx == 0:
                    cost_saving = self.W[route[0]][route[1]] if len(route) > 1 else 0
                else:
                    cost_saving = self.W[route[-2]][route[-1]] if len(route) > 1 else 0
                
                if demand_loss > 0 and cost_saving > 0:
                    score = cost_saving / demand_loss
                    candidates.append((score, vertex_idx))
        
        if candidates:
            # Выбор на основе вероятностей (предпочтение большему score)
            candidates.sort(reverse=True)
            total_score = sum(score for score, _ in candidates)
            if total_score > 0:
                rand_val = random.random() * total_score
                cumulative = 0
                for score, vertex_idx in candidates:
                    cumulative += score
                    if cumulative >= rand_val:
                        del route[vertex_idx]
                        break
        
        return new_route_set
    
    def calculate_route_demand(self, route: List[int]) -> float:
        """Расчет общего спроса, обслуживаемого маршрутом"""
        total_demand = 0
        for i in range(len(route)):
            for j in range(i, len(route)):
                total_demand += self.D[route[i]][route[j]] + self.D[route[j]][route[i]]
        return total_demand
    
    def is_route_feasible(self, route: List[int]) -> bool:
        return self.m_min <= len(route) <= self.m_max
    
    # ========== ГИПЕРЭВРИСТИКИ ==========
    
    def roulette_wheel_selection(self) -> str:
        """Выбор оператора по стратегии рулетки"""
        total_score = sum(self.llh_scores.values())
        if total_score == 0:
            return random.choice(list(self.operators.keys()))
        
        rand_val = random.random() * total_score
        cumulative = 0
        
        for llh_name, score in self.llh_scores.items():
            cumulative += score
            if cumulative >= rand_val:
                return llh_name
        
        return random.choice(list(self.operators.keys()))
    
    def update_llh_scores(self, llh_name: str, success: bool):
        """Обновление scores операторов"""
        self.llh_applications[llh_name] += 1
        
        if success:
            self.llh_successes[llh_name] += 1
            if self.llh_scores[llh_name] < 100:
                self.llh_scores[llh_name] += 1
        else:
            if self.llh_scores[llh_name] > 1:
                self.llh_scores[llh_name] -= 1
    
    def amalgam_selection(self, generation_successes: Dict[str, int], 
                         generation_applications: Dict[str, int]) -> Dict[str, float]:
        """AMALGAM стратегия для population-based поиска"""
        selection_probs = {}
        total_success_ratio = 0
        
        for llh_name in self.operators.keys():
            applications = generation_applications.get(llh_name, 1)
            successes = generation_successes.get(llh_name, 0)
            success_ratio = successes / applications if applications > 0 else 0
            total_success_ratio += success_ratio
        
        for llh_name in self.operators.keys():
            applications = generation_applications.get(llh_name, 1)
            successes = generation_successes.get(llh_name, 0)
            success_ratio = successes / applications if applications > 0 else 0
            
            if total_success_ratio > 0:
                selection_probs[llh_name] = (success_ratio / total_success_ratio * 
                                           (1 - 0.03)) + 0.03  # минимальная вероятность 3%
            else:
                selection_probs[llh_name] = 1.0 / len(self.operators)
        
        return selection_probs
    
    # ========== DBMOSA ==========
    
    def dbmosa(self, max_iterations: int = 1000, initial_temp: float = 1.0, 
               cooling_rate: float = 0.95) -> Tuple[List[List[int]], float, float]:
        """
        Dominance-based Multi-Objective Simulated Annealing
        """
        # Генерация начального решения
        initial_solutions = self.generate_initial_solutions(num_solutions=1)
        if not initial_solutions:
            raise ValueError("Не удалось сгенерировать начальное решение")
        
        current_solution = initial_solutions[0]
        archive = [current_solution]  # архив недоминируемых решений
        temperature = initial_temp
        
        best_att = self.calculate_att(current_solution)
        best_trt = self.calculate_trt(current_solution)
        
        for iteration in range(max_iterations):
            # Выбор оператора через гиперэвристику
            llh_name = self.roulette_wheel_selection()
            operator = self.operators[llh_name]
            
            # Генерация соседнего решения
            # new_solution = operator(current_solution)
            new_solution = self.apply_operator_with_repair(operator, current_solution)
            
            if not self.is_feasible(new_solution):
                self.update_llh_scores(llh_name, False)
                continue
            
            new_att = self.calculate_att(new_solution)
            new_trt = self.calculate_trt(new_solution)
            
            # Расчет энергии (доминирования)
            current_dominated = self.count_dominated(current_solution, archive)
            new_dominated = self.count_dominated(new_solution, archive)
            delta_energy = new_dominated - current_dominated
            
            # Правило принятия Metropolis
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / temperature):
                current_solution = new_solution
                success = True
                
                # Обновление архива
                if new_dominated == 0:
                    # Удаляем доминируемые решения
                    archive = [sol for sol in archive if not self.dominates(new_solution, sol)]
                    archive.append(new_solution)
                    
                    if new_att < best_att and new_trt < best_trt:
                        best_att = new_att
                        best_trt = new_trt
            else:
                success = False
            
            self.update_llh_scores(llh_name, success)
            
            # Охлаждение
            temperature *= cooling_rate
            
            if iteration % 100 == 0:
                print(f"DBMOSA Iteration {iteration}: Temp={temperature:.4f}, "
                      f"Archive size={len(archive)}, Best ATT={best_att:.2f}")
        
        # Возвращаем лучшее решение из архива
        best_solution = min(archive, key=lambda x: (self.calculate_att(x), self.calculate_trt(x)))
        return best_solution, self.calculate_att(best_solution), self.calculate_trt(best_solution)
    
    def count_dominated(self, solution: List[List[int]], archive: List[List[List[int]]]) -> int:
        """Подсчет количества решений в архиве, которые доминируют данное решение"""
        count = 0
        sol_att = self.calculate_att(solution)
        sol_trt = self.calculate_trt(solution)
        
        for arch_sol in archive:
            arch_att = self.calculate_att(arch_sol)
            arch_trt = self.calculate_trt(arch_sol)
            
            if (arch_att <= sol_att and arch_trt <= sol_trt and 
                (arch_att < sol_att or arch_trt < sol_trt)):
                count += 1
        
        return count
    
    def dominates(self, sol1: List[List[int]], sol2: List[List[int]]) -> bool:
        """Проверка, доминирует ли sol1 над sol2"""
        att1, trt1 = self.calculate_att(sol1), self.calculate_trt(sol1)
        att2, trt2 = self.calculate_att(sol2), self.calculate_trt(sol2)
        
        return (att1 <= att2 and trt1 <= trt2 and 
                (att1 < att2 or trt1 < trt2))
    
    # ========== NSGA-II ==========
    
    def nsga_ii(self, population_size: int = 50, generations: int = 100, 
                crossover_prob: float = 0.8, mutation_prob: float = 0.9) -> List[Tuple[List[List[int]], float, float]]:
        """
        Non-dominated Sorting Genetic Algorithm II
        """
        # Инициализация популяции
        population = self.generate_initial_solutions(num_solutions=population_size)
        
        for generation in range(generations):
            # Оценка fitness
            evaluated_pop = [(sol, self.calculate_att(sol), self.calculate_trt(sol)) 
                           for sol in population if self.is_feasible(sol)]
            
            if len(evaluated_pop) < 2:
                continue
            
            # Non-dominated sorting
            fronts = self.non_dominated_sorting(evaluated_pop)
            
            # Selection, crossover, mutation
            new_population = []
            generation_successes = defaultdict(int)
            generation_applications = defaultdict(int)
            
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self.tournament_selection(fronts)
                parent2 = self.tournament_selection(fronts)
                
                # Crossover
                if random.random() < crossover_prob:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])
                
                # Mutation с гиперэвристикой
                if random.random() < mutation_prob:
                    llh_name = self.roulette_wheel_selection()
                    operator = self.operators[llh_name]
                    mutated_child = operator(child)
                    
                    generation_applications[llh_name] += 1
                    
                    if self.is_feasible(mutated_child):
                        child = mutated_child
                        generation_successes[llh_name] += 1
                
                if self.is_feasible(child):
                    new_population.append(child)
            
            population = new_population[:population_size]
            
            if generation % 10 == 0:
                best_front = fronts[0] if fronts else []
                if best_front:
                    # Находим решение с наилучшим компромиссом (минимум суммы)
                    best_solution = min(best_front, key=lambda x: (x[1], x[2]))
                    best_att = best_solution[1]
                    best_trt = best_solution[2]
                    
                    # Также показываем диапазон значений в фронте
                    all_att = [att for _, att, _ in best_front]
                    all_trt = [trt for _, _, trt in best_front]
                    print(f"NSGA-II Generation {generation}: "
                          f"Best compromise: ATT={best_att:.2f}, TRT={best_trt:.2f} | "
                          f"Front: ATT=[{min(all_att):.2f}-{max(all_att):.2f}], "
                          f"TRT=[{min(all_trt):.2f}-{max(all_trt):.2f}], size={len(best_front)}")
        
        # Возвращаем Pareto front
        evaluated_pop = [(sol, self.calculate_att(sol), self.calculate_trt(sol)) 
                       for sol in population if self.is_feasible(sol)]
        fronts = self.non_dominated_sorting(evaluated_pop)
        
        return fronts[0] if fronts else []
    
    def non_dominated_sorting(self, population: List[Tuple]) -> List[List[Tuple]]:
        """Non-dominated sorting для NSGA-II"""
        fronts = [[]]
        n = len(population)
        
        # Создаем структуры для хранения информации о доминировании
        dominated_by = [0] * n
        dominates = [[] for _ in range(n)]
        
        # Заполняем матрицы доминирования
        for i in range(n):
            sol1, att1, trt1 = population[i]
            
            for j in range(n):
                if i == j:
                    continue
                
                sol2, att2, trt2 = population[j]
                
                # Проверяем, доминирует ли sol1 над sol2
                if (att1 <= att2 and trt1 <= trt2 and 
                    (att1 < att2 or trt1 < trt2)):
                    dominates[i].append(j)
                # Проверяем, доминирует ли sol2 над sol1  
                elif (att2 <= att1 and trt2 <= trt1 and 
                    (att2 < att1 or trt2 < trt1)):
                    dominated_by[i] += 1
            
            # Если ни одно решение не доминирует sol1, добавляем в первый фронт
            if dominated_by[i] == 0:
                fronts[0].append(population[i])
        
        # Строим последующие фронты
        i = 0
        while fronts[i]:
            next_front = []
            
            for sol_data in fronts[i]:
                sol_idx = population.index(sol_data)
                
                # Для каждого решения, которое доминирует текущее
                for dominated_idx in dominates[sol_idx]:
                    dominated_by[dominated_idx] -= 1
                    if dominated_by[dominated_idx] == 0:
                        next_front.append(population[dominated_idx])
            
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def tournament_selection(self, fronts: List[List[Tuple]]) -> List[List[int]]:
        """Tournament selection для NSGA-II"""
        if not fronts or not fronts[0]:
            return self.generate_initial_solutions(num_solutions=1)[0]
        
        # Выбираем случайный фронт (предпочтение более высоким фронтам)
        front_idx = min(len(fronts)-1, int(random.expovariate(1.0)))
        selected_front = fronts[front_idx]
        
        if not selected_front:
            return self.generate_initial_solutions(num_solutions=1)[0]
        
        # Выбираем случайное решение из выбранного фронта
        selected_solution, _, _ = random.choice(selected_front)
        return selected_solution

    def crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """Кроссовер для NSGA-II (улучшенная версия)"""
        child = []
        
        # Чередуем маршруты от родителей
        min_len = min(len(parent1), len(parent2))
        for i in range(min_len):
            if random.random() < 0.5:
                if parent1[i] not in child:
                    child.append(parent1[i].copy())
            else:
                if parent2[i] not in child:
                    child.append(parent2[i].copy())
        
        # Добавляем оставшиеся маршруты
        remaining_slots = self.r - len(child)
        if remaining_slots > 0:
            all_routes = parent1 + parent2
            available_routes = [route for route in all_routes if route not in child]
            
            while remaining_slots > 0 and available_routes:
                route = random.choice(available_routes)
                if route not in child:
                    child.append(route.copy())
                    available_routes.remove(route)
                    remaining_slots -= 1
        
        # Если все еще не хватает маршрутов, создаем новые
        while len(child) < self.r:
            new_route = self._generate_random_route()
            if new_route not in child:
                child.append(new_route)
        
        return child[:self.r]

    def _generate_random_route(self) -> List[int]:
        """Генерация случайного маршрута"""
        start = random.randint(0, self.n-1)
        route = [start]
        current_vertex = start
        
        while len(route) < self.m_min:
            neighbors = []
            for v in range(self.n):
                if (v not in route and 
                    self.W[current_vertex][v] < float('inf')):
                    neighbors.append(v)
            
            if not neighbors:
                break
                
            next_vertex = random.choice(neighbors)
            route.append(next_vertex)
            current_vertex = next_vertex
        
        return route

# Пример использования
def create_sample_problem(n: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    W = np.full((n, n), float('inf'))
    np.fill_diagonal(W, 0)
    
    # Создаем сеть
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.4:
                W[i][j] = W[j][i] = random.uniform(1, 15)
    
    D = np.random.randint(0, 50, (n, n))
    np.fill_diagonal(D, 0)
    
    return W, D

def generate_alternative_matrix(weight_matrix) -> np.ndarray:
    n = len(weight_matrix)
    dist_matrix = weight_matrix.copy()
    
    # Алгоритм Флойда-Уоршелла для нахождения всех кратчайших путей
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])
    
    # 2. Создаем матрицу альтернатив
    alt_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                alt_matrix[i][j] = 0
            elif dist_matrix[i][j] == float('inf'):
                alt_matrix[i][j] = float('inf')
            else:
                alt_matrix[i][j] = (dist_matrix[i][j])
                
    return alt_matrix

if __name__ == "__main__":
    # Создаем тестовую задачу
    W, D = create_sample_problem()
    A = generate_alternative_matrix(W)
    
    # Инициализируем решатель
    utrp_solver = EnhancedUTRP(
        weight_matrix=W,
        demand_matrix=D,
        alt_matrix=A,
        min_vertices=3,
        max_vertices=8,
        num_routes=5
    )
    
    print("Запуск DBMOSA...")
    dbmosa_solution, dbmosa_att, dbmosa_trt = utrp_solver.dbmosa(max_iterations=500)
    print(f"DBMOSA Результат: ATT={dbmosa_att:.2f}, TRT={dbmosa_trt:.2f}")
    
    print("\nЗапуск NSGA-II...")
    nsga_front = utrp_solver.nsga_ii(population_size=30, generations=50)
    if nsga_front:
        best_nsga = min(nsga_front, key=lambda x: (x[1], x[2]))
        print(f"NSGA-II Лучшее: ATT={best_nsga[1]:.2f}, TRT={best_nsga[2]:.2f}")
        print(f"Размер Pareto front: {len(nsga_front)}")
    
    print("\nСтатистика операторов:")
    for llh_name in utrp_solver.operators.keys():
        apps = utrp_solver.llh_applications[llh_name]
        success = utrp_solver.llh_successes[llh_name]
        success_rate = success / apps if apps > 0 else 0
        print(f"Оператор {llh_name}: Применений={apps}, Успехов={success}, "
              f"Успешность={success_rate:.2f}")