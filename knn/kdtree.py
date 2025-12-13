
# ============================================
# KD-TREE ИМПЛЕМЕНТАЦИЯ
# ============================================

import math

class KDNode:
    """Възел в kd-дървото"""
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point      # точката (feature vector)
        self.label = label      # класът на точката
        self.axis = axis        # оста (измерението), по която се дели
        self.left = left        # ляво поддърво
        self.right = right      # дясно поддърво


def euclidean_distance(p, q):
    """
    Изчислява Евклидово разстояние между две точки.
    
    p, q: списъци с координати
    Връща: разстояние (float)
    """
    s = 0
    for i in range(len(p)):
        s += (p[i] - q[i]) ** 2
    return math.sqrt(s)


def build_kdtree(points, labels, depth=0):
    """
    Изгражда kd-дърво рекурсивно.
    
    Алгоритъм:
    1. Избираме ос за разделяне: axis = depth % k (циклично редуване)
    2. Сортираме точките по тази ос
    3. Избираме медианата като корен
    4. Рекурсивно строим ляво и дясно поддърво
    
    Args:
        points: списък от точки (feature vectors)
        labels: списък от етикети (класове)
        depth: дълбочината в дървото (определя оста на разделяне)
    
    Returns:
        корен на построеното kd-дърво (KDNode) или None
    """
    n = len(points)
    
    # Базов случай: няма точки
    if n == 0:
        return None
    
    # Определяме по коя ос ще разделяме
    # Ако имаме k измерения, редуваме осите: 0, 1, 2, ..., k-1, 0, 1, ...
    k = len(points[0])
    axis = depth % k
    
    # Комбинираме точките и етикетите за сортиране
    combined = list(zip(points, labels))
    
    # Сортираме по текущата ос
    # Пример: ако axis=0, сортираме по първата координата
    combined.sort(key=lambda x: x[0][axis])
    
    # Намираме медианата (средният елемент след сортиране)
    median_idx = n // 2
    
    # Разделяме данните
    median_point, median_label = combined[median_idx]
    
    # Рекурсивно строим лявото поддърво (точки < медиана)
    left_points = [p for p, l in combined[:median_idx]]
    left_labels = [l for p, l in combined[:median_idx]]
    
    # Рекурсивно строим дясното поддърво (точки >= медиана)
    right_points = [p for p, l in combined[median_idx + 1:]]
    right_labels = [l for p, l in combined[median_idx + 1:]]
    
    # Създаваме възел
    node = KDNode(
        point=median_point,
        label=median_label,
        axis=axis,
        left=build_kdtree(left_points, left_labels, depth + 1),
        right=build_kdtree(right_points, right_labels, depth + 1)
    )
    
    return node


def kdtree_search_recursive(node, query_point, k, best):
    """
    Помощна рекурсивна функция за търсене в kd-дърво.
    
    Обхожда дървото и намира k най-близки съседи.
    
    Args:
        node: текущият възел в дървото
        query_point: точката, за която търсим съседи
        k: брой на търсените съседи
        best: списък с текущите най-добри k съседи (модифицира се)
    """
    if node is None:
        return
    
    # Изчисляваме разстоянието до текущия възел
    dist = euclidean_distance(query_point, node.point)
    
    # Добавяме към най-добрите
    best.append((dist, node.label))
    best.sort(key=lambda x: x[0])  # сортираме по разстояние (възходящо)
    
    # Запазваме само най-близките k
    if len(best) > k:
        best.pop()  # премахваме най-далечния
    
    # Определяме в кое поддърво трябва да търсим първо
    axis = node.axis
    diff = query_point[axis] - node.point[axis]
    
    # Избираме по-близкото поддърво спрямо разделящата хиперравнина
    if diff < 0:
        # query_point е вляво от хиперравнината
        close_subtree = node.left
        far_subtree = node.right
    else:
        # query_point е вдясно от хиперравнината
        close_subtree = node.right
        far_subtree = node.left
    
    # Търсим в по-близкото поддърво
    kdtree_search_recursive(close_subtree, query_point, k, best)
    
    # PRUNING: Проверяваме дали трябва да търсим и в другото поддърво
    # Условия за търсене в далечното поддърво:
    # 1. Имаме по-малко от k съседи ИЛИ
    # 2. Разстоянието до хиперравнината < разстоянието до k-тия най-близък съсед
    #    (защото може да има по-близки точки от другата страна)
    if len(best) < k or abs(diff) < best[-1][0]:
        kdtree_search_recursive(far_subtree, query_point, k, best)


def kdtree_knn_search(root, query_point, k):
    """
    Намира k най-близки съседи използвайки kd-дърво.
    
    Алгоритъм:
    1. Обхождаме дървото рекурсивно
    2. За всеки възел изчисляваме разстоянието до query точката
    3. Поддържаме списък с k най-близки съседи
    4. Използваме "pruning" - отрязваме клонове, които не могат да съдържат по-близки съседи
    
    Pruning логика:
    - Изчисляваме разстоянието до хиперравнината (разделящата равнина)
    - Ако това разстояние > разстоянието до k-тия най-близък съсед,
      то всички точки в другото поддърво са по-далечни → пропускаме го!
    
    Args:
        root: коренът на kd-дървото
        query_point: точката, за която търсим съседи
        k: брой на търсените съседи
    
    Returns:
        списък от k двойки (разстояние, етикет)
    """
    # Списък за най-близките k съседи
    best = []
    
    # Стартираме рекурсивното търсене
    kdtree_search_recursive(root, query_point, k, best)
    
    return best


def knn_predict_kdtree(kdtree, test_point, k):
    """
    Прави предсказание използвайки kd-дърво.
    
    Args:
        kdtree: коренът на kd-дървото
        test_point: точката за предсказване
        k: брой съседи
    
    Returns:
        предсказания клас (етикет)
    """
    # Намираме k най-близки съседи
    neighbors = kdtree_knn_search(kdtree, test_point, k)
    
    # Гласуване: броим колко пъти се среща всеки клас
    votes = {}
    for _, label in neighbors:
        if label not in votes:
            votes[label] = 0
        votes[label] += 1
    
    # Връщаме класа с най-много гласове
    return max(votes, key=votes.get)