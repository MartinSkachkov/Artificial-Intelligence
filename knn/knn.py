# ============================================
# ГЛАВЕН ФАЙЛ
# ============================================

from ucimlrepo import fetch_ucirepo
import random
import math
import copy
import matplotlib.pyplot as plt

# Импортираме kd-tree модула
from kdtree import build_kdtree, knn_predict_kdtree

# ЗАРЕЖДАНЕ НА ДАННИТЕ
iris = fetch_ucirepo(id=53)
X_df = iris.data.features
y_df = iris.data.targets

X = X_df.values.tolist() # [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], ...]
y = y_df["class"].values.tolist() # ["Iris-setosa", "Iris-setosa", "Iris-versicolor", ...]


# НОРМАЛИЗАЦИЯ (Min-Max)
def min_max_normalize(data):
    """
    Min-Max нормализация: скалира всяка стойност в [0, 1].
    
    Формула: x_norm = (x - min) / (max - min)
    
    Защо е нужна за kNN:
    - kNN използва разстояния между точки
    - Ако една характеристика има стойности 0-100, а друга 0-1,
      първата ще доминира разстоянието
    - Нормализацията прави всички характеристики равнопоставени
    """
     
    normalized = copy.deepcopy(data)
    cols = len(data[0])
    rows = len(data)

    # Намираме min и max за всяка колона
    mins = [min(row[i] for row in data) for i in range(cols)]
    maxs = [max(row[i] for row in data) for i in range(cols)]

    # Нормализираме всяка стойност
    for r in range(rows):
        for c in range(cols):
            if maxs[c] == mins[c]:
                normalized[r][c] = 0
            else:
                normalized[r][c] = (data[r][c] - mins[c]) / (maxs[c] - mins[c])

    return normalized


X = min_max_normalize(X) # [[0.2, 0.6, 0.06, 0.04], ...]


# СТРАТИФИЦИРАНО 80/20 РАЗДЕЛЯНЕ
# Разделяме по класове, за да се запази пропорцията (50, 50, 50).
def stratified_split(X, y, train_ratio=0.8):
    """
    Стратифицирано разделяне на данните.
    
    Запазва пропорциите на класовете в train и test множествата.
    Пример: ако имаме 50-50-50 за 3 класа, и в train и в test
    ще имаме същото съотношение (40-40-40 в train, 10-10-10 в test).
    """

    # Списък за уникалните класове
    classes = list(sorted(set(y))) # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Разделяме по клас
    X_by_class = {c: [] for c in classes}
    y_by_class = {c: [] for c in classes}

    rows_count = len(X)

    for i in range(rows_count):
        current_class = y[i]
        X_by_class[current_class].append(X[i])
        y_by_class[current_class].append(y[i])

    # Създаваме train/test
    X_train, y_train = [], []
    X_test, y_test = [], []

    # За всеки клас разделяме 80/20
    for c in classes:
        data_c = list(zip(X_by_class[c], y_by_class[c])) # [([...], 'Iris-setosa'), ([...], 'Iris-setosa'), ...]
        random.shuffle(data_c)

        split_index = int(len(data_c) * train_ratio)

        train_part = data_c[:split_index]
        test_part  = data_c[split_index:]

        for pair in train_part:
            X_train.append(pair[0])
            y_train.append(pair[1])
        for pair in test_part:
            X_test.append(pair[0])
            y_test.append(pair[1])

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = stratified_split(X, y)


# ЕВКЛИДОВО РАЗСТОЯНИЕ
def euclidean(p, q):
    s = 0
    for i in range(len(p)):
        s += (p[i] - q[i]) ** 2
    return math.sqrt(s)


# kNN – НАЙ-БЛИЗКИ СЪСЕДИ
def knn_predict(train_X, train_y, test_point, k):
    """
    Наивна kNN имплементация: проверява всички точки.
    
    Сложност: O(n) за едно предсказание
    """
        
    # намираме всички разстояния
    distances = []
    train_rows_count = len(train_X)
    for i in range(train_rows_count):
        d = euclidean(train_X[i], test_point)
        distances.append((d, train_y[i])) # [(0.0, 'Iris-setosa'), (0.18, 'Iris-setosa'), ...]

    # сортиране по разстояние
    distances.sort(key=lambda x: x[0])

    # вземаме първите k съседи
    neighbors = distances[:k]

    # гласуване
    votes = {}
    for _, label in neighbors:
        if label not in votes:
            votes[label] = 0
        votes[label] += 1

    # печелившият клас
    return max(votes, key=votes.get)


# ФУНКЦИЯ ЗА ТОЧНОСТ
def accuracy(pred, real):
    """Изчислява точността: % правилни предсказания"""
    
    correct = 0
    for i in range(len(pred)):
        if pred[i] == real[i]:
            correct += 1
    return correct / len(pred)


# 10-FOLD CROSS-VALIDATION
# Правим 10 приблизително равни части
def cross_validation_10(X_data, Y_data, k, use_kdtree=False):
    """
    10-кратна кръстосана проверка.
    
    Разделя данните на 10 части (folds).
    За всяка част: тренира на останалите 9, тества на нея.
    """

    fold_size = len(X_data) // 10

    accuracies = []

    for fold in range(10):
        start = fold * fold_size
        end = start + fold_size

        X_test = X_data[start:end]
        Y_test = Y_data[start:end]

        X_train_cv = X_data[:start] + X_data[end:]
        Y_train_cv = Y_data[:start] + Y_data[end:]

        preds = []
        if use_kdtree:
            # Изграждаме kd-дърво
            kdtree = build_kdtree(X_train_cv, Y_train_cv)
            for i in range(len(X_test)):
                pred = knn_predict_kdtree(kdtree, X_test[i], k)
                preds.append(pred)
        else:
            # Наивна имплементация
            for i in range(len(X_test)):
                pred = knn_predict(X_train_cv, Y_train_cv, X_test[i], k)
                preds.append(pred)


        acc = accuracy(preds, Y_test)
        accuracies.append(acc)

        print(f"Accuracy Fold {fold+1}: {round(acc*100,2)}%")

    avg = sum(accuracies) / len(accuracies)

    # стандартно отклонение
    variance = sum((a - avg)**2 for a in accuracies) / len(accuracies)
    std = math.sqrt(variance)

    return avg, std


# ============================================
# ТЕСТВАНЕ С НАИВНАТА ИМПЛЕМЕНТАЦИЯ
# ============================================

k = int(input("\nВъведете k: "))

print("\n" + "="*60)
print("ТЕСТВАНЕ С НАИВНА kNN ИМПЛЕМЕНТАЦИЯ")
print("="*60)

# 1. ТОЧНОСТ ВЪРХУ TRAIN (наивна)
train_predictions = []
train_rows_count = len(X_train)
for i in range(train_rows_count):
    pred = knn_predict(X_train, y_train, X_train[i], k)
    train_predictions.append(pred)

train_acc = accuracy(train_predictions, y_train)
print("\n1. Train Accuracy:", round(train_acc * 100, 2), "%")

# 2. 10-FOLD CROSS-VALIDATION (наивна)
print("\n2. 10-Fold Cross-Validation Results:\n")
avg_acc, std_acc = cross_validation_10(X_train, y_train, k, use_kdtree=False)

print("\nAverage Accuracy:", round(avg_acc * 100, 2), "%")
print("Standard Deviation:", round(std_acc * 100, 2), "%")

# 3. ТОЧНОСТ ВЪРХУ TEST (наивна)
test_preds = []
test_rows_count = len(X_test)
for i in range(test_rows_count):
    pred = knn_predict(X_train, y_train, X_test[i], k)
    test_preds.append(pred)

test_acc = accuracy(test_preds, y_test)
print("\n3. Test Set Accuracy:", round(test_acc * 100, 2), "%")


# ============================================
# ТЕСТВАНЕ С KD-TREE
# ============================================

print("\n" + "="*60)
print("ТЕСТВАНЕ С KD-TREE ИМПЛЕМЕНТАЦИЯ")
print("="*60)

# Изграждаме kd-дърво от тренировъчните данни
kdtree = build_kdtree(X_train, y_train)

# 1. ТОЧНОСТ ВЪРХУ TRAIN (kd-tree)
train_predictions_kd = []
for i in range(train_rows_count):
    pred = knn_predict_kdtree(kdtree, X_train[i], k)
    train_predictions_kd.append(pred)

train_acc_kd = accuracy(train_predictions_kd, y_train)
print("\n1. Train Accuracy (kd-tree):", round(train_acc_kd * 100, 2), "%")

# 2. 10-FOLD CROSS-VALIDATION (kd-tree)
print("\n2. 10-Fold Cross-Validation Results (kd-tree):\n")
avg_acc_kd, std_acc_kd = cross_validation_10(X_train, y_train, k, use_kdtree=True)

print("\nAverage Accuracy (kd-tree):", round(avg_acc_kd * 100, 2), "%")
print("Standard Deviation (kd-tree):", round(std_acc_kd * 100, 2), "%")

# 3. ТОЧНОСТ ВЪРХУ TEST (kd-tree)
test_preds_kd = []
for i in range(test_rows_count):
    pred = knn_predict_kdtree(kdtree, X_test[i], k)
    test_preds_kd.append(pred)

test_acc_kd = accuracy(test_preds_kd, y_test)
print("\n3. Test Set Accuracy (kd-tree):", round(test_acc_kd * 100, 2), "%")


# ============================================
# ГРАФИКА
# ============================================

def plot_accuracy_k(X_train, y_train, X_test, y_test, max_k=20):
    """Генерира графика на точността за различни стойности на k"""
    ks = list(range(1, max_k + 1))
    accuracies_naive = []
    accuracies_kdtree = []

    for k in ks:
        # Наивна имплементация
        preds_naive = []
        for i in range(len(X_test)):
            pred = knn_predict(X_train, y_train, X_test[i], k)
            preds_naive.append(pred)
        acc_naive = accuracy(preds_naive, y_test)
        accuracies_naive.append(acc_naive)
        
        # kd-tree имплементация
        kdtree = build_kdtree(X_train, y_train)
        preds_kd = []
        for i in range(len(X_test)):
            pred = knn_predict_kdtree(kdtree, X_test[i], k)
            preds_kd.append(pred)
        acc_kd = accuracy(preds_kd, y_test)
        accuracies_kdtree.append(acc_kd)

    # Преобразуваме точността в проценти
    accuracies_naive_percent = [acc * 100 for acc in accuracies_naive]
    accuracies_kdtree_percent = [acc * 100 for acc in accuracies_kdtree]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, accuracies_naive_percent, marker='o', label='Наивна kNN', linewidth=2)
    plt.plot(ks, accuracies_kdtree_percent, marker='s', label='kd-tree kNN', linewidth=2, linestyle='--')
    plt.xlabel("k (брой съседи)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Сравнение на точността: Наивна kNN vs kd-tree kNN", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Изпълнение на графиката
plot_accuracy_k(X_train, y_train, X_test, y_test, max_k=20)