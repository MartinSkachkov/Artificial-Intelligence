import random
import math

# # X = [
# #     [5.1, 3.5, 1.4, 0.2],
# #     [4.9, 3.0, 1.4, 0.2],
# #     [6.0, 2.2, 4.0, 1.0]
# # ]

# # y = [
# #     "Iris-setosa",
# #     "Iris-setosa",
# #     "Iris-versicolor"
# # ]

# X = []
# y = []

# n = int(input("rows: "))

# print(f"features {n}:")
# for _ in range(n):
#     feature = list(map(float, input().split()))
#     X.append(feature)

# print(f"targets {n}:")
# for _ in range(n):
#     target = input()
#     y.append(target)

# print(X)
# print(y)

# print(list(zip(X, y))) # [([5.1, 3.5, 1.4, 0.2], 'Iris-setosa'), ([4.9, 3.0, 1.4, 0.2], 'Iris-setosa'), ([6.0, 2.2, 4.0, 1.0], 'Iris-versicolor')]

# def normalize_value(x, min_val, max_val):
#     if min_val == max_val:
#         return 0
#     return (x - min_val) / (max_val - min_val)

# def get_col_min_max(data): # data = [ [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [6.0, 2.2, 4.0, 1.0] ]
#     cols = len(data[0]) # those are the features -> 4

#     mins = []
#     maxs = []

#     for col in range(cols):
#         column_data = [row[col] for row in data] # [5.1, 4.9, 6.0]
#         mins.append(min(column_data))
#         maxs.append(max(column_data)) 

#     return mins, maxs # [4.9, 2.2, 1.4, 0.2], [6.0, 3.5, 4.0, 1.0]

# def min_max_normalization(data):
#     mins, maxs = get_col_min_max(data)

#     normalized_data = []

#     for row in data:
#         normalized_row = []
#         for i in range(len(row)):
#             normalized_value = normalize_value(row[i], mins[i], maxs[i])
#             normalized_row.append(normalized_value)

#         normalized_data.append(normalized_row)
    
#     return normalized_data # [[0.18181818181818124, 1.0, 0.0, 0.0], [0.0, 0.6153846153846153, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]]


# # y = ["A", "A", "B", "B", "B"]

# # class_indices = {
# #     "A": [0, 1],
# #     "B": [2, 3, 4]
# # }

# def stratified_split(X, y, train_ratio=0.8):
#     class_indecies = {}

#     for i, label in enumerate(y):
#         if label not in class_indecies:
#             class_indecies[label] = []
#         class_indecies[label].append(i)

#     X_train, X_test = [], []
#     y_train, y_test = [], []

#     for label, indecies in class_indecies.items():
#         random.shuffle(indecies)

#         split_point = int(len(indecies) * train_ratio)

#         train_idxs = indecies[:split_point]
#         test_idxs = indecies[split_point:]

#         for i in train_idxs:
#             X_train.append(X[i])
#             y_train.append(y[i])

#         for i in test_idxs:
#             X_test.append(X[i])
#             y_test.append(y[i])

#     return X_train, X_test, y_train, y_test

# def euclidean(p, q): # p, q - n dimentional points
#     s = 0

#     for i in range(len(p)):
#         s += (p[i] - q[i]) ** 2
#     return math.sqrt(s)

# def accuracy(pred, real):
#     correct = 0
#     all_entries_count = len(pred)

#     for i in range(all_entries_count):
#         if pred[i] == real[i]:
#             correct += 1
    
#     return correct / all_entries_count

# def knn_predict(train_X, train_y, test_point, k):
#     distances = []

#     train_rows_count = len(train_X)

#     for i in range(train_rows_count):
#         dist = euclidean(train_X[i], test_point)
#         distances.append((dist, train_y[i])) # [(0.0, 'Iris-setosa'), (0.18, 'Iris-setosa'), ...]

#     distances.sort(key=lambda x: x[0])

#     neighbours = distances[:k]

#     votes = {}
#     for _, label in neighbours:
#         if label not in votes:
#             votes[label] = 0
#         votes += 1

#     return max(votes, key=votes.get)

# def coss_validation_10(X, y, k):
#     fold_size = len(X) // 10

#     accuracies = []

#     for fold in range(10):
#         start = fold * fold_size
#         end = start + fold_size

#         X_test = X[start:end]
#         y_test = y[start:end]

#         X_train = X[:start] + X[end:]
#         y_train = y[:start] + y[end:]

#         predictions = []

#         for i in range(len(X_test)):
#             pred = knn_predict(X_train, y_train, X_test[i], k)
#             predictions.append(pred)

#         acc = accuracy(predictions, y_test)
#         accuracies.append(acc)

#         print(f"Accuracy Fold {fold+1}: {round(acc*100,2)}%")
    
#     avg = sum(accuracies) / len(accuracies)
#     variance = sum((a - avg)**2 for a in accuracies) / len(accuracies)
#     std = math.sqrt(variance)

#     return avg, std


# k = int(input())
# X_train, X_test, y_train, y_test = stratified_split(X, y, train_ratio=0.8)
# test_predictions = []
# test_rows_count = len(X_test)

# for i in range(test_rows_count):
#     pred = knn_predict(X_train, y_train, X_test[i], k)
#     test_predictions.append(pred)

# test_accuracy = accuracy(test_predictions, y_test)
# print("\n3. Test Set Accuracy:", round(test_accuracy * 100, 2), "%")

# ----------------------------------------------------------------------

def entropy(labels):
    total = len(labels)

    counts = {}

    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent
        
y = ["A", "A", "B", "B", "B","B","B"]
print(entropy(y))

def information_gain(X, y, feature_index, threshold):
    parent_entropy = entropy(y)

    left_y = []
    right_y = []

    for i in range(len(X)):
        if X[i][feature_index] <= threshold:
            left_y.append(y[i])
        else:
            right_y.append(y[i])

    total = len(y)

    weighted_entropy = 0

    if len(left_y) > 0:
        weighted_entropy += (len(left_y) / total) * entropy(left_y)
    
    if len(right_y) > 0:
        weighted_entropy += (len(right_y) / total) * entropy(right_y)

    return parent_entropy - weighted_entropy

X = [
    [2.5],
    [3.5],
    [1.5],
    [4.5]
]

y = ["A", "A", "B", "B"]

ig = information_gain(X, y, feature_index=0, threshold=3.0)
print(ig)
