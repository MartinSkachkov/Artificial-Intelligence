import random
from collections import defaultdict, Counter
from numpy import indices
from ucimlrepo import fetch_ucirepo

from naive_bayes import NaiveBayes

def load_data():
    """
    Зарежда набора от данни Congressional Voting Records от UCI.
    """
    dataset = fetch_ucirepo(id=105)

    features_data = []
    labels_data = []

    features = dataset.data.features.values.tolist() # [['n', 'y', 'n',...], ...]
    labels = dataset.data.targets['Class'].values.tolist() # ['republican', 'democrat', ...]

    for row in features:
        features_data.append(list(row))

    for label in labels:
        labels_data.append(label)

    return features_data, labels_data

def handle_missing_mode_0(features_data):
    """
    Обработва липсващите стойности (nan), като ги приема
    за трета категория: 'abstain'.
    """
    new_features_data = []

    for row in features_data:
        new_features_row = []
        for v in row:
            if v != v:          # това означава v е nan
                new_features_row.append('abstain')
            else:
                new_features_row.append(v)
        new_features_data.append(new_features_row)

    return new_features_data

def handle_missing_mode_1(features_data, labels_data):
    """
    Запълва липсващите стойности според класа.

    Алгоритъм:
    1. За всеки клас (democrat / republican):
       - намира всички редове, принадлежащи на този клас
    2. За всеки атрибут:
       - намира най-често срещаната стойност в този клас
    3. При обхождане на всички данни:
       - ако се срещне nan
       - той се заменя с най-честата стойност
         за съответния атрибут и клас
    """
    new_features_data = [row[:] for row in features_data] # копираме старите features и ще ги заменим където е nan
    num_features = len(features_data[0]) # 16 
    classes = set(labels_data) #  # {'republican', 'democrat'}

    most_common = {} # most_common[(клас, индекс_на_атрибут)] = най-честа_стойност

    for cls in classes:
        indices = [i for i, label in enumerate(labels_data) if cls == label] # индексите на редовете в features_data, които принадлежат към този клас

        for j in range(num_features):
            values = []

            for i in indices:
                if features_data[i][j] == features_data[i][j]:  # НЕ е nan
                    values.append(features_data[i][j])

            most_common[(cls, j)] = Counter(values).most_common(1)[0][0]

    for i in range(len(features_data)):
        for j in range(num_features):
            if features_data[i][j] != features_data[i][j]: # e nan
                new_features_data[i][j] = most_common[(labels_data[i], j)]

    return new_features_data

def accuracy(real_labels, predicted_labels):
    """
    Изчислява точността на класификацията.

    Алгоритъм:
    - сравнява елемент по елемент реалния и предсказания клас
    - брои колко от тях съвпадат
    - дели броя на съвпаденията на общия брой примери
    """
    correct = 0
    total_num_labels = len(real_labels)

    for i in range(total_num_labels):
        if real_labels[i] == predicted_labels[i]:
            correct += 1

    return correct / total_num_labels

def stratified_split(features_data, labels_data, test_ratio=0.2):
    """
    Разделя данните на тренировъчно и тестово множество,
    като запазва първоначалното съотношение на класовете.

    Алгоритъм:
    1. Смесва данните на случаен принцип
    2. Групира ги по клас
    3. За всеки клас:
       - отделя 80% за обучение
       - отделя 20% за тестване
    4. Обединява всички класове обратно
    """
    data = list(zip(features_data, labels_data)) # data = [([...], 'democrat'), ([...], 'republican'), ([...], 'democrat')]
    random.shuffle(data)

    group_by_class = defaultdict(list)
    for features, label in data:
        group_by_class[label].append((features,label)) # by_class['democrat'] = [([...], 'democrat'), ([...], 'democrat'), ...]

    train, test = [], []

    for cls in group_by_class:
        cls_data = group_by_class[cls]
        split = int(len(cls_data) * (1 - test_ratio))
        train.extend(cls_data[:split])
        test.extend(cls_data[split:])       

    features_train, label_train = zip(*train)
    features_test, labels_test = zip(*test)

    return list(features_train), list(label_train), list(features_test), list(labels_test)

def cross_validation(features_data, labels_data, folds=10):
    """
    Реализира 10-кратна кръстосана проверка.

    Алгоритъм:
    1. Разбърква всички данни
    2. Разделя ги на 10 приблизително равни части (folds)
    3. За всяка итерация:
       - използва 1 fold за тестване
       - използва останалите за обучение
       - тренира нов модел
       - измерва точността
    4. Запазва точността за всеки fold
    """
    data = list(zip(features_data, labels_data))
    random.shuffle(data)

    fold_size = len(data) // folds
    accuracies = []

    for i in range(folds):
        test = data[i*fold_size:(i+1)*fold_size]
        train = data[:i*fold_size] + data[(i+1)*fold_size:]

        features_train, labels_train = zip(*train)
        features_test, labels_test = zip(*test)

        model = NaiveBayes()
        model.fit(list(features_train), list(labels_train))
        preds = model.predict(list(features_test))

        accuracies.append(accuracy(list(labels_test), preds))

    return accuracies

def main():
    """
    Управлява цялата програма.

    Последователност:
    1. Чете входния режим (0 или 1)
    2. Зарежда данните
    3. Обработва липсващите стойности според режима
    4. Разделя данните на train и test
    5. Обучава Наивен Бейсов класификатор
    6. Изчислява:
       - точност върху тренировъчните данни
       - 10-fold cross validation
       - точност върху тестовите данни
    7. Извежда резултатите
    """
    mode = int(input())

    features_data, label_data = load_data()

    if mode == 0:
        features_data = handle_missing_mode_0(features_data)
    else:
        features_data = handle_missing_mode_1(features_data, label_data)

    features_train, labels_train, features_test, labels_test = stratified_split(features_data, label_data)

    model = NaiveBayes(laplace=1.0)
    model.fit(features_train, labels_train)

    train_acc = accuracy(labels_train, model.predict(features_train))
    test_acc = accuracy(labels_test, model.predict(features_test))

    cv_scores = cross_validation(features_train, labels_train)

    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = (sum((x - mean_cv) ** 2 for x in cv_scores) / len(cv_scores)) ** 0.5

    print("1. Train Set Accuracy:")
    print(f"   Accuracy: {train_acc * 100:.2f}%\n")

    print("10-Fold Cross-Validation Results:\n")
    for i, acc in enumerate(cv_scores, 1):
        print(f"   Accuracy Fold {i}: {acc * 100:.2f}%")

    print(f"\n   Average Accuracy: {mean_cv * 100:.2f}%")
    print(f"   Standard Deviation: {std_cv * 100:.2f}%\n")

    print("2. Test Set Accuracy:")
    print(f"   Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()