import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self, laplace=1.0):
        """
        laplace = λ (параметър за заглаждане на Лаплас)
        """
        self.laplace = laplace
        self.class_priors = {}        # log P(class)
        self.likelihoods = {}         # log P(value | class)
        self.classes = set()
        self.num_features = 0

    def fit(self, features_data, labels_data):
        """
        Обучение на модела
        features_data – списък от списъци 
        labels_data – списък с класове
        """
        self.classes = set(labels_data)
        self.num_features = len(features_data[0])
        total_samples = len(labels_data)

        # 1. Prior probabilities: P(class)
        for cls in self.classes:
            count_cls = labels_data.count(cls)
            self.class_priors[cls] = math.log(count_cls / total_samples)

        # 2. Likelihoods: P(feature_value | class)
        for cls in self.classes:
            # всички редове от този клас
            features_data_by_cls = [features_data[i] for i in range(len(features_data)) if labels_data[i] == cls]
            total_cls = len(features_data_by_cls)

            for j in range(self.num_features):
                # всички възможни стойности за атрибута j
                values = set(row[j] for row in features_data)

                k = len(values)  # брой възможни стойности

                for v in values:
                    count = sum(1 for row in features_data_by_cls if row[j] == v)

                    prob = (count + self.laplace) / (total_cls + self.laplace * k)
                    self.likelihoods[(cls, j, v)] = math.log(prob)

    def predict(self, features_data):
        """
        Предсказване на класове
        """
        predictions = []

        for row in features_data:
            scores = {}

            for cls in self.classes:
                score = self.class_priors[cls]

                for j, value in enumerate(row):
                    score += self.likelihoods.get((cls, j, value), 0)

                scores[cls] = score

            # класът с най-голяма лог-вероятност
            predictions.append(max(scores, key=scores.get))

        return predictions
