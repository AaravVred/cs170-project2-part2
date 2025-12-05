import math
import time


def load_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            row = [float(x) for x in parts]
            data.append(row)
    return data


def split_labels_and_features(data):
    labels = []
    feats = []
    for row in data:
        labels.append(row[0])
        feats.append(row[1:])
    return labels, feats


def pick_feature_subset(features, subset):
    new_feats = []
    for row in features:
        new_row = []
        for idx in subset:
            new_row.append(row[idx - 1])  
        new_feats.append(new_row)
    return new_feats


def normalize_features(features):
    if len(features) == 0:
        return features

    n = len(features)
    d = len(features[0])

    means = [0.0] * d
    stds = [0.0] * d

    for j in range(d):
        s = 0.0
        for i in range(n):
            s += features[i][j]
        means[j] = s / n

    for j in range(d):
        s = 0.0
        for i in range(n):
            diff = features[i][j] - means[j]
            s += diff * diff
        std = math.sqrt(s / n)
        if std == 0.0:      
            std = 1.0
        stds[j] = std

    norm = []
    for i in range(n):
        row = []
        for j in range(d):
            value = (features[i][j] - means[j]) / stds[j]
            row.append(value)
        norm.append(row)

    return norm



class Classifier:
    def __init__(self):
        self.train_data = []
        self.train_labels = []

    def train(self, data, labels):
        self.train_data = data
        self.train_labels = labels

    def _distance(self, x, y):
        total = 0.0
        for a, b in zip(x, y):
            diff = a - b
            total += diff * diff
        return math.sqrt(total)

    def test(self, x):
        best_dist = None
        best_label = None

        for vec, lab in zip(self.train_data, self.train_labels):
            d = self._distance(x, vec)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_label = lab

        return best_label


class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def evaluate(self, data, feature_subset):
        labels, feats = split_labels_and_features(data)

        feats = pick_feature_subset(feats, feature_subset)

        feats = normalize_features(feats)

        n = len(feats)
        correct = 0

        start = time.time()

        for i in range(n):
            train_data = []
            train_labels = []
            for j in range(n):
                if j == i:
                    continue
                train_data.append(feats[j])
                train_labels.append(labels[j])

            test_instance = feats[i]
            true_label = labels[i]

            self.classifier.train(train_data, train_labels)
            pred = self.classifier.test(test_instance)

            if pred == true_label:
                correct += 1

        elapsed = time.time() - start
        accuracy = correct / n

        print("Feature subset:", feature_subset)
        print(f"Correct {correct} / {n}  ->  accuracy = {accuracy:.3f}")
        print(f"Time: {elapsed:.3f} seconds\n")

        return accuracy



def main():
    small_file = "small-test-dataset-2-2.txt"
    large_file = "large-test-dataset-2.txt"

    small_data = load_data(small_file)
    clf1 = Classifier()
    val1 = Validator(clf1)
    print("Testing small dataset with features {3,5,7}:")
    acc_small = val1.evaluate(small_data, [3, 5, 7])
    

    large_data = load_data(large_file)
    clf2 = Classifier()
    val2 = Validator(clf2)
    print("Testing large dataset with features {1,15,27}:")
    acc_large = val2.evaluate(large_data, [1, 15, 27])
    

    


if __name__ == "__main__":
    main()
