import numpy as np
from collections import Counter
from typing import Optional

class Node:
    def __init__(self, feature: Optional[int] = None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None, split_using="entropy"):
        if split_using not in ('entropy', 'gini', 'train_error'):
            raise ValueError(f"split_using argument must be one of ('entropy', 'gini', 'train_error')")
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.split_using = split_using

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.leaves = []
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_labels = len(y), len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            self.leaves.append(leaf_value)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if best_gain == 0:
            leaf_value = self._most_common_label(y)
            self.leaves.append(leaf_value)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold, best_gain

    def _gain(self, y, X_column, threshold):
        parent_metric = self._entropy(y) if self.split_using == 'entropy' else self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        metric_l = self._entropy(y[left_idxs]) if self.split_using == 'entropy' else self._gini(y[left_idxs])
        metric_r = self._entropy(y[right_idxs]) if self.split_using == 'entropy' else self._gini(y[right_idxs])
        child_metric = (n_l / n) * metric_l + (n_r / n) * metric_r

        information_gain = parent_metric - child_metric
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        counter = Counter(y)
        ps = np.array([count / len(y) for count in counter.values()])
        return -np.sum(ps * np.log2(ps + 1e-9))  # Added epsilon to avoid log(0)

    def _gini(self, y):
        counter = Counter(y)
        ps = np.array([count / len(y) for count in counter.values()])
        return 1 - np.sum(ps ** 2)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
