#this code works but takes too long to run
import pandas as pd
from collections import Counter
from typing import Optional
import numpy as np

class Node:
    def __init__(self, feature: str = None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, split_using="entropy"):
        if split_using not in ('entropy', 'gini', 'train_error'):
            raise ValueError(f"split_using argument must be one of ('entropy', 'gini', 'train_error')")
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.split_using = split_using

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.leaves = []
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(y.unique())

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            self.leaves.append(leaf_value)
            return Node(value=leaf_value)

        feat_idxs = X.sample(n=self.n_features, axis=1, replace=False).columns
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if best_gain == 0:
            leaf_value = self._most_common_label(y)
            self.leaves.append(leaf_value)
            return Node(value=leaf_value)

        print(f"column: {best_feature}, thr: {best_thresh}, type: {self.datatype}")
        left_idxs, right_idxs = self._split(X[best_feature], best_thresh)
        left = self._grow_tree(X.loc[left_idxs], y.loc[left_idxs], depth + 1)
        right = self._grow_tree(X.loc[right_idxs], y.loc[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X: pd.DataFrame, y: pd.Series, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[feat_idx]
            #we don't need this handling anymore since pandas can do unique for cols with null
            thresholds = X_column.astype(str).unique() if X_column.dtype == 'object' else X_column.unique()

            for thr in thresholds:
                gain = self._gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    self.datatype = X_column.dtype
        return split_idx, split_threshold, best_gain

    def _gain(self, y: pd.Series, X_column: pd.Series, threshold):
        if self.split_using == 'entropy':
            parent_entropy = self._entropy(y)
            left_idxs, right_idxs = self._split(X_column, threshold)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0

            n = len(y)
            n_l, n_r = len(left_idxs), len(right_idxs)
            e_l, e_r = self._entropy(y.loc[left_idxs]), self._entropy(y.loc[right_idxs])
            child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

            information_gain = parent_entropy - child_entropy
            return information_gain

        elif self.split_using == 'gini':
            parent_gini = self._gini(y)
            left_idxs, right_idxs = self._split(X_column, threshold)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0

            n = len(y)
            n_l, n_r = len(left_idxs), len(right_idxs)
            gini_l, gini_r = self._gini(y.iloc[left_idxs]), self._gini(y.iloc[right_idxs])
            child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

            information_gain = parent_gini - child_gini
            return information_gain

        elif self.split_using == 'train_error':
            parent_error = self._train_error(y)
            left_idxs, right_idxs = self._split(X_column, threshold)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0

            n = len(y)
            n_l, n_r = len(left_idxs), len(right_idxs)
            error_l, error_r = self._train_error(y.iloc[left_idxs]), self._train_error(y.iloc[right_idxs])
            child_error = (n_l / n) * error_l + (n_r / n) * error_r

            information_gain = parent_error - child_error
            return information_gain

    def _split(self, X_column: pd.Series, split_thresh):
        if X_column.dtype == "float64" or X_column.dtype == "int64":
            left_idxs = X_column.index[X_column <= split_thresh]
            right_idxs = X_column.index[X_column > split_thresh]

        else:
            split_thresh = str(split_thresh)
            X_column = X_column.astype(str)
            #print(f"str column")
            left_idxs = X_column.index[X_column == split_thresh]
            right_idxs = X_column.index[X_column != split_thresh]

        return left_idxs, right_idxs

    def _train_error(self, y: pd.Series):
        counter = Counter(y)
        most_common_label, count = counter.most_common(1)[0]
        return 1 - (count / len(y))

    def _entropy(self, y: pd.Series):
        counter = Counter(y)
        ps = pd.Series([count / len(y) for count in counter.values()])
        return -sum(ps * ps.apply(lambda p: p * np.log2(p) if p > 0 else 0))

    def _gini(self, y: pd.Series):
        counter = Counter(y)
        ps = pd.Series([count / len(y) for count in counter.values()])
        return 1 - sum(ps ** 2)

    def _most_common_label(self, y: pd.Series):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X: pd.DataFrame):
        y_pred = pd.Series([self._traverse_tree(x, self.root) for _, x in X.iterrows()])
        return y_pred

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] == node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
