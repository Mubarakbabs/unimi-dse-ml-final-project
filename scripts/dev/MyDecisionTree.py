import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Tuple
from enum import Enum

class Node:

    """
    Class representing a node in the decision tree.

    Attributes:
    -----------
    feature : Optional[int]
        The feature index used for splitting at this node.
    threshold : Optional[float]
        The threshold value used for splitting at this node.
    left : Optional[Node]
        The left child node.
    right : Optional[Node]
        The right child node.
    value : Optional[int]
        The class label if this node is a leaf.
    """
    left: 'Node'
    right: 'Node'
    
    #req: constructor that initiliazes the node
    def __init__(self, feature:str=None, threshold=None, left=None, right=None,*,value=None):
        #the asterisk forces us to pass value by name to ensure leaf nodes are obvious
        self.feature = feature
        self.threshold = threshold
        #req: left and right children. Should both be nodes        
        self.left = left 
        self.right = right
        self.value = value


    #req: flag to check if node is a leaf
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:

    """
    Class representing a Decision Tree classifier.

    Parameters:
    -----------
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    max_depth : int, optional (default=100)
        The maximum depth of the tree.
    n_features : Optional[int], optional (default=None)
        The number of features to consider when looking for the best split. If None, then all features are considered.
    split_using : string, optional (default=entropy)
        The splitting criterion for defining impurity/information gain. Options: entropy, gini
    """
    #req: a constructor initializing the tree predictor 
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, split_using="entropy"):
        if split_using not in ('entropy', 'gini', 'train_error'):
            raise ValueError(f"split_using argument must be one of ('entropy', 'gini', 'train_error)")

        #req: stopping criteria: we may split these to create different trees instead of choosing the best
        #initialize with different stopping criteria
        self.min_samples_split= min_samples_split
        self.max_depth=max_depth
        #nfeatures is a way to add some randomness to the tree
        self.n_features=n_features
        self.root=None
        self.split_using=split_using

    def __str__(self):
        #tell python what to print if the decision tree is called.
        #this should include the split criterion, the stop criterion applied, the depth and the accuracy
        return f"Decision Tree with depth {self.depth}, split using {self.split_using} and stopped upon {self.stop_criterion}"


    #req: a procedure for training the tree on a given data set (this includes fit and grow tree).
    def fit(self, X,y):
        self.leaves = []
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        

    #stop tree when either purity is achieved or the stopping criterion is reached
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            self.depth = depth
            self.n_labels = n_labels
            self.n_samples = n_samples
            #print(f"depth: {depth}, n_labels: {n_labels}, n_samples: {n_samples}")
            self.stop_criterion = f"Depth: {depth}" if depth>=self.max_depth else f"reaching a pure label" if n_labels==1 else f"N_samples less than: {n_samples}"
            leaf_value = self._most_common_label(y) 
            self.leaves.append(leaf_value)
            return Node(value=leaf_value)  
            
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        #find the best split
        #TAKES the result from best split function
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if best_gain == 0:
            leaf_value = self._most_common_label(y)
            self.leaves.append(leaf_value)
            return Node(value=leaf_value)  
       
        #create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1 #initialize with -1 to ensure any value of gain we pick will be bigger
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate the information gain
                gain = self._gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        #debugging
        #print(f"Split index: {split_idx}, split thresh: {split_threshold}, best gain: {best_gain}")
        return split_idx, split_threshold, best_gain
    
    #needed for best split
    #req: information gain calculated based on entropy or gini criterion depending on which was supplied
    def _gain(self, y, X_column, threshold):
        if self.split_using == 'entropy':
            # parent entropy
            parent_entropy = self._entropy(y)

            #create children
            left_idxs, right_idxs = self._split(X_column, threshold)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0
            
            #calculate the weighted average of children entropy
            n = len(y)
            n_l, n_r = len(left_idxs), len(right_idxs)
            e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

            information_gain = parent_entropy - child_entropy

            return information_gain
        
        elif self.split_using == 'gini':
            parent_gini = self._gini(y)

            
            left_idxs, right_idxs = self._split(X_column, threshold)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0

            n = len(y)
            n_l, n_r = len(left_idxs), len(right_idxs)
            gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
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
            error_l, error_r = self._train_error(y[left_idxs]), self._train_error(y[right_idxs])
            child_error = (n_l / n) * error_l + (n_r / n) * error_r

            information_gain = parent_error - child_error

            return information_gain



    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_idxs

    def _train_error(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)

        return np.min(ps)


    def _entropy(self, y):
        #entropy = #x/n
        hist = np.bincount(y)
        ps = hist / len(y)
        #how do we ensure the log is in base 2
        return -np.sum([p * np.log(p) for p in ps if p> 0])
    
    def _gini(self, y):
        #2*(#x/n) * (1-#x/n)
        hist = np.bincount(y)
        ps = hist / len(y)
        #general formula
        #gini = 1 - np.sum([p**2 for p in ps])
        #binary case
        gini = 2*np.prod([p for p in ps])
        return gini
        #wrong function return 1 - np.sum([2*p*(1-p) for p in ps])

    def _most_common_label(self,y):
        counter = Counter(y) #check what a counter data structure is

        value = counter.most_common(1)[0][0]
        return value
    
    #req: a procedure for to evaluate the predictor on a test set
    def predict(self, X):
        y_pred = np.array([self._traverse_tree(x, self.root) for x in X])
        y_pred = y_pred.reshape((len(y_pred)),)
        return y_pred
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
    

