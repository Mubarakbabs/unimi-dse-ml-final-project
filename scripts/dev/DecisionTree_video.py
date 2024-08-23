#first Decision Tree created while watching the video
import numpy as np
from collections import Counter

class Node:

    # left: Node
    # right: Node
    
    #req: constructor that initiliazes the node
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
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
    
    #req: decision criterion used by the current node. This sounds like the traverse tree function. Can we shift it into the Node isntead?
    
class DecisionTree:
    #req: a constructor initializing the tree predictor 
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        #req: stopping criteria: we may split these to create different trees instead of choosing the best
        #initialize with different stopping criteria
        self.min_samples_split= min_samples_split
        self.max_depth=max_depth
        #nfeatures is a way to add some randomness to the tree
        self.n_features=n_features
        self.root=None

    #req: a procedure for training the tree on a given data set (this includes fit and grow tree). We should generalize them to be able to use the different splitting and stopping criteria
    #this may involve including an argument of splitting/stopping criterion
    def fit(self, X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #check stopping criteria
        if(depth >= self.max_depth or n_labels==1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        #find the best split
        #TAKES the result from best split function
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
       
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
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold
    
    #needed for best split
    #req: splttign criteria. We should define a different version that uses gini index instead of entropy
    def _information_gain(self, y, X_column, threshold):
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



    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_idxs

    def _entropy(self, y):
        #entropy = #x/n
        hist = np.bincount(y)
        ps = hist / len(y)
        #how do we ensure the log is in base 2
        return -np.sum([p * np.log(p) for p in ps if p> 0])

    def _most_common_label(self,y):
        counter = Counter(y) #check what a counter data structure is
        value = counter.most_common(1)[0][0]
        return value
    #req: a procedure for to evaluate the predictor on a test set
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
    
    #compute the training error according to the 0-1 loss. 
    #hyperparameter tuning to maximize the threshold on at least one of them

