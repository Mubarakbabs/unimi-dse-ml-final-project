#This is my submission for my Machine Learning Project

##Project Question
Project 2: Tree predictors for binary classification
Download the Mushroom dataset. The main task of this project is the implementation from scratch of tree predictors for binary classification to determine whether mushrooms are poisonous. The tree predictors must use single-feature binary tests as the decision criteria at any internal node (as seen in the lectures). More precisely, consider thresholds on a single feature in the case of a numerical/ordinal feature or membership tests in the case of a categorical feature. We suggest the following guidelines to aid your work on this project.

First, implement a basic class/structure for the nodes of the tree predictors. It should possess the following attributes/procedures:

A constructor that initializes the node (empty or with given attributes)
Left and right children, which should be nodes themselves
A flag to check if the node is a leaf
The decision criterion/test used by the current node: it should be a function taking a data point (e.g., a numpy vector) as input and returning a Boolean value as output.
Second, implement a class/structure for the (binary) tree predictor. It should contain the following attributes/procedures:

A constructor initializing the tree predictor (possibly passing the information on which decision criteria/tests can be adopted on each single feature)
A splitting criterion for selecting both the leaf to expand and the decision criterion to adopt in the new internal node (e.g., Gini index, scaled entropy, etc.)
A stopping criterion to halt the construction of the decision tree. For example, these could be maximum tree depth, maximum number of nodes/leaves, a constraint on the weight of leaves (e.g., number of samples reaching the leaf, entropy/impurity of the leaf), minimum entropy/impurity decrease
A procedure for training the tree predictor on a given training set
A procedure to evaluate the tree predictor on a given validation/test set
Feel free to add extra attributes/procedures if deemed necessary for the task.

Train the tree predictors adopting at least 3 reasonable criteria for the expansion of the leaves, and at least 2 reasonable stopping criteria. Compute the training error of each tree predictor according to the 0-1 loss.

Perform hyperparameter tuning according to the splitting criteria and the stopping criteria adopted (e.g., tune the threshold on the maximum size of the tree) for at least one of the tree predictors. Keep in mind that the relevant hyperparameter tuning is an important part of the project and must be performed using a sound procedure.

Write a report where you discuss your findings, with particular care about the adopted methodology, and a thorough discussion about the models’ performance (with comments on the eventual presence of over/underfitting). In the case of overfitting, some ways to tackle it would be by pruning the tree predictors (see the references below) or by an appropriate stopping criterion.

We suggest the following resources, in addition to the lecture notes of the course, as further reading for the interested student:

Chapter 18 of the book “Understanding Machine Learning: From Theory to Algorithms” by Shalev-Shwartz and Ben-David
Section 9.3.3 of the book “Foundations of Machine Learning” by Mohri, Rostamizadeh, and Talwalkar
The Decision Forests course by Google for Developers: https://developers.google.com/machine-learning/decision-forests
Optional: Implement random forest by reusing the already implemented tree predictor class/structure.