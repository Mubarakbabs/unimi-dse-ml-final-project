#adding other utility functions that aren't necessarily tied to the decision tree
import numpy as np
import pandas as pd
from scripts.final.DecisionTree import DecisionTree

def train_test_split(X,y,test_size=0.2, random_state=None):
    X = np.array(X)
    y = np.array(y)
    np.random.seed(random_state)
    test_indices = np.random.choice(X.shape[0], size=round(test_size*X.shape[0]), replace=False)
    X_test = X[test_indices]
    y_test = y[test_indices]
    y_test = y_test.reshape((len(y_test)),).astype(int)

    all_indices = np.arange(X.shape[0])
    train_indices = np.setdiff1d(all_indices, test_indices)
    X_train = X[train_indices]
    y_train = y[train_indices]
    y_train = y_train.reshape((len(y_train)),).astype(int)

    return X_train, X_test, y_train, y_test

def encode_labels(df):
    """modifies the dataframe in place and returns the encoding dictionary"""
    encoding_dict = {}
    for col in df.columns:
        if col in df.select_dtypes(exclude='number').columns:
            unique_values = df[col].unique()
            unique_values = pd.Series(unique_values).sort_values().values
            value_to_int = {value: idx for idx, value in enumerate(unique_values)}
            encoding_dict[col] = value_to_int
            df.loc[:,col] = df[col].map(value_to_int)
            df[col] =pd.to_numeric(df.loc[:,col], errors='coerce').values
    return encoding_dict

def accuracy(y_actual, y_predicted):
    if len(y_actual) != len(y_predicted):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    correct_predictions = np.sum(y_actual == y_predicted)
    total_predictions = len(y_predicted)
    accuracy = correct_predictions/total_predictions

    return f"Accuracy: {accuracy}"

def recall(y_actual, y_predicted):
    if len(y_actual) != len(y_predicted):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    actual_positives = y_actual==1
    actual_negatives = y_actual==0
    predicted_positives = y_predicted==1
    predicted_negatives = y_predicted==0
    true_positives = np.sum(actual_positives & predicted_positives)
    false_negatives = np.sum(actual_positives & predicted_negatives)
    recall = true_positives/(true_positives+false_negatives)
    return f"Recall: {recall}"

def precision(y_actual, y_predicted):
    if len(y_actual) != len(y_predicted):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    actual_positives = y_actual==1
    actual_negatives = y_actual==0
    predicted_positives = y_predicted==1
    predicted_negatives = y_predicted==0
    true_positives = np.sum(actual_positives & predicted_positives)
    false_positives = np.sum(actual_negatives & predicted_positives)
    precision = true_positives/(true_positives+false_positives)
    return f"Precision: {precision}"

    #compute the training error according to the 0-1 loss. 

def zero_one_loss(y_train, y_pred):
    incorrect_predictions = np.sum(y_train != y_pred)
    training_error = incorrect_predictions / len(y_train)
    return training_error



    #hyperparameter tuning to maximize the threshold on at least one of them
def tune(X, y, tune_on, split_using, start, stop):
    best_error = 1
    X_train, X_validate, y_train, y_validate = train_test_split(X,y, test_size=0.2)
    for i in range(start, stop):
        params = {tune_on: i, 'split_using': split_using }
        tree = DecisionTree(**params)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_validate)
        validation_error = zero_one_loss(y_validate, y_pred)
        print(f"tree depth: {tree.max_depth}, validation error: {validation_error}")
        if validation_error < best_error:
            improvement = validation_error - best_error
            #for some reason it prints the next tree after the
            best_error = validation_error
            best_tree = tree
            best_depth = i
            # if improvement < 0.001 and best_error != 1:
            #     break
    print(f"best depth: {best_depth} validation error: {best_error}")
    return best_tree