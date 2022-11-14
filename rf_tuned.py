import numpy as np
import pandas as pd
import joblib
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Displays the performance metrics of the model
def display_metrics(model, test_features, test_labels):
    tn = fp = fn = tp = 0
    test_len = len(test_labels)
    test_pred = model.predict(test_features)

    # Computes the accuracy, precision, recall, and f1 score
    for i in range(test_len):
        if test_pred[i] == 0 and test_labels[i] == 0:
            tn += 1
        elif test_pred[i] == 0 and test_labels[i] == 1:
            fn += 1
        elif test_pred[i] == 1 and test_labels[i] == 0:
            fp += 1
        elif test_pred[i] == 1 and test_labels[i] == 1:
            tp += 1
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (recall * precision) / (recall + precision)

    # Displays the performance metrics
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 Score: {}'.format(f1_score))


# Return the best random forest model using random search
def random_search(train_features, train_labels):
    # Number of trees in the random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

    # Maximum number of levels in a tree
    max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]

    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 8]

    # Minimum number of samples required to split a leaf
    min_samples_leaf = [1, 2, 4]

    # Number of features considered for each split
    max_features = ['sqrt', 'log2']

    # Method for selecting samples
    bootstrap = [True, False]

    # Parameters used for the random search model
    rf_param = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'bootstrap': bootstrap}

    # Perform five-fold cross validation on the random forest classifier using the random search model
    rf = ensemble.RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param,
                                   n_iter=100, cv=5, random_state=42, n_jobs=3)

    # Fit the random forest model
    rf_random.fit(train_features, train_labels)

    # Display the parameters of the best model from the random search
    print(rf_random.best_params_)

    # Sets the random forest model as the best model from the random search
    return rf_random.best_estimator_


# Return the best random forest model using grid search
def grid_search(train_features, train_labels):
    # Parameters used for the grid search model
    rf_param = {'n_estimators': [450, 500, 550],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2],
                'min_samples_leaf': [2],
                'max_features': ['log2'],
                'bootstrap': [False]}

    # Perform five-fold cross validation on the random forest classifier using the grid search model
    rf = ensemble.RandomForestClassifier()
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5, n_jobs=3)

    # Fit the random forest model
    rf_grid.fit(train_features, train_labels)

    # Display the parameters of the best model from the grid search
    print(rf_grid.best_params_)

    # Sets the random forest model as the best model from the grid search
    return rf_grid.best_estimator_


if __name__ == '__main__':
    # Loads the Chicago crime data set
    crimes_filepath = 'D:/Users/Eric/Google Drive/Colab Notebooks/total_df.csv'
    crimes_df = pd.read_csv(crimes_filepath)

    # Drops the index column
    crimes_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Sort the crimes by year
    crimes_df.sort_values(by=['Year'], inplace=True)

    # Uses the 'Arrest' feature as the target output for training
    crime_features = np.array(crimes_df.drop(['Arrest'], axis=1))
    crime_labels = np.array(crimes_df['Arrest'])

    # Sets the percentage of data set as the test set
    test_split = 0.25

    # Generate a random uniform mask to separate the training set and the test set
    test_len = int(test_split * len(crime_labels))
    test_mask = np.zeros(len(crime_labels)).astype(bool)
    for _ in range(test_len):
        loop = True
        while loop:
            # Get the random index value to set as True
            index = np.random.randint(0, len(crime_labels))

            # Break out of the while loop if the index for the mask is False
            # Used to prevent repeats of index values
            if not test_mask[index]:
                test_mask[index] = True
                loop = False

    # Gets the test set using the test mask
    X_test = crime_features[test_mask]
    y_test = crime_labels[test_mask]

    # Gets the training set using the inverse of the test mask
    X_train = crime_features[~test_mask]
    y_train = crime_labels[~test_mask]

    # Create the random forest classifier
    rf_model = ensemble.RandomForestClassifier(n_estimators=550, max_depth=20, min_samples_leaf=2,
                                               min_samples_split=2, max_features='log2', bootstrap=True)

    # Train the random forest model
    rf_model.n_jobs = 3
    rf_model.fit(X_train, y_train)

    # Display the performance metrics of the random forest model
    display_metrics(rf_model, X_test, y_test)

    # Save the random forest model
    rf_file = 'random_forest.pkl'
    joblib.dump(rf_model, rf_file)
