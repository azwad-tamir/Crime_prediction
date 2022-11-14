import numpy as np
import pandas as pd
import joblib
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV


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


# Return the best adaboost model using grid search
def grid_search(train_features, train_labels):
    # Parameters used for the grid search model
    ada_param = {'n_estimators': [400, 500, 600],
                 'learning_rate': [1, 1.2, 1.4, 1.6, 1.8, 2]}

    # Perform five-fold cross validation on the adaboost classifier using the grid search model
    ada = ensemble.AdaBoostClassifier()
    ada_grid = GridSearchCV(estimator=ada, param_grid=ada_param, cv=5, n_jobs=3)

    # Fit the adaboost model
    ada_grid.fit(train_features, train_labels)

    # Display the parameters of the best model from the grid search
    print(ada_grid.best_params_)

    # Sets the adaboost model as the best model from the grid search
    return ada_grid.best_estimator_


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

    # Create the adaboost classifier
    ada_model = ensemble.AdaBoostClassifier(learning_rate=1.8, n_estimators=600)

    # Train the adaboost model
    ada_model.n_jobs = 3
    ada_model.fit(X_train, y_train)

    # Display the performance metrics of the adaboost model
    display_metrics(ada_model, X_test, y_test)

    # Save the adaboost model
    ada_file = 'adaboost.pkl'
    joblib.dump(ada_model, ada_file)
