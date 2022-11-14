import numpy as np
import pandas as pd
import joblib
from sklearn import neighbors
from sklearn.model_selection import RandomizedSearchCV


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


# Return the best k-NN model using random search
def random_search(train_features, train_labels):
    # Number of neighbors to use
    n_neighbors = [3, 5, 7, 9, 11, 13]

    # Weight function used for prediction
    weights = ['uniform', 'distance']

    # Algorithm to use
    algorithm = ['ball_tree', 'kd_tree']

    # Leaf size used for ball tree or kd tree
    leaf_size = [20, 30, 40]

    # Power parameter used for Minkowski metric
    p = [1, 2]

    # Parameters used for the random search model
    knn_param = {'n_neighbors': n_neighbors,
                 'weights': weights,
                 'algorithm': algorithm,
                 'leaf_size': leaf_size,
                 'p': p}

    # Perform five-fold cross validation on the k-NN classifier using the random search model
    knn = neighbors.KNeighborsClassifier()
    knn_random = RandomizedSearchCV(estimator=knn, param_distributions=knn_param,
                                    n_iter=100, cv=5, random_state=42, n_jobs=3)

    # Fit the k-NN model
    knn_random.fit(train_features, train_labels)

    # Display the parameters of the best model from the random search
    print(knn_random.best_params_)

    # Sets the k-NN model as the best model from the random search
    return knn_random.best_estimator_


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

    # Create the k-NN classifier
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=11, weights='distance',
                                               algorithm='ball_tree', leaf_size=20, p=1)

    # Train the k-NN model
    knn_model.n_jobs = 3
    knn_model.fit(X_train, y_train)

    # Display the performance metrics of the k-NN model
    display_metrics(knn_model, X_test, y_test)

    # Save the k-NN model
    knn_file = 'k-NN.pkl'
    joblib.dump(knn_model, knn_file)
