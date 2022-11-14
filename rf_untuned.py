import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import cross_validate

if __name__ == '__main__':
    # Loads the Chicago crime data set
    crimes_filepath = 'D:/Users/Eric/Google Drive/Colab Notebooks/total_df.csv'
    crimes_df = pd.read_csv(crimes_filepath)

    # Drops the index column
    crimes_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Uses the 'Arrest' feature as the target output for training
    X = crimes_df.drop(['Arrest'], axis=1)
    y = crimes_df['Arrest']

    # Trains the random forest classifier using five-fold cross validation
    k_fold = 5
    clf = ensemble.RandomForestClassifier()
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=clf, X=X, y=y, scoring=scoring, cv=k_fold, n_jobs=2, verbose=0)

    # Display the scores for each fold
    for i in range(k_fold):
        print('Accuracy (Fold = {}): {}'.format(i + 1, results['test_accuracy'][i]))
        print('Precision (Fold = {}): {}'.format(i + 1, results['test_precision'][i]))
        print('Recall (Fold = {}): {}'.format(i + 1, results['test_recall'][i]))
        print('F1 Score (Fold = {}): {}\n'.format(i + 1, results['test_f1'][i]))

    # Display the average scores
    print('Accuracy (Average): {}'.format(np.mean(results['test_accuracy'])))
    print('Precision (Average): {}'.format(np.mean(results['test_precision'])))
    print('Recall (Average): {}'.format(np.mean(results['test_recall'])))
    print('F1 Score (Average): {}'.format(np.mean(results['test_f1'])))
