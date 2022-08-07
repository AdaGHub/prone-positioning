"""Script to perform k-NN time series classification. Includes hyperparameter optimization.

Make sure to run processing.py once before calling this."""
import os
import sys
import pickle as pkl
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tslearn.utils import to_time_series_dataset
from scripts.utils import RANDOM_STATE, load_data_raw
from scripts.utils_prediction import set_seed, CustomScaler, N_SPLITS_OUTER, N_SPLITS_INNER, LONGI
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from statistics import mean, stdev
from utils import SCORES_PATH

SCALER = CustomScaler(scaler=StandardScaler())


def main(outcome='X81_Primary_out'):
    # Set seed for reproducibility
    set_seed(RANDOM_STATE)

    # Print option
    np.set_printoptions(threshold=np.inf)

    # Display settings
    pd.options.display.max_columns = None
    pd.options.display.max_rows = 50

    # Load non imputed data, select longi variables of interest
    inputs, targets = load_data_raw()
    inputs = inputs[LONGI]

    # Drop rows with NaN values
    nan_rows = [index for index, row in inputs.iterrows() if row.isnull().any()]  # Indices of rows with NaN
    inputs = inputs.drop(nan_rows)
    targets = targets.drop(nan_rows)

    # Store time series into a list
    list_ts = []
    for idx in inputs.index.get_level_values('id').unique():
        list_ts.append(inputs.loc[(idx,), :].to_numpy())

    # Convert to tslearn time series dataset format
    x = to_time_series_dataset(list_ts)

    # Select labels
    y = targets[outcome].groupby('id').head(1).to_numpy().astype(int)

    # Our pipeline for the inner GridSearchCV consists of two phases.
    # First, data will be normalized using standard normalization.
    # Afterwards, it is fed to a KNN classifier.
    # For the KNN classifier, we tune the n_neighbors and weights hyperparameters.
    n_splits = N_SPLITS_INNER
    pipeline = GridSearchCV(
        Pipeline([
            ('normalize', SCALER),
            ('knn', KNeighborsTimeSeriesClassifier())
        ]),
        {'knn__n_neighbors': [5, 10, 25, 100], 'knn__weights': ['uniform', 'distance']},
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    )

    # Outer loop split
    kf_outer = StratifiedKFold(n_splits=N_SPLITS_OUTER, random_state=RANDOM_STATE, shuffle=True)
    fold_number = 0
    list_aucs = [[], [], [], []]
    for train_idx, test_idx in kf_outer.split(x, y):
        fold_number += 1
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        # Fit our pipeline
        print(end='Performing hyper-parameter tuning of KNN classifier... ')
        pipeline.fit(x_train, y_train)
        results = pipeline.cv_results_

        # Print each possible configuration parameter and the out-of-fold accuracies
        print('Done!')
        print()
        print('Got the following accuracies on the test set for each fold:')
        header_str = '|'
        columns = ['n_neighbors', 'weights']
        columns += ['score_fold_{}'.format(i + 1) for i in range(n_splits)]
        for col in columns:
            header_str += '{:^12}|'.format(col)
        print(header_str)
        print('-' * (len(columns) * 13))
        for i in range(len(results['params'])):
            s = '|'
            s += '{:>12}|'.format(results['params'][i]['knn__n_neighbors'])
            s += '{:>12}|'.format(results['params'][i]['knn__weights'])
            for k in range(n_splits):
                score = results['split{}_test_score'.format(k)][i]
                score = np.around(score, 5)
                s += '{:>12}|'.format(score)
            print(s.strip())
        best_comb = np.argmax(results['mean_test_score'])
        best_params = results['params'][best_comb]
        print()
        print('Best parameter combination:')
        print('weights={}, n_neighbors={}'.format(best_params['knn__weights'],
                                                  best_params['knn__n_neighbors']))

        # Scale training and testing data
        scaler = SCALER
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Fit TS KNN classifier
        knn = KNeighborsTimeSeriesClassifier(best_params['knn__n_neighbors'], best_params['knn__weights'])
        knn.fit(x_train, y_train)

        # Predict probabilities for different test TS lengths (early classification)
        x_test_truncated = x_test.copy()
        ts_length = 4
        for i in range(4):
            if i != 0:
                for j in range(x_test_truncated.shape[0]):
                    x_test_truncated[j, -i, :] = np.nan
            y_scores = knn.predict_proba(x_test_truncated.copy())
            fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
            roc_auc = auc(fpr, tpr)
            ts_length -= 1
            list_aucs[ts_length].append(roc_auc)

    for i in range(len(list_aucs)):
        list_aucs[i] = (mean(list_aucs[i]), stdev(list_aucs[i]))

    # Save list of aucs
    with open(os.path.join(SCORES_PATH, 'list_aucs_knn.pkl'), 'wb') as f:
        pkl.dump(list_aucs, f)


if __name__ == '__main__':
    main()
