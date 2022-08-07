"""Script to prepare data for simple RNN model.
Uses preprocessed data from processing.py
Use the load_data_raw function from utils.py to load everything.
"""
import os
import pickle as pkl
from statistics import mean, stdev
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from utils import load_data_raw, RANDOM_STATE, SCORES_PATH
from utils_prediction import get_rnn, scale_process_inputs, set_seed, N_SPLITS_INNER, N_SPLITS_OUTER, \
    VERBOSE, PARAM_GRID, padding, LONGI, STATIC_BINARY, STATIC_CONTINUOUS
import pandas as pd
from keras.callbacks import EarlyStopping
from kerashypetune import KerasGridSearch
import sys

BINARIZATION = False


def main(outcome='X81_Primary_out'):
    # Hide TensorFlow warnings
    tf.get_logger().setLevel('ERROR')

    # Set seed for reproducibility
    set_seed(RANDOM_STATE)

    # Display settings
    pd.options.display.max_columns = None
    pd.options.display.max_rows = 50
    # np.set_printoptions(threshold=np.inf)

    param_grid = PARAM_GRID

    # Load non imputed data
    x, targets = load_data_raw()

    # Select longitudinal and static variables to be included (most correlated with outcomes)
    x = x[LONGI]
    c1 = targets[STATIC_BINARY].groupby('id').head(1)
    c2 = targets[STATIC_CONTINUOUS].groupby('id').head(1)

    # Select end points
    y = targets[outcome]

    # Remove patients with Missing Values (MD) in c1 or c2 (no meaning, unlike NaN values in time series data)
    nan_rows_c1 = [index for index, row in c1.iterrows() if row.isnull().any()]
    nan_rows_c2 = [index for index, row in c2.iterrows() if row.isnull().any()]
    for nan_row in nan_rows_c1 + nan_rows_c2:
        x = x.drop(nan_row[0], level=0)
        c1 = c1.drop(nan_row[0], level=0)
        c2 = c2.drop(nan_row[0], level=0)
        y = y.drop(nan_row[0], level=0)

    # Convert x to 3D numpy array and c1, c2, y to 2D arrays
    n_samples = len(x.index.get_level_values('id').unique())
    n_timesteps = len(x.index.get_level_values('dse').unique())
    n_features = len(x.columns)
    x = x.values.reshape(n_samples, n_timesteps, n_features)
    print(f'Shape of x array: {x.shape}, type = {x.dtype}')
    y = y.values.reshape(n_samples, n_timesteps, 1)
    print(f'Shape of y array: {y.shape}, type = {y.dtype}')
    c1 = np.array(c1)
    print(f'Shape of c1 array: {c1.shape}, type = {c1.dtype}')
    c2 = np.array(c2)
    print(f'Shape of c2 array: {c2.shape}, type = {c2.dtype}')

    # Grid Search Initialization
    kgs = KerasGridSearch(get_rnn, param_grid, monitor='val_loss', greater_is_better=False, tuner_verbose=VERBOSE)
    es = EarlyStopping(patience=10, verbose=1, min_delta=0.0001, monitor='val_loss', mode='auto',
                       restore_best_weights=True)
    best_combis = []  # Store best combinations found at each outer loop
    results = []  # Store their associated performance scores
    kf_outer = StratifiedKFold(n_splits=N_SPLITS_OUTER, random_state=RANDOM_STATE, shuffle=True)
    fold_number = 0
    list_aucs = [[], [], [], []]
    for train_idx, test_idx in kf_outer.split(x, y[:, 0, :].flatten()):
        fold_number += 1
        x_train, c1_train, c2_train, y_train = x[train_idx], c1[train_idx], c2[train_idx], y[train_idx]
        x_test, c1_test, c2_test, y_test = x[test_idx], c1[test_idx], c2[test_idx], y[test_idx]
        # Inner loop
        kf_inner = StratifiedKFold(n_splits=N_SPLITS_INNER, random_state=RANDOM_STATE, shuffle=True)
        folds_trials = []  # A list of lists of dicts, listing the combinations tested on each fold
        folds_scores = []  # A list of list, first list = scores fold 1 etc.
        for train_idx_inner, test_idx_inner in kf_inner.split(x_train, y_train[:, 0, :].flatten()):
            x_train_inner, c1_train_inner, c2_train_inner = x[train_idx_inner], c1[train_idx_inner], c2[train_idx_inner]
            y_train_inner = y[train_idx_inner]
            x_test_inner, c1_test_inner, c2_test_inner = x[test_idx_inner], c1[test_idx_inner], c2[test_idx_inner]
            y_test_inner = y[test_idx_inner]
            # Scale and process before input
            x_train_inner, c2_train_inner, x_test_inner, c2_test_inner = scale_process_inputs(x_train=x_train_inner,
                                                                                              c2_train=c2_train_inner,
                                                                                              x_test=x_test_inner,
                                                                                              c2_test=c2_test_inner,
                                                                                              binarization=BINARIZATION)
            # The 'epoch' given in the optimal combination is the one with the lowest validation loss during training
            kgs.search(x_train_inner, y_train_inner,
                       validation_data=(x_test_inner, y_test_inner),
                       callbacks=[es])
            print("Combinations : ", kgs.trials)
            folds_trials.append(kgs.trials)
            print("Scores of each combi on this fold: ", kgs.scores)
            folds_scores.append(kgs.scores)
        print("Scores on every fold: ", folds_scores)
        avg_scores = [sum(col) / float(len(col)) for col in zip(*folds_scores)]
        print("Average scores: ", avg_scores)
        # Return the avg best combi (only 'epochs' varies)
        best_combi_folds = [i[avg_scores.index(min(avg_scores))] for i in folds_trials]
        print("Overall best combination:", best_combi_folds)
        # Stick to the combi with the mean number of epochs to test it in the outer loop (not optimal, just an estimate)
        # best_combi = sorted(best_combi_folds, key=lambda i: i['epochs'], reverse=True)[0] # MAX INSTEAD OF MEAN
        mean_epoch = int(mean([i['epochs'] for i in best_combi_folds]))
        best_combi = best_combi_folds[0]
        best_combi['epochs'] = mean_epoch
        print("Best combi to use for the outer loop test:", best_combi)
        best_combis.append(best_combi)
        model = get_rnn(best_combi)
        # Scale and process before testing
        print('Shapes of x_train and c2_train before processing: ', x_train.shape, c2_train.shape)
        x_train, c2_train, x_test, c2_test = scale_process_inputs(x_train=x_train,
                                                                  c2_train=c2_train,
                                                                  x_test=x_test,
                                                                  c2_test=c2_test,
                                                                  binarization=BINARIZATION)
        print('Shapes of x_train and c2_train after processing: ', x_train.shape, c2_train.shape)
        model.fit(x_train, y_train, epochs=best_combi['epochs'],
                  batch_size=best_combi['batch_size'])
        result = model.evaluate(x_test, y_test)
        print("Accuracy: %.2f%%" % (result[1] * 100))
        results.append(result[1])

        # Predict probabilities for different test TS lengths (early classification)
        for ts_length in range(1, 5):
            x_test_truncated = x_test.copy()
            x_test_truncated = x_test_truncated[:, :ts_length]
            y_scores = model.predict([x_test_truncated, c1_test, c2_test])[:, -1, :].ravel()
            fpr, tpr, threshold = roc_curve(y_test[:, -1, :].ravel(), y_scores)
            roc_auc = auc(fpr, tpr)
            list_aucs[ts_length - 1].append(roc_auc)

    for i in range(len(list_aucs)):
        list_aucs[i] = (mean(list_aucs[i]), stdev(list_aucs[i]))

    print("List of the best combinations tested accross the outer loop: ", best_combis)
    print("Generalized accuracy of the architecture: %.2f%%" % (mean(results) * 100))

    # Save list of aucs
    with open(os.path.join(SCORES_PATH, 'list_aucs_rnn.pkl'), 'wb') as f:
        pkl.dump(list_aucs, f)
    print(list_aucs)


if __name__ == '__main__':
    main()
