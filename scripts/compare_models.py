import os
from matplotlib import pyplot as plt
import knn
import condrnn
import rnn
import condrnn_b
from scripts.utils import SCORES_PATH
import pickle as pkl
from time import perf_counter

t0 = perf_counter()
for outcome in ['X83_Death28d', 'X84_Intub', 'X81_Primary_out']:

    knn.main(outcome)
    condrnn.main(outcome)
    condrnn_b.main(outcome)
    rnn.main(outcome)

    # Retrieve scores
    with open(os.path.join(SCORES_PATH, 'list_aucs_knn.pkl'), 'rb') as f:
        list_aucs_knn = pkl.load(f)
    with open(os.path.join(SCORES_PATH, 'list_aucs_condrnn.pkl'), 'rb') as f:
        list_aucs_condrnn = pkl.load(f)
    with open(os.path.join(SCORES_PATH, 'list_aucs_rnn.pkl'), 'rb') as f:
        list_aucs_rnn = pkl.load(f)
    with open(os.path.join(SCORES_PATH, 'list_aucs_condrnn_b.pkl'), 'rb') as f:
        list_aucs_condrnn_b = pkl.load(f)

    # Plot AUCs of various models for different time series lengths
    plt.plot([1, 2, 3, 4],
             [list_aucs_knn[0][0],
              list_aucs_knn[1][0],
              list_aucs_knn[2][0],
              list_aucs_knn[3][0]
              ],
             color='r', linewidth=1, linestyle='dotted', label='kNN')
    plt.plot([1, 2, 3, 4],
             [list_aucs_condrnn[0][0],
              list_aucs_condrnn[1][0],
              list_aucs_condrnn[2][0],
              list_aucs_condrnn[3][0]
              ],
             color='b', linewidth=1, linestyle='-', label='CondRNN')
    plt.plot([1, 2, 3, 4],
             [list_aucs_condrnn_b[0][0],
              list_aucs_condrnn_b[1][0],
              list_aucs_condrnn_b[2][0],
              list_aucs_condrnn_b[3][0]
              ],
             color='royalblue', linewidth=1, linestyle='-', label='Binary CondRNN')
    plt.plot([1, 2, 3, 4],
             [list_aucs_rnn[0][0],
              list_aucs_rnn[1][0],
              list_aucs_rnn[2][0],
              list_aucs_rnn[3][0]
              ],
             color='c', linewidth=1, linestyle='-', label='SimpleRNN')

    # Round to 2 decimals to show in table
    list_aucs_knn = [f'{round(tup[0], 4)} ± {round(tup[1], 2)}' for tup in list_aucs_knn]
    list_aucs_condrnn = [f'{round(tup[0], 4)} ± {round(tup[1], 2)}' for tup in list_aucs_condrnn]
    list_aucs_condrnn_b = [f'{round(tup[0], 4)} ± {round(tup[1], 2)}' for tup in list_aucs_condrnn_b]
    list_aucs_rnn = [f'{round(tup[0], 4)} ± {round(tup[1], 2)}' for tup in list_aucs_rnn]

    # Produce tables for numerical comparison
    rcolors = ['r', 'b', 'royalblue', 'c']
    cellText = [list_aucs_knn, list_aucs_condrnn, list_aucs_condrnn_b, list_aucs_rnn]
    plt.table(cellText=cellText,
              rowLabels=['kNN', 'CondRNN', 'Binary CondRNN', 'SimpleRNN'],
              rowColours=rcolors,
              colLabels=['after day 0', 'after day 1', 'after day 2', 'after day 3'],
              colColours=['silver', 'silver', 'silver', 'silver'],
              loc='bottom')

    # Make space for the table
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylabel('AUROC')
    plt.xticks([])
    plt.legend(loc='lower right')
    plt.title(f'Early AUROC scores of 4 models for {outcome}', fontsize=9)
    plt.show()
t1 = perf_counter()
print("Time elapsed: ", t1 - t0)
