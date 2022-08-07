import operator
import random
import sys
import numpy as np
import os
import pandas as pd
import scipy
import seaborn as sns
import meld
from adjustText import adjust_text
from matplotlib import pyplot as plt, cm as cm
from random import choices
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from pyclustering.cluster.kmedoids import kmedoids

# Tune the variable SUBSET to select a particular subset
# Set SUBSET='MEX_OUT' to exclude Mexico, 'MEX_ONLY' to consider Mexico dataset only, 'ALL' for the whole database
SUBSET = 'MEX_ONLY'

# Project constants
# FIX RANDOM SEED FOR MODEL COMPARISON. SET TO 'None' TO PRODUCE THE FINAL MODEL.
RANDOM_STATE = 999  # Set to None for deployment, otherwise any of your favorite numbers to compare models
DATA_PATH = os.path.join('..', 'data')
RESULTS_PATH = os.path.join('..', 'results')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PKL_PATH = os.path.join(DATA_PATH, 'pkl')
PROCESSED_PATH = os.path.join(PKL_PATH, 'processed')
SCORES_PATH = os.path.join('..', 'scores')


def load_data_phate():
    # Returns inputs and targets dataframes as defined in processing.py
    x = pd.read_pickle(os.path.join(PROCESSED_PATH, 'inputs.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_PATH, 'targets.pkl'))
    return x, y


def load_data_phate_con():
    # Returns inputs and targets dataframes as defined in processing_control.py
    x = pd.read_pickle(os.path.join(PROCESSED_PATH, 'inputs_control.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_PATH, 'targets_control.pkl'))
    return x, y

def load_data_raw():
    # Returns inputs and targets dataframes (NaN values included)
    x = pd.read_pickle(os.path.join(PROCESSED_PATH, 'inputs_nan.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_PATH, 'targets_nan.pkl'))
    return x, y


def forward_filling(df):
    for index in df.index.get_level_values('id').unique():
        df.loc[index, :].ffill(inplace=True)
        df.loc[index, :].bfill(inplace=True)
    return df


def remove_unavailable_first_visits(df):
    for index in df.index.get_level_values('id').unique():
        if df.loc[(index, 0), :].isnull().any():
            df = df.drop(index)
    return df


def remove_false_measurements(df):
    for index in df.index.get_level_values('id').unique():
        if (df.loc[(index,), 'SpO2'] < 60).any() or (df.loc[(index,), 'SpO2'] > 100).any() or \
                (df.loc[(index,), 'RR'] < 5).any() or (df.loc[(index,), 'RR'] > 60).any():
            df = df.drop(index)
    return df


def variable_panel(z, data, col_wrap=4, z_prefix='PHATE', save=False, file_name=None, cmap=None, col_size=5,
                   row_size=5):
    """Visualize all variables in data Dataframe over the z coordinates (ndarray)."""
    data = data.copy()  # Copy data to protect raw_pkl dataframe
    z = z.copy()

    # Check variables and add coordinates to dataframe
    variables = data.columns.copy()
    n_variables = len(variables)
    data[[f'{z_prefix} 1', f'{z_prefix} 2']] = z

    # Compute number of rows and columns
    n_rows = max(n_variables // col_wrap + (1 if n_variables % col_wrap != 0 else 0), 1)
    n_cols = min(col_wrap, n_variables)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * col_size, n_rows * row_size))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i, c_name in enumerate(variables):
        if cmap is None:
            if data[c_name].dtype in ('category', 'bool', 'boolean') or data[c_name].isin([0., 1., np.nan]).all():
                cmap_c = 'Dark2'
            else:
                cmap_c = None
        else:
            cmap_c = cmap
        sns.scatterplot(data=data, x=f'{z_prefix} 1', y=f'{z_prefix} 2', ax=axes[i], hue=c_name, palette=cmap_c)
        axes[i].set_title(c_name)
        axes[i].get_legend().set_title('')
        if not i % n_cols == 0:
            axes[i].set_ylabel('')
        if i < len(axes) - n_cols:
            axes[i].set_xlabel('')

    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.set_facecolor('white')
        a.set_xticks([])
        a.set_yticks([])

    f.tight_layout()

    if save:
        if file_name is None:
            raise Exception('Need to define a file name.')
        plt.savefig(os.path.join(RESULTS_PATH, f'{file_name}.pdf'))
    plt.show()


def meld_panel(z, data, variable, pipeline=None, col_wrap=4, z_prefix='PHATE', save=False, file_name=None, cmap=None,
               col_size=5,
               row_size=5, meld_args=dict()):
    """Visualize variable and meld likelhoods of different classes on a z embedding."""
    data = data.copy()  # Copy data to protect raw_pkl dataframe
    z = z.copy()

    # Check variables and add coordinates to dataframe
    z_keys = [f'{z_prefix} 1', f'{z_prefix} 2']
    inputs = data.columns.copy()  # Keep input columns
    data[variable.name] = pd.Categorical(variable)
    data[z_keys] = z

    # Drop mising values
    data = data.dropna().copy()   # Copy to avoid annoying warning

    # Compute meld likelihoods
    meld_op = meld.MELD(**meld_args, random_state=RANDOM_STATE)
    x = pipeline.fit_transform(data[inputs]) if pipeline is not None else data[inputs]
    lik = meld.utils.normalize_densities(meld_op.fit_transform(x, data[variable.name])).set_index(data.index)
    data.loc[:, lik.columns] = lik

    n_plots = lik.shape[1]

    # Compute number of rows and columns
    n_rows = max(n_plots // col_wrap + (1 if n_plots % col_wrap != 0 else 0), 1)
    n_cols = min(col_wrap, n_plots)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * col_size, n_rows * row_size))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])

    axes = axes.flatten()
    hue_norm = (lik.values.min(), lik.values.max())

    for i, c_name in enumerate(lik):
        sns.scatterplot(data=data, x=f'{z_prefix} 1', y=f'{z_prefix} 2', ax=axes[i], hue=c_name,
                        palette=meld.utils.get_meld_cmap(), hue_norm=hue_norm)
        axes[i].set_title(str(variable.name) + ' = ' + str(c_name) + ' (MELD)')
        axes[i].get_legend().set_title('')
        if not i % n_cols == 0:
            axes[i].set_ylabel('')
        if i < len(axes) - n_cols:
            axes[i].set_xlabel('')

    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.set_facecolor('white')
        a.set_xticks([])
        a.set_yticks([])

    f.tight_layout()

    if save:
        if file_name is None:
            raise Exception('Need to define a file name.')
        plt.savefig(os.path.join(RESULTS_PATH, f'{file_name}.pdf'))
    plt.show()


def view_trajectories_group(z, groupby, z_prefix='PHATE '):
    """Visualize 2D average trajectories of each group using dse as time points
    z should be a ndarray of 2D coordinates (i.e. a 2D PHATE embedding)
    groupby should be a pandas series to identify groups with a proper ('id', 'dse') index.
    Recall dse stands for days since enrolment.
    """
    # List of colors to distinguish each group trajectory
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Time points
    time_points = groupby.index.unique(level='dse')

    # Copy data to protect raw_pkl dataframe
    groupby = groupby.copy()
    z = z.copy()

    # Tranfer 2D data and groupby into a dataframe to facilitate manipulation
    data = pd.DataFrame({groupby.name: pd.Categorical(groupby)}, index=groupby.index)
    z_keys = [f'{z_prefix} 1', f'{z_prefix} 2']
    data[z_keys] = z

    # Groups
    data = data.dropna()
    data[groupby.name] = data[groupby.name].cat.remove_unused_categories()
    groups = list(data[groupby.name].cat.categories)

    # Check if there are not too many groups to assign each color
    if len(groups) > len(colors):
        raise Exception('Too many different groups to visualize trajectories.')

    # Plot a neutral PHATE embedding
    plt.scatter(z[:, 0], z[:, 1], marker='o', s=4, color='grey')

    # Compute and plot each group's 2D trajectory
    for c, g in enumerate(groups):
        avg_trajectory = np.zeros(shape=(time_points.size, 2))
        for i, t in enumerate(time_points):
            avg_trajectory[i, 0] = data[data[groupby.name] == g][z_keys[0]].xs(t, level=1, axis=0).mean()
            avg_trajectory[i, 1] = data[data[groupby.name] == g][z_keys[1]].xs(t, level=1, axis=0).mean()

        plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], marker='o', markersize=5, linestyle='dashed',
                 color=colors[c], label=groupby.name + " = " + str(g))
        for i, label in enumerate(map(str, time_points)):
            plt.annotate('Day ' + label, (avg_trajectory[i, 0], avg_trajectory[i, 1]))

    # Annotations
    plt.xlabel(z_keys[0])
    plt.ylabel(z_keys[1])
    plt.title("Average trajectories of " + groupby.name + " groups", fontsize=10)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.legend(fontsize=8)
    plt.grid(True)
    plt.show()


def view_random_trajectories(z, groupby, n_traj=5, z_prefix="PHATE ", annotate=False):
    """Visualize motion of each patient
    n_traj is the number of drawn trajetories (preferably small to better visualize them)
    z should be a ndarray of 2D coordinates (i.e. a 2D PHATE embedding)
    groupby should be a pandas series to identify groups with a proper ('id', 'dse') index.
    Recall dse stands for days since enrolment.
    """
    # Copy
    z = z.copy()
    groupby = groupby.copy()
    # List of colors to distinguish each group
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Check if there are not too many groups to assign each color
    if groupby.dropna().unique().shape[0] > len(colors):
        raise Exception('Too many groups to distinguish them.')
    # Map each group to a color
    group_color = {}
    for c, group in enumerate(groupby.dropna().unique()):
        group_color[group] = colors[c]
    # Tranfer 2D data into dataframe to associate embedded points with their indices
    data = pd.DataFrame(index=groupby.index)
    z_keys = [f'{z_prefix} 1', f'{z_prefix} 2']
    data[z_keys] = z
    # Plot a neutral PHATE embedding
    plt.scatter(z[:, 0], z[:, 1], marker='o', s=4, color='grey')
    # Plot random trajectories of patients with an available label
    random_patients = choices(groupby.dropna().index.get_level_values('id').unique(), k=n_traj)
    for patient in random_patients:
        patient_label = groupby.loc[patient].head(1).item()
        plt.plot(data.loc[patient, z_keys[0]], data.loc[patient, z_keys[1]], marker='o',
                 markersize=5, linestyle='-', linewidth=0.4, color=group_color[patient_label],
                 label=groupby.name + " = " + str(patient_label))
        if annotate:
            for index, row in data.loc[patient, :].iterrows():
                plt.annotate('Day ' + str(index),  # this is the text
                             (row[z_keys[0]], row[z_keys[1]]),  # these are the coordinates to position the label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 5),  # distance from text to points (x,y)
                             ha='center')
    # Legend
    plt.xlabel(z_keys[0])
    plt.ylabel(z_keys[1])
    plt.title(f'{n_traj} random trajectories colored by {groupby.name}', fontsize=10)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # Avoid duplicated items in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys())
    # Increase legend linewidth
    for line in leg.get_lines():
        line.set_linewidth(1)
    plt.grid(True)
    plt.show()


def view_kmeans_trajectories(x, x_sub, z, metric, n_clusters=4, z_prefix="PHATE ", random_state=RANDOM_STATE):
    """Visualize k-means trajectories clusters and their corresponding barycenters
    x is the original high-dimensional data with indices ('id', 'dse') on which we apply TS KMeans
    x_sub is a subgroup from x. Set to None if we do not want a subgroup clustering
    z is the PHATE embedding of x to project the trajectories in 2D
    """
    # Copy
    z = z.copy()
    x = x.copy()
    if x_sub is not None:
        x_sub = x_sub.copy()

    # List of colors to distinguish each cluster
    colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k']

    # Store and transform the points to cluster with TimeSeriesKMeans
    if x_sub is None:
        to_cluster = x
    else:
        to_cluster = x_sub
    n_ts = len(to_cluster.index.get_level_values('id').unique())
    n_timesteps = len(to_cluster.index.get_level_values('dse').unique())
    n_features = len(to_cluster.columns)
    ts = to_cluster.values.reshape(n_ts, n_timesteps, n_features)
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state)
    ts_clusters = km.fit_predict(ts)

    # Store 2D PHATE embbedings into dataframe to associate embedded points with their indices to facilitate iteration
    data = pd.DataFrame(index=x.index)
    z_keys = [f'{z_prefix}1', f'{z_prefix}2']
    data[z_keys] = z

    # Check if there are not too many clusters to assign each color
    if n_clusters > len(colors):
        raise Exception('Too many clusters to distinguish them.')

    # Map each cluster to a color
    cluster_color = {}
    for c, cluster in enumerate(np.unique(ts_clusters)):
        cluster_color[cluster] = colors[c]

    # Plot a neutral PHATE embedding
    plt.scatter(z[:, 0], z[:, 1], marker='o', s=4, color='grey')

    # Plot trajectories
    for i, patient in enumerate(to_cluster.index.get_level_values('id').unique()):
        patient_label = ts_clusters[i]
        plt.plot(data.loc[patient, z_keys[0]], data.loc[patient, z_keys[1]], marker='o',
                 markersize=0, linestyle='-', linewidth=0.7, color=cluster_color[patient_label],
                 label="Cluster " + str(patient_label))

    # Legend
    plt.xlabel(z_keys[0])
    plt.ylabel(z_keys[1])
    plt.title(f'Plot of {n_clusters} distinct trends from {metric}-k-means', fontsize=10)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # Avoid duplicated items in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys())
    # Increase legend linewidth
    for line in leg.get_lines():
        line.set_linewidth(1.5)
    plt.grid(True)
    plt.show()

    # Return the list of cluster assignment for each patient visit
    return np.repeat(ts_clusters, 4)


def view_kmedoids_trajectories(x, x_proc, z, metric, n_clusters=4, z_prefix='PHATE '):
    """Visualize k-medoids trajectories clusters and their corresponding medoids
    x is the original high-dimensional scaled data with indices ('id', 'dse') on which we apply TS KMeans
    z is the PHATE embedding of x to project to visualize the trajectories in 2D
    metric should either 'euclidean' or 'dtw'
    """
    """Visualize k-medoids trajectories clusters and their corresponding medoids
    x is the original high-dimensional data with indices ('id', 'dse') on which we apply TS KMedoids
    x_proc is the numpy array processed from x
    z is the PHATE embedding of x to visualize the trajectories in 2D
    metric should 'dtw' or other supported distances
    """
    # Copy
    z = z.copy()
    x = x.copy()
    x_proc = pd.DataFrame(data=x_proc, index=x.index)  # Put x_proc in a DataFrame to separate each time series

    # List of colors to distinguish each cluster
    colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k']

    # Check if there are not too many clusters to assign each color
    n_ts = len(x.index.get_level_values('id').unique())
    if n_clusters > len(colors):
        raise Exception('Too many clusters to distinguish them.')

    # Transform x_proc_df into list of ts to compute distance matrix with cdist
    list_ts = []
    for idx in x.index.get_level_values('id').unique():
        list_ts.append(x_proc.loc[(idx,), :].to_numpy())

    # Choose k initial medoids at random  /!\ NOT OPTIMAL /!\
    initial_medoids = random.sample(range(n_ts), n_clusters)

    # Apply the k-medoids process on ts
    if metric == 'dtw':
        distance_matrix = cdist_dtw(list_ts)
    else:
        raise Exception('Metric unsupported.')

    # Process KMedoids
    km = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
    km.process()

    # Store clusters and medoids
    ts_clusters = km.get_clusters()
    medoids = km.get_medoids()

    # Tranfer 2D data into dataframe to associate embedded points with their indices to facilitate iteration
    z_df = pd.DataFrame(index=x.index)
    z_keys = [f'{z_prefix}1', f'{z_prefix}2']
    z_df[z_keys] = z

    # Map each patient id to a cluster number 0, 1, ..., n_clusters
    list_patient_ids = x.index.get_level_values('id').unique().tolist()
    patient_cluster = {}
    for cluster_number, cluster in enumerate(ts_clusters):
        for i in cluster:
            patient_cluster[list_patient_ids[i]] = cluster_number

    # Add a column to 2D data corresponding to the cluster number of each patient
    for patient in list_patient_ids:
        z_df.loc[(patient,), 'cluster_number'] = patient_cluster[patient]
    z_df['cluster_number'] = z_df['cluster_number'].astype('int')

    # Retrieve the original patient ids of the computed medoids
    medoid_patients = []
    for medoid_id in medoids:
        medoid_patients.append(list_patient_ids[medoid_id])

    # Plot a neutral PHATE embedding
    plt.scatter(z[:, 0], z[:, 1], marker='o', s=4, color='grey')

    # Plot clusters of trajectories
    for i, patient in enumerate(z_df.index.get_level_values('id').unique()):
        patient_label = z_df.loc[patient, 'cluster_number'].head(1).item()
        color = colors[patient_label]
        plt.plot(z_df.loc[patient, z_keys[0]], z_df.loc[patient, z_keys[1]], marker='o',
                 markersize=0, linestyle='-', linewidth=0.4, color=color,
                 label="TS Cluster " + str(patient_label))

    # Plot corresponding medoids
    for medoid_patient in medoid_patients:
        patient_label = z_df.loc[medoid_patient, 'cluster_number'].head(1).item()
        plt.plot(z_df.loc[medoid_patient, z_keys[0]], z_df.loc[medoid_patient, z_keys[1]], marker='o',
                 markersize=4, linestyle='-', linewidth=2, color=colors[patient_label])
        for i in z_df.loc[medoid_patient].index.get_level_values('dse').unique():
            # Plot labels
            texts = list()
            texts.append(
                plt.text(z_df.loc[(medoid_patient, i), z_keys[0]], z_df.loc[(medoid_patient, i), z_keys[1]], str(i),
                         color="white",
                         bbox=dict(facecolor=colors[patient_label], edgecolor=colors[patient_label], boxstyle='circle'),
                         fontsize=8))
            adjust_text(texts,
                        # arrowprops=dict(arrowstyle='->', color='black'),
                        autoalign=True,
                        avoid_points=False,
                        # only_move={'points':'y', 'text':'y', 'objects':'y'},
                        ha='center', va='center')

    # Legend
    plt.xlabel(f"{z_prefix}1")
    plt.ylabel(f"{z_prefix}2")
    plt.title(f'Plot of {n_clusters} distinct trends from {metric.upper()}-K-Medoids and\n'
              f'their respective medoids in bold lines',
              fontsize=10)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # Avoid duplicated items in the legend
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys())
    # Increase legend linewidth
    for line in leg.get_lines():
        line.set_linewidth(1.5)
    plt.grid(True)
    plt.show()

    return z_df['cluster_number']


def cluster_box_plots(df, n_cols=5, cmap=None, kw_test=False, significance_level=0.05):
    """ Box plots (numerical variables) with or without Krustal-Wallis test
    df is a DataFrame of the original data n_samples x n_features, with one of the columns named 'clusters'
    df_proc is the processed version of df, used to produce box plot for ordered categorical features
    """
    df = df.copy()  # Protect original data
    df = df.groupby('id').head(1)
    clusters = sorted(df['ts_clusters'].unique())
    if kw_test:
        for cluster in clusters:
            nb_samples = df.loc[df['ts_clusters'] == cluster].shape[0]
            if nb_samples < 5:
                raise Exception(f'Cluster {cluster} has only {nb_samples} < 5 which is too few for Krustal-Wallis.')
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis', lut=len(clusters)).colors
    features = df.select_dtypes(include='float32').columns.tolist()
    n_features = len(features)
    n_rows = -(-n_features // n_cols)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
    axes = axes.flatten()
    for i in range(n_features):
        sequence = []
        for cluster in clusters:
            data = df.loc[df['ts_clusters'] == cluster][features[i]].dropna().tolist()
            sequence.append(data)
        if kw_test:
            _, p_value = scipy.stats.kruskal(*sequence)
            if p_value <= significance_level:
                axes[i].set_title(f'{features[i]}', fontsize=10, fontweight='bold', backgroundcolor='grey',
                                  color='white')
            else:
                axes[i].set_title(f'{features[i]}', fontsize=10, fontweight='bold')
            axes[i].legend(title=f'K.-W. test p-value = {p_value:.3f}', loc='upper right', fancybox=True, framealpha=0.5,
                           title_fontsize=8)
        else:
            axes[i].set_title(f'{features[i]}', fontsize=10, fontweight='bold')
        # Plot boxes
        bplot = axes[i].boxplot(sequence, positions=range(0, len(clusters)), patch_artist=True)
        # Color boxes
        for patch, color in zip(bplot['boxes'], cmap):
            patch.set_facecolor(color)
        axes[i].yaxis.grid(False)
        axes[i].set_xlabel('TS Cluster')
    # Remove extra plots
    for j in range(n_features, len(axes)):
        axes[j].set_axis_off()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()


def cluster_bar_plots(df, n_cols=3, bar_width=0.5):
    # Bar charts (categorical variables)
    features = df.select_dtypes(include='category').columns.tolist()
    features.remove('ts_clusters')
    n_features = len(features)
    n_rows = -(-n_features // n_cols)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axes = axes.flatten()
    # Bar plots for categorical variables
    clusters = sorted(df['ts_clusters'].unique())
    for j in range(n_features):
        categories = df[features[j]].cat.categories
        colors = plt.cm.OrRd(np.linspace(0.3, 0.9, len(categories)))
        index = np.arange(0, len(clusters))  # Number of clusters
        # Plot bars and create text labels for the table
        list_perc_per_cluster_sum = np.zeros(len(clusters))
        for k, cat in enumerate(categories):  # Plot bar perc for each category
            list_perc_per_cluster = []
            for cluster in clusters:
                nb_cat = df.loc[df['ts_clusters'] == cluster].loc[df[features[j]] == cat].shape[0]
                nb_tot = df.loc[df['ts_clusters'] == cluster].shape[0]
                perc = nb_cat / nb_tot * 100
                list_perc_per_cluster.append(perc)
            axes[j].bar(index, list_perc_per_cluster, bar_width, bottom=list_perc_per_cluster_sum,
                        color=colors[k], label=cat)
            list_perc_per_cluster_sum = list_perc_per_cluster_sum + np.array(list_perc_per_cluster)
        axes[j].set_xlabel('TS Cluster')
        axes[j].set_ylabel('Frequency')
        axes[j].set_xticks(np.arange(min(index), max(index) + 1, 1.0))
        axes[j].set_title(f'{features[j]}', fontsize=12, fontweight='bold')
        axes[j].legend(loc='upper right', fancybox=True, framealpha=0.5)
    # Remove extra plots
    for j in range(n_features, len(axes)):
        axes[j].set_axis_off()
    # Adjust x-axis margins
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()
