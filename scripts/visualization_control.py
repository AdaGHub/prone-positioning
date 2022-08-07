"""Script for several visualizations with PHATE (Control group).

MAKE SURE to run processing_control.py once before calling this."""

from openpyxl import load_workbook
import sys
import random
import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
import phate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.decomposition  # PCA
import sklearn.manifold  # t-SNE
from umap import UMAP
from utils import load_data_phate_con, variable_panel, meld_panel, RANDOM_STATE, view_trajectories_group, \
    view_random_trajectories, view_kmeans_trajectories, view_kmedoids_trajectories, \
    cluster_box_plots, cluster_bar_plots
from processing import DATA_FILE, SUBSET

SAVE = False  # Save some plots as individuals pdfs for figures

# Number of clusters (4 for MEX out / MEX only, 6 for ALL)
N_CLUSTERS = 4
N_TS_CLUSTERS = 4

# PHATE parameters (mex: 30/30/7/0)
KNN = 40  # Increase connectivity
T = 30  # Increase to tighten the branches
ALPHA = 8  # Decrease to tighten and decrease distances between clusters (inverse to KNN)
GAMMA = 0  # Default = 1

SCALER = StandardScaler()

# Set seed for 'random' reproducibility
random.seed(RANDOM_STATE)

# MELD parameter
BETA = KNN

# Display settings
pd.set_option('display.max_rows', 500)

# Load pkl inputs and targets
x, y = load_data_phate_con()

# Data scaling
scaler = SCALER
x_proc = x.copy()
for column in x_proc:
    scaler.fit(x_proc[[column]])
    x_proc[[column]] = scaler.fit_transform(x_proc[[column]])

print("%%%%%%%% Max/Min OF SCALED INPUTS: %%%%%%%%\n", x_proc.max(), x_proc.min())

# Instantiate the PHATE estimator and transform data
phate_operator = phate.PHATE(n_jobs=-2, knn=KNN, t=T, decay=ALPHA, random_state=RANDOM_STATE, gamma=GAMMA)
z = phate_operator.fit_transform(x_proc)

# Instantiate PCA and t-SNE and tranform data
pca_operator = sklearn.decomposition.PCA(n_components=2, random_state=RANDOM_STATE)
z_pca = pca_operator.fit_transform(x_proc)
tsne_operator = sklearn.manifold.TSNE(n_components=2, random_state=RANDOM_STATE)
z_tsne = tsne_operator.fit_transform(x_proc)
umap_operator = UMAP(n_components=2, init='random', random_state=RANDOM_STATE)
z_umap = umap_operator.fit_transform(x_proc)

# Visualize inputs
variable_panel(z=z, data=x, z_prefix="PHATE", col_wrap=3)
variable_panel(z=z_pca, data=x, z_prefix="PCA", col_wrap=3)
variable_panel(z=z_tsne, data=x, z_prefix="t-SNE", col_wrap=3)
variable_panel(z=z_umap, data=x, z_prefix="UMAP", col_wrap=3)

# Add clusters to target with constrained KMeans to avoid ending up with very small clusters
y['clusters'] = pd.Categorical(
    KMeansConstrained(n_clusters=N_CLUSTERS, size_min=10, random_state=RANDOM_STATE).fit_predict(x_proc))

# Select some targets to visualize
targets = [
    'Days_since_Enrol',
    "Nb_Comorbidities",
    "Mexican_trial",
    'X04_Gender',
    'X05_Age',
    'X08_BMI',
    'X17_MAP_enrol',
    'X18_RR_enrol',
    'X20_ROX_enrol',
    'X200_SF_enrol',
    'X22_CCD',
    'X23_CLD',
    'X24_DM',
    'X25_CKD',
    'X26_Cancer',
    'X27_Obesity',
    'X28_CLF',
    'X30_Steroid',
    'X81_Primary_out',
    # 'X82_Alive28d',
    'X83_Death28d',
    'X84_Intub'
    # 'clusters'
]

# Visualize targets
variable_panel(z=z, data=y[targets])


"""# MELD panels
meld_panel(z=z, data=x, pipeline=None, variable=y['X81_Primary_out'],
           meld_args=dict(knn=KNN, beta=BETA))"""

if SAVE:
    for lab in x.columns:
        variable_panel(z=z, data=x[[lab]], save=True, file_name=f'{SUBSET}_x_raw_{lab}')
    for lab in y.columns:
        variable_panel(z=z, data=y[[lab]], save=True, file_name=f'{SUBSET}_y_raw_{lab}')
        if lab in ('X81_Primary_out', 'X83_Death28d', 'X84_Intub'):
            # Compute MELD for all categorical variables
            meld_panel(z=z, data=x, variable=y[lab], pipeline=None, meld_args=dict(knn=KNN, beta=BETA),
                       save=True, file_name=f'{SUBSET}_y_meld_{lab}')

# Set backend for interactive plots
# matplotlib.use('Qt5Agg')

# Time-Series clustering
ts_clusters = view_kmedoids_trajectories(x=x, x_proc=x_proc, z=z, metric='dtw', n_clusters=N_TS_CLUSTERS)
ts_clusters_pca = view_kmedoids_trajectories(x=x, x_proc=x_proc, z=z_pca, metric='dtw', n_clusters=N_TS_CLUSTERS, z_prefix='PCA ')
ts_clusters_tsne = view_kmedoids_trajectories(x=x, x_proc=x_proc, z=z_tsne, metric='dtw', n_clusters=N_TS_CLUSTERS, z_prefix='t-SNE ')
ts_clusters_umap = view_kmedoids_trajectories(x=x, x_proc=x_proc, z=z_umap, metric='dtw', n_clusters=N_TS_CLUSTERS, z_prefix='UMAP ')

# Add ts_clusters to targets
y['ts_clusters'] = pd.Categorical(ts_clusters)

continuous = [
    'X05_Age',
    'X08_BMI',
    'X17_MAP_enrol',
    'X18_RR_enrol',
    'X20_ROX_enrol',
    'X200_SF_enrol'
    # 'Do_not_Intubate_order' NOT FOUND
]

categorical = [
    'Mexican_trial',
    'X04_Gender',
    'X22_CCD',
    'X23_CLD',
    'X24_DM',
    'X25_CKD',
    'X26_Cancer',
    'X27_Obesity',
    'X28_CLF',
    'X30_Steroid',
    'X81_Primary_out',
    # 'X82_Alive28d',
    'X83_Death28d',
    'X84_Intub'
    # 'Do_not_Intubate_order' NOT FOUND
]

continuous.append('ts_clusters')
categorical.append('ts_clusters')
cluster_box_plots(y[continuous], kw_test=True, n_cols=5)
cluster_bar_plots(y[categorical], n_cols=4)

# Quantitative characteristics of the time series clusters
table = pd.DataFrame(index=y['ts_clusters'].unique()).sort_index()
for index in table.index:
    cluster = y.loc[y['ts_clusters'] == index].groupby('id').head(1)
    nb_patients = cluster.shape[0]
    table.loc[index, "Nb_Patients"] = nb_patients
    table.loc[index, "Mexican_trial_Perc"] = (cluster.loc[cluster['Mexican_trial'] == 1].shape[
                                                  0] / nb_patients) * 100
    table.loc[index, 'X04_Female_Perc'] = (cluster.loc[cluster['X04_Gender'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X05_Age_Median'] = cluster["X05_Age"].median(skipna=True)
    table.loc[index, 'X05_Age_IQR'] = cluster["X05_Age"].quantile(0.75) - cluster["X05_Age"].quantile(0.25)
    table.loc[index, 'X08_BMI_Median'] = cluster["X08_BMI"].median(skipna=True)
    table.loc[index, 'X08_BMI_IQR'] = cluster["X08_BMI"].quantile(0.75) - cluster["X08_BMI"].quantile(0.25)
    table.loc[index, 'X17_MAP_enrol_Mean'] = cluster["X17_MAP_enrol"].mean(skipna=True)
    table.loc[index, 'X17_MAP_enrol_SD'] = cluster["X17_MAP_enrol"].std(skipna=True)
    table.loc[index, 'X18_RR_enrol_Mean'] = cluster["X18_RR_enrol"].mean(skipna=True)
    table.loc[index, 'X18_RR_enrol_SD'] = cluster["X18_RR_enrol"].std(skipna=True)
    table.loc[index, 'X20_ROX_enrol_Mean'] = cluster["X20_ROX_enrol"].mean(skipna=True)
    table.loc[index, 'X20_ROX_enrol_SD'] = cluster["X20_ROX_enrol"].std(skipna=True)
    table.loc[index, 'X200_SF_enrol_Mean'] = cluster["X200_SF_enrol"].mean(skipna=True)
    table.loc[index, 'X200_SF_enrol_SD'] = cluster["X200_SF_enrol"].std(skipna=True)
    table.loc[index, 'X22_CCD_Perc'] = (cluster.loc[cluster['X22_CCD'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X23_CLD_Perc'] = (cluster.loc[cluster['X23_CLD'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X24_DM_Perc'] = (cluster.loc[cluster['X24_DM'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X25_CKD_Perc'] = (cluster.loc[cluster['X25_CKD'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X26_Cancer_Perc'] = (cluster.loc[cluster['X26_Cancer'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X27_Obesity_Perc'] = (cluster.loc[cluster['X27_Obesity'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X28_CLF_Perc'] = (cluster.loc[cluster['X28_CLF'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'Nb_Comorbidities_Mean'] = cluster["Nb_Comorbidities"].mean(skipna=True)
    table.loc[index, 'Nb_Comorbidities_SD'] = cluster["Nb_Comorbidities"].std(skipna=True)
    table.loc[index, 'X281_1+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 0].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X282_2+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 1].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X283_3+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 2].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X284_4+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 3].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X285_5+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 4].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X286_6+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 5].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X287_7+_Comorbidities_Perc'] = (cluster.loc[cluster['Nb_Comorbidities'] > 6].shape[
                                                          0] / nb_patients) * 100
    table.loc[index, 'X30_Steroid_Perc'] = (cluster.loc[cluster['X30_Steroid'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X81_Primary_out_Perc'] = (cluster.loc[cluster['X81_Primary_out'] == 1].shape[
                                                    0] / nb_patients) * 100
    # table.loc[index, 'X82_Alive28d_Perc'] = (cluster.loc[cluster['X82_Alive28d'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X83_Death28d_Perc'] = (cluster.loc[cluster['X83_Death28d'] == 1].shape[0] / nb_patients) * 100
    table.loc[index, 'X84_Intub_Perc'] = (cluster.loc[cluster['X84_Intub'] == 1].shape[0] / nb_patients) * 100
# Import the table to an Excel sheet in the global database Excel file
with pd.ExcelWriter(DATA_FILE, mode='a', if_sheet_exists="replace") as writer:
    table.set_index('CLUSTER_' + table.index.astype(str)).transpose().to_excel(writer, float_format="%.2f",
                                                                               sheet_name='TS_Clusters_Table_CON')
