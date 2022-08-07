"""Script for processing (Control group). Handle data imputation (forward filling) (WORK ON 2021.1.3 IDE VERSION).
Saves two dataframes : inputs.pkl and targets.pkl. Use the load_data function from utils.py to load everything."""
import sys
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from utils import RAW_PATH, PROCESSED_PATH, forward_filling, remove_unavailable_first_visits, \
    remove_false_measurements, SUBSET

# Forward filling
FFILL = True

# Display settings
# pd.set_option('display.max_rows', 5000)
# pd.set_option('display.max_columns', 5000)

# Import data
DATA_FILE = os.path.join(RAW_PATH, 'database.xlsx')
data = pd.read_excel(DATA_FILE,
                     na_values=['Na', 'NA', 'NaN', 'MD']).rename(columns={'X02_Identification': 'id'}).set_index('id')

# Convert categorical variables to numerical
data = data.replace(to_replace=['Yes', 'No', 'Female', 'Male', 'APP', 'Control'], value=[1, 0, 1, 0, 1, 0])

# Consider only Control patients
data = data[data['X03_Group'] == 0]

# Remove non per-protocal patients
data = data[data['X89_Per_protocol'] == 1]

# Add Mexican_trial co-variable to distinguish mexican population (useful when working on the whole database)
conditions = [(data['X01_COUNTRY'] == "Mexico"), (data['X01_COUNTRY'] != "Mexico")]
values = [1, 0]
data['Mexican_trial'] = np.select(conditions, values)

# Selection of a subgroup of interest
if SUBSET == 'MEX_ONLY':
    print('Mexican subset only.')
    data = data[data['X01_COUNTRY'] == 'Mexico']
elif SUBSET == 'MEX_OUT':
    print('Mexican subset excluded.')
    data = data[data['X01_COUNTRY'] != 'Mexico']
elif SUBSET == 'ALL':
    print('Whole database selected.')
else:
    raise Exception('Please select a valid subset.')

# Drop X01_COUNTRY (useless column)
data = data.drop(columns=["X01_COUNTRY"])

# Convert all entries to float for numerical application
data = data.astype('float32')

# Add some useful variables
# Delta_Per_PP_ for the difference between values during PP and before PP
# Delta_Post_PP_ for the difference between values post PP and before PP
data["Nb_Comorbidities"] = data.loc[:, "X22_CCD": "X28_CLF"].sum(axis=1)

# Generate ROX and SF ratios for D0-D1-D2-D3
data["X200_SF_enrol"] = data["X12_SpO2_enrol"] / data["X14_FiO2_enrol"]
data['SF_D1'] = data['X65_SpO2_D1'] / data['X68_FiO2_D1']
data['SF_D2'] = data['X66_SpO2_D2'] / data['X69_FiO2_D2']
data['SF_D3'] = data['X67_SpO2_D3'] / data['X70_FiO2_D3']
data["X20_ROX_enrol"] = data["X200_SF_enrol"] / data["X18_RR_enrol"]
data['ROX_D1'] = data['SF_D1'] / data['X74_RR_D1']
data['ROX_D2'] = data['SF_D2'] / data['X75_RR_D2']
data['ROX_D3'] = data['SF_D3'] / data['X76_RR_D3']

# Select longitudinal variables to use as input
longi_data = data.loc[:, [
                             'X20_ROX_enrol', 'ROX_D1', 'ROX_D2', 'ROX_D3',
                             'X200_SF_enrol', 'SF_D1', 'SF_D2', 'SF_D3',
                             'X12_SpO2_enrol', 'X65_SpO2_D1', 'X66_SpO2_D2', 'X67_SpO2_D3',
                             'X14_FiO2_enrol', 'X68_FiO2_D1', 'X69_FiO2_D2', 'X70_FiO2_D3',
                             'X18_RR_enrol', 'X74_RR_D1', 'X75_RR_D2', 'X76_RR_D3'
                         ]]

# Extract batches for time-series analysis (D1/D2/D3)
longi_D0 = longi_data[[
                       'X12_SpO2_enrol',
                       'X14_FiO2_enrol',
                       'X18_RR_enrol',
                       'X200_SF_enrol',
                       'X20_ROX_enrol'
                       ]]
longi_D1 = longi_data[[
                       'X65_SpO2_D1',
                       'X68_FiO2_D1',
                       'X74_RR_D1',
                       'SF_D1',
                       'ROX_D1'
                       ]]
longi_D2 = longi_data[[
                       'X66_SpO2_D2',
                       'X69_FiO2_D2',
                       'X75_RR_D2',
                       'SF_D2',
                       'ROX_D2'
                       ]]
longi_D3 = longi_data[[
                       'X67_SpO2_D3',
                       'X70_FiO2_D3',
                       'X76_RR_D3',
                       'SF_D3',
                       'ROX_D3'
                       ]]

# Rename columns to get consistent column names between batches
longi_D0 = longi_D0.rename(columns={
                                    'X12_SpO2_enrol': 'SpO2',
                                    'X14_FiO2_enrol': 'FiO2',
                                    'X18_RR_enrol': 'RR',
                                    'X200_SF_enrol': 'SF',
                                    'X20_ROX_enrol': 'ROX'
                                    })
longi_D1 = longi_D1.rename(columns={
                                    'X65_SpO2_D1': 'SpO2',
                                    'X68_FiO2_D1': 'FiO2',
                                    'X74_RR_D1': 'RR',
                                    'SF_D1': 'SF',
                                    'ROX_D1': 'ROX'
                                    })
longi_D2 = longi_D2.rename(columns={
                                    'X66_SpO2_D2': 'SpO2',
                                    'X69_FiO2_D2': 'FiO2',
                                    'X75_RR_D2': 'RR',
                                    'SF_D2': 'SF',
                                    'ROX_D2': 'ROX',
                                    })
longi_D3 = longi_D3.rename(columns={
                                    'X67_SpO2_D3': 'SpO2',
                                    'X70_FiO2_D3': 'FiO2',
                                    'X76_RR_D3': 'RR',
                                    'SF_D3': 'SF',
                                    'ROX_D3': 'ROX'
                                    })

# Add corresponding dse to index
longi_D0['dse'], longi_D1['dse'], longi_D2['dse'], longi_D3['dse'] = 0, 1, 2, 3
batches = [longi_D0.reset_index().set_index(['id', 'dse']), longi_D1.reset_index().set_index(['id', 'dse']),
           longi_D2.reset_index().set_index(['id', 'dse']), longi_D3.reset_index().set_index(['id', 'dse'])]

# Gather everything
inputs = pd.concat(batches).sort_index()

# We assume that the first visit D0 does not contain any NaN values
inputs = remove_unavailable_first_visits(inputs)

# Remove patients with wrong measurements
inputs = remove_false_measurements(inputs)

# Append targets to each visit
targets = inputs.join(data, on='id', rsuffix='_t')

# Big list of interesting targets
targets_list = [
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
    'X82_Alive28d',
    'X83_Death28d',
    'X84_Intub',
    # 'Do_not_Intubate_order' NOT FOUND
]

print(inputs.max(), inputs.min())

# Forward filling to have time series of same length (only for visualization)
if FFILL:
    inputs = forward_filling(inputs)

# Drop visits with some NaN values
nan_rows = [index for index, row in inputs.iterrows() if row.isnull().any()]  # Indices of rows with NaN
if nan_rows:
    inputs = inputs.drop(nan_rows)  # Drop rows NaN value(s)
    targets = targets.drop(nan_rows)  # Fit targets to have the same dimension
    targets = targets[targets_list]  # Separate target variables from time series variables
else:
    targets = targets[targets_list]  # Separate target variables from time series variables

# Add 'dse' to targets
targets.insert(0, 'Days_since_Enrol', inputs.index.get_level_values(1))

# Cast some target columns to categorical
for cat_col in [
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
    'X82_Alive28d',
    'X83_Death28d',
    'X84_Intub'
]:
    targets[cat_col] = pd.Categorical(targets[cat_col])

print(targets.dtypes)

# Save to pickle
nb_patients = inputs.groupby('id').head(1).shape[0]
print("Remaining patients after pre-processing: ", nb_patients)
inputs.to_pickle(os.path.join(PROCESSED_PATH, 'inputs_control.pkl'))
targets.to_pickle(os.path.join(PROCESSED_PATH, 'targets_control.pkl'))
print(inputs, targets)
print(inputs.max(), inputs.min())
