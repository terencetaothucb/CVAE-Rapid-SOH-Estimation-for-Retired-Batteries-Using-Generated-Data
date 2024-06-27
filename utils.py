import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100, np.std(np.abs((y_true - y_pred) / y_true)) * 100

def split_SOC_data(condition, Fts, train_SOC_values):

    # Assuming SOC is the first column
    train_mask = np.isin(condition[:, 0], train_SOC_values)
    # print("train_mask",train_mask)
    test_mask = ~train_mask
    # print("test_mask",test_mask)
    train_Fts = Fts[train_mask]
    train_condition = condition[train_mask]
    test_Fts = Fts[train_mask + test_mask]
    test_condition = condition[train_mask + test_mask]
    test_condition2 = condition[test_mask]

    return train_Fts, train_condition, test_Fts, test_condition, test_condition2, test_mask

# --- Load Data ---
def load_data(hyperparams):
    if (hyperparams['battery'] == "NMC2.1"):
        data = pd.read_excel('battery_data/NMC_2.1Ah_W_3000.xlsx', sheet_name="Sheet1")
    if (hyperparams['battery'] == "LFP"):
        data = pd.read_excel('battery_data/LFP_35Ah_W_3000.xlsx', sheet_name="SOC ALL")
    if (hyperparams['battery'] == "LMO"):
        data = pd.read_excel('battery_data/LMO_10Ah_W_3000.xlsx', sheet_name="SOC ALL")
    if (hyperparams['battery'] == "NMC21"):
        data = pd.read_excel('battery_data/NMC_21Ah_W_3000.xlsx', sheet_name="SOC ALL")


    data['SOC'] /= 100  # Normalize SOC by dividing by 100
    # Filter data to include only rows where SOC is less than or equal to 50% (0.5 after normalization)
    filtered_data = data[data['SOC'] <= 0.5]
    print("Data shape after filtering:", filtered_data.shape)

    features = filtered_data.loc[:, 'U1':'U21'].values
    soc = filtered_data['SOC'].values
    soh = filtered_data['SOH'].values

    print("Features shape:", features.shape)
    print("SOC shape:", soc.shape)
    print("SOH shape:", soh.shape)

    # Combining SOC and SOH as conditional input
    condition = np.column_stack((soc, soh))
    return features, condition, filtered_data