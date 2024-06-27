
from sklearn.ensemble import RandomForestRegressor
import matplotlib.colors as mcolors
from Conditional_VAE import generate_data, conditional_vae
from utils import *

def run_SOH_experiment(masked_generated_Fts_dir, masked_generated_SOH_dir, case_index , hyperparams,train_SOC_values, test_SOC_index, data, generated_data, masked_generated_Fts, test_condition):
    # train_SOC_values is a list of SOC levels used for training
    # test_SOC_index is the SOC level used for testing
    # Filtering training data for the specified list of SOC levels
    train_data = data[data['SOC'].isin(train_SOC_values)]
    test_data = data[data['SOC'] == test_SOC_index]

    X_train = train_data.loc[:, 'U1':'U21'].values
    y_train = train_data['SOH'].values
    X_test = test_data.loc[:, 'U1':'U21'].values
    y_test = test_data['SOH'].values

    # Generate generated data for the test_SOC_index
    data = generated_data[test_condition[:, 0] == test_SOC_index]
    print("test_SOC_index:",test_SOC_index)

    if (case_index > hyperparams['mode']):
        # Assuming features are the first 21 columns
        X_generated = np.array(masked_generated_Fts_dir[test_SOC_index])
        SOH_generated = np.array(masked_generated_SOH_dir[test_SOC_index])

    else:
        # Assuming features are the first 21 columns
        X_generated = data[:, 0:21]
        SOH_generated = data[:, -1]


    # Phase 1: Train Model on Available Data for Training SOC Levels
    model_phase1 = RandomForestRegressor(n_estimators=20,max_depth=64,bootstrap=False).fit(X_train, y_train)
    y_pred_phase1 = model_phase1.predict(X_test)
    mape_phase1, std_phase1 = mean_absolute_percentage_error(y_test, y_pred_phase1)

    # Phase 2: Train Model on Generated Data for Selected Testing SOC
    model_phase2 = RandomForestRegressor(n_estimators=20,max_depth=64,bootstrap=False).fit(X_generated, SOH_generated)
    y_pred_phase2 = model_phase2.predict(X_test)
    mape_phase2, std_phase2 = mean_absolute_percentage_error(y_test, y_pred_phase2)

    return mape_phase1, std_phase1, mape_phase2, std_phase2

def run_SOH_experiments(masked_generated_Fts_dir, masked_generated_SOH_dir, case_index , hyperparams,train_SOC_values, all_SOC_values,data, generated_data, masked_generated_Fts, test_condition):
    mape_results_phase1 = []
    mape_results_phase2 = []
    std_results_phase1 = []
    std_results_phase2 = []
    test_SOC_indices = [soc for soc in all_SOC_values if soc not in train_SOC_values]

    for test_SOC_index in test_SOC_indices:

        mape_phase1, std_phase1, mape_phase2, std_phase2 = run_SOH_experiment(
            masked_generated_Fts_dir, masked_generated_SOH_dir,
            case_index, hyperparams,
            train_SOC_values, test_SOC_index, data, generated_data, masked_generated_Fts, test_condition
        )
        mape_results_phase1.append(mape_phase1)
        mape_results_phase2.append(mape_phase2)
        std_results_phase1.append(std_phase1)
        std_results_phase2.append(std_phase2)


    return mape_results_phase1, mape_results_phase2, std_results_phase1, std_results_phase2

def preprocess(case_index, hyperparams, test_condition, test_Fts, train_SOC_values, train_SOC_values_cases, generated_Fts):

    repeated_test_condition = np.repeat(test_condition[:, 0:2], hyperparams['sampling_multiplier'], axis=0)

    # Loop over testing SOC values and features
    # Exclude the training SOC values from the all SOC values
    test_SOCs = [soc for soc in hyperparams['all_SOC_values'] if soc not in train_SOC_values]
    test_result = []
    rate = 0
    masked_generated_Fts_dir = dict.fromkeys(test_SOCs)
    masked_generated_SOH_dir = dict.fromkeys(test_SOCs)

    for test_SOC in test_SOCs:

        rate = rate + 1
        print(f"Testing SOC: {test_SOC}")
        print(test_SOC)


        # Create a boolean mask for the specific SOC value
        mask_soc = train_SOC_values[1] if (len(train_SOC_values_cases)) != 1 else (train_SOC_values[0])
        Fts_mask = repeated_test_condition[:, 0] == mask_soc
        SOH_mask = repeated_test_condition[:, 0] == test_SOC

        # Apply the mask to generated_Fts
        masked_generated_Fts = generated_Fts[Fts_mask]
        # Apply the mask to generated SOH
        masked_generated_SOH = repeated_test_condition[SOH_mask, 1]

        test_result.append(masked_generated_Fts)
        n = len(masked_generated_Fts)
        m = len(masked_generated_Fts[0])
        # print(n,m)
        k = 0
        # physic_weight is the mean difference between the corresponding U of two adjacent SOC
        if (hyperparams['battery'] == "NMC2.1"):
            physic_weight = 0.043
        if (hyperparams['battery'] == "LFP"):
            physic_weight = 0.0295
        if (hyperparams['battery'] == "LMO"):
            physic_weight = 0.095
        if (hyperparams['battery'] == "NMC21"):
            physic_weight = 0.04

        while (k < len(masked_generated_Fts)):
            l = 0
            while (l < len(masked_generated_Fts[0])):

                if (case_index > hyperparams['mode']):
                    # 20 is the proportional coefficient
                    masked_generated_Fts[k][l] = masked_generated_Fts[k][l] + ((test_SOC - mask_soc) * physic_weight * 20)
                else:
                    break
                l = l + 1
            k = k + 1

        masked_generated_Fts_dir[test_SOC] = masked_generated_Fts
        masked_generated_SOH_dir[test_SOC] = masked_generated_SOH


    return masked_generated_Fts, masked_generated_SOH, masked_generated_Fts_dir, masked_generated_SOH_dir
