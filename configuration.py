# Define Hyperparameters and Constants
hyperparams = {
    'battery': 'NMC2.1', # NMC2.1，NMC21，LMO，LFP
    'file_path': 'battery_data/NMC_2.1Ah_W_3000.xlsx',
    'sampling_multiplier': 1,
    'feature_dim': 21,  # Dimension of the main input features
    'condition_dim': 2,  # Dimension of the conditional input (SOC + SOH)
    'embedding_dim': 64,
    'intermediate_dim': 64,
    'latent_dim': 2,
    'batch_size': 32,
    'epochs': 50,
    'num_heads': 1,
    'train_SOC_values': [0.05, 0.15, 0.25, 0.35, 0.45, 0.50],  # SOC values to use for training
    'all_SOC_values': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],  # All SOC values in the dataset
    'mode': 3,  # when case > 3, interpolation ends; set mode to 99 for only interpolation, to -1 for only extrapolation
}

train_SOC_values_cases = [
    [0.05, 0.15, 0.25, 0.35, 0.45, 0.50],  # case0
    [0.05, 0.10, 0.30, 0.40, 0.50],  # case1
    [0.05, 0.25, 0.40, 0.50],  # case2
    [0.05, 0.25, 0.50],  # case3
    [0.35, 0.40, 0.45, 0.50],  # case4
    [0.20, 0.25, 0.30],  # case5
    [0.05, 0.10, 0.15, 0.20],  # case6
    [0.05, 0.10]  # case7
]