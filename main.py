# -*- coding: utf-8 -*-

# --- Import Libraries ---
import os
import tensorflow as tf
from Conditional_VAE import generate_data, conditional_vae
from downstream_task import *
from configuration import hyperparams, train_SOC_values_cases

# --- Set Seed for Reproducibility ---

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)  # For numpy
    tf.random.set_seed(seed)  # For tensorflow
    tf.keras.utils.set_random_seed(seed) # For keras
    tf.config.experimental.enable_op_determinism()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":

    # Set seeds for reproducibility
    set_seeds()
    all_mape_results_phase1 =[]
    all_mape_results_phase2 =[]
    # Loop through the cases

    for case_index, train_SOC_values in enumerate(train_SOC_values_cases):
        print(f"\nRunning case {case_index}")
        print(f"Training SOC levels: {train_SOC_values}")
        # Load data and prepare it for the VAE
        Fts, condition, data = load_data(hyperparams)
        train_Fts, train_condition, test_Fts, test_condition, test_condition2, test_mask = split_SOC_data(condition, Fts, train_SOC_values)
        print("train_Fts shape:", train_Fts.shape)
        print("train_condition shape:", train_condition.shape)
        # Build the VAE
        vae, encoder, decoder = conditional_vae(
            feature_dim=hyperparams['feature_dim'],
            condition_dim=hyperparams['condition_dim'],
            embedding_dim=hyperparams['embedding_dim'],
            intermediate_dim=hyperparams['intermediate_dim'],
            latent_dim=hyperparams['latent_dim'],
            num_heads=hyperparams['num_heads']
        )

        # Generate data
        generated_data, generated_Fts, repeated_conditions_denormalized, history, train_generated_features = generate_data(
            vae,
            train_Fts,
            train_condition,
            test_condition,  # Pass the test conditions here
            encoder,
            decoder,
            hyperparams['sampling_multiplier'],
            batch_size=hyperparams['batch_size'],
            epochs=hyperparams['epochs'],
            latent_dim=hyperparams['latent_dim']
        )
        masked_generated_Fts, masked_generated_SOH, masked_generated_Fts_dir, masked_generated_SOH_dir = preprocess(
            case_index, hyperparams, test_condition, test_Fts, train_SOC_values, train_SOC_values_cases, generated_Fts)


        print(len(generated_data),len(masked_generated_Fts),len(masked_generated_SOH))

        # Run SOH experiments and print results
        mape_results_phase1, mape_results_phase2, std_results_phase1, std_results_phase2 = run_SOH_experiments(
            masked_generated_Fts_dir, masked_generated_SOH_dir,
            case_index, hyperparams,
            train_SOC_values_cases[case_index],
            hyperparams['all_SOC_values'], data, generated_data, masked_generated_Fts, test_condition)

        all_mape_results_phase1.append(mape_results_phase1)
        all_mape_results_phase2.append(mape_results_phase2)

        # Print the results for the current case
        print(f"\nResults for Case {case_index}:")
        print("MAPE Phase 1:", mape_results_phase1)
        print("MAPE Phase 2:", mape_results_phase2)
        print("Standard Deviation Phase 1:", std_results_phase1)
        print("Standard Deviation Phase 2:", std_results_phase2)
