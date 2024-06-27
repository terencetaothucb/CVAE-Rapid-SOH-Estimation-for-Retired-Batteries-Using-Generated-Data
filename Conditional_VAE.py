import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MultiHeadAttention, Embedding, Flatten
from sklearn.preprocessing import MinMaxScaler


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1, seed=0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def conditional_vae(feature_dim, condition_dim, embedding_dim, intermediate_dim, latent_dim, num_heads):
    # Embedding layer for conditional input (SOC + SOH)
    condition_input = Input(shape=(condition_dim,))
    condition_embedding = Dense(embedding_dim, activation='relu')(condition_input)
    condition_embedding_expanded = tf.expand_dims(condition_embedding, 2)

    # Main input (21-dimensional features)
    x = Input(shape=(feature_dim,))
    # VAE Encoder
    h = Dense(intermediate_dim, activation='relu')(x)
    h_expanded = tf.expand_dims(h, 2)

    # Cross-attention in Encoder
    attention_to_encode = MultiHeadAttention(num_heads, key_dim=embedding_dim)(
        query=h_expanded,
        key=condition_embedding_expanded,
        value=condition_embedding_expanded
    )
    attention_output_squeezed = tf.squeeze(attention_to_encode, 2)

    z_mean = Dense(latent_dim)(attention_output_squeezed)
    z_log_var = Dense(latent_dim)(attention_output_squeezed)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(inputs=[x, condition_input], outputs=[z_mean, z_log_var, z])

    # VAE Decoder
    z_input = Input(shape=(latent_dim,))
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(feature_dim, activation='sigmoid')
    h_decoded = decoder_h(z_input)
    h_decoded_expanded = tf.expand_dims(h_decoded, 2)

    # Cross-attention in Decoder
    attention_to_decoded = MultiHeadAttention(num_heads, key_dim=embedding_dim)(
        query=h_decoded_expanded,
        key=condition_embedding_expanded,
        value=condition_embedding_expanded
    )
    attention_output_decoded_squeezed = tf.squeeze(attention_to_decoded, 2)
    _x_decoded_mean = decoder_mean(attention_output_decoded_squeezed)
    decoder = Model(inputs=[z_input, condition_input], outputs=_x_decoded_mean)

    # VAE Model
    _, _, z = encoder([x, condition_input])
    vae_output = decoder([z, condition_input])
    vae = Model(inputs=[x, condition_input], outputs=vae_output)

    # VAE Loss
    xent_loss = feature_dim * mse(x, vae_output)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    w_xent = 0.5
    w_kl = 0.5
    vae_loss = K.mean(w_xent * xent_loss + w_kl * kl_loss)
    vae.add_loss(vae_loss)
    vae.add_metric(xent_loss, name='xent_loss', aggregation='mean')
    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    vae.compile(optimizer=Adam())
    return vae, encoder, decoder

def generate_data(vae, train_features, train_condition, test_condition, encoder, decoder, sampling_multiplier, batch_size, epochs, latent_dim):
    # Normalize feature data (training)
    feature_scaler = MinMaxScaler().fit(train_features)
    train_features_normalized = feature_scaler.transform(train_features)

    # Combine training and testing conditional data for scaling
    combined_conditions = np.vstack([train_condition, test_condition])
    # Normalize conditional data (training and testing using the same scaler)
    condition_scaler = MinMaxScaler().fit(combined_conditions)
    train_condition_normalized = condition_scaler.transform(train_condition)
    test_condition_normalized = condition_scaler.transform(test_condition)
    # Fit the VAE model using training data
    history = vae.fit([train_features_normalized, train_condition_normalized], train_features_normalized,
                      epochs=epochs, batch_size=batch_size, verbose=0)
    # Generate new samples based on testing conditions
    num_samples = len(test_condition_normalized) * sampling_multiplier
    print("num_samples",num_samples)
    random_latent_values_new = K.random_normal(shape=(num_samples, latent_dim), seed=0)
    random_latent_values_train = K.random_normal(shape=(len(train_condition_normalized) * sampling_multiplier, latent_dim), seed=0)

    # Use the testing conditional input for generating data
    repeated_conditions = np.repeat(test_condition_normalized, sampling_multiplier, axis=0)

    new_features_normalized = decoder.predict([random_latent_values_new, repeated_conditions])

    # Denormalize the generated feature data
    generated_features = feature_scaler.inverse_transform(new_features_normalized)

    repeated_conditions_train = np.repeat(train_condition_normalized, sampling_multiplier, axis=0)

    train_features_normalized = decoder.predict([random_latent_values_train, repeated_conditions_train])

    # Denormalize the generated feature data
    train_generated_features = feature_scaler.inverse_transform(train_features_normalized)

    train_generated_features = np.vstack([train_generated_features, generated_features])

    # Denormalize the repeated conditions to return them to their original scale
    repeated_conditions_denormalized = condition_scaler.inverse_transform(repeated_conditions)
    # Combine generated features with their corresponding conditions for further analysis
    generated_data = np.hstack([generated_features, repeated_conditions_denormalized])

    return generated_data, generated_features, repeated_conditions_denormalized, history, train_generated_features