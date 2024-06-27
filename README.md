# Generative-learning-assisted Rapid State-of-Health Estimation for Sustainable Battery Recycling with Random Retirement Conditions
Rapid and accurate pretreatment for state of health (SOH) estimation in retired batteries is crucial for recycling sustainability. Data-driven approaches, while innovative in SOH estimation, require exhaustive data collection and are sensitive to retirement conditions. Here we show that the generative machine learning strategy can alleviate such a challenge, validated through a unique dataset of 2700 retired lithium-ion battery samples, covering 3 cathode material types, 3 physical formats, 4 capacity designs, and 4 historical usages. With generated data, a simple regressor realizes an accurate pretreatment, with mean absolute percentage errors below 6%, even under unseen retirement conditions.

# 1. Setup
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.11.5
* numpy=1.26.4
* tensorflow=2.15.0
* keras=2.15.0
* matplotlib=3.9.0
* scipy=1.13.1
* scikit-learn=1.3.1
* pandas=2.2.2

# 2. Datasets
* We physically tested 270 retired lithium-ion batteries, covering 3 cathode types, 4 historical usages, 3 physical formats, and 4 capacity designs. See more details on [Pulse-Voltage-Response-Generation](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation).
## 2.1 Battery Types
|Cathode Material|Nominal Capacity (Ah)|Physical Format|Historical Usage|Quantity|
|:--|:--|:--|:--|:--|
|NMC|2.1|Cylinder|Lab Accelerated Aging|67 (from 12 physical batteries)|
|LMO|10|Pouch|HEV1|95|
|NMC|21|Pouch|BEV1|52|
|LFP|35|Square Aluminum Shell|HEV2|56|

# 3. Experiment
## 3.1 Settings
* Python file "configuration" contains all the hyperparameters. Change these parameters to choose battery type, model size and testing conditions.
```python
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
```
## 3.2 Run
* After changing the experiment settings, __run `main.py` directly.__
* The experiment contains two parts:
    * Leverage generative machine learning to generate data under unseen retirement conditions based on already-measured data.
    * Use the generated data to supervise a random forest regressor which estimates the battery SOH.

# 4. Experiment Details
The entire experiment consists of three steps: 
* Design and train the Conditional-VAE (CVAE) model.
* Latent space scaling and sampling to generate the data.
* Perform downstream tasks by using generated data.

First, we design a VAE model with attention mechanism. Then, we select the SOC values for training and filter the corresponding data from folder __data__ to train the VAE. After obtaining the VAE model, we perform scaling on the latent space informed by prior knowledge and sample from the scaled latent space to generate data. Finally, we use the generated data to train a random forest model to predict SOH.
## 4.1 VAE with cross attention for data generation
To allow the network to focus on relevant aspects of the voltage response matrix $x$ conditioned by the additional retirement condition information $cond$, we introduced the attention mechanism in both the encoder and decoder of the VAE. Here, we use the encoder as an example to illustrate.

The encoder network in the variational autoencoder is designed to process and compress input data into a latent space. It starts by taking the 21-dimensional battery voltage response feature matrix $x$ as main input and retirement condition matrix of the retired batteries $cond=[SOC,SOH]$ as conditional input. The condition input is first transformed into an embedding $C$, belonging to a larger latent space with 64-dimension. The conditional embedding $C$ is formulated as: 
$$C = \text{ReLU} \left( cond \cdot W_c^T + b_c \right)$$
where, $W_c$, $b_c$ are the condition embedding neural network weighting matrix and bias matrix, respectively. Here is the implementation:
```python
  # Embedding layer for conditional input (SOC + SOH)
    condition_input = Input(shape=(condition_dim,))
    condition_embedding = Dense(embedding_dim, activation='relu')(condition_input)
    condition_embedding_expanded = tf.expand_dims(condition_embedding, 2)
```
The main input matrix $x$, representing battery pulse voltage response features, is also transformed into this 64-dimensional latent space:
$$H = \text{ReLU} \left( x \cdot W_h^T + b_h \right)$$
where,  $W_h$, $b_h$ are the main input embedding neural network weighting matrix and bias matrix, respectively. Here is the implementation:
```python
  # Main input (21-dimensional features)
    x = Input(shape=(feature_dim,))
    # VAE Encoder
    h = Dense(intermediate_dim, activation='relu')(x)
    h_expanded = tf.expand_dims(h, 2)
```

Both $H$ and $C$ are then integrated via a cross-attention mechanism, allowing the network to focus on relevant aspects of the voltage response matrix $x$ conditioned by the additional retirement condition information $cond$:
$$AttenEncoder = \text{Attention}(H,C,C)$$ 
Here is the implementation:
```python
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
```
The primary function of the Decoder Network is to transform the sampled latent variable $z$ back into the original dataspace, reconstructing the input data or generating new data attended on the original or unseen retirement conditions. The first step in the decoder is a dense layer that transforms $z$ into an intermediate representation:
$$H^{'} = \text{ReLU} \left( z \cdot W_d^T + b_d \right)$$
```python
    # VAE Decoder
    z_input = Input(shape=(latent_dim,))
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(feature_dim, activation='sigmoid')
    h_decoded = decoder_h(z_input)
    h_decoded_expanded = tf.expand_dims(h_decoded, 2)
```
$H^{'}$ is then integrated via a cross-attention mechanism, allowing the network to focus on relevant aspects of the voltage response matrix $x$ conditioned by the additional retirement condition information $cond$:
$$AttenDecoder = \text{Attention}(H^{'},C^{'},C^{'})$$ 
```python
    # Cross-attention in Decoder
    attention_to_decoded = MultiHeadAttention(num_heads, key_dim=embedding_dim)(
        query=h_decoded_expanded,
        key=condition_embedding_expanded,
        value=condition_embedding_expanded
    )
    attention_output_decoded_squeezed = tf.squeeze(attention_to_decoded, 2)
    _x_decoded_mean = decoder_mean(attention_output_decoded_squeezed)
    decoder = Model(inputs=[z_input, condition_input], outputs=_x_decoded_mean)
```
With both the encoder and the decoder, the construction of the VAE model is
```python
    # VAE Model
    _, _, z = encoder([x, condition_input])
    vae_output = decoder([z, condition_input])
    vae = Model(inputs=[x, condition_input], outputs=vae_output)
```

See the Methods section of the paper for more details.
## 4.2 Latent space scaling and sampling to generate the data
After training the VAE model, it is necessary to sample its latent space to generate new data. This section will specifically explain how to perform scaling and sampling in the latent space.
### 4.2.1 Latent space scaling informed by prior knowledge
Certain retirement conditions, e.g., extreme SOH and SOC can be under-represented in the battery recycling pretreatment due to practical constraints. Specifically, the retired batteries exhibit concentrated SOH and SOC, leading to poor estimation performance when confronted with out-of-distribution (OOD) batteries.  This phenomenon results from the fact that retired electric vehicle batteries are collected in batches with similar historical usages and, thus similar SOH conditions. With a stationary rest following, the voltage values of the collected retired batteries are discharged lower than a certain threshold due to the safety concerns of the battery recyclers, resulting in a stationary rest SOC lower than 50%. Even if the explicit battery retirement conditions are still unknown, we can use this approximated prior knowledge to generate enough synthetic data to cover the actual retirement conditions. 

Given two data generation settings, namely, interpolation and extrapolation, we use different latent space scaling strategies. In the interpolation setting, the scaling matrix $T$ is an identity matrix $I$ assuming the encoder network and decoder network can learn the inherited data structures without taking advantage of any prior knowledge. In the extrapolation setting, however, the assumption cannot be guaranteed due to the OOD issue, a general challenge of machine learning models. Here we use the means of training and testing SOC distributions to define the scaling matrix, a prior knowledge of the battery retirement conditions, then the latent space is scaled as:
$$z_{\text{mean}}^{'} = T_{\text{mean}} \cdot z_{\text{mean}}$$
$$z_{\text {log-var }}^{'}=T_{\text {log-var}} \cdot z_{\text {log-var}}$$
where, $T_{\text{mean}}$ and $T_{\text{log-var}}$ are the scaling matrices defined by the broadcasted mean, and variance ratio between the testing and training SOC distributions. We emphasize that the SOH distributions are irrelevant to such a scaling. This is because these identical SOH values could be seen as representing physically distinct batteries, i.e., they do not affect the scaling process. Thus, feeding the model with the same SOH values during training and reconstruction does not present an OOD problem. On the other hand, for the SOC dimension, our goal is to generate data under unseen SOC conditions, where physical tests cannot be exhausted.
### 4.2.2 Sampling in the scaled latent space 
The sampling step in the VAE is a bridge between the deterministic output of the encoder neural network and the stochastic nature of the scaled latent space. It allows the model to capture the hidden structure of the input data, specifically the pulse voltage response $x$ and $cond$ to explore similar data points. The sampling procedure can be formulated as:
$$z = z_{\text{mean}} + e^{\frac{1}{2}z_{\text {log-var }}} \cdot \boldsymbol{\epsilon}$$
where, $\boldsymbol{\epsilon}$, is a Gaussian noise vector sampled from $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The exponential term $e^{\frac{1}{2}z_{\text {log-var }}}$ turns the log variance vector to a positive variance vector. $z$ is the sampled latent variable. 

The implementation of data generation process based on latent space scaling and sampling is as follows.
```python
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
```
## 4.3 Random forest regressor for SOH estimation
Since the data has been generated, the next step is to use the generated data to predict the SOH. We use the generated data to train a random forest model to predict SOH，and the random forest for regression can be formulated as:
$$\overline{y} = \overline{h}(\mathbf{X}) = \frac{1}{K} \sum_{k=1}^{K} h(\mathbf{X}; \vartheta_k, \theta_k)$$
where $\overline{y}$ is the predicted SOH value vector. $K$ is the tree number in the random forest. $\vartheta_k$ and $\theta_k$ are the hyperparameters. i.e., the minimum leaf size and the maximum depth of the $k$ th tree in the random forest, respectively. In this study, the hyperparameters are set as equal across different battery retirement cases, i.e., $K=20$ , $\vartheta_k=1$, and $\theta_k=64$, for a fair comparison with the same model capability. 

The Implementations are based on the ensemble method of the Sklearn Package (version 1.3.1) in the Python 3.11.5 environment, with a random state at 0.
```python
    # Phase 2: Train Model on Generated Data for Selected Testing SOC
    model_phase2 = RandomForestRegressor(n_estimators=20,max_depth=64,bootstrap=False).fit(X_generated, SOH_generated)
    y_pred_phase2 = model_phase2.predict(X_test)
    mape_phase2, std_phase2 = mean_absolute_percentage_error(y_test, y_pred_phase2)
```
# 5. Access
Access the raw data and processed features [here](https://zenodo.org/uploads/11671216) under the [MIT licence](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation/blob/main/LICENSE). Correspondence to [Terence (Shengyu) Tao](terencetaotbsi@gmail.com) and CC Prof. [Xuan Zhang](xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
# 6. Acknowledgements
[Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) and [Zixi Zhao](zhaozx23@mails.tsinghua.edu.cn)  at Tsinghua Berkeley Shenzhen Institute designed the model and algorithms, developed and tested the experiments, uploaded the model and experimental code, revised the testing experiment plan, and wrote this instruction document based on supplementary materials.  

