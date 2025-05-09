# Wind Speed Prediction using LSTM

## Introduction

Wind speed prediction (WSP) is essential for optimizing the performance and maintenance of wind turbines, as it directly influences power generation. Accurate WSP can significantly improve energy efficiency and reduce maintenance costs. However, wind speed is inherently variable and depends on a range of weather conditions, such as pressure, temperature, and humidity.

### Motivation

Brazil's vast geographical area and diverse climate pose unique challenges for wind power forecasting. In this project, we leverage a large time series dataset from Vitoria, Brazil, containing hourly weather data (e.g., wind speed, pressure, temperature). Due to the sequential nature of the data and the need to capture long-term dependencies, we chose Long Short-Term Memory (LSTM) networks, a specialized form of Recurrent Neural Networks (RNNs), to model and predict wind speed.

## Dataset Description

The dataset contains hourly weather data from 2012 to 2014, including:

* Wind Speed (wdsp)
* Pressure (stp)
* Temperature (tmax, tmin)
* Humidity (hmax, hmin)
* Various other meteorological features

### Preprocessing

* Data Imputation: Missing values are replaced using the mean of each feature.
* Normalization: Standard scaling and Min-Max scaling are applied.
* Time Series Generation: Sequences of 120 hours (5 days) are created, predicting the wind speed 24 hours after the sequence ends.

## Model Architecture

The model uses an LSTM network implemented with Keras and TensorFlow:

* **Input Layer:** Takes sequences of shape (120, number of features)
* **LSTM Layer:** 32 units with recurrent dropout
* **Dropout Layer:** Dropout rate of 0.5 to prevent overfitting
* **Dense Layer:** Output a single value (wind speed)
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Metrics:** Mean Absolute Error (MAE)


## Results

The model predicts wind speed with high accuracy over validation and test datasets. It captures long-term dependencies efficiently, leveraging LSTM's ability to retain memory over extended periods.


## Contributions

This project demonstrates the application of LSTM networks for wind speed forecasting, a critical task for wind power generation. Accurate predictions can lead to optimized turbine operation and maintenance, ultimately contributing to the efficient use of renewable energy resources.

