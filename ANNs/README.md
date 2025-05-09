# Laminar Burning Velocieties Predictions

## Introduction

This is a project designed to predict laminar burning velocities (LBV) of hydrocarbon and oxygenated hydrocarbon fuels using artificial neural networks (ANNs). The model is trained to estimate LBVs based on the fuel structure and physical conditions (pressure, temperature, equivalence ratio), leveraging a set of molecular groups to capture structural dependencies.

## Methodology

The project uses a Quantitative Structure-Property Relationship (QSPR) approach to correlate the LBVs with the molecular structure of fuels. The ANN model, implemented with Keras and TensorFlow, takes as input the molecular group data along with temperature, pressure, and equivalence ratio. Key features of the model include:

* Utilization of group contribution methods inspired by Joback and Reid (1987).
* Incorporation of non-cyclic and cyclic carbon groups, as well as oxygenated groups (alcohol, ether, carbonyl, ester).
* Data normalization for optimal training convergence.


## Usage

To train the model, create two CSV. files containing the model hyperparameters and chemistry data, respectively. Then use the following command:

```bash
python execute_plan.py --data data/fuel_data.csv --data data/hyp.csv
```

To evaluate the model:

```bash
python evaluate.py --model models/ann_model.h5
```

## Data Format

The input data file should be a CSV containing columns for molecular groups, temperature, pressure, and equivalence ratio. Example:

| Molecular Group | Temperature | Pressure | Equivalence Ratio |
| --------------- | ----------- | -------- | ----------------- |
| CH3             | 298         | 1        | 1.0               |

## Results

The trained model predicts the LBV for a given fuel under specified conditions. Prediction accuracy was validated through k-fold cross-validation using experimental data and was found to generalize well within the data range. For more details on predictive accuracy and results please visit our paper.

## References

* Original Paper: [https://www.sciencedirect.com/science/article/abs/pii/S0010218021002686]

