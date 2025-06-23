import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib


scaler = joblib.load('hybrid_scaler.pkl')
model = keras.models.load_model('gnn_hybrid.keras')
matplotlib.use('Agg')
test_data_path = os.path.join(os.getcwd(), 'test_data_features.pkl')
test_data_target_path = os.path.join(os.getcwd(), 'test_data_target.pkl')
test_set = joblib.load(test_data_path)
Y1_test = joblib.load(test_data_target_path)

y_test_pred = scaler.inverse_transform(model.predict(test_data_path))
# Convert Y1_test (tf.data.Dataset) to a NumPy array
y_test_true = np.concatenate([y.numpy() for y in Y1_test], axis=0)

# Flatten the array
y_test_true = y_test_true.reshape(-1)

# Print statistics for Y1_test
print("Y1_test shape:", y_test_true.shape)
print("Standard Deviation:", np.std(y_test_true))
print("Variance:", np.var(y_test_true))
print("Mean:", np.mean(y_test_true))
print("Min and Max:", [min(y_test_true), max(y_test_true)])
print("Min and Max Predictions:", [min(y_test_pred), max(y_test_pred)])

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.absolute(y_test_true - y_test_pred))
mean_abs_test = np.mean(np.absolute(y_test_true))

# Normalized error (relative to data range)
data_range = np.max(y_test_true) - np.min(y_test_true)
range_normalized_error = (mae / data_range) * 100

# Relative error as a percentage
relative_error = (mae / mean_abs_test) * 100

# Parity plot
fig, ax = plt.subplots()
ax.scatter(y_test_true, y_test_pred, s=10, label='Predictions')

# Add statistics to the plot
plt.text(0.05, 0.85, f'MAE: {mae:.2f} eV', transform=plt.gca().transAxes)
plt.text(0.05, 0.81, f'MAE w.r.t. Mean [%]: {relative_error:.2f} %', transform=plt.gca().transAxes)
plt.text(0.05, 0.77, f'MAE w.r.t. Range [%]: {range_normalized_error:.2f} %', transform=plt.gca().transAxes)
plt.text(0.05, 0.73, f'Mean: {np.mean(y_test_true):.2f}, Std: {np.std(y_test_true):.2f}',
         transform=plt.gca().transAxes)

# Add 45-degree line
line = np.linspace(min(y_test_true.min(), y_test_pred.min()),
                   max(y_test_true.max(), y_test_pred.max()), 100)
ax.plot(line, line, color='red', linestyle='--', label='y = x')

# Plot formatting
plt.xlabel('Adsorption Energy DFT (eV)')
plt.ylabel('Adsorption Energy NN (eV)')
plt.title('Parity Plot')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("/presentations/results/parity_plot.png")
plt.show()

# GET SOMEHOW OR CONNECT WITH MAIN, PROBABLY IMPORT THIS SCRIPT THERE
# scaler = StandardScaler()  # this should be done after separating train/test
# y_test_pred = scaler.inverse_transform(nn.predict(artificial_alloy))

print(f'The adsorption energy of OH on NiFeCoMoW is: {y_test_pred[0][0]} [eV]')
