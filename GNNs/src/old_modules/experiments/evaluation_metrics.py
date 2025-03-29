import numpy as np

# Assume y_test_true and y_test_pred are numpy arrays
mae = np.mean(np.abs(y_test_true - y_test_pred))  # MAE
mean_abs_test = np.mean(np.abs(y_test_true))      # Mean of true values

# Relative error as a percentage
relative_error = (mae / mean_abs_test) * 100

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((y_test_true - y_test_pred)**2))

# R-squared
ss_total = np.sum((y_test_true - np.mean(y_test_true))**2)
ss_residual = np.sum((y_test_true - y_test_pred)**2)
r_squared = 1 - (ss_residual / ss_total)

# Print results
print(f"MAE: {mae}")
print(f"Relative Error (%): {relative_error}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")