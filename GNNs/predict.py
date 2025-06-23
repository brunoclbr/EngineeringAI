from fastapi import FastAPI, HTTPException
import torch
from src.data_processing import PreProcess
from src.utils.serialize_torch import smiles_to_graph
import mlflow.pytorch
import pandas as pd
from rdkit import Chem
import joblib
from main_torch import CombinedDataset, custom_collate_fn  # needed for loading test loader
import pickle
import json
import numpy as np
from sklearn.metrics import mean_absolute_error
from pydantic import BaseModel
import uvicorn
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-GUI backend

# --- Best Practice: Define global variables for resources but leave them empty ---
# The app will populate these variables during its startup sequence.
model = None
scaler = None

# Initialize FastAPI app
app = FastAPI(title="API for Adsorption Energy Prediction")


@app.on_event("startup")
def load_startup_resources():
    """
    This function is called when the FastAPI application starts up.
    It loads the ML model and the scaler, preventing the server from hanging
    and providing clear error messages if a resource fails to load.
    """
    global model, scaler

    print("--- FastAPI application starting up: Loading resources ---")

    # --- 1. Load the ML model from MLflow ---
    # Make sure your MLflow tracking server is running before you start this app.
    try:
        print("Connecting to MLflow tracking server...")
        mlflow.set_tracking_uri("http://localhost:5000")
        MODEL_URI = 'runs:/9e089241b7834f9899560884e1a13493/hybrid_model_artifact_path'

        print(f"Loading model from URI: {MODEL_URI}...")
        model = mlflow.pytorch.load_model(MODEL_URI)
        model.eval()  # Set to evaluation mode
        print("--- ML model loaded successfully ---")

    except Exception as e:
        # If this fails, the server will still start, but 'model' will be None.
        # The endpoints will then return an error, which is better than a silent crash.
        print(f"!!! CRITICAL ERROR: Failed to load model from MLflow.")
        print(f"!!! Please ensure the MLflow server is running at 'http://localhost:5000' and the run ID is correct.")
        print(f"!!! Error details: {e}")

    # --- 2. Load the scaler using a robust file path ---
    # This builds an absolute path to the file, making it independent of where you run the 'uvicorn' command.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(script_dir, 'pkl', 'torch_scaler.pkl')

        print(f"Loading scaler from path: {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            scaler = joblib.load(f)
        print("--- Scaler loaded successfully ---")

    except FileNotFoundError:
        print(f"!!! CRITICAL ERROR: Scaler file not found at the expected path.")
        print(f"!!! Expected path: {scaler_path}")
    except Exception as e:
        print(f"!!! CRITICAL ERROR: An error occurred while loading the scaler.")
        print(f"!!! Error details: {e}")

    print("--- Startup resource loading complete ---")


class PredictionInput(BaseModel):
    """
    team should be able to compute adsorptions energies in the most efficient and simple way. just follow the format
        {"catalyst": {"Ni": 0.2, "Fe": 0.2, "Co": 0.2, "Mo": 0.2, "W": 0.2},
        "miller_indices": [1, 1, 1, 0],
        "adsorbate": "[OH]"}
    """
    catalyst: dict
    miller_indices: list
    adsorbate: str
    catalyst_output: dict | None = None
    adsorbate_output: str | None = None

    def to_input_format(self):
        """Convert lists to NumPy arrays"""
        self.catalyst_output = self.catalyst
        catalyst = pd.DataFrame({'Concentration': [self.catalyst]})
        catalyst = PreProcess.encoded_alloy(catalyst)
        self.catalyst = np.array(catalyst, dtype=np.float32)
        self.miller_indices = np.array(self.miller_indices, dtype=np.float32)
        mol = Chem.MolFromSmiles(self.adsorbate, sanitize=False)
        self.adsorbate_output = self.adsorbate
        self.adsorbate = smiles_to_graph(mol)


class BatchPredictionInput(BaseModel):
    """
    this class is not meant to be used by the team, only for error quantification of the test set
    """
    file_path: str
    generate_plot: bool = False


@app.post("/predict/")
def predict(data: PredictionInput):
    # Add a check to ensure the model and scaler were loaded correctly on startup.
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Model or scaler is not available. Check server logs for startup errors."
        )

    try:
        data.to_input_format()
        x1_tensor = torch.tensor(data.catalyst, dtype=torch.float32)
        x2_tensor = torch.tensor(data.miller_indices, dtype=torch.float32).unsqueeze(0)
        graph_data = data.adsorbate

        # Run inference
        with torch.no_grad():  # Good practice for inference
            y_pred = model(x1_tensor, x2_tensor, graph_data)

        y_pred = y_pred.cpu().numpy().reshape(-1, 1)

        # Apply inverse transformation using the scaler
        ads_energy_nn = scaler.inverse_transform(y_pred)

        # Save results to CSV log
        log_data = {
            "catalyst": str(data.catalyst_output),
            "miller_indices": str(data.miller_indices.tolist()),  # Convert numpy array to list for JSON
            "adsorbate": data.adsorbate_output,
            "predicted_energy [eV]": float(ads_energy_nn[0][0])
        }
        df = pd.DataFrame([log_data])
        df.to_csv("prediction_log.csv", mode='a', header=not os.path.exists("prediction_log.csv"), index=False)

        return log_data

    except Exception as e:
        # Return a more specific error to the user
        raise HTTPException(status_code=400, detail=f"An error occurred during prediction: {str(e)}")


@app.post("/batch_predict/")
def batch_predict(batch_data: BatchPredictionInput):
    # Add a check to ensure the model and scaler were loaded correctly on startup.
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Model or scaler is not available. Check server logs for startup errors."
        )

    try:
        with open(batch_data.file_path, "rb") as file:
            test_loader = pickle.load(file)

        all_y_true = []
        all_y_pred = []
        running_val_loss = 0.0
        loss_fn = torch.nn.MSELoss()

        with torch.no_grad():
            for (x1_batch, x2_batch), y_batch, x3_batch in test_loader:
                pred = model(x1_batch, x2_batch, x3_batch)
                val_loss = loss_fn(pred, y_batch)
                running_val_loss += val_loss.item()
                all_y_true.append(y_batch)
                all_y_pred.append(pred)

        all_y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
        all_y_pred = torch.cat(all_y_pred, dim=0).cpu().numpy()

        all_y_pred = scaler.inverse_transform(all_y_pred.reshape(-1, 1))
        all_y_true = scaler.inverse_transform(all_y_true.reshape(-1, 1))

        mae = float(mean_absolute_error(all_y_true, all_y_pred))
        std_dev = float(np.std(all_y_pred))

        results = {
            "MAE (eV)": mae,
            "Standard Deviation": std_dev,
            "Validation Loss": running_val_loss / len(test_loader),
        }

        if batch_data.generate_plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(all_y_true, all_y_pred, label="Predictions", color="blue", edgecolors="black", s=20, alpha=0.7)
            plt.plot(all_y_true, all_y_true, 'r', linestyle="--", linewidth=2, label="Ideal Fit")
            plt.xlabel("DFT Values (eV)", fontsize=12)
            plt.ylabel("Neural Network Values (eV)", fontsize=12)
            plt.title("Parity Plot", fontsize=14, fontweight="bold")
            plt.text(min(all_y_true), max(all_y_pred), f"MAE: {mae:.3f} eV\nStd Dev: {std_dev:.3f} eV", fontsize=10,
                     verticalalignment="top", bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            plt.legend()
            plt.savefig("parity_plot.png", dpi=300, bbox_inches="tight")
            results["plot_path"] = "parity_plot.png"

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred during batch prediction: {str(e)}")


if __name__ == "__main__":
    # To run this file directly for simple testing, use: python your_script_name.py
    # For robust development, use the command: uvicorn your_script_name:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

