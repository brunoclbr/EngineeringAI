from fastapi import FastAPI
import torch
from src.data_processing import PreProcess
from src.serialize_torch import smiles_to_graph
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

# Initialize FastAPI app
app = FastAPI(title="API for Adsorption Energy Prediction")

# Load the trained model from MLflow
mlflow.set_tracking_uri("http://localhost:5000")
MODEL_URI = 'runs:/9e089241b7834f9899560884e1a13493/hybrid_model_artifact_path'
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()  # Set to evaluation mode

# Load the scaler
scaler = joblib.load('torch_scaler.pkl')  # pickle.load(f)


#--- Define input data structure as a Pydantic Model. For catalyst definition a pd.DF is expected in the form:
#                Concentration
#0                 {'Ag': 1.0} --> this dict for single prediction
#1      {'Au': 0.5, 'Co': 0.5} --> from here on for batch prediction
class PredictionInput(BaseModel):
    """
    team should be able to compute adsorptions energies in the most efficient and simple way. just follow the format
        {"catalyst": {"Ni": 0.2, "Fe": 0.2, "Co": 0.2, "Mo": 0.2, "W": 0.2},
        "miller_indices": [1, 1, 1, 0],
        "adsorbate": "[OH]"}
    """
    # main_torch expects a str for catalyst composition/adsorbate and then handles transformation automatically.
    catalyst: dict  # e.g., {'Ag': 1.0}
    miller_indices: list  # e.g., "[1, 1, 1, 0]"
    adsorbate: str  # e.g., "OH" --> SMILES NOTATION and then to Chem.mol
    # since input data needs to be preprocessed to a non-(human)-readable format, I added these output variables
    catalyst_output: dict | None = None
    adsorbate_output: str | None = None

    def to_input_format(self):  # here I should use the methods defined in pre_processing
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
    file_path: str  # Path to test_loader.pkl
    generate_plot: bool = False  # Whether to generate a parity plot

# Predefined dropdown options
#AVAILABLE_CATALYSTS = ["Pt", "PtAg", "PtAu", "Pd", "Ag"]
#AVAILABLE_MILLER_INDICES = ["[1 1 1]", "[1 1 0]", "[1 0 0]"]
#AVAILABLE_ADSORBATES = ["OH*", "O*", "H*"]

#@app.get("/")
#def home():
#    return {"message": "GNN Adsorption Energy Prediction API"}

@app.post("/predict/")
def predict(data: PredictionInput):
    try:
        data.to_input_format()  # Convert x1 and x2 to valid input format
        # Convert inputs to tensors
        x1_tensor = torch.tensor(data.catalyst, dtype=torch.float32)
        x2_tensor = torch.tensor(data.miller_indices, dtype=torch.float32).unsqueeze(0)
        graph_data = data.adsorbate
        print(f"x1: {x1_tensor.shape}, n/ x2: {x2_tensor.shape} n/ x3: {graph_data} ")

        # Run inference
        y_pred = model(x1_tensor, x2_tensor, graph_data)
        y_pred = y_pred.detach().cpu().numpy().reshape(-1, 1)

        # Apply inverse transformation using the scaler
        y_pred = y_pred.reshape(-1, 1)
        ads_energy_nn = scaler.inverse_transform(y_pred)

        # Save results to CSV log
        log_data = {
            "catalyst": str(data.catalyst_output),
            "miller_indices": str(data.miller_indices),
            "adsorbate": data.adsorbate_output,
            "predicted_energy [eV]": float(ads_energy_nn[0][0])
        }
        df = pd.DataFrame([log_data])  # this will add every predicted output
        df.to_csv("prediction_log.csv", mode='a',
                  header=not os.path.exists("prediction_log.csv"), index=False)

        # Debugging: Check if log_data is JSON serializable
        try:
            json.dumps(log_data)  # This should not raise an error
        except TypeError as e:
            print("JSON Serialization Error:", e)

        # Return log_data in FastAPI route
        return log_data

    except Exception as e:
        return {"error": str(e)}


@app.post("/batch_predict/")
def batch_predict(batch_data: BatchPredictionInput):
    try:
        # Load the test loader
        with open(batch_data.file_path, "rb") as file:
            test_loader = pickle.load(file)
            all_y_true = []
            all_y_pred = []
            running_val_loss = 0.0

            loss_fn = torch.nn.MSELoss()

            # Disable gradient computation for inference
            with torch.no_grad():
                for (x1_batch, x2_batch), y_batch, x3_batch in test_loader:
                    # Forward pass
                    pred = model(x1_batch, x2_batch, x3_batch)

                    # Compute batch validation loss
                    val_loss = loss_fn(pred, y_batch)
                    running_val_loss += val_loss.item()

                    # Collect true and predicted values
                    all_y_true.append(y_batch)
                    all_y_pred.append(pred)

            # Convert lists to tensors
            all_y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
            all_y_pred = torch.cat(all_y_pred, dim=0).cpu().numpy()

            # Apply inverse transformation (assuming predictions are scaled)
            all_y_pred = scaler.inverse_transform(all_y_pred.reshape(-1, 1))
            all_y_true = scaler.inverse_transform(all_y_true.reshape(-1, 1))

            # Compute Metrics
            mae = float(mean_absolute_error(all_y_true, all_y_pred))
            std_dev = float(np.std(all_y_pred))

            results = {
                "MAE (eV)": mae,
                "Standard Deviation": std_dev,
                "Validation Loss": running_val_loss / len(test_loader),
            }

            # Generate Parity Plot
            if batch_data.generate_plot:
                plt.figure(figsize=(6, 6))
                plt.scatter(all_y_true, all_y_pred, label="Predictions", color="blue", edgecolors="black", s=20,
                            alpha=0.7)
                plt.plot(all_y_true, all_y_true, 'r', linestyle="--", linewidth=2, label="Ideal Fit")
                plt.xlabel("DFT Values (eV)", fontsize=12)
                plt.ylabel("Neural Network Values (eV)", fontsize=12)
                plt.title("Parity Plot", fontsize=14, fontweight="bold")
                plt.text(min(all_y_true), max(all_y_pred), f"MAE: {mae:.3f} eV\nStd Dev: {std_dev:.3f} eV",
                         fontsize=10, verticalalignment="top",
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
                plt.legend()
                plt.savefig("parity_plot.png", dpi=300, bbox_inches="tight")
                results["plot_path"] = "parity_plot.png"

            return {"mae": mae, "std_dev": std_dev}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # once running open http://127.0.0.1:8000/docs → Auto-generated Swagger UI to test API.
    uvicorn.run(app, host="0.0.0.0", port=8000)
