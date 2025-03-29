from fastapi import FastAPI
import torch
import mlflow.pytorch
import pandas as pd
import pickle
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from pydantic import BaseModel
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from MLflow
mlflow.set_tracking_uri("http://localhost:5000")
MODEL_URI = 'runs:/d27eef3991114170942c4a034a0f0648/hybrid_model_artifact_path'
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()  # Set to evaluation mode

# Load the scaler
with open('torch_scaler.pkl', 'rb') as f:
    print(f.read())
    scaler = joblib.load('torch_scaler.pkl')  # pickle.load(f)


# Define input data structure
class PredictionInput(BaseModel):
    catalyst: str  # e.g., "PtAg"
    miller_indices: str  # e.g., "[1 1 1 0]"
    adsorbate: str  # e.g., "OH*"
    x1: list  # 1D feature list
    x2: list  # 1D feature list
    graph_data: dict  # Graph data as dict {"x": [], "edge_index": [], "edge_attr": []}


class BatchPredictionInput(BaseModel):
    samples: list  # List of PredictionInput dictionaries
    generate_plot: bool = False  # Whether to generate a parity plot


# Predefined dropdown options
AVAILABLE_CATALYSTS = ["Pt", "PtAg", "PtAu", "Pd", "Ag"]
AVAILABLE_MILLER_INDICES = ["[1 1 1]", "[1 1 0]", "[1 0 0]"]
AVAILABLE_ADSORBATES = ["OH*", "O*", "H*"]


@app.get("/")
def home():
    return {"message": "GNN Adsorption Energy Prediction API"}


@app.post("/predict/")
def predict(data: PredictionInput):
    try:
        # Convert inputs to tensors
        x1_tensor = torch.tensor(data.x1, dtype=torch.float32)
        x2_tensor = torch.tensor(data.x2, dtype=torch.float32)
        graph_x = torch.tensor(data.graph_data["x"], dtype=torch.float32)
        graph_edge_index = torch.tensor(data.graph_data["edge_index"], dtype=torch.long)
        graph_edge_attr = torch.tensor(data.graph_data["edge_attr"], dtype=torch.float32)

        graph_data = {"x": graph_x, "edge_index": graph_edge_index, "edge_attr": graph_edge_attr}

        # Run inference
        y_pred = model(x1_tensor.unsqueeze(0), x2_tensor.unsqueeze(0), graph_data)
        y_pred = y_pred.detach().cpu().numpy().reshape(-1, 1)

        # Apply inverse transformation using the scaler
        ads_energy_nn = scaler.inverse_transform(y_pred)

        # Save results to CSV log
        log_data = {
            "catalyst": data.catalyst,
            "miller_indices": data.miller_indices,
            "adsorbate": data.adsorbate,
            "predicted_energy": ads_energy_nn[0][0]
        }
        df = pd.DataFrame([log_data])
        df.to_csv("prediction_log.csv", mode='a', header=not os.path.exists("prediction_log.csv"), index=False)

        return {"adsorption_energy_eV": ads_energy_nn[0][0]}

    except Exception as e:
        return {"error": str(e)}


@app.post("/batch_predict/")
def batch_predict(batch_data: BatchPredictionInput):
    try:
        all_preds = []
        true_values = []  # If available
        for sample in batch_data.samples:
            # Convert inputs to tensors
            x1_tensor = torch.tensor(sample.x1, dtype=torch.float32)
            x2_tensor = torch.tensor(sample.x2, dtype=torch.float32)
            graph_x = torch.tensor(sample.graph_data["x"], dtype=torch.float32)
            graph_edge_index = torch.tensor(sample.graph_data["edge_index"], dtype=torch.long)
            graph_edge_attr = torch.tensor(sample.graph_data["edge_attr"], dtype=torch.float32)

            graph_data = {"x": graph_x, "edge_index": graph_edge_index, "edge_attr": graph_edge_attr}

            # Run inference
            y_pred = model(x1_tensor.unsqueeze(0), x2_tensor.unsqueeze(0), graph_data)
            y_pred = y_pred.detach().cpu().numpy().reshape(-1, 1)

            # Apply inverse transformation
            ads_energy_nn = scaler.inverse_transform(y_pred)
            all_preds.append(ads_energy_nn[0][0])

        # Calculate metrics if true values are provided
        mae = mean_absolute_error(true_values, all_preds) if true_values else None
        std_dev = np.std(all_preds)

        results = {"predictions": all_preds, "MAE": mae, "Std Dev": std_dev}

        if batch_data.generate_plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(true_values, all_preds, color='y', label='Parity Plot')
            plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--',
                     label="Perfect Fit")
            plt.title('Parity Plot')
            plt.xlabel('DFT [eV]')
            plt.ylabel('ANN [eV]')
            plt.legend()
            plot_path = "parity_plot.png"
            plt.savefig(plot_path)
            results["parity_plot"] = plot_path

        return results
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # once running open http://127.0.0.1:8000/docs → Auto-generated Swagger UI to test API.
    uvicorn.run(app, host="0.0.0.0", port=8000)
