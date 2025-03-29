import os
import optuna
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.metrics import mean_absolute_error
from src.model_torch import Hybrid, GraphEncoder
from src.data_processing import EDA, PreProcess
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.models import infer_signature
import pickle
import joblib


def custom_collate_fn(batch):
    # Separate the batch into features (X1, X2), targets (Y), and graphs (X3)
    batch1, batch2, y_batch, batch3 = [], [], [], []

    for sample in batch:
        x1, x2 = sample[0]  # Features
        y = sample[1]  # Target
        x3 = sample[2]  # Graph data

        batch1.append(x1)
        batch2.append(x2)
        y_batch.append(y)
        batch3.append(x3)

    # Stack the features and target into batches
    batch1 = torch.stack(batch1)
    batch2 = torch.stack(batch2)
    y_batch = torch.stack(y_batch)

    # Use PyG's Batch to handle batching of graph data
    batch3 = Batch.from_data_list(batch3)  # Use Data.from_data_list to create a batch of graphs

    return (batch1, batch2), y_batch, batch3

class CombinedDataset(Dataset):
    # Custom Dataset (without calling iter() on PyGDataLoader, which was returning an error)
    def __init__(self, feat1, feat2, trgt, graph_data):
        self.feat1 = feat1
        self.feat2 = feat2
        self.trgt = trgt
        self.graph_data = graph_data  # Graph data (X3)

    def __len__(self):
        return len(self.trgt)

    def __getitem__(self, idx):
        # Return the features, target and the corresponding graph data
        return (self.feat1[idx], self.feat2[idx]), self.trgt[idx], self.graph_data[idx]


# --- Explanatory Data Analysis ---
def run_eda():

    # Load the data
    if False:
        cat_data = pd.read_excel('../small_data.xlsx')
    else:
        cat_data = pd.read_excel('../cat_data.xlsx')  # REMEMBER TO CHOOSE FULL_PROCESSED_DATA.PKL AND FULL.TFRECORDS
    #  EDA

    eda = EDA(cat_data)
    alloys = eda.alloy_elements
    products = eda.products_df
    miller = eda.miller
    energy = eda.energy

    data = {
            'alloys': alloys,
            'miller': miller,
            'products': products,
            'energy': energy
        }

    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(data, f)


# load EDA data, preprocess it and train model
def load_and_preprocess(serialize_bool=False):
    with open('full_processed_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    alloys, miller, products, energy = (loaded_data['alloys'], loaded_data['miller'],
                                        loaded_data['products'], loaded_data['energy'])

    prepsd_data = PreProcess(alloys, miller, products, energy, serialize_bool)

    print('TENSOR MANAGEMENT BEGINS')

    valid_mask = prepsd_data.mask_tensor
    prepsd_data.alloy_tensor = prepsd_data.alloy_tensor[valid_mask]
    prepsd_data.miller_tensor = prepsd_data.miller_tensor[valid_mask]
    prepsd_data.energy_tensor = prepsd_data.energy_tensor[valid_mask]

    # Convert numpy arrays to PyTorch tensors
    X1 = torch.tensor(prepsd_data.alloy_tensor, dtype=torch.float32).clone().detach()  # Shape: (134, 32)
    X2 = torch.tensor(prepsd_data.miller_tensor, dtype=torch.float32).clone().detach()  # Shape: (134, 4)
    Y1 = torch.tensor(prepsd_data.energy_tensor, dtype=torch.float32).clone().detach().reshape(-1, 1)  # Shape: (134, 1)

    def split_list(dataset):
        n = len(dataset)
        split1 = int(n * 0.80)  # First 80%
        split2 = int(n * 0.85)  # Next 5% (up to 85%)

        #list1 = slicing_data[:split1]  # First 80%
        #list2 = slicing_data[split1:split2]  # Next 5%
        #list3 = slicing_data[split2:]  # Last 15%

        return dataset[:split1], dataset[split1:split2], dataset[split2:]

    # Split into train/val/test with same proportions as in graph_preprocess.py
    X1_tr, X1_test, X1_val = split_list(X1)
    X2_tr, X2_test, X2_val = split_list(X2)
    Y1_tr, Y1_test, Y1_val = split_list(Y1)

    # Normalize the target variable (Y1) with training set
    scaler = StandardScaler()
    Y1_tr = torch.tensor(scaler.fit_transform(Y1_tr.reshape(-1, 1)), dtype=torch.float32).clone().detach()
    joblib.dump(scaler, 'torch_scaler.pkl')
    Y1_val = torch.tensor(scaler.transform(Y1_val.reshape(-1, 1)), dtype=torch.float32).clone().detach()
    Y1_test = torch.tensor(scaler.transform(Y1_test.reshape(-1, 1)), dtype=torch.float32).clone().detach()

    # Load preprocessed PyG graphs (converted earlier)
    train_path = os.path.join(os.getcwd(), "..", "torch_data", "train_20.pt")
    val_path = os.path.join(os.getcwd(), "..", "torch_data", "val_20.pt")
    test_path = os.path.join(os.getcwd(), "..", "torch_data", "test_20.pt")

    # Load graphs. They don't need parsing with valid_mask as graphs with non-valid molecules were not saved
    X3_tr = torch.load(train_path, weights_only=False)
    X3_val = torch.load(val_path, weights_only=False)
    X3_test = torch.load(test_path, weights_only=False)

    # Custom Dataset (without calling iter() on PyGDataLoader, which was returning an error)
    combined_train_dataset = CombinedDataset(X1_tr, X2_tr, Y1_tr, X3_tr)
    combined_val_dataset = CombinedDataset(X1_val, X2_val, Y1_val, X3_val)
    combined_test_dataset = CombinedDataset(X1_test, X2_test, Y1_test, X3_test)

    # Define batch size
    batch_size = 132
    # DataLoader with shuffle=True ensures consistent shuffling
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # save data loaders
    joblib.dump(train_loader, 'train_loader.pkl')
    joblib.dump(val_loader, 'val_loader.pkl')
    joblib.dump(test_loader, 'test_loader.pkl')


def compute_metrics(y_true, y_model):
    """
        Compute MAE and R² Score for given true and predicted values.
    """
    y_true = y_true.detach().cpu().numpy()
    y_model = y_model.detach().cpu().numpy()

    mae = mean_absolute_error(y_true, y_model)
    mean = np.mean(y_true)
    ss_res = np.sum(abs(y_true-y_model))**2
    ss_tot = np.sum(y_true-mean)**2
    r2 = 1 - ss_res/ss_tot

    return mae, r2


def train_and_eval(model, optimizer, loss_fn, training_loader, validation_loader):

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    # Early stopping parameters
    patience = 3  # Number of consecutive increasing epochs before stopping
    counter = 0
    best_val_loss = float('inf')  # Set to a very high value initially
    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        train_y_true = []
        train_y_pred = []

        # Wrap train_loader with tqdm for a progress bar
        train_loader_tqdm = tqdm(training_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)

        # Training batch loop
        for (x1_batch, x2_batch), y_batch, x3_batch in train_loader_tqdm:
            optimizer.zero_grad()  # Zero the gradients
            # Forward pass
            pred = model(x1_batch, x2_batch, x3_batch)
            # Compute the loss
            loss = loss_fn(pred, y_batch)
            # Backpropagation
            loss.backward()
            optimizer.step()
            # Accumulate loss
            running_train_loss += loss.item()
            train_y_true.append(y_batch)
            train_y_pred.append(pred)

            # Update tqdm progress bar with loss
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        # Compute average training loss per batch in one epoch
        perbatch_avgtrain_loss = running_train_loss / len(train_loader_tqdm)
        train_losses.append(perbatch_avgtrain_loss)

        print(f"Epoch {epoch + 1}: Avg Train Loss Per Batch = {perbatch_avgtrain_loss:.4f}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        all_y_true = []
        all_y_pred = []
        # Disable gradient calculation for validation
        with torch.no_grad():
            for (x1_batch, x2_batch), y_batch, x3_batch in validation_loader:
                # Forward pass through the model
                pred = model(x1_batch, x2_batch, x3_batch)
                # Compute the validation loss
                val_loss = loss_fn(pred, y_batch)
                running_val_loss += val_loss.item()
                # Collect true and predicted values for metrics
                all_y_true.append(y_batch)
                all_y_pred.append(pred)

        # Convert lists to tensors
        # EVEN IF I USE MAE THIS DOESN'T HAVE eV AS UNIT BECAUSE Y IS SCALED!!!
        train_y_true = torch.cat(train_y_true, dim=0)
        train_y_pred = torch.cat(train_y_pred, dim=0)

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        # Compute metrics
        train_mae, _ = compute_metrics(train_y_true, train_y_pred)
        val_mae, _ = compute_metrics(all_y_true, all_y_pred)

        train_metrics.append(train_mae)
        val_metrics.append(val_mae)

        # Compute average validation loss per batch in one epoch
        perbatch_avgval_loss = running_val_loss / len(validation_loader)
        print(f"Epoch {epoch + 1}: Avg Val Loss Per Batch = {perbatch_avgval_loss:.4f}")
        val_losses.append(perbatch_avgval_loss)

        # Early stopping logic
        if perbatch_avgval_loss < best_val_loss:
            best_val_loss = perbatch_avgval_loss  # Update best loss
            counter = 0  # Reset counter
        else:
            counter += 1  # Increase counter if val_loss is not improving
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                print(counter)
                break  # Stop training

    return train_losses, train_metrics, val_losses, val_metrics, val_losses[epoch-counter], epoch-counter

# --- Optuna HP-optimization ---


def hp_tuning(tr_loader, vl_loader):

    tr_loader = tr_loader
    vl_loader = vl_loader

    def objective(trial):
        # Suggest hyperparameters to tune
        lr_opt = trial.suggest_float("lr", 1e-7, 1e-5)
        dropout_optuna = trial.suggest_float("dropout", 0.3, 0.5)
        emb = trial.suggest_int("emb", 4, 16)
        # n_l2_opt = trial.suggest_int("n_l2", 274, 562)
        # n_l3_opt = trial.suggest_int("n_l3", 12, 273)
        # hidden_dim = trial
        # .suggest_int("hidden_dim", 32, 256, step=32)

        # Create the model with suggested hyperparameters
        gnn_model_optuna = GraphEncoder(embedding_dim=emb)
        hybrid_model_optuna = Hybrid(gnn_model=gnn_model_optuna, dropout_rate=dropout_optuna)

        # Define optimizer and loss function
        optimizer_tuna = torch.optim.Adam(hybrid_model_optuna.parameters(), lr=lr_opt)

        loss_fn_tuna = torch.nn.MSELoss()

        # Train and evaluate model
        _, _, _, _, opt_val_loss, _ = train_and_eval(hybrid_model_optuna, optimizer_tuna,
                                                     loss_fn_tuna, tr_loader, vl_loader)

        return opt_val_loss  # Optuna minimizes validation loss

    study = optuna.create_study(direction="minimize")  # We want to minimize validation loss
    study.optimize(objective, n_trials=2)  # Run 50 trials
    # Print the best hyperparameters
    print("Best Hyperparameters:", study.best_params)


if __name__ == '__main__':

    # --- Config Booleans ---
    eda_bool = False
    serialize_data = False  # saves molecules2graphs as tfrecords in cwd, move to src.
    # Change output_file_names accordingly in graph_preprocess.py line 223
    train_bool = True
    hp_bool = False

    if train_bool:
        load_and_preprocess(serialize_data)

    # --- Load Data Loaders ---

    with open('train_loader.pkl', 'rb') as f:
        train_loader = pickle.load(f)

    with open('val_loader.pkl', 'rb') as f:
        val_loader = pickle.load(f)

    if hp_bool:
        hp_tuning(train_loader, val_loader)

    # --- Model Instantiation ---

    # Define HP (Replace with config file?)
    params = {"lr": 1e-6,
              "dropout_rate": 0.5,
              "atomic_number_embedding": 12,
              "latent_graph_dim": 32}

    # Define GraphEncoder (GNN) model first
    gnn_model = GraphEncoder(embedding_dim=params["atomic_number_embedding"],
                             latent_feat_dim=params["latent_graph_dim"])

    # Define the complete Hybrid model
    hybrid_model = Hybrid(gnn_model, dropout_rate=params["dropout_rate"])

    optimizer_real = torch.optim.Adam(hybrid_model.parameters(), lr=params["lr"])

    loss_fn_real = torch.nn.MSELoss()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- TRAIN ---
    mlflow_tr_loss, mlflow_tr_metr, mlflow_val_loss, mlflow_val_metr, _, epochs = train_and_eval(hybrid_model,
                                                                                                 optimizer_real,
                                                                                                 loss_fn_real,
                                                                                                 train_loader,
                                                                                                 val_loader)

    # --- MLFLOW ---
    # mlflow ui --host 127.0.0.1 --port 5000
    # https://www.youtube.com/watch?v=6ngxBkx05Fs
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Second Experiment")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Training Info", "GNNs for adsorption energy on catalyst surfaces")

        for epo in range(epochs):

            mlflow.log_metric(key="Train Loss", value=mlflow_tr_loss[epo], step=epo)
            mlflow.log_metric(key="Train MAE", value=mlflow_tr_metr[epo], step=epo)
            mlflow.log_metric(key="Validation Loss", value=mlflow_val_loss[epo], step=epo)
            mlflow.log_metric(key="Validation MAE", value=mlflow_val_metr[epo], step=epo)

        # IT MAKES SENSE TO DO THIS WITH BEST MODEL USE IF WITH BEST LOSS
        # Infer signature
        # Dummy graph input - Convert PyG tensors to lists

        # Dummy graph input - Convert PyG tensors to lists
        dummy_graph_data = {
            "x": torch.randn(2, 2).tolist(),  # Convert to Python list
            "edge_index": torch.randint(0, 2, (2, 1)).tolist(),
            "edge_attr": torch.randn(2, 2).tolist()
        }

        # Convert other tensors to lists
        sample_input = [
            torch.randn(1, 61).tolist(),  # Convert tensor to list
            torch.randn(1, 4).tolist(),
            dummy_graph_data
        ]

        # Infer signature with scalar output (must be float, not tensor)
        signature = infer_signature(sample_input, [0.0])

        # Log model with MLflow
        mlflow.pytorch.log_model(
            hybrid_model,
            "hybrid_model_artifact_path",
            signature=signature,
            #input_example=sample_input  # Now fully JSON-compatible
        )

        mlflow.register_model(model_uri='runs:/d27eef3991114170942c4a034a0f0648/model_name',
                              name="hybrid_gnn_model")
