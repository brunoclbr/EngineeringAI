# **GNNs for Adsorption Energy Prediction in Catalyst Layers**

## **Project Overview**

### **Why This Project?**
In fuel cell technology, selecting the right catalyst composition is crucial for optimizing performance and durability. The **Catalyst-Coated Membrane (CCM)** plays a vital role in facilitating the **Oxygen Reduction Reaction (ORR)**, which directly impacts the efficiency of fuel cells. Traditionally, computational methods like **Density Functional Theory (DFT)** are used to predict **adsorption energies** (how well a molecule binds to a catalyst). However, DFT simulations are computationally expensive and time-consuming, making large-scale catalyst screening impractical.

To address this challenge, I explored the intersection of **quantum chemistry, electrochemistry, and deep learning**, leading to the development of a **Graph Neural Network (GNN) model**. This model predicts adsorption energies rapidly, making high-throughput catalyst screening feasible. The result is a machine-learning-powered tool that helps researchers identify optimal catalyst compositions more efficiently.

---

## **How It Works**

### **Machine Learning Approach**
- The project uses a **Hybrid GNN model**, which combines atomic and structural features to predict adsorption energies.
- Training data consists of catalyst surfaces, their **Miller indices**, and different **adsorbates** (e.g., OH*, O*, H*).
- The model learns to approximate DFT-calculated adsorption energies while significantly reducing computational costs.
- The predictions can be used to construct **volcano plots**, which help determine the best catalyst compositions for ORR in fuel cells.

---

## **Project Structure**
The repository contains three key Python scripts that form the backbone of the project:

### **1. `main.py` - Model Training & MLflow Integration**
This script is responsible for training the GNN model and logging the results using **MLflow** for experiment tracking.
- Defines and trains the **Hybrid GNN model**.
- Logs **hyperparameters**, **training metrics**, and the **trained model** to MLflow.
- Saves the best-performing model, which is later used for inference.

### **2. `predict.py` - FastAPI Inference Server**
This script turns the trained model into a **REST API** using FastAPI, enabling real-time predictions.
- Loads the trained model from **MLflow**.
- Accepts catalyst composition, Miller indices, and adsorbate as inputs via a **POST request**.
- Returns predicted adsorption energies in **electron volts (eV)**.
- Supports **batch predictions** and can generate **parity plots** comparing model predictions to DFT values.

Here you can see some of the prelimnary results of the test set (data not involved in training/validation) of this approach:

<p align="center">
  <img src="https://github.com/brunoclbr/EngineeringAI/blob/main/GNNs/images/parity_plot.png?raw=true" width="450" alt="results">
</p>

#### **How to Run:**
```bash
uvicorn predict:app --host 0.0.0.0 --port 8000
```

### **3. `cat_web_app.py` - Streamlit Web Interface**
This script provides a user-friendly **web application** for the research team (no programming knowledge).

- Users can **input different catalyst compositions, miller indices, and adsorbates** for single-sample predictions
- To do: implement FastAPI's bacth prediction options:
    - Support for **CSV file uploads** for batch predictions.
    - Display of results **instantly**, including **parity plots** for batch comparisons.

The web page looks like this:

<p align="center">
  <img src="https://github.com/brunoclbr/EngineeringAI/blob/main/GNNs/images/web_app.png?raw=true" width="450" alt="results">
</p>

#### **How to Run:**
```bash
streamlit run cat_web_app.py
```

## **How to Use This Project**

### **1. Set Up the Environment**
Ensure you have Python installed, then install the required dependencies:
```bash
pip install torch mlflow fastapi uvicorn streamlit pandas matplotlib requests
```
### **2. Train the Model (Optional, if not using a pre-trained model)**

Run the training script to generate a new model:

```bash
python main_torch.py
```
### **3. Start the FastAPI Backend**

This serves the model as an API for real-time predictions:

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000

```
### **4. Launch the Streamlit Web App**

To make predictions via a web interface:

```bash
streamlit cat_web_app.py

```
---

## **Future Improvements**

- Extending model training with larger DFT datasets.
- Deploying the FastAPI backend on a **cloud server** for public access.
- Enhancing the Streamlit UI with **data visualization** and **volcano plot** generation.

---
