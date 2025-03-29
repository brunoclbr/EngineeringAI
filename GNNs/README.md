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

#### **How to Run:**
```bash
uvicorn predict:app --reload
