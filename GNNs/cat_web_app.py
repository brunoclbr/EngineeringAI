import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json

# FastAPI backend URL
API_URL = "http://localhost:8000"

st.title("GNN Adsorption Energy Predictor")

# Dropdown menus for user selection
catalyst = st.selectbox("Select Catalyst", ["Pt", "PtAg", "PtAu", "Pd", "Ag"])
miller_indices = st.selectbox("Select Miller Indices", ["[1 1 1]", "[1 1 0]", "[1 0 0]"])
adsorbate = st.selectbox("Select Adsorbate", ["OH*", "O*", "H*"])

st.subheader("Enter Input Features")
x1_input = st.text_area("Enter x1 features (comma-separated)")
x2_input = st.text_area("Enter x2 features (comma-separated)")
graph_x = st.text_area("Graph Node Features (comma-separated)")
graph_edge_index = st.text_area("Graph Edge Index (comma-separated pairs)")
graph_edge_attr = st.text_area("Graph Edge Attributes (comma-separated)")

if st.button("Predict"):
    try:
        # Convert input text to lists
        x1 = [float(i) for i in x1_input.split(",")]
        x2 = [float(i) for i in x2_input.split(",")]
        graph_data = {
            "x": [list(map(float, graph_x.split(",")))],
            "edge_index": [list(map(int, graph_edge_index.split(",")))],
            "edge_attr": [list(map(float, graph_edge_attr.split(",")))]
        }

        # Send request to FastAPI
        response = requests.post(f"{API_URL}/predict/", json={
            "catalyst": catalyst,
            "miller_indices": miller_indices,
            "adsorbate": adsorbate,
            "x1": x1,
            "x2": x2,
            "graph_data": graph_data
        })

        result = response.json()
        st.success(f"Predicted Adsorption Energy: {result['adsorption_energy_eV']} eV")
    except Exception as e:
        st.error(f"Error: {e}")

st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        samples = df.to_dict(orient='records')
        response = requests.post(f"{API_URL}/batch_predict/", json={"samples": samples, "generate_plot": True})
        result = response.json()

        if "parity_plot" in result:
            st.image("parity_plot.png", caption="Parity Plot")
        st.write("Batch Prediction Results:")
        st.write(result)
