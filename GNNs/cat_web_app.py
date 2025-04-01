import streamlit as st
import requests
import json

# FastAPI backend URL
API_URL = "http://localhost:8000"

st.title("GNN Adsorption Energy Predictor")

# User input for prediction
st.subheader("Input Features")

# Catalyst selection (user-defined composition)
st.write("Enter Catalyst Composition (Element: Fraction)")
catalyst_input = st.text_area("Example: {'Pt': 0.5, 'Au': 0.5}")

# Miller indices selection
miller_input = st.text_area("Enter Miller Indices (comma-separated)", "1,1,1,0")
adsorbate = st.text_input("Enter Adsorbate (SMILES format)", "[OH]")

if st.button("Predict"):
    try:
        # Parse catalyst input from text to dictionary
        catalyst = json.loads(catalyst_input.replace("'", '"'))  # Ensure valid JSON format

        # Convert Miller indices to a list
        miller_indices = list(map(int, miller_input.split(",")))

        # Prepare JSON payload
        payload = {
            "catalyst": catalyst,
            "miller_indices": miller_indices,
            "adsorbate": adsorbate
        }

        # Send request to FastAPI
        response = requests.post(f"{API_URL}/predict/", json=payload)

        # Handle response
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Adsorption Energy: {result['predicted_energy [eV]']} eV")
        else:
            st.error(f"Error: {response.text}")

    except Exception as e:
        st.error(f"Invalid input format. Please check your entries.\n\nError: {e}")