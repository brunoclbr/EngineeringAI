import numpy as np
from spektral.layers import GCNConv, GlobalAveragePooling
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate
from keras.models import Model, Sequential

## IF I DEFINE ALL 3 FEATURES I HAVE AS NODES INTERACTINC WITH EACH OTHER
# Example usage:
def encode_product(product, num_products):
    """One-hot encode a product name."""

    fixed_products = ['CH2CH2', 'HO2', 'HO', 'O', 'CH', 'CH2C', 'CHCH', 'CH3', 'CH3CH2', 'CH3CH', 'CH2CH', 'CH3C',
                      'CO', 'C6H6', 'H', 'C2H6', 'C3H8', 'CH4', 'H2O', 'I', 'NO', 'NH3', 'C', 'H3O', 'OH', 'N',
                      'CO2', 'OOH', 'SH', 'S', 'CH2', 'NH', 'CH3CH2CH3', 'CH3CHCH2', 'CH3CCH', 'Ru', 'COOH', 'HCOO',
                      'COH', 'CHO', 'OCCOH', 'Cu', 'OCCO', 'OCCHO', 'C3H7O3', 'C3H6O3', 'C3H8O3', 'O2', 'LiO2',
                      'Li2O2', 'LiO', 'NaO2', 'NaO', 'Na2O2', 'KO2', 'K2O2', 'KO', '', '@LS-133', '@HS-133', 'CHCO',
                      'CCO', 'CHOH', 'Re', 'OCHO', 'CHC', 'CH3O', 'Rh', 'Ir', 'Pt', 'Ag', 'Pd', 'Os', 'Au', 'Zn',
                      'Ni', 'SiH2', 'NH2', 'NNH', 'HN', 'NNH2', 'SiH3', 'SiHCl', 'SiCl2', 'SiH2Cl', 'CHCl2',
                      'SiHCl2', 'CH2Cl', 'CCl2', 'H2N', 'CCl', 'CHCl', 'SiCl3', 'CCl3', 'SiCl', 'Si', 'SiH', 'HS',
                      'CHCHCH3', 'CCH2CH3', 'CH2CH2CH3', 'CH2CH3', 'C2H3', 'CH3CH3', 'CCH3', 'CHCH2CH3', 'C5H11',
                      'C6H12', 'C28H28P2', 'C28H30P2', 'C18H37N', 'C18H38N', 'OCHCH2O', 'Co', 'Zr', 'Rb', 'Y', 'Sr',
                      'Nb', 'Tc', 'Cd', 'Mo', 'OHOH', 'OHOOH', 'OHO', 'OCCH2O', 'CH3CHO', 'HCN', 'CH3CH2OH',
                      'OCHCHO', 'OCH2CH2O']

    product_to_idx = {prod: idx for idx, prod in enumerate(sorted(fixed_products))}
    product_idx = product_to_idx.get(product, -1)  # Map product to its index
    if product_idx == -1:
        raise ValueError(f"Unknown product: {product}")
    one_hot = np.zeros(num_products, dtype=np.float32)
    one_hot[product_idx] = 1.0
    return one_hot


def preprocess_reaction_data(reactions, product_encoding_func=encode_product, num_products=131):
    """
    Preprocess each reaction into a graph structure with nodes and edges.

    Args:
        reactions (list of dict): List of reactions, where each reaction is a dictionary:
                                  {'alloy': ..., 'miller': ..., 'product': ...}.
        product_encoding_func (function): Function to encode product features.
        num_products (int): Total number of products (for one-hot encoding size).

    Returns:
        node_features (list of np.ndarray): List of node feature matrices for each reaction graph.
        adjacency_matrices (list of np.ndarray): List of adjacency matrices for each reaction graph.
    """
    node_features = []
    adjacency_matrices = []

    for reaction in range(0, len(reactions[0]['product'])):
        # Extract features for the nodes
        alloy_features = np.array(reactions[0]['alloy'][reaction], dtype=np.float32)  # Example: Alloy feature vector
        miller_features = np.array(reactions[0]['miller'][reaction], dtype=np.float32)  # Example: Miller indices vector
        product_features = product_encoding_func(reactions[0]['product'][reaction], num_products)  # Encode product as one-hot

        # Create the node feature matrix (stack all node features)
        reaction_node_features = np.stack([alloy_features, miller_features, product_features], axis=0)
        node_features.append(reaction_node_features)

        # Create a simple adjacency matrix (fully connected graph with uniform weights)
        adj_matrix = np.ones((3, 3))  # 3 nodes for each graph
        np.fill_diagonal(adj_matrix, 0)  # No self-loops
        adjacency_matrices.append(adj_matrix)

    return node_features, adjacency_matrices


# Example reaction data
reactions = [
    {'alloy': [0.5, 0.5, 0.0, ...], 'miller': [1, 0, 0, ...], 'product': 'COOH'},
    {'alloy': [0.2, 0.8, 0.0, ...], 'miller': [1, 1, 1, ...], 'product': 'OH'},
]

# Preprocess
num_products = 131
node_features, adjacency_matrices = preprocess_reaction_data(reactions, num_products)

# Convert to TensorFlow tensors
node_features = [tf.convert_to_tensor(f, dtype=tf.float32) for f in node_features]
adjacency_matrices = [tf.convert_to_tensor(a, dtype=tf.float32) for a in adjacency_matrices]


def gnn(alloy_dense, miller_dense, feature_dim):
    # Define GNN Layers
    gnn_input_node_features = Input(shape=(3, feature_dim), name="node_features")  # 3 nodes per reaction

    gnn_input_adjacency = Input(shape=(3, 3), name="adjacency_matrix")
    alloys_input = Input(shape=(61,), name="alloys")
    miller_indices_input = Input(shape=(4,), name="miller_indices")

    x = GCNConv(64, activation="relu")([gnn_input_node_features, gnn_input_adjacency])
    x = GCNConv(32, activation="relu")([x, gnn_input_adjacency])
    x = GlobalAveragePooling()(x)  # Combine node outputs into a single vector

    # Rest of your model
    concatenated_inputs = Concatenate()([alloy_dense, miller_dense, x])
    output = Dense(1, activation="linear")(concatenated_inputs)

    model = Model(inputs=[alloys_input, miller_indices_input, gnn_input_node_features, gnn_input_adjacency], outputs=output)
    # Compile the Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    # Model Summary
    model.summary()
