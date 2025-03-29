import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn
from tensorflow_gnn.models import graph_sage
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow_gnn.keras.layers import MapFeatures, Pool
from matplotlib import pyplot as plt
from tensorflow_gnn.models.gcn.gcn_conv import GCNConv
from rdkit import Chem
import numpy as np

# 1. DEFINE INPUTS (MOLECULES) AND OUTPUTS
# CCO is encoded as [6, 6, 8], so 3 nodes. max len of node set per graph == largest molecule. Find this out to
# create your graph_spec. With this I have to declare nodes size but accepts variable node sets.
# How do they map their molecules with the graph_schema?


mol = Chem.MolFromSmiles("CCO")  # Load ethanol C2H6O
adjacency_matrix = Chem.GetAdjacencyMatrix(mol)  # Get adjacency matrix
atom_number = []
for atom in mol.GetAtoms():
    atom_number.append(tf.convert_to_tensor(atom.GetAtomicNum(), dtype=tf.float32))  # Print atomic numbers
print(f'C2H6O atomic numbers: {atom_number}')
print(f'AM shape: {adjacency_matrix.shape} and its values: {adjacency_matrix}')

# Extract source and target indices from adjacency matrix
source_indices, target_indices = np.nonzero(adjacency_matrix)
print("Source indices:", source_indices)
print("Target indices:", target_indices)

# Construct NodeSet & EdgeSet for TensorFlow GNN. This should be replaced using graphSchema ?
# not necessarily but i can use this graphSchema to define the graphspec
# features etc expect RAGGED TENSOR
node_features = atom_number  # Shape: [3, 1]
bond_features = tf.constant([[1.0], [1.0], [1.0], [1.0]], dtype=tf.float32)  # Shape: [4, 1]

graph1 = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={"label": tf.constant([0.7], dtype=tf.float32)}  # Add a label feature
    ),
    node_sets={
        "atoms": tfgnn.NodeSet.from_fields(
            sizes=[3],
            features={"hidden_state": node_features}
        )
    },
    edge_sets={
        'bonds': tfgnn.EdgeSet.from_fields(
            sizes=[len(source_indices)],
            features={"hidden_state": bond_features},
            adjacency=tfgnn.Adjacency.from_indices(
                source=("atoms", source_indices.tolist()),
                target=("atoms", target_indices.tolist())
            )
        )
    }
)


