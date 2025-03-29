from rdkit import Chem
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np

schema_ptbx = """
context {
  features {
    key: "label"
    value: {
      description: "compound entropy"
      dtype: DT_FLOAT
    }
  }
}
node_sets {
  key: "atoms"
  value {
    features {
      key: "hidden_state"
      value {
        description: "atomic_number representing atom type"
        dtype: DT_FLOAT
        shape {dim { size: -1 }}  # <-- Allow variable-sized tensors
      }
    }
  }
}
edge_sets {
  key: "bonds"
  value {
    source: "atoms"
    target: "atoms"
    features {
      key: "hidden_state"
      value {
        description: "bond type."
        dtype: DT_FLOAT
        shape {dim { size: -1 }}  # <-- Allow variable-sized tensors
      }
    }
  }
}
"""


# Define a function to process molecules
def smiles_to_graph(smiles: str, label: float):
    mol = Chem.MolFromSmiles(smiles)
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)

    # Get atomic numbers and create RaggedTensor for variable-sized atom features
    max_atomic_num = 118
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features = tf.one_hot(atom_numbers, max_atomic_num, dtype=tf.float32)

    # Convert atom features to RaggedTensor (variable number of atoms per molecule)
    atom_features_ragged = tf.RaggedTensor.from_tensor(atom_features, ragged_rank=1)

    # Extract source and target indices from adjacency matrix (keep them as dense tensors)
    source_indices, target_indices = np.nonzero(adjacency_matrix)

    # Bond features as RaggedTensor (variable number of bonds per molecule)
    bond_types = [int(bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()]
    bond_features = tf.one_hot(bond_types, depth=4, dtype=tf.float32)
    bond_features_ragged = tf.RaggedTensor.from_tensor(bond_features, ragged_rank=1)

    # Create GraphTensor with correct tensor types
    graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={"label": tf.constant([label], dtype=tf.float32)}
        ),
        node_sets={
            "atoms": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([len(atom_numbers)], dtype=tf.int32),
                features={"hidden_state": atom_features_ragged}
            )
        },
        edge_sets={
            "bonds": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(source_indices)], dtype=tf.int32),
                features={"hidden_state": bond_features_ragged},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("atoms", tf.constant(source_indices, dtype=tf.int32)),
                    target=("atoms", tf.constant(target_indices, dtype=tf.int32))
                )
            )
        }
    )

    return graph


def save_graphs_to_tfrecord(smiles_list, graph_labels, output_file):
    graph_tensors = [smiles_to_graph(smiles, label) for smiles, label in zip(smiles_list, graph_labels)]

    with tf.io.TFRecordWriter(output_file) as writer:
        for graph in graph_tensors:
            serialized_example = tfgnn.write_example(graph)
            writer.write(serialized_example.SerializeToString())

    print(f"Saved {len(graph_tensors)} graphs to {output_file}")


smiles_data = ["CCO", "C1=CC=CC=C1", "CC(=O)O", "CCO", "C1=CC=CC=C1", "CC(=O)O"]  # Ethanol, Benzene, Acetic Acid
labels = [0.7, 0.9, 0.5, 0.7, 0.9, 0.5]  # Example entropy values

save_graphs_to_tfrecord(smiles_data, labels, "molecules.tfrecord")
