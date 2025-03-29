from rdkit import Chem
from google.protobuf import text_format
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import os

# Define the graph schema
# context {
#   features {
#     key: "label"
#     value {
#       description: "compound entropy"
#       dtype: DT_FLOAT
#     }
#   }
# }
schema_ptbx = """
node_sets {
  key: "atoms"
  value {
    features {
      key: "atomic_number"
      value {description: "atomic_number representing atom type" dtype: DT_FLOAT shape { dim { size: 118 } } }
      }
      features {
      key: "electro_negativity"
      value {description: "Defines electron-pulling ability" dtype: DT_FLOAT shape { dim { size: 1 } } }
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
        shape { dim { size: 4 } }
      }
    }
  }
}
"""
graph_schema = text_format.Merge(schema_ptbx, schema_pb2.GraphSchema())
graph_tensor_spec1 = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

# Convert SMILES string to graph with padding for one "big sparse molecule"
def smiles_to_graph(mol: object, label: float=0.1, max_nodes: int = 50, max_edges: int = 100):
    """
    DOUBLE CHECK PADDED MOLECULE, AM I TRULY GETTING ONE BIGGER, SPARSE MOLECULE OR DOES IT HAVE VARIABLE SIZE?
    :param mol: str with SMILES convention encoding molecule
    :param label:  float with target property of molecule (entropy, adsorption energy etc.)
    :param max_nodes: max number of atoms permitted within molecule
    :param max_edges: max number of bonds permitted within molecule
    :return: graph tensor
    """
    #mol = Chem.MolFromSmiles(smiles)  # THIS IS THE CONNECTING PART, MAIN ALREADY RETURNS A PROCESSED mol OBJECT
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    num_bonds = adjacency_matrix.shape[0]
    def add_self_loops(adj_matrix):
        adj_with_self_loops = adj_matrix.copy()
        np.fill_diagonal(adj_with_self_loops, 1)
        return adj_with_self_loops

    #adjacency_matrix = add_self_loops(adjacency_matrix)
    # Get atomic numbers and one-hot encode them (up to 118 elements)
    max_atomic_num = 118  # THIS NUMBER IS THEORETICALLY RIGHT; MIGHT BE BETTER TO SET IT EQUAL TO THE AMOUNT OF
    # UNIQUE ATOMS IN THE PRODUCTS OF THE DATABASE
    max_electro_neg_num = max_nodes  # Electronegativity feature dimension → 1 (since it's a single scalar per atom)
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features = tf.one_hot(atom_numbers, max_atomic_num, dtype=tf.float32)
    num_atoms = tf.shape(atom_features)[0]  # Get the number of atoms (outer dimension)
    #atom_features_ragged = tf.RaggedTensor.from_tensor(atom_features, ragged_rank=1)

    # ========== ADDITIONAL ATOMIC FEATURES ==========

    # Hybridization (sp, sp2, sp3, sp3d, sp3d2)
    #hybridization_types = [Chem.rdchem.HybridizationType.SP,
    #                       Chem.rdchem.HybridizationType.SP2,
    #                       Chem.rdchem.HybridizationType.SP3,
    #                       Chem.rdchem.HybridizationType.SP3D,
    #                       Chem.rdchem.HybridizationType.SP3D2]
    #hybridization_map = {hyb: i for i, hyb in enumerate(hybridization_types)}

    #hybridization_list = [hybridization_map.get(atom.GetHybridization(), -1) for atom in mol.GetAtoms()]
    #hybridization_one_hot = tf.one_hot(hybridization_list, depth=len(hybridization_types), dtype=tf.float32)

    # Electronegativity (Normalized between 0 and 1)
    pauling_electronegativities = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16,
        35: 2.96, 53: 2.66  # Add more elements as needed
    }
    max_en = max(pauling_electronegativities.values())
    electronegativity = [pauling_electronegativities.get(atom.GetAtomicNum(), 0) / max_en for atom in mol.GetAtoms()]
    electronegativity = tf.expand_dims(tf.convert_to_tensor(electronegativity, dtype=tf.float32), axis=-1)
    #electronegativity_ragged = tf.RaggedTensor.from_tensor(electronegativity_tensor, ragged_rank=1)
    #electronegativity_tensor = tf.ragged.constant(electronegativity_tensor)
    #ragged_tensor = tf.RaggedTensor.from_tensor(electronegativity_tensor)
    #electronegativity_tensor = tf.ragged.constant(tf.convert_to_tensor(electronegativity, dtype=tf.float32))
    # Extract source and target indices from adjacency matrix
    # AM I MAKING THIS SYMMETRIC???
    #adjacency_matrix[end, start] = adjacency_matrix[start, end]
    def adjacency_to_edges(adj_matrix):
        """Convert an adjacency matrix to source and target edge indices."""
        source, target = np.where(adj_matrix > 0)
        return source, target

    source_indices, target_indices = adjacency_to_edges(adjacency_matrix)
    source_indices = np.array(source_indices)
    target_indices = np.array(target_indices)
    #source_indices = tf.reshape(tf.expand_dims(tf.convert_to_tensor(source_indices, dtype=tf.int32), axis=0), [-1])
    #source_indices = tf.RaggedTensor.from_tensor(source_indices, ragged_rank=1)
    #target_indices = tf.reshape(tf.expand_dims(tf.convert_to_tensor(target_indices, dtype=tf.int32), axis=0), [-1])
    #target_indices = tf.RaggedTensor.from_tensor(target_indices, ragged_rank=1)
    #source_indices, target_indices = np.nonzero(adjacency_matrix)

    # Bond features (simple one-hot for single, double, triple bonds)
    bond_features = tf.one_hot([int(bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()], depth=4, dtype=tf.float32)

    # Create GraphTensor
    graph = tfgnn.GraphTensor.from_pieces(
        #context=tfgnn.Context.from_fields(
        #    features={"label": tf.constant([label], dtype=tf.float32)}  # Add a label feature
        #),
        node_sets={
            "atoms": tfgnn.NodeSet.from_fields(
                sizes=[len(atom_numbers)],  # CAMBIAR ESTO A RAGGED?
                features={
                    "atomic_number": atom_features,
                    "electro_negativity": electronegativity,
                }
            )
        },
        edge_sets={
            "bonds": tfgnn.EdgeSet.from_fields(
                sizes=[num_bonds-1],
                features={"hidden_state": bond_features},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("atoms", source_indices), # these are including self-attention of nodes so i should put sizes=number of atoms insead of bonds?
                    target=("atoms", target_indices)
                )
            )
        }
    )

    return graph


# Serialize graphs
def save_graphs_to_tfrecord(mol_obj, target, output_file, schema, max_nodes=50, max_edges=100):
    """
    serialize graphs into .tfrecord files
    """
    print("IF RUNNING FROM OTHER FILE I SHOULD NOT SEE THIS")
    graph_tensors = [smiles_to_graph(smiles, label, max_nodes, max_edges) for smiles, label in zip(mol_obj, target)]

    with tf.io.TFRecordWriter(output_file) as writer:
        for graph in graph_tensors:
            tfgnn.check_compatible_with_schema_pb(graph, schema)
            serialized_example = tfgnn.write_example(graph)
            writer.write(serialized_example.SerializeToString())

    print(f"Saved {len(graph_tensors)} graphs to {output_file}")


# Decode tfrecords into graph tensors

def decode_fn(record_bytes):
    """
        Parses a single serialized tf.Example proto into a GraphTensor
        Handles ragged tensors and validates inputs.
        :param record_bytes:
        :param graph_spec:
        :return: new_graph, label
    """
    graph = tfgnn.parse_single_example(graph_tensor_spec1, record_bytes, validate=True)

    return graph


train_path = os.path.join(os.getcwd(), "TEST_molecules.tfrecord")
val_path = os.path.join(os.getcwd(), "TEST_molecules.tfrecord")

train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
val_ds = tf.data.TFRecordDataset([val_path]).map(decode_fn)
#g, y = train_ds.take(1).get_single_element()
#print(g.node_sets['atoms'])
#print(g.node_sets['atoms'].features[tfgnn.HIDDEN_STATE])

if __name__ == "__main__":

    print("Serializing Molecules in serialize.py")

    smiles_data = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1=CC=CC=C1"),
                   Chem.MolFromSmiles("CC(=O)O"), Chem.MolFromSmiles("CCO"),
                   Chem.MolFromSmiles("C1=CC=CC=C1"), Chem.MolFromSmiles("CC(=O)O")]  # Ethanol, Benzene, Acetic Acid
    labels = [0.7, 0.9, 0.5, 0.7, 0.9, 0.5]  # Example entropy values
    save_graphs_to_tfrecord(smiles_data, labels, "TEST_molecules.tfrecord", graph_schema)

    train_path = os.path.join(os.getcwd(), "TEST_molecules.tfrecord")
    val_path = os.path.join(os.getcwd(), "TEST_molecules.tfrecord")

    train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
    val_ds = tf.data.TFRecordDataset([val_path]).map(decode_fn)
    #g = train_ds.take(1).get_single_element()
    for graph in train_ds:
        print(graph.node_sets['atoms'])
        #node_set[tfgnn.HIDDEN_STATE] deberia printear esto para ver que le paso al set_initial_node_state
        print(graph.node_sets['atoms'].features["atomic_number"])
        print(graph.node_sets['atoms'].features["electro_negativity"])
