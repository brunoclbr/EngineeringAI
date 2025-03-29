import tensorflow as tf
import numpy as np
import tensorflow_gnn as tfgnn

def build_graph_tf(product_atoms, product_bonds, catalyst_nodes, miller_indices):
    """
    Build a graph for a single reaction with product atoms, catalyst, and adsorption edges.

    Args:
    - product_atoms (np.ndarray): Node features for product atoms (shape: [num_product_atoms, feature_dim]).
    - product_bonds (list of tuples): Edges between product atoms (list of (src, dst)).
    - catalyst_nodes (np.ndarray): Node features for catalyst (shape: [num_catalyst_nodes, feature_dim]).
    - miller_indices (np.ndarray): Miller indices (edge feature vector for adsorption edges).

    Returns:
    - tfgnn.GraphTensor: A graph representation compatible with TensorFlow GNN.
    """
    # Number of nodes
    num_product_atoms = len(product_atoms)
    num_catalyst_nodes = len(catalyst_nodes)

    # Combine node features: product + catalyst
    node_features = tf.constant(
        np.vstack([product_atoms, catalyst_nodes]).astype(np.float32)
    )

    # Define product bonds (edges within the product)
    product_edge_sources, product_edge_targets = zip(*product_bonds)

    # Adsorption edges: connect each product atom to each catalyst node
    adsorption_edges = [
        (i, num_product_atoms + j)
        for i in range(num_product_atoms)
        for j in range(num_catalyst_nodes)
    ]
    adsorption_edge_sources, adsorption_edge_targets = zip(*adsorption_edges)

    # Combine edges
    edge_sources = tf.constant(product_edge_sources + list(adsorption_edge_sources), dtype=tf.int32)
    edge_targets = tf.constant(product_edge_targets + list(adsorption_edge_targets), dtype=tf.int32)

    # Edge features:
    # - Zero vectors for product bonds
    # - Miller indices for adsorption edges
    num_product_edges = len(product_bonds)
    product_edge_features = tf.zeros([num_product_edges, len(miller_indices)], dtype=tf.float32)
    adsorption_edge_features = tf.tile(
        tf.expand_dims(tf.constant(miller_indices, dtype=tf.float32), axis=0),
        [len(adsorption_edges), 1],
    )
    edge_features = tf.concat([product_edge_features, adsorption_edge_features], axis=0)

    # Create a graph schema
    node_set = tfgnn.NodeSet.from_fields(
        features={"atom_features": node_features}, sizes=[len(node_features)]
    )
    edge_set = tfgnn.EdgeSet.from_fields(
        features={"bond_features": edge_features},
        sizes=[len(edge_sources)],
        adjacency=tfgnn.Adjacency.from_indices(
            source=("atoms", edge_sources),
            target=("atoms", edge_targets),
        ),
    )
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={"atoms": node_set}, edge_sets={"bonds": edge_set}
    )

    return graph


# Example usage
# Product atoms (5 atoms, 3 features each)
product_atoms = np.random.rand(5, 3)

# Bonds between product atoms
product_bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]

# Catalyst nodes (3 nodes, 4 features each)
catalyst_nodes = np.random.rand(3, 4)

# Miller indices (edge feature for adsorption)
miller_indices = np.array([1, 0, 0])  # Example: Miller index [1, 0, 0]

# Build the graph
graph = build_graph_tf(product_atoms, product_bonds, catalyst_nodes, miller_indices)

# Print the graph structure
print(graph)
