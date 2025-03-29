from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import tensorflow as tf
def preprocess_chemical_data(products, fixed_products):
    """
    Preprocess the chemical data for GNN.

    Args:
        products (list of str): List of product names to encode.
        fixed_products (list of str): Full list of known products (used for consistent encoding).

    Returns:
        node_features (np.ndarray): Node feature matrix of shape (num_products, num_products).
        adjacency_matrix (np.ndarray): Adjacency matrix of shape (num_products, num_products).

        For a Graph Neural Network (GNN), the data must be structured as a graph, which consists of:

    Each Reaction as a Graph:

    A node in this graph represents an individual aspect of the reaction:
    Catalyst alloy composition (e.g., Ag3Pt encoded as a node feature vector).
    Surface Miller indices (encoded as another node feature vector).
    Adsorbed product (the one-hot encoding or other features of the product).
    Each reaction forms a single graph with three nodes:
    Node 1: Alloy composition.
    Node 2: Miller indices.
    Node 3: Product.
    Edges:

    Since the nodes represent distinct entities of a reaction, all three nodes can be connected with edges.
    You can use a fully connected graph (complete graph) with arbitrary weights or even assign uniform edge weights
    (e.g., all edges = 1.0).
    Features:

    Alloy Node: Encoded features (composition, as in your existing setup).
    Miller Index Node: Encoded features (already processed via dense layers).
    Product Node: Encoded features from your encoded_product function.
    Batching Graphs:

    Since each reaction is independent, you will process a batch of graphs in the GNN during training.
    """
    # Ensure consistent ordering of fixed_products
    fixed_products = sorted(fixed_products)
    product_to_idx = {product: idx for idx, product in enumerate(fixed_products)}

    # Generate one-hot encoded features
    num_products = len(fixed_products)
    node_features = np.eye(num_products)

    # Generate adjacency matrix based on molecular similarity
    fingerprints = []
    for product in fixed_products:
        try:
            mol = Chem.MolFromSmiles(product)  # Convert product to RDKit Mol object
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # Compute fingerprint
            fingerprints.append(fp)
        except:
            # Handle molecules that cannot be processed (e.g., empty strings)
            fingerprints.append(None)

    # Compute pairwise similarities to construct adjacency matrix
    adjacency_matrix = np.zeros((num_products, num_products))
    for i in range(num_products):
        for j in range(num_products):
            if fingerprints[i] is not None and fingerprints[j] is not None:
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                adjacency_matrix[i, j] = similarity
            else:
                adjacency_matrix[i, j] = 0.0  # No similarity if fingerprint is missing

    return node_features, adjacency_matrix


fixed_products = ['CH2CH2', 'HO2', 'HO', 'O', 'CH', 'CH2C', 'CHCH', ...]  # Your full product list
products = ['COOH', 'OH', 'CH4', ...]  # Example products to encode

node_features, adjacency_matrix = preprocess_chemical_data(products, fixed_products)

# Convert to TensorFlow tensors for input
node_features = tf.convert_to_tensor(node_features, dtype=tf.float32)
adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float32)