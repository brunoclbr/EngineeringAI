import torch
import numpy as np
import rdkit.Chem as Chem
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
#import networkx as nx
#import matplotlib.pyplot as plt
#from torch_geometric.utils import to_networkx

# Define embedding layer globally (only once)
embedding_layer = nn.Embedding(119, 16)  # 119 elements (including index 0 for padding)
# mapped to 16D vector


def smiles_to_graph(mol: Chem.Mol, label: float = 0.1):
    """
    Converts an RDKit molecule to a PyTorch Geometric Data object.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        label (float): Target value (e.g., entropy, adsorption energy).

    Returns:
        Data: A PyTorch Geometric Data object.
    """

    # 1. **Convert adjacency matrix to edge index (PyG format)**
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    source_indices, target_indices = np.where(adjacency_matrix > 0)  # Edge list

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)  # Shape (2, num_edges)
    #edge_index = torch.tensor([
    #                           [0, 0, 1, 2],  # Source nodes
    #                           [1, 2, 0, 0]   # Target nodes
    #                                       ])

    # 2. **Node Features (Atomic Numbers & Electronegativity)**
    atom_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    max_atomic_num = 118  # One-hot encoding size
    # Instead of one-hot encoding, use embeddings
    #embedded_atom_number = embedding_layer(atom_numbers)  # Shape: (num_nodes, 16)

    # One-hot encode atomic numbers
    atomic_one_hot = F.one_hot(torch.tensor(atom_numbers, dtype=torch.long), num_classes=max_atomic_num).float()

    # ========== ADDITIONAL ATOMIC FEATURES ==========

    # Hybridization (sp, sp2, sp3, sp3d, sp3d2)
    #hybridization_types = [Chem.rdchem.HybridizationType.S,
    #                       Chem.rdchem.HybridizationType.SP,
     #                      Chem.rdchem.HybridizationType.SP2,
      #                     Chem.rdchem.HybridizationType.SP3,
       #                    Chem.rdchem.HybridizationType.SP3D,
        #                   Chem.rdchem.HybridizationType.SP3D2]
    #hybridization_map = {hyb: i for i, hyb in enumerate(hybridization_types)}

    #hybridization_list = [hybridization_map.get(atom.GetHybridization(), -1) for atom in mol.GetAtoms()]
    #print(hybridization_list)
    #hybridization_one_hot = F.one_hot(torch.tensor(hybridization_list, dtype=torch.long), num_classes=6).float()

    # Electronegativity values (normalized)
    pauling_electronegativities = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16,
                                   35: 2.96, 53: 2.66}  # Extend as needed
    max_en = max(pauling_electronegativities.values())

    electronegativities = [pauling_electronegativities.get(atom.GetAtomicNum(), 0) / max_en for atom in mol.GetAtoms()]
    electronegativities = torch.tensor(electronegativities, dtype=torch.float).unsqueeze(-1)  # Shape (num_atoms, 1)

    #is_aromatic = torch.tensor([1.0 if  mol.GetAtoms().GetIsAromatic() else 0.0], dtype=torch.float)

    # Combine atomic features (Concatenating One-hot and electronegativity)
    node_features = torch.cat([atomic_one_hot, electronegativities], dim=-1)  # Shape (num_atoms, X)
    # each attribute is a vector of size 4. [
    #                                        [[,a,b,c,d],[x,y]],
    #                                        [[,a,b,c,d],[x,y]]
    #                                                           ]   --> 2 atoms, X=2 columns for nodes (2 means 2 features)

    # 3. **Edge Features (Bond Type One-hot Encoding)**
    bond_types = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}  # Single, double, triple, aromatic
    bond_features = [bond_types.get(bond.GetBondTypeAsDouble(), 0) for bond in mol.GetBonds()]

    edge_attr = F.one_hot(torch.tensor(bond_features, dtype=torch.long), num_classes=4).float()  # Shape (num_edges, 1)
    # each attribute is a vector of size 4. [[,a,b,c,d],
    #                                        [,a,b,c,d]]--> 2 edges, one column for edge_attr

    # 4. **Graph Data Object**
    """
    Node (torch.Tensor, optional) feature matrix x with shape [num_nodes, num_node_features]
    edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]
    edge_attr (torch.Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]
    """
    graph_data = Data(
        x=node_features,  # Node features
        edge_index=edge_index,  # Edge connections
        edge_attr=edge_attr,  # Edge features (bond type)
    )

    return graph_data



def save_graphs_to_torch(mol_obj, output_file):
    """
    Save graphs as a PyTorch file.
    Args:
        mol_obj (list[Chem.Mol]): List of RDKit molecules.
        target (list[float]): Target values.
        output_file (str): File path to save.
    """
    graph_list = [smiles_to_graph(mol) for mol in mol_obj]

    torch.save(graph_list, output_file)  # Save as a Torch file
    print(f"Saved {len(graph_list)} graphs to {output_file}")


if __name__ == "__main__":

    # Define iso-octane molecular graph (C8H18)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 1, 4, 1, 5, 2, 6, 2, 7, 3, 8, 3, 9, 4, 10, 4, 11,
         5, 12, 5, 13, 6, 14, 6, 15, 7, 16, 7, 17, 8, 18, 8, 19, 9, 20, 9, 21,
         10, 22, 11, 23, 12, 24, 13, 25],  # Source
        [1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 2, 7, 2, 8, 3, 9, 3, 10, 4, 11, 4,
         12, 5, 13, 5, 14, 6, 15, 6, 16, 7, 17, 7, 18, 8, 19, 8, 20, 9, 21, 9,
         22, 10, 23, 11, 24, 12, 25, 13]  # Target
    ], dtype=torch.long)

    # Node Features (Atomic number + Electronegativity)
    x = torch.tensor([
        [6, 2.55], [6, 2.55], [6, 2.55], [6, 2.55], [6, 2.55], [6, 2.55], [6, 2.55], [6, 2.55],  # 8 Carbons
        [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20],  # 18 Hydrogens
        [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20], [1, 2.20],
        [1, 2.20], [1, 2.20]
    ], dtype=torch.float)

    # Create PyG Data Object
    #data = Data(x=x, edge_index=edge_index)

    # Convert to NetworkX for visualization
    #G = to_networkx(data, to_undirected=True)

    # Plot the graph
    #plt.figure(figsize=(8, 6))
    #nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=800, font_size=10)
    #plt.title("Iso-Octane Molecular Graph (C8H18)")
    #plt.show()
    #print("Serializing Molecules in serialize.py")

    # Example SMILES: Ethanol, Benzene, Acetic Acid
    smiles_data = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1=CC=CC=C1"),
                   Chem.MolFromSmiles("CC(=O)O"), Chem.MolFromSmiles("CCO"),
                   Chem.MolFromSmiles("C1=CC=CC=C1"), Chem.MolFromSmiles("CC(=O)O")]

    #labels = [0.7, 0.9, 0.5, 0.7, 0.9, 0.5]  # Example entropy values

    # Save graphs as a Torch file
    file_name = "TEST_molecules.pt"
    save_graphs_to_torch(smiles_data, file_name)

    # Load the saved graphs
    train_graphs = torch.load(file_name, weights_only=False)

    # Verify one graph
    for graph in train_graphs:
        print("\nGraph Structure:")
        print("Node Features Shape:", graph.x.shape)
        print("Edge Index Shape:", graph.edge_index.shape)
        print("Edge Attributes Shape:", graph.edge_attr.shape)
        print("Label:", graph.y)

        # Print node features (Atomic number one-hot + electronegativity)
        print("Sample Node Features:", graph.x[:3])  # Print first 3 nodes

        # Print edge connectivity
        print("Sample Edge Index:", graph.edge_index[:, :5])  # Print first 5 edges

        break  # Only check the first graph

