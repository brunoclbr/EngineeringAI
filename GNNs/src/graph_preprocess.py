import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import SanitizeMol
from rdkit.Chem.MolStandardize import rdMolStandardize
import torch
from src.serialize_torch import smiles_to_graph


def preprocess_chemical_data_atomic(molecules, serialize=True):
    """
    Preprocess product DataFrames into graph-like representations where each atom is a node.

    Args:
    - molecule: DataFrame with 'ProductNames' column containing SMILES strings.
    - serialize: Boolean that tells to transform and store molecules2graph

   a molecule encodes:
    - node_features: NumPy array with atomic counts (batch_size, num_atoms).
    - adjacency_matrix: NumPy array with adjacency matrices (batch_size, num_atoms, num_atoms).
    - to ensure consistent dimensions across all molecules, smaller matrices are padded to match the size of the
    largest molecule's atom count. Adjacency matrices are padded with zeros to align dimensions for batch processing.

    returns:
    - a mask list with indexes of non-valid molecules to filter out those molecules in other input features before
    model training
    """

    mask_list = []  # List to hold the masks for valid/invalid molecules
    # First pass: Determine max_atoms
    graph_tensors = []
    for smiles in molecules['ProductNames']:
        if smiles is None:
            mask_list.append(False)  # Invalid molecule (None)
            continue

        mol = Chem.MolFromSmiles(smiles, sanitize=False)  # Avoid automatic correction
        if mol is None:
            mask_list.append(False)  # Invalid molecule (could not parse)
        else:
            mask_list.append(True)  # Valid molecule
        # Manually sanitize to compute valences
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        mol = Chem.AddHs(mol)  # add hydrogens
        #mol.UpdatePropertyCache(strict=False)  # Force ring info update
        #rdmolops.FastFindRings(mol)  # Explicitly detect rings
        #for atom in mol.GetAtoms():
        #    print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}")

        # Second pass: Generate node features and adjacency matrices and store in list
        if serialize:
            graph = smiles_to_graph(mol)
            # RETURNS GRAPH WITH NODES AND Adjacency Matrix
            graph_tensors.append(graph)

    def split_list(slicing_data):
        """
        HERE DO SPLIT in 70/20/10, if I do it consistently then easy
        :param slicing_data:
        :return tuple with train/val/test:
        """
        n = len(slicing_data)
        split1 = int(n * 0.80)  # First 80%
        split2 = int(n * 0.85)  # Next 5% (up to 85%)

        list1 = slicing_data[:split1]  # First 80%
        list2 = slicing_data[split1:split2]  # Next 5%
        list3 = slicing_data[split2:]  # Last 15%

        return list1, list2, list3

    # graphs in list after for loop from before
    if serialize:
        split_list = split_list(graph_tensors)
        output_file_names = ["train_20.pt",
                             "test_20.pt",
                             "val_20.pt"]

        for idx, graphs in enumerate(split_list):
            torch.save(graphs, output_file_names[idx])  # Save as a Torch file

    return mask_list


if __name__ == '__main__':
    # Example inputs
    alloy_numpy = np.random.rand(500, 61)  # Example alloy tensor (50000 samples, 61 features)
    miller_numpy = np.random.rand(500, 4)  # Example Miller indices (50000 samples, 4 features)
    products1 = ["[O]O", "[OH]", "[O]", "[CH]", "N=N[H]", "O=O", "[K][O][O]"]  # Product strings
    products1 = pd.DataFrame(products1, columns=["ProductNames"])
    energy_numpy = np.random.rand(500, 1)  # Reaction energy labels

    # Full list of fixed products
    #smiles = ["C=C", "[O]O", "[OH]", "[O]", "[CH]", "N=N[H]", "C=C", "C#C", "C", "O=CC(=O)", "C(CO)O", "O=O",
    #          "[K][O][O]"]

    smiles_df = ['CH2CH2', 'HO2', 'HO', 'O', 'CH', 'CH2C', 'CHCH', 'CH3', 'CH3CH2', 'CH3CH', 'CH2CH', 'CH3C',
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

    smiles_df = pd.DataFrame(smiles_df, columns=["ProductNames"])
    # Example data
    data = {'ProductNames': ['C1OC1', 'CCO', 'C=C']}
    products = pd.DataFrame(data)

    #node_features, adjacency_matrix = preprocess_chemical_data_atomic(products)

    #print("Node Features Shape:", node_features.shape)  # Should be (3, 118)
    #print("Adjacency Matrix Shape:", adjacency_matrix.shape)  # Should be (3, max_atoms, max_atoms)

    # Preprocess products into node features and adjacency matrices
    #product_node_features, product_adjacency_matrices = preprocess_chemical_data(products1, smiles)
    product_node_features, product_adjacency_matrices, _ = preprocess_chemical_data_atomic(smiles_df)

    print(product_node_features.shape)
    print(product_adjacency_matrices.shape)

    # Convert alloy and miller tensors to TensorFlow tensors
    alloy_tensor = tf.convert_to_tensor(alloy_numpy, dtype=tf.float32)
    miller_tensor = tf.convert_to_tensor(miller_numpy, dtype=tf.float32)
    energy_tensor = tf.convert_to_tensor(energy_numpy, dtype=tf.float32)
