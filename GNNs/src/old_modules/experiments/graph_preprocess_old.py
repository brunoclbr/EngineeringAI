import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import SanitizeMol
from rdkit.Chem.MolStandardize import rdMolStandardize
from gnn.tfrecords.serialize import smiles_to_graph, graph_schema
import tensorflow_gnn as tfgnn
from matplotlib.colors import ColorConverter


def preprocess_and_remove_hydrogens(smiles: str):
    """
    Process the molecule from SMILES string, validate its structure,
    and safely remove explicit hydrogens if applicable.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        mol: An RDKit Mol object with hydrogens removed if appropriate.
    """
    # Step 1: Generate molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecule: Could not parse SMILES")

    # Step 2: Sanitize and standardize the molecule
    try:
        SanitizeMol(mol)  # Ensure the molecule is valid
        mol = rdMolStandardize.Cleanup(mol)  # Clean and standardize
    except Exception as e:
        print(f"Sanitization or standardization failed: {e}")
        return None

    # Step 3: Inspect and handle hydrogens without neighbors
    problematic_hydrogens = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and len(atom.GetNeighbors()) == 0:
            problematic_hydrogens.append(atom.GetIdx())

    if problematic_hydrogens:
        print(f"Found {len(problematic_hydrogens)} problematic hydrogens.")
        # Remove these problematic hydrogens
        editable_mol = Chem.EditableMol(mol)
        for idx in reversed(problematic_hydrogens):  # Remove in reverse order to avoid index shift
            editable_mol.RemoveAtom(idx)
        mol = editable_mol.GetMol()

    # Step 4: Remove explicit hydrogens safely
    if mol.GetNumAtoms() > mol.GetNumHeavyAtoms():
        try:
            mol = Chem.RemoveHs(mol)  # Remove explicit hydrogens
        except Exception as e:
            print(f"Error during hydrogen removal: {e}")

    return mol


def preprocess_chemical_data_atomic(molecule):
    """
    Preprocess product DataFrames into graph-like representations where each atom is a node.

    Args:
    - molecule: DataFrame with 'ProductNames' column containing SMILES strings.

   a molecule encodes:
    - node_features: NumPy array with atomic counts (batch_size, num_atoms).
    - adjacency_matrix: NumPy array with adjacency matrices (batch_size, num_atoms, num_atoms).
    - to ensure consistent dimensions across all molecules, smaller matrices are padded to match the size of the
    largest molecule's atom count. Adjacency matrices are padded with zeros to align dimensions for batch processing.

    returns:
    - a mask list with indexes of non-valid molecules to filter out those molecules in other input features before
    model training
    """

    #atom_types = [1, 6, 7, 8]  # Hydrogen, Carbon, Nitrogen, Oxygen...
    atom_types = [
        1,  # Hydrogen
        2,  # Helium
        3,  # Lithium
        4,  # Beryllium
        5,  # Boron
        6,  # Carbon
        7,  # Nitrogen
        8,  # Oxygen
        9,  # Fluorine
        10,  # Neon
        11,  # Sodium
        12,  # Magnesium
        13,  # Aluminum
        14,  # Silicon
        15,  # Phosphorus
        16,  # Sulfur
        17,  # Chlorine
        18,  # Argon
        19,  # Potassium
        20,  # Calcium
        21,  # Scandium
        22,  # Titanium
        23,  # Vanadium
        24,  # Chromium
        25,  # Manganese
        26,  # Iron
        27,  # Cobalt
        28,  # Nickel
        29,  # Copper
        30,  # Zinc
        31,  # Gallium
        32,  # Germanium
        33,  # Arsenic
        34,  # Selenium
        35,  # Bromine
        36,  # Krypton
        37,  # Rubidium
        38,  # Strontium
        39,  # Yttrium
        40,  # Zirconium
        41,  # Niobium
        42,  # Molybdenum
        43,  # Technetium
        44,  # Ruthenium
        45,  # Rhodium
        46,  # Palladium
        47,  # Silver
        48,  # Cadmium
        49,  # Indium
        50,  # Tin
        51,  # Antimony
        52,  # Tellurium
        53,  # Iodine
        54,  # Xenon
        55,  # Cesium
        56,  # Barium
        57,  # Lanthanum
        58,  # Cerium
        59,  # Praseodymium
        60,  # Neodymium
        61,  # Promethium
        62,  # Samarium
        63,  # Europium
        64,  # Gadolinium
        65,  # Terbium
        66,  # Dysprosium
        67,  # Holmium
        68,  # Erbium
        69,  # Thulium
        70,  # Ytterbium
        71,  # Lutetium
        72,  # Hafnium
        73,  # Tantalum
        74,  # Tungsten
        75,  # Rhenium
        76,  # Osmium
        77,  # Iridium
        78,  # Platinum
        79,  # Gold
        80,  # Mercury
        81,  # Thallium
        82,  # Lead
        83,  # Bismuth
        84,  # Polonium
        85,  # Astatine
        86,  # Radon
        87,  # Francium
        88,  # Radium
        89,  # Actinium
        90,  # Thorium
        91,  # Protactinium
        92,  # Uranium
        93,  # Neptunium
        94,  # Plutonium
        95,  # Americium
        96,  # Curium
        97,  # Berkelium
        98,  # Californium
        99,  # Einsteinium
        100,  # Fermium
        101,  # Mendelevium
        102,  # Nobelium
        103,  # Lawrencium
        104,  # Rutherfordium
        105,  # Dubnium
        106,  # Seaborgium
        107,  # Bohrium
        108,  # Hassium
        109,  # Meitnerium
        110,  # Darmstadtium
        111,  # Roentgenium
        112,  # Copernicium
        113,  # Nihonium
        114,  # Flerovium
        115,  # Moscovium
        116,  # Livermorium
        117,  # Tennessine
        118  # Oganesson
    ]
    atom_type_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}
    node_features_list = []
    adjacency_matrix_list = []
    mask_list = []  # List to hold the masks for valid/invalid molecules

    max_atoms = 0  # Maximum number of atoms in a molecule

    # First pass: Determine max_atoms
    for smiles in molecule['ProductNames']:
        if smiles is None:
            mask_list.append(False)  # Invalid molecule (None)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mask_list.append(False)  # Invalid molecule (could not parse)
            continue

        num_atoms = len(mol.GetAtoms())
        max_atoms = max(max_atoms, num_atoms)
        mask_list.append(True)  # Valid molecule

    # Second pass: Generate node features and adjacency matrices
    output_file_names = ["graph_molecules_train.tfrecord",
                         "graph_molecules_val.tfrecord",
                         "graph_molecules_test.tfrecord"]

    graph_tensors = []
    for smiles in molecule['ProductNames']:
        if smiles is None:
            continue

        mol = preprocess_and_remove_hydrogens(smiles)
        ############################################
        # USE SYMBOLIC LABELS AS REACTION ENERGY WILL BE USED AFTER CONCATENATION WITH OTHER ANN INPUTS
        label = 0.1
        # INSERT HERE SMILES2GRAPH
        graph = smiles_to_graph(mol, label=label)
        graph_tensors.append(graph)
        # RETURNS GRAPH WITH NODES AND AM

    # HERE DO SPLIT in 70/20/10, if i do it consistently then easy
    def split_list(slicing_data):
        n = len(slicing_data)
        split1 = int(n * 0.70)  # First 70%
        split2 = int(n * 0.90)  # Next 20% (up to 90%)

        list1 = slicing_data[:split1]  # First 70%
        list2 = slicing_data[split1:split2]  # Next 20%
        list3 = slicing_data[split2:]  # Last 10%

        return list1, list2, list3

    splitted_list = split_list(graph_tensors)

    for idx, output_file in enumerate(output_file_names):
        with tf.io.TFRecordWriter(output_file) as writer:
            for graph in splitted_list[idx]:
                tfgnn.check_compatible_with_schema_pb(graph, graph_schema)
                serialized_example = tfgnn.write_example(graph)
                writer.write(serialized_example.SerializeToString())

        ############################################

    def old_encoding():
        # Node feature: One-hot encoding of atom types
        num_atoms = len(mol.GetAtoms())
        node_features = np.zeros((num_atoms, len(atom_types)))
        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            if atomic_num in atom_type_to_idx:
                node_features[i, atom_type_to_idx[atomic_num]] = 1.0
        node_features_list.append(node_features)

        # Adjacency matrix with bond types
        adjacency_matrix = np.zeros((num_atoms, num_atoms))
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # Assign bond type values
            if bond_type == Chem.rdchem.BondType.SINGLE:
                adjacency_matrix[start, end] = 1.0
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                adjacency_matrix[start, end] = 2.0
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                adjacency_matrix[start, end] = 3.0
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                adjacency_matrix[start, end] = 1.5

            # Make symmetric
            adjacency_matrix[end, start] = adjacency_matrix[start, end]

        adjacency_matrix_list.append(adjacency_matrix)

    # Pad node features and adjacency matrices to uniform size
    padded_node_features = []
    padded_adjacency_matrices = []

    def old_padding():
        for node_features, adjacency_matrix in zip(node_features_list, adjacency_matrix_list):
            # Pad node features
            padded_features = np.zeros((max_atoms, len(atom_types)))
            padded_features[:node_features.shape[0], :] = node_features
            padded_node_features.append(padded_features)

            # Pad adjacency matrix
            size = adjacency_matrix.shape[0]
            padded_matrix = np.zeros((max_atoms, max_atoms))
            padded_matrix[:size, :size] = adjacency_matrix
            padded_adjacency_matrices.append(padded_matrix)

    # Convert to tensors
    node_features_tensor = np.array(padded_node_features, dtype=np.float32)
    adjacency_matrix_tensor = np.array(padded_adjacency_matrices, dtype=np.float32)

    #return node_features_tensor, adjacency_matrix_tensor, mask_list
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
        'OCHCHO', 'OCH2CH2O'
    ]
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
