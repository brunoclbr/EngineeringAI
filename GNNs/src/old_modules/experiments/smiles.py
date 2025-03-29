from rdkit import Chem
import rdkit


def parse_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {smiles}")
    else:
        print(f"Successfully parsed SMILES: {smiles}")
    return mol

# Test SMILES
smiles_list = [
    'C1CCCCC1PCCPCCPCCP1CCCCC1C'
]

# Test each SMILES string
for smiles in smiles_list:
    parse_smiles(smiles)