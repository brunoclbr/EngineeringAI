# Define the mapping of fixed products to their corresponding SMILES notation
fixed_products_to_smiles = {
    "CH2CH2": "C=C",
    "HO2": "[O]O",
    "HO": "[OH]",
    "O": "[O]",
    "CH": "[CH]",
    "CH2C": "C=C",
    "CHCH": "C#C",
    "CH3": "C",
    "CH3CH2": "CC",
    "CH3CH": "C[CH]",
    "CH2CH": "C=C",
    "CH3C": "C[C]",
    "CO": "C=O",
    "C6H6": "c1ccccc1",
    "H": "[H]",
    "C2H6": "CC",
    "C3H8": "CCC",
    "CH4": "C",
    "H2O": "O",
    "I": "[I]",
    "NO": "N=O",
    "NH3": "N",
    "C": "[C]",
    "H3O": "[OH3+]",
    "OH": "[OH]",
    "N": "[N]",
    "CO2": "O=C=O",
    "OOH": "[O]O",
    "SH": "[SH]",
    "S": "[S]",
    "CH2": "[CH2]",
    "NH": "[NH]",
    "CH3CH2CH3": "CCC",
    "CH3CHCH2": "C=CC",
    "CH3CCH": "C#CC",
    "Ru": "[Ru]",
    "COOH": "C(=O)O",
    "HCOO": "O=CO",
    "COH": "C=O",
    "CHO": "C=O",
    "OCCOH": "O=CC(=O)",
    "Cu": "[Cu]",
    "OCCO": "OCC=O",
    "OCCHO": "OC=CO",
    "C3H7O3": "CC(CO)O",
    "C3H6O3": "CCC(=O)O",
    "C3H8O3": "C(CO)O",
    "O2": "O=O",
    "LiO2": "[Li][O][O]",
    "Li2O2": "[Li]O[Li]",
    "LiO": "[Li]O",
    "NaO2": "[Na][O][O]",
    "NaO": "[Na]O",
    "Na2O2": "[Na]O[Na]",
    "KO2": "[K][O][O]",
    "K2O2": "[K]O[K]",
    "KO": "[K]O",
    "": None,
    "@LS-133": None,
    "@HS-133": None,
    "CHCO": "[CH]=C=O",
    "CCO": "CC=O",
    "CHOH": "CO",
    "Re": "[Re]",
    "OCHO": "O=CO",
    "CHC": "[CH]=C",
    "CH3O": "CO",
    "Rh": "[Rh]",
    "Ir": "[Ir]",
    "Pt": "[Pt]",
    "Ag": "[Ag]",
    "Pd": "[Pd]",
    "Os": "[Os]",
    "Au": "[Au]",
    "Zn": "[Zn]",
    "Ni": "[Ni]",
    "SiH2": "[SiH2]",
    "NH2": "[NH2]",
    "NNH": "N=N[H]",
    "HN": "[H]N",
    "NNH2": "N[NH2]",
    "SiH3": "[SiH3]",
    "SiHCl": "SiHCl",
    "SiCl2": "[Si](Cl)Cl",
    "SiH2Cl": "[Si]-H-[H]-Cl",
    "CHCl2": "C(Cl)Cl",
    "SiHCl2": "[Si](Cl)(Cl)[H]",
    "CH2Cl": "C[Cl]",
    "CCl2": "[C](Cl)Cl",
    "H2N": "[NH2]",
    "CCl": "C[Cl]",
    "CHCl": "C[Cl]",
    "SiCl3": "[Si](Cl)(Cl)Cl",
    "CCl3": "C(Cl)(Cl)Cl",
    "SiCl": "[Si][Cl]",
    "Si": "[Si]",
    "SiH": "[Si][H]",
    "HS": "[SH]",
    "CHCHCH3": "C=C[CH3]",
    "CCH2CH3": "C=CC",
    "CH2CH2CH3": "CCC",
    "CH2CH3": "CC",
    "C2H3": "C=C",
    "CH3CH3": "CC",
    "CCH3": "CC",
    "CHCH2CH3": "CCC",
    "C5H11": "CCCCC",
    "C6H12": "C1CCCCC1",
    "C28H28P2": "C1CCCCC1PCCPCCPCCP1CCCCC1",
    "C28H30P2": "C1CCCCC1PCCPCCPCCP1CCCCC1C",
    "C18H37N": "CCCCCCCCCCCCCCCCCCCCCCCN",
    "C18H38N": "NCCCCCCCCCCCCCCCCCCCCCC",
    "OCHCH2O": "O=C=O",
    "Co": "[Co]",
    "Zr": "[Zr]",
    "Rb": "[Rb]",
    "Y": "[Y]",
    "Sr": "[Sr]",
    "Nb": "[Nb]",
    "Tc": "[Tc]",
    "Cd": "[Cd]",
    "Mo": "[Mo]",
    "OHOH": "OOO",
    "OHOOH": "OO[O]",
    "OHO": "OO",
    "OCCH2O": "C=O",
    "CH3CHO": "CC=O",
    "HCN": "C#N",
    "CH3CH2OH": "CCO",
    "OCHCHO": "O=C=C",
    "OCH2CH2O": "OCCO"
}

# Original list of fixed products
fixed_products = [
    'CH2CH2', 'HO2', 'HO', 'O', 'CH', 'CH2C', 'CHCH', 'CH3', 'CH3CH2', 'CH3CH', 'CH2CH', 'CH3C',
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


# Function to map the list of fixed products to SMILES
def map_to_smiles(products):
    return [fixed_products_to_smiles.get(molecule, None) for molecule in products]


# Get the SMILES corresponding to each product
mapped_smiles = map_to_smiles(fixed_products)


if __name__ == "__main__":

    # Print the result
    mapped_smiles = map_to_smiles(fixed_products)
    i = 1
    for product, smile in zip(fixed_products, mapped_smiles):
        print(f"{product}: {smile}")
        print(i)
        i += 1

fixed_products = [
    "C=C",  # CH2CH2 (ethene)
    "[O]O",  # HO2 (hydroperoxyl radical)
    "[OH]",  # HO (hydroxyl radical)
    "[O]",  # O (oxygen atom)
    "[CH]",  # CH (methylidyne radical)
    "C=C",  # CH2C (assumed ethene, alias)
    "C#C",  # CHCH (acetylene)
    "C",  # CH3 (methane radical)
    "CC",  # CH3CH2 (ethane)
    "C[CH]",  # CH3CH (ethyl radical)
    "C=C",  # CH2CH (ethene)
    "C[C]",  # CH3C (methyl radical)
    "C=O",  # CO (carbon monoxide)
    "c1ccccc1",  # C6H6 (benzene)
    "[H]",  # H (hydrogen atom)
    "CC",  # C2H6 (ethane)
    "CCC",  # C3H8 (propane)
    "C",  # CH4 (methane)
    "O",  # H2O (water)
    "[I]",  # I (iodine atom)
    "N=O",  # NO (nitric oxide)
    "N",  # NH3 (ammonia)
    "[C]",  # C (carbon atom)
    "[H3O+]",  # H3O (hydronium ion)
    "[OH]",  # OH (hydroxyl radical)
    "[N]",  # N (nitrogen atom)
    "O=C=O",  # CO2 (carbon dioxide)
    "[O]O",  # OOH (peroxyl radical)
    "[SH]",  # SH (mercapto radical)
    "[S]",  # S (sulfur atom)
    "[CH2]",  # CH2 (methylene radical)
    "[NH]",  # NH (amino radical)
    "CCC",  # CH3CH2CH3 (propane)
    "C=CC",  # CH3CHCH2 (propene)
    "C#CC",  # CH3CCH (propyne)
    "[Ru]",  # Ru (ruthenium atom)
    "C(=O)O",  # COOH (carboxylic acid)
    "O=CO",  # HCOO (formate)
    "C=O",  # COH (formaldehyde)
    "C=O",  # CHO (formaldehyde)
    "O=CC(=O)",  # OCCOH (glyoxal)
    "[Cu]",  # Cu (copper atom)
    "OCC=O",  # OCCO (glyoxal variant)
    "OC=CO",  # OCCHO (glyoxal variant)
    "CC(CO)O",  # C3H7O3 (glycerol radical)
    "CCC(=O)O",  # C3H6O3 (pyruvic acid)
    "C(CO)O",  # C3H8O3 (glycerol)
    "O=O",  # O2 (molecular oxygen)
    "[Li][O][O]",  # LiO2 (lithium peroxide)
    "[Li]O[Li]",  # Li2O2 (lithium oxide)
    "[Li]O",  # LiO (lithium oxide)
    "[Na][O][O]",  # NaO2 (sodium peroxide)
    "[Na]O",  # NaO (sodium oxide)
    "[Na]O[Na]",  # Na2O2 (sodium oxide)
    "[K][O][O]",  # KO2 (potassium peroxide)
    "[K]O[K]",  # K2O2 (potassium oxide)
    "[K]O",  # KO (potassium oxide)
    None,  # Invalid (empty string)
    None,  # Invalid (@LS-133)
    None,  # Invalid (@HS-133)
    "[CH]=C=O",  # CHCO (ketene)
    "CC=O",  # CCO (acetaldehyde)
    "CO",  # CHOH (methanol radical)
    "[Re]",  # Re (rhenium atom)
    "O=CO",  # OCHO (formic acid)
    "[CH]=C",  # CHC (vinylidene radical)
    "CO",  # CH3O (methanol radical)
    "[Rh]",  # Rh (rhodium atom)
    "[Ir]",  # Ir (iridium atom)
    "[Pt]",  # Pt (platinum atom)
    "[Ag]",  # Ag (silver atom)
    "[Pd]",  # Pd (palladium atom)
    "[Os]",  # Os (osmium atom)
    "[Au]",  # Au (gold atom)
    "[Zn]",  # Zn (zinc atom)
    "[Ni]",  # Ni (nickel atom)
    "[Si][H][H]",  # SiH2 (silylene radical)
    "[NH2]",  # NH2 (amino radical)
    "N=N[H]",  # NNH (hydrazyl radical)
    "[H]N",  # HN (imino radical)
    "N=N[H][H]",  # NNH2 (hydrazine radical)
    "[Si][H][H][H]",  # SiH3 (silyl radical)
    "[Si][H][Cl]",  # SiHCl (chlorosilyl radical)
    "[Si][Cl][Cl]",  # SiCl2 (dichlorosilylene radical)
    "[Si][H][Cl][Cl]",  # SiH2Cl (dichlorosilyl radical)
    "C[Cl][Cl]",  # CHCl2 (dichloromethyl radical)
    "[Si][H][Cl][Cl]",  # SiHCl2 (dichlorosilyl radical)
    "C[Cl]",  # CH2Cl (chloromethyl radical)
    "[C][Cl][Cl]",  # CCl2 (dichlorocarbene radical)
    "[H2N]",  # H2N (ammonia radical)
    "C[Cl]",  # CCl (methyl chloride)
    "[CHCl]",  # CHCl (chloromethyl radical)
    "[Si][Cl][Cl][Cl]",  # SiCl3 (trichlorosilyl radical)
    "[C][Cl][Cl][Cl]",  # CCl3 (trichloromethyl radical)
    "[Si][Cl]",  # SiCl (chlorosilyl radical)
    "[Si]",  # Si (silicon atom)
    "[Si][H]",  # SiH (silyl radical)
    "[SH]",  # HS (mercapto radical)
    "C=C[CH3]",  # CHCHCH3 (propene)
    "C=CC",  # CCH2CH3 (propene alias)
    "CCC",  # CH2CH2CH3 (propane)
    "CC",  # CH2CH3 (ethane)
    "C=C",  # C2H3 (ethene)
    "C=C",  # CH3CH3 (ethene alias)
    "CC",  # CCH3 (ethane)
    "CCC",  # CHCH2CH3 (propane)
    "C5H11",  # C5H11 (pentyl radical)
    "C6H12",  # C6H12 (cyclohexane)
    "C28H28P2",  # C28H28P2 (custom phosphine molecule)
    "C28H30P2",  # C28H30P2 (custom phosphine molecule)
    "C18H37N",  # C18H37N (custom amine molecule)
    "C18H38N",  # C18H38N (custom amine molecule)
    "O=C=O",  # OCHCH2O (glyoxal-derived radical)
    "[Co]",  # Co (cobalt atom)
    "[Zr]",  # Zr (zirconium atom)
    "[Rb]",  # Rb (rubidium atom)
    "[Y]",  # Y (yttrium atom)
    "[Sr]",  # Sr (strontium atom)
    "[Nb]",  # Nb (niobium atom)
    "[Tc]",  # Tc (technetium atom)
    "[Cd]",  # Cd (cadmium atom)
    "[Mo]",  # Mo (molybdenum atom)
    "OOO",  # OHOH (peroxide variant)
    "OO[O]",  # OHOOH (hydroperoxide radical)
    "OO",  # OHO (peroxide)
    "C=O",  # OCCH2O (glyoxal variant)
    "CC=O",  # CH3CHO (acetaldehyde)
    "HC#N"  # HCN (hydrogen cyanide)
]



