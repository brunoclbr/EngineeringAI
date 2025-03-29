import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from src.graph_preprocess import preprocess_chemical_data_atomic
from src.to_smiles import map_to_smiles


class EDA:

    def __init__(self, data):
        """
        Class for performing exploratory data analysis and transform raw excel data into meaningful DataFrame
        :data: DataFrame with raw excel data
        """
        self.data = data
        self.alloy_dict = {}
        self.element_data = []
        self.product_dict_count = Counter()
        self.miller = [self.data['facet_1'],
                       self.data['facet_2'],
                       self.data['facet_3'],
                       self.data['facet_4']]
        self.energy = self.data['ReactionEnergy']
        # self. ... add

        # Count Elements & Alloys in Catalyst
        surface_df = self.data['Concentration'].apply(ast.literal_eval)
        for element_dict in surface_df:
            alloy_number = len(element_dict)
            # Update alloy_dict with the alloy number
            if alloy_number in self.alloy_dict:
                self.alloy_dict[alloy_number] += 1
            else:
                self.alloy_dict[alloy_number] = 1

            for element, concentration in element_dict.items():
                self.element_data.append({'Element': element, 'Concentration': concentration})

        # Count Products per catalyst reaction
        # 2 simplifications. Only one adsorbate and no stoichiometry is being
        # taken into account here for simplicity. Maybe expand in the future
        self.products_df = self.data['Products']
        products_df_count = self.products_df.apply(
            lambda x: ast.literal_eval(x.strip('"')) if isinstance(x, str) else x)

        # Initialize dictionary to store species count in products
        # get dict from series
        products_names = []
        for product in products_df_count:
            self.product_dict_count.update(product.keys())
            key = next(iter(product.keys())).replace('star', '')
            products_names.append(key)
        self.products_df = pd.DataFrame(products_names, columns=['ProductNames'])
        #print(self.products_df)
        self.product_dict_count = {key.replace('star', ''): value for key, value in self.product_dict_count.items()}

        # how many alloys?
        #print(f"Alloys and their occurrences: {self.alloy_dict}")
        # how many elements?
        self.df_elements = pd.DataFrame(self.element_data)
        unique_elements = self.df_elements['Element'].unique()
        #print(f'Elements in database: {unique_elements}')

        # variable with alloys and their concentrations for training. First row is number of elements in database
        self.alloy_elements = pd.concat([pd.DataFrame(
            {'Concentration': [unique_elements]}), surface_df]).reset_index(drop=True)
        #print(self.alloy_elements)
        #print(type(self.alloy_elements))
        # how many products? len(unique_elements),
        #print(f'Adsorbed Molecules and their occurrences: {self.product_dict_count}')
        # print(f'Adsorbed Molecules: {self.products_df}')

    def plt_count_element(self):
        # Plot histogram of element occurrences
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df_elements, x='Element', palette='viridis')
        plt.title('Occurrences of Each Element')
        plt.xlabel('Element')
        plt.ylabel('Count')
        plt.show()

    def plt_mean_concentration(self):
        # Mean concentration per element and bar plot
        mean_concentration = self.df_elements.groupby('Element')['Concentration'].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=mean_concentration.index, y=mean_concentration.values, palette='viridis')
        plt.title('Mean Concentration of Each Element')
        plt.xlabel('Element')
        plt.ylabel('Mean Concentration')
        plt.show()

    def plt_dict_composition(self, xlabel='Alloy Constituents', ylabel='Count', alloy_bool=True):

        """ xlabel either 'Alloy Constituents' or 'Products' """
        plt_dict = self.alloy_dict if alloy_bool else self.product_dict_count

        # this should erase 'star' from products for better visibility
        if not alloy_bool:
            cleaned_data = {key.replace('star', ''): value for key, value in plt_dict.items()}
            threshold = 10
            grouped_dict = {k: v for k, v in cleaned_data.items() if v >= threshold}
            grouped_dict["Other"] = sum(v for k, v in cleaned_data.items() if v < threshold)
        else:  # this is workaround for alloy composition
            grouped_dict = plt_dict

        # alloy compositions plot
        dict2df = pd.DataFrame(list(grouped_dict.items()), columns=[xlabel, ylabel])
        dict2df = dict2df.sort_values(by=xlabel)  # Sort by the number of constituents
        plt.figure(figsize=(10, 6))
        sns.barplot(x=xlabel, y='Count', data=dict2df, palette='viridis')
        plt.title(f'{xlabel} and Occurrence')
        plt.yscale('log')  # Log scale for y-axis
        plt.xlabel(f'Number of {xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.xticks(rotation=60)
        plt.show()


class PreProcess:
    """
    get the features from EDA and transform them to tensors, normalize them etc.
    :param _alloys: object containing alloy composition
    :param _miller: object containing miller indices
    :param _products: object containing adsorbate data. 1) Only string 2) Later add more chemical descriptors
    :param _energy: object containing reaction energy data (output)

    The objects generated in this class need to be Numpy Arrays for further preprocessing
    - data normalization, split into train/test - and later be converted into
    trainable tf. tensors floats .32
    """
    def __init__(self, _alloys, _miller, _products, _energy, serialize):
        """
        when __init__ objects are instantiated from EDA class, these are their types
        print(type(self.alloy_elements)) -- <class 'pandas.core.frame.DataFrame'>
        print(type(self.products_df)) -- <class 'pandas.core.frame.DataFrame'>
        print(type(self.miller)) -- <class 'list'>
        print(type(self.energy)) -- <class 'pandas.core.series.Series'>
        """

        print('PREPROCESS BEGINS')

        elements_from_alloys = sorted(_alloys.iloc[0]['Concentration'])  # first row with unique names of
        # catalyst materials
        #print(f"SORTED ELEMENTS IN DATABASE: {elements_from_alloys}")
        self.alloys = _alloys.drop(0).reset_index(drop=True)  # get rid of first row with unique names of catalyst
        self.miller = _miller
        self.products = _products
        self.energy = _energy

        # 1 CATALYST ALLOYS
        # loop over elements, know the alphabetical order, keep it strict. Validate. Assign in each iteration the
        # corresponding concentration to the spot. Assign each unique name an integer to go to

        alloy_matrix = self.encoded_alloy(self.alloys, elements_from_alloys)
        self.alloy_tensor = np.array(alloy_matrix, dtype=np.float32)

        # 2. MILLER
        # LATER THINK ABOUT ELIMINATING 4TH DIMENSION SINCE TOO LITTLE INFO
        self.miller[0] = np.array(self.miller[0], dtype=np.float32)
        self.miller[1] = np.array(self.miller[1], dtype=np.float32)
        self.miller[2] = np.array(self.miller[2], dtype=np.float32)
        self.miller[3] = np.array(self.miller[3], dtype=np.float32)

        # Expand dimensions of each array to make them 2D
        self.miller = [np.expand_dims(array, axis=1) for array in self.miller]

        # Concatenate along the second axis
        self.miller_tensor = np.concatenate(self.miller, axis=1)

        # 3. PRODUCTS
        self.products['ProductNames'] = map_to_smiles(self.products['ProductNames'])
        """
        this now serializes the products and stores them in local folder. They have to be loaded and encoded later 
        before training begins. It returns "mask_tensor" that keeps track of not parsed molecules, 
        which means less data. With this mask I can keep track of the other variables and erase them. I think like
        this I'll always need the mask_tensor, but not always the serialization? avoid with if clause at some point?
        """
        mask_tensor = preprocess_chemical_data_atomic(self.products, serialize)
        self.mask_tensor = mask_tensor

        # 4. ENERGY
        self.energy_tensor = np.array(self.energy, dtype=np.float32)

    @staticmethod
    def encoded_alloy(alloys_to_encode, elements):

        """
        Elements in database: ['Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Bi', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cs',
        'Cu', 'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'In', 'Ir', 'K', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb',
        'Nd', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'Sb', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Tc',
        'Te', 'Ti', 'Tl', 'V', 'W', 'Y', 'Zn', 'Zr']
        """

        # Define the number of elements
        num_elements = len(elements)
        print(f'THERE ARE {num_elements} ELEMENTS IN THE ALLOYS')

        # Initialize a matrix to store the concentrations, with rows for each alloy and columns for each element.
        alloy_matrix = np.zeros((len(alloys_to_encode), num_elements))

        # Create a mapping from element symbols to column indices based on their position in `elements`
        element_to_idx = {element: idx for idx, element in enumerate(elements)}

        # Iterate over each alloy composition dictionary in the DataFrame and fill the alloy matrix

        for i, row in alloys_to_encode.iterrows():
            try:
                # Parse the concentration string into a dictionary if needed
                if isinstance(row['Concentration'], str):
                    concentration_dict = ast.literal_eval(row['Concentration'])

                elif isinstance(row['Concentration'], dict):
                    concentration_dict = row['Concentration']

                else:
                    continue  # Skip rows that aren't dictionaries or strings
                # Fill in the alloy_matrix
                for elem, concentration in concentration_dict.items():
                    if elem in element_to_idx:  # Only map known elements in `elements`
                        col_idx = element_to_idx[elem]
                        alloy_matrix[i, col_idx] = concentration
            except Exception as e:
                print(f"Error processing row {i}: {e}")

        return alloy_matrix

    # THIS WAS BEFORE USING GNNS TO REPRESENT THE PRODUCTS
    @staticmethod
    def encoded_product(to_encode_products, manual_products=False):
        """
        The 131 products have been assigned with a unique integer. So if I want to predict
        Ag3Pt it needs to go through this function and output a 1x131 array with 2 non-zero
        elements indicating Ag3Pt.
        SORT PRODUCTS ALPHABETICALLY FOR REPRODUCIBILITY. MIND TRAINING AND ADDING NEW PRODUCTS
        BECAUSE IT WILL CHANGE ORDER OF MAPPING.
        """

        # use manual_products for predicting artificial ads energies
        fixed_products = ['CH2CH2', 'HO2', 'HO', 'O', 'CH', 'CH2C', 'CHCH', 'CH3', 'CH3CH2', 'CH3CH', 'CH2CH', 'CH3C',
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

        if manual_products:
            unique_products = fixed_products
        else:
            unique_products = list(to_encode_products['ProductNames'].unique())

        unique_products = sorted(unique_products)
        #print(unique_products)
        #print(f"UNIQUE PRODUCTS NUMBER: {len(unique_products)}")
        product_to_int = {product: idx for idx, product in enumerate(unique_products)}
        to_encode_products['ProductNames'] = to_encode_products['ProductNames'].map(product_to_int)
        #print(f"MAPPED PRODUCTS: {to_encode_products}")
        # Convert to a one-hot encoded NumPy array
        encoded_products = np.eye(len(unique_products))[to_encode_products['ProductNames']]
        return encoded_products
