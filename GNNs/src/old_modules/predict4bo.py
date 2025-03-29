import numpy as np
import pandas as pd
from typing import Callable
import tensorflow as tf
# 1 DEFINE ALLOY


def artificial_alloy(encoded_alloy: Callable, encoded_product: Callable):

    elements_from_alloys = ['Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Bi', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cs',
                            'Cu', 'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'In', 'Ir', 'K', 'La', 'Li', 'Mg', 'Mn',
                            'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'Rh',
                            'Ru', 'Sb', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Tc', 'Te', 'Ti', 'Tl', 'V', 'W', 'Y', 'Zn', 'Zr']

    ptag3 = {'Pt': 0.25, 'Ag': 0.75}
    pt3Co = {'Pt': 0.75, 'Co': 0.25}
    ptru = {'Pt': 0.5, 'Ru': 0.5}
    ptpd = {'Pt': 0.5, 'Pd': 0.5}
    aupt = {'Pt': 0.5, 'Au': 0.5}
    nicofe = {'Ni': 0.333, 'Co': 0.333, 'Fe': 0.333}
    nifecomow = {'Ni': 0.2, 'Fe': 0.2, 'Co': 0.2, 'Mo': 0.2, 'W': 0.2}
    create_alloys = pd.DataFrame({'Concentration': [nifecomow, ptag3, aupt, ptpd, pt3Co, ptru, nicofe]})

    encoded_alloy_tensor = encoded_alloy(create_alloys, elements_from_alloys)
    #print(encoded_alloy_tensor)
    #print(encoded_alloy_tensor.shape)

    # 2. MILLER
    # Helper function to check for prime numbers
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # Miller and Miller2 arrays setup
    miller = [np.expand_dims(np.array([1], dtype=np.float32), axis=1),
              np.expand_dims(np.array([1], dtype=np.float32), axis=1),
              np.expand_dims(np.array([1], dtype=np.float32), axis=1),
              np.expand_dims(np.array([0], dtype=np.float32), axis=1)]

    miller2 = [np.expand_dims(np.array([1], dtype=np.float32), axis=1),
               np.expand_dims(np.array([0], dtype=np.float32), axis=1),
               np.expand_dims(np.array([0], dtype=np.float32), axis=1),
               np.expand_dims(np.array([0], dtype=np.float32), axis=1)]

    # DataFrame initialization for 'OH'
    oh_ads = pd.DataFrame(['OH'], columns=['ProductNames'])

    # List to store miller tensors
    miller_tensor = []

    # Looping through and checking if i is prime
    for i in range(7):
        if is_prime(i):  # Check if i is prime
            miller_tensor.append(np.concatenate(miller, axis=1))
        else:
            miller_tensor.append(np.concatenate(miller2, axis=1))

        if i > 0:  # Double the size of oh_ads DataFrame for every iteration after the first
            oh_ads = pd.concat([oh_ads, oh_ads], ignore_index=True)

    encoded_oh_ads = encoded_product(oh_ads, manual_products=True)

    # PREDICT ARTIFICIAL ALLOY WITH ADS ENERGY
    x_test_artificial = [encoded_alloy_tensor,
                         miller_tensor,
                         encoded_oh_ads]

    return x_test_artificial

