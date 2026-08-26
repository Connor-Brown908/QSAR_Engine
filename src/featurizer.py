import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class MoleculeFeaturizer:
    def __init__(self, n_bits=2048, radius=2):
        self.n_bits = n_bits
        self.radius = radius

    def _get_ecfp4(self, mol):
        """Generates ECFP4 (Morgan Fingerprint) as a bit vector."""
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _get_2d_descriptors(self, mol):
        """Extracts continuous physical property descriptors."""
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
        return np.array(descriptors, dtype=float)

    def featurize_smiles_list(self, smiles_list):
        """Converts a series of SMILES into a unified feature matrix."""
        features = []
        valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # Filter out invalid chemical representations
            
            ecfp4 = self._get_ecfp4(mol)
            desc2d = self._get_2d_descriptors(mol)
            combined = np.concatenate([ecfp4, desc2d])
            
            features.append(combined)
            valid_indices.append(idx)

        return np.array(features), valid_indices