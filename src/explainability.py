import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

class SubstructureExplainer:
    def __init__(self, model, X_background):
        self.model = model
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(model)
        
    def generate_atomic_weights(self, target_smiles, featurizer):
        """Maps SHAP bit contributions back to individual atoms in a molecule."""
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            raise ValueError("Invalid SMILES input string.")

        # Compute prediction baseline via feature pipeline
        X_mol, _ = featurizer.featurize_smiles_list([target_smiles])
        shap_values = self.explainer.shap_values(X_mol)[0]
        
        # Helper function mapping atomic environment vectors to local model probabilities
        def atom_weight_fn(fp_matrix):
            # fp_matrix shape expected by RDKit SimilarityMaps: (N, 2048)
            # Append zero-padded 2D descriptors to match expected model dimension (2054)
            padded_input = np.hstack([fp_matrix, np.zeros((fp_matrix.shape[0], 6))])
            return self.model.predict_proba(padded_input)[:, 1]

        # Generate atomic contribution heatmaps directly onto the RDKit canvas
        fig, max_weight = SimilarityMaps.GetAtomicWeightsForModel(
            mol, atom_weight_fn
        )
        return fig