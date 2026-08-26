from src.featurizer import MoleculeFeaturizer
from src.train import QSARModelTrainer
from src.explainability import SubstructureExplainer
from sklearn.metrics import roc_auc_score, average_precision_score

if __name__ == "__main__":
    print("[1/4] Initializing Chemical Featurizer...")
    featurizer = MoleculeFeaturizer(n_bits=2048)

    print("[2/4] Loading TDC Dataset & Performing Scaffold Split...")
    trainer = QSARModelTrainer(dataset_name="hERG")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = trainer.prepare_datasets(featurizer)

    print("[3/4] Tuning Hyperparameters & Training Model...")
    model = trainer.tune_and_train_xgboost(X_tr, y_tr, X_va, y_va)

    # Evaluate on held-out scaffold test set
    test_probs = model.predict_proba(X_te)[:, 1]
    print(f"[+] Test ROC-AUC : {roc_auc_score(y_te, test_probs):.4f}")
    print(f"[+] Test PR-AUC  : {average_precision_score(y_te, test_probs):.4f}")

    print("[4/4] Generating Substructure Explainability Visuals...")
    explainer = SubstructureExplainer(model, X_tr)
    # Example toxicophore molecule (Aspirin sample)
    sample_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    fig = explainer.generate_atomic_weights(sample_smiles, featurizer)
    
    print("[+] Pipeline Execution Complete.")