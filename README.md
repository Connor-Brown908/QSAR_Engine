# QSAR Engine

A production-grade quantitative structure-activity relationship (QSAR) modeling engine for predicting molecular properties and drug toxicity. This pipeline combines RDKit molecular featurization, XGBoost hyperparameter tuning, and SHAP-based explainability to build interpretable predictive models for pharmaceutical and chemical research.

## What it does

QSAR Engine automates the end-to-end workflow for molecular property prediction:
1. **Featurizes** molecules using ECFP4 Morgan fingerprints + 2D physical descriptors (molecular weight, logP, TPSA, H-donors/acceptors, rotatable bonds)
2. **Loads datasets** from the Therapeutics Data Commons (TDC) with strict Bemis-Murcko scaffold splitting
3. **Optimizes and trains** an XGBoost classifier using Optuna hyperparameter tuning
4. **Evaluates** on held-out test sets with ROC-AUC and PR-AUC metrics
5. **Explains predictions** by mapping SHAP feature importance back to individual atoms in molecules

## Stack

- **Language:** Python 3
- **Chemistry:** RDKit (molecular parsing, fingerprinting, visualization)
- **Data:** PyTDC (Therapeutics Data Commons), pandas, numpy
- **ML:** scikit-learn, XGBoost, Optuna (hyperparameter optimization), SHAP (explainability)
- **Visualization:** matplotlib

## How it's organized

```
QSAR_Engine/
  main.py                 Entry point: runs full pipeline (featurize → train → evaluate → explain)
  requirements.txt        Python dependencies
  
  src/
    featurizer.py         MoleculeFeaturizer: ECFP4 fingerprints & 2D descriptors
    train.py              QSARModelTrainer: data loading, scaffold splitting, XGBoost tuning
    explainability.py     SubstructureExplainer: atomic-level SHAP attribution maps
    data_ingestion.py     DataIngestionPipeline: production data loader with SMILES standardization
```

### How it fits together

The pipeline is centered on chemical data flow:

1. **Data ingestion** (`DataIngestionPipeline`) downloads raw SMILES strings from TDC, standardizes molecules (removes salts, neutralizes charges), and applies scaffold splitting
2. **Featurization** (`MoleculeFeaturizer`) converts valid SMILES to fixed-size feature vectors (2048-bit Morgan fingerprints + 6 physical descriptors)
3. **Training** (`QSARModelTrainer`) prepares train/validation/test splits and runs Optuna-based XGBoost hyperparameter optimization
4. **Explainability** (`SubstructureExplainer`) uses SHAP TreeExplainer to attribute predictions back to individual atoms, generating heat maps on the molecular structure

Entry point (`main.py`) orchestrates all four steps: initialize featurizer → load and split TDC dataset → tune and train model → generate explainability visuals.

## Getting started

### Prerequisites

- Python 3.8 or higher
- RDKit (requires conda or pip install; see [RDKit docs](https://www.rdkit.org/docs/Install.html))

### Installation

```bash
git clone https://github.com/Connor-Brown908/QSAR_Engine.git
cd QSAR_Engine
pip install -r requirements.txt
```

### Running the pipeline

```bash
python main.py
```

This will:
- Initialize a molecular featurizer (2048-bit Morgan fingerprints)
- Fetch the hERG dataset from TDC and apply scaffold splitting
- Tune XGBoost hyperparameters using Optuna (15 trials)
- Print test ROC-AUC and PR-AUC scores
- Generate atomic weight explainability plots

### Running data ingestion separately

You can also prepare data independently:

```bash
python src/data_ingestion.py --type ADME --dataset hERG --outdir ./data
```

Supported datasets: any TDC ADME or Toxicity dataset (hERG, CYP2D6, Ames, etc.).

## Example output

```
[1/4] Initializing Chemical Featurizer...
[2/4] Loading TDC Dataset & Performing Scaffold Split...
[3/4] Tuning Hyperparameters & Training Model...
[+] Test ROC-AUC : 0.7432
[+] Test PR-AUC  : 0.6891
[4/4] Generating Substructure Explainability Visuals...
[+] Pipeline Execution Complete.
```

## Key features

- **Rigorous scaffolding:** Prevents data leakage by splitting train/val/test on Bemis-Murcko scaffolds
- **Hyperparameter optimization:** Optuna-based Bayesian search over XGBoost parameters
- **Explainability:** SHAP TreeExplainer with atomic attribution maps for model transparency
- **Production-ready data pipeline:** Automatic SMILES standardization, salt removal, and charge neutralization
- **Robust chemistry:** RDKit-based featurization with invalid molecule filtering

## How to extend

- **Custom datasets:** Update `src/train.py` to load from CSV or other sources instead of TDC
- **Alternative models:** Replace XGBoost in `tune_and_train_xgboost()` with sklearn RandomForest, LightGBM, or neural networks
- **Different features:** Modify `MoleculeFeaturizer` to add graph-based fingerprints (ECFP, RDKit hashing) or 3D descriptors
- **Ensemble methods:** Combine multiple trained models for improved robustness

## Requirements

See `requirements.txt` for full dependency list:
- rdkit
- PyTDC
- numpy
- pandas
- scikit-learn
- xgboost
- shap
- optuna
- matplotlib

## License

This project is provided as-is for research and development purposes.

## References

- [RDKit Documentation](https://www.rdkit.org/)
- [Therapeutics Data Commons (TDC)](https://tdcommons.ai/)
- [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/)
- [Optuna Hyperparameter Optimization](https://optuna.readthedocs.io/)
