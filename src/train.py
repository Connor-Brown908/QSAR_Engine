import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
from tdc.single_pred import ADME

class QSARModelTrainer:
    def __init__(self, dataset_name="hERG"):
        # Load dataset from Therapeutics Data Commons
        self.data_loader = ADME(name=dataset_name)
        # Apply strict Bemis-Murcko scaffold splitting (80/10/10)
        self.splits = self.data_loader.get_split(method="scaffold", seed=42)
        
    def prepare_datasets(self, featurizer):
        train_df = self.splits["train"]
        val_df = self.splits["valid"]
        test_df = self.splits["test"]

        X_train, valid_idx_tr = featurizer.featurize_smiles_list(train_df["Drug"].values)
        y_train = train_df["Y"].iloc[valid_idx_tr].values

        X_val, valid_idx_v = featurizer.featurize_smiles_list(val_df["Drug"].values)
        y_val = val_df["Y"].iloc[valid_idx_v].values

        X_test, valid_idx_te = featurizer.featurize_smiles_list(test_df["Drug"].values)
        y_test = test_df["Y"].iloc[valid_idx_te].values

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def tune_and_train_xgboost(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eval_metric": "logloss",
                "random_state": 42
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=15)
        
        print(f"[+] Best Validation ROC-AUC: {study.best_value:.4f}")
        
        best_model = xgb.XGBClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        return best_model