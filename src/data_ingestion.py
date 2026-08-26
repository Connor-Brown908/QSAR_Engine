import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tdc.single_pred import ADME, Tox


class DataIngestionPipeline:
    """Production-grade data loader, standardizer, and splitter for TDC datasets."""

    def __init__(self, problem_type="ADME", dataset_name="hERG", output_dir="./data"):
        self.problem_type = problem_type
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, dataset_name.lower())
        os.makedirs(self.output_dir, exist_ok=True)

        # Standardizer components from RDKit
        self.uncharger = rdMolStandardize.Uncharger()

    def fetch_data(self):
        """Downloads raw data from Therapeutics Data Commons (PyTDC)."""
        print(f"[1/4] Fetching dataset '{self.dataset_name}' from TDC ({self.problem_type})...")
        if self.problem_type.upper() == "ADME":
            data_loader = ADME(name=self.dataset_name)
        elif self.problem_type.upper() in ["TOX", "TOXICITY"]:
            data_loader = Tox(name=self.dataset_name)
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

        # Extract scaffold splits directly from TDC
        splits = data_loader.get_split(method="scaffold", seed=42, frac=[0.8, 0.1, 0.1])
        return splits

    def sanitize_molecule(self, smiles: str):
        """Applies chemical standardization:

        1. Parses SMILES to RDKit Mol object
        2. Clears salts and disconnects metal complexes
        3. Neutralizes charges
        4. Re-canonicalizes SMILES
        """
        if not smiles or not isinstance(smiles, str):
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # 1. Remove salts / get largest fragment
            mol = rdMolStandardize.FragmentParent(mol)

            # 2. Neutralize charges where possible
            mol = self.uncharger.unread(mol) if hasattr(self.uncharger, 'unread') else self.uncharger.uncharge(mol)

            # 3. Canonicalize SMILES
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return canonical_smiles
        except Exception:
            return None

    def process_and_save_split(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Cleans SMILES and filters out invalid structures."""
        print(f"[2/4] Standardizing SMILES in {split_name} split ({len(df)} rows)...")

        # Rename columns to standard schema
        df = df.rename(columns={"Drug": "SMILES", "Y": "Target"})

        # Sanitize SMILES
        df["SMILES_clean"] = df["SMILES"].apply(self.sanitize_molecule)

        # Drop invalid SMILES or missing target values
        initial_count = len(df)
        df = df.dropna(subset=["SMILES_clean", "Target"]).reset_index(drop=True)
        dropped_count = initial_count - len(df)

        if dropped_count > 0:
            print(f"      dropped {dropped_count} invalid/unparseable SMILES.")

        # Save to disk as clean CSV
        output_path = os.path.join(self.output_dir, f"{split_name}.csv")
        df[["SMILES_clean", "Target"]].to_csv(output_path, index=False)
        print(f"[+] Saved clean '{split_name}' split to {output_path}")

        return df

    def run(self):
        """Executes the complete data ingestion sequence."""
        splits = self.fetch_data()
        
        processed_splits = {}
        for split_name in ["train", "valid", "test"]:
            if split_name in splits:
                processed_splits[split_name] = self.process_and_save_split(
                    splits[split_name], split_name
                )
                
        print(f"\n[✓] Data Ingestion Complete! Cleaned files ready in: {self.output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and clean SMILES datasets for QSAR modeling.")
    parser.add_argument("--type", type=str, default="ADME", help="Problem type: ADME or Tox")
    parser.add_argument("--dataset", type=str, default="hERG", help="Target dataset name (e.g., hERG, CYP2D6, Ames)")
    parser.add_argument("--outdir", type=str, default="./data", help="Output directory")

    args = parser.parse_args()

    ingestor = DataIngestionPipeline(
        problem_type=args.type,
        dataset_name=args.dataset,
        output_dir=args.outdir
    )
    ingestor.run()