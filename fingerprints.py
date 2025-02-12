import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from tqdm import tqdm
import os

# Load CSV
zinc_df = pd.read_csv(os.path.expanduser('~/250k_zinc.csv'))[['smiles']]  # Keep only 'SMILES' column
print(zinc_df.head())

# Create output directory
output_dir = "fingerprints_output"
os.makedirs(output_dir, exist_ok=True)

def calculate_morgan_fingerprint(smiles, radius=2, nBits=1024):
    """Calculate Morgan fingerprints for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp, dtype=np.int8)
    return np.zeros(nBits, dtype=np.int8)

def calculate_ap_fingerprint(smiles, nBits=2048):
    """Calculate Atom Pair fingerprints for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
        return np.array(fp, dtype=np.int8)
    return np.zeros(nBits, dtype=np.int8)

def calculate_rdk5_fingerprint(smiles, nBits=2048):
    """Calculate RDKit 5-bit hashed fingerprints for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = Chem.RDKFingerprint(mol)
        arr = np.zeros((nBits,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        return arr
    return np.zeros(nBits, dtype=np.int8)

def get_fingerprints_stream(df, smiles_column='smiles', fingerprint_type="morgan", nBits=1024, radius=2):
    """
    Generator function that yields fingerprint rows instead of storing all in memory.

    Parameters:
    - df (pd.DataFrame): DataFrame with a column containing SMILES strings
    - smiles_column (str): Column name containing SMILES strings
    - fingerprint_type (str): "morgan", "ap" (Atom Pair), or "rdk5" (RDKit 5-bit)
    - nBits (int): Number of bits for the fingerprint
    - radius (int): Radius for Morgan fingerprint (only used if fingerprint_type="morgan")

    Yields:
    - list: A row of fingerprint values
    """
    for smiles in tqdm(df[smiles_column], desc=f"Generating {fingerprint_type} fingerprints"):
        if fingerprint_type == "morgan":
            yield calculate_morgan_fingerprint(smiles, radius, nBits).tolist()
        elif fingerprint_type == "ap":
            yield calculate_ap_fingerprint(smiles, nBits).tolist()
        elif fingerprint_type == "rdk5":
            yield calculate_rdk5_fingerprint(smiles, nBits).tolist()

def save_fingerprints_to_csv(df, output_path, fingerprint_type="morgan", nBits=1024, radius=2):
    """
    Saves fingerprints to CSV as they are generated.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing SMILES strings
    - output_path (str): Path to save the CSV
    - fingerprint_type (str): Type of fingerprint ("morgan", "ap", or "rdk5")
    - nBits (int): Number of bits for the fingerprint
    - radius (int): Radius for Morgan fingerprints (ignored for others)
    """
    columns = [f'{fingerprint_type}_fp_{i}' for i in range(nBits)]
    
    # Open the file and write the header
    with open(output_path, 'w') as f:
        f.write(",".join(columns) + "\n")

    # Append rows as they are generated
    with open(output_path, 'a') as f:
        for row in get_fingerprints_stream(df, fingerprint_type=fingerprint_type, nBits=nBits, radius=radius):
            f.write(",".join(map(str, row)) + "\n")

# Generate and save fingerprints
save_fingerprints_to_csv(zinc_df, os.path.join(output_dir, "morgan_fingerprints.csv"), fingerprint_type="morgan", nBits=1024, radius=2)
save_fingerprints_to_csv(zinc_df, os.path.join(output_dir, "ap_fingerprints.csv"), fingerprint_type="ap", nBits=2048)
save_fingerprints_to_csv(zinc_df, os.path.join(output_dir, "rdk5_fingerprints.csv"), fingerprint_type="rdk5", nBits=2048)

print("Fingerprint generation and saving completed.")
