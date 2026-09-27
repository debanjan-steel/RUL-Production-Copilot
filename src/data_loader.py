"""NASA C-MAPSS FD001-FD004 Data Ingestion and Dataset Management Module.

Handles loading, parsing, and automated acquisition of the turbofan
run-to-failure degradation benchmark datasets across all operating conditions.
"""

from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

# Standard 26 column schema of C-MAPSS dataset
COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"s{i}" for i in range(1, 22)]

# Sensors with zero or near-zero variance in single-condition benchmarks (FD001, FD003)
FLAT_SENSORS_SINGLE_COND = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Informative 14 sensors for FD001/FD003
INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

# For multi-condition benchmarks (FD002, FD004), all 21 sensors exhibit variation
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]

HUGGINGFACE_MIRROR_BASE = (
    "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main/"
)


def get_data_dir() -> Path:
    """Returns absolute path to the data directory, creating it if needed."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_cmapss_file(filename: str, dest_path: Path) -> bool:
    """Attempts to download a specific dataset file from the verified mirror."""
    url = HUGGINGFACE_MIRROR_BASE + filename
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"[Warning] Failed to download {filename} from mirror: {e}")
        return False


def load_cmapss_raw(dataset_id: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Loads raw C-MAPSS training, test, and ground-truth RUL files for any subset.
    
    Args:
        dataset_id: One of 'FD001', 'FD002', 'FD003', or 'FD004'.
        
    Returns:
        Tuple of (df_train, df_test, y_test_rul)
    """
    valid_subsets = ["FD001", "FD002", "FD003", "FD004"]
    if dataset_id not in valid_subsets:
        raise ValueError(f"Invalid dataset_id '{dataset_id}'. Choose from {valid_subsets}")

    data_dir = get_data_dir()
    train_path = data_dir / f"train_{dataset_id}.txt"
    test_path = data_dir / f"test_{dataset_id}.txt"
    rul_path = data_dir / f"RUL_{dataset_id}.txt"

    # Automatically fetch if missing
    for p, prefix in [(train_path, "train_"), (test_path, "test_"), (rul_path, "RUL_")]:
        if not p.exists():
            fname = f"{prefix}{dataset_id}.txt"
            print(f"[Data Loader] Fetching {fname} from mirror...")
            download_cmapss_file(fname, p)

    df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    y_test_rul = np.loadtxt(rul_path, dtype=np.float32)

    return df_train, df_test, y_test_rul
