"""Turbofan Signal Preprocessing, Savitzky-Golay Denoising, and Windowing.

Prepares raw engine cycle telemetry into temporal window tensors for
CNN-Transformer ingestion with zero data leakage.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import INFORMATIVE_SENSORS


def compute_piecewise_rul(df: pd.DataFrame, max_rul: float = 125.0) -> pd.DataFrame:
    """Calculates ground truth RUL per engine trajectory capped at max_rul.
    
    Piecewise linear degradation assumes machinery health remains constant
    during initial operating life before measurable fatigue/wear onset.
    """
    df = df.copy()
    # Compute maximum cycle reached per engine
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id", how="left")
    
    # Calculate raw and piecewise linear RUL
    raw_rul = df["max_cycle"] - df["cycle"]
    df["RUL"] = raw_rul.clip(upper=max_rul).astype(np.float32)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def apply_savitzky_golay(
    df: pd.DataFrame,
    features: list[str] = INFORMATIVE_SENSORS,
    window_length: int = 15,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Applies Savitzky-Golay polynomial filtering per engine trajectory.
    
    Preserves degradation inflection points and peaks while removing
    high-frequency sensor measurement jitter without phase lag.
    """
    df = df.copy()
    smoothed_dfs = []
    
    for _, group in df.groupby("engine_id"):
        group = group.sort_values("cycle").copy()
        n_samples = len(group)
        
        # Window length must be odd and <= number of samples
        wl = window_length if window_length % 2 != 0 else window_length + 1
        if n_samples < wl:
            wl = n_samples if n_samples % 2 != 0 else max(3, n_samples - 1)
            
        if wl > polyorder:
            for feat in features:
                group[feat] = savgol_filter(group[feat].values, window_length=wl, polyorder=polyorder)
        smoothed_dfs.append(group)
        
    return pd.concat(smoothed_dfs, ignore_index=True)


class CmapssPreprocessor:
    """End-to-end preprocessor managing normalization and tensor windowing."""

    def __init__(self, sequence_length: int = 30, max_rul: float = 125.0):
        self.sequence_length = sequence_length
        self.max_rul = max_rul
        self.features = INFORMATIVE_SENSORS
        self.scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        self.is_fitted = False

    def fit_transform(
        self, df_train: pd.DataFrame, use_sg: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fits scaler on training set, applies SG filter, and creates train tensors.
        
        Returns:
            X_train: Array of shape (N, sequence_length, n_features)
            y_train: Array of shape (N,)
        """
        df_train = compute_piecewise_rul(df_train, max_rul=self.max_rul)
        
        if use_sg:
            df_train = apply_savitzky_golay(df_train, features=self.features)

        # Fit scaler ONLY on training data to avoid data leakage
        df_train[self.features] = self.scaler.fit_transform(df_train[self.features])
        self.is_fitted = True

        X_list, y_list = [], []
        for _, group in df_train.groupby("engine_id"):
            data_arr = group[self.features].values
            labels_arr = group["RUL"].values
            n_samples = len(group)

            if n_samples >= self.sequence_length:
                for i in range(n_samples - self.sequence_length + 1):
                    X_list.append(data_arr[i : i + self.sequence_length])
                    y_list.append(labels_arr[i + self.sequence_length - 1])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def transform_test(
        self, df_test: pd.DataFrame, use_sg: bool = True
    ) -> np.ndarray:
        """Transforms test set taking the last sequence_length cycles per engine.
        
        Returns:
            X_test: Array of shape (num_engines, sequence_length, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming test data.")

        df_test = df_test.copy()
        if use_sg:
            df_test = apply_savitzky_golay(df_test, features=self.features)

        df_test[self.features] = self.scaler.transform(df_test[self.features])

        X_test = []
        for _, group in df_test.groupby("engine_id"):
            data_arr = group[self.features].values
            n_samples = len(data_arr)

            if n_samples >= self.sequence_length:
                seq = data_arr[-self.sequence_length :]
            else:
                # Pad sequence with edge replication if trajectory is shorter than window
                pad_len = self.sequence_length - n_samples
                pad = np.repeat(data_arr[:1], pad_len, axis=0)
                seq = np.vstack([pad, data_arr])
            X_test.append(seq)

        return np.array(X_test, dtype=np.float32)

    def transform_single_window(self, sequence: list[list[float]]) -> np.ndarray:
        """Preprocesses a single sequence window for FastAPI production serving.
        
        Args:
            sequence: List of cycle sensor readings, shape [time_steps, n_sensors]
        """
        arr = np.array(sequence, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # If incoming data has all 21 sensors or full 26 columns, extract the 14 informative ones
        if arr.shape[1] == len(INFORMATIVE_SENSORS):
            scaled = self.scaler.transform(arr) if self.is_fitted else arr
        elif arr.shape[1] >= 21:
            # Assumes standard C-MAPSS 21 sensors order
            informative_indices = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]
            extracted = arr[:, informative_indices]
            scaled = self.scaler.transform(extracted) if self.is_fitted else extracted
        else:
            scaled = arr

        # Ensure sequence length matches
        if len(scaled) >= self.sequence_length:
            final_seq = scaled[-self.sequence_length :]
        else:
            pad_len = self.sequence_length - len(scaled)
            pad = np.repeat(scaled[:1], pad_len, axis=0)
            final_seq = np.vstack([pad, scaled])

        return np.expand_dims(final_seq, axis=0).astype(np.float32)
