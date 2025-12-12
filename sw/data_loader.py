import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic mouse motion features: dt, dx, dy, speed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: ["timestamp_ms", "x", "y"]

    Returns
    -------
    pd.DataFrame
        DataFrame with added motion features.
    """
    df = df.copy()

    # ensure numeric
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.sort_values("timestamp_ms")

    df["dt"] = df["timestamp_ms"].diff().fillna(1).clip(lower=1)
    df["dx"] = df["x"].diff().fillna(0)
    df["dy"] = df["y"].diff().fillna(0)
    df["speed"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5 / df["dt"]

    return df

def parse_custom_event(row: dict) -> dict:
    """
    Normalize a raw JSON event into a uniform schema.

    Parameters
    ----------
    row : dict
        Raw event dict loaded from JSON.

    Returns
    -------
    dict
        Normalized row with:
        - timestamp
        - event type
        - coordinates
        - process info (pid, process_name)
        - label placeholder fields
    """
    event_type = row.get("event_type")

    base = {
        "timestamp_ms": row.get("monotonic_ms") or row.get("wall_time_ms"),
        "session_id": row.get("session_id"),
        "event_type": event_type,
        "pid": row.get("foreground", {}).get("pid"),
        "process_name": row.get("foreground", {}).get("process_name"),
        "source": "custom",
        "x": None,
        "y": None,
        "key": None,
        "type": "other",
    }

    if event_type in {"pointermove", "click", "pointerdown", "pointerup"}:
        base["type"] = "mouse"
        base["x"] = row.get("x_screen")
        base["y"] = row.get("y_screen")

    elif isinstance(event_type, str) and "key" in event_type:
        base["type"] = "key"
        base["key"] = row.get("key")

    return base


def load_custom_json_file(path: Path) -> pd.DataFrame:
    """
    Load .json or .jsonl mouse/key logs from your bot or from PMC.

    Parameters
    ----------
    path : Path
        Path to JSON line file.

    Returns
    -------
    pd.DataFrame
        Normalized event list.
    """
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(parse_custom_event(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue

    return pd.DataFrame(records)

def load_pmc_file(path: Path) -> pd.DataFrame:
    """
    Load PMC CSV format and normalize column names.

    Parameters
    ----------
    path : Path
        CSV file path.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)

    df["timestamp_ms"] = df["client_timestamp"]
    df["x"] = df.get("x")
    df["y"] = df.get("y")

    df["button"] = df.get("button")
    df["state"] = df.get("state")
    df["window"] = df.get("window")

    df["source"] = "pmc"
    df["session_id"] = df.get("session_id", path.stem)

    return df


def load_single_file(path: Path) -> pd.DataFrame:
    """
    Load a single dataset file, automatically detecting format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    suffix = path.suffix.lower()

    if suffix in {".json", ".jsonl"}:
        return load_custom_json_file(path)

    if suffix == ".csv":
        return load_pmc_file(path)

    return pd.DataFrame()


def load_folder(
    folder: str | Path,
    *,
    features: bool = True,
    label: Optional[int] = None,
    test_split: Optional[float] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load an entire folder of JSON/CSV files.

    Parameters
    ----------
    folder : str | Path
        Root folder with logs.

    features : bool
        Whether to add basic dx/dy/speed features.

    label : int, optional
        If provided → assign label to each session.

    test_split : float, optional
        If provided → split into (train, test).

    Returns
    -------
    (train_df, test_df)
        test_df = None if no split is requested.
    """
    folder = Path(folder)

    dfs = []
    for file in folder.rglob("*"):
        if not file.is_file():
            continue

        df = load_single_file(file)
        if df.empty:
            continue

        if label is not None:
            df["label"] = label

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid dataset files found in {folder}")

    df = pd.concat(dfs, ignore_index=True)

    df = df.dropna(subset=["timestamp_ms"])

    if features:
        df = add_basic_features(df)

    df = df.reset_index(drop=True)

    if test_split:
        train_df, test_df = train_test_split(
            df, test_size=test_split, shuffle=False
        )
        return train_df, test_df

    return df, None

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Standard tabular train/test split and scaling.

    Parameters
    ----------
    X : np.ndarray
    y : pd.Series

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), scaler

def df_to_sequence_windows(
    df: pd.DataFrame,
    window_size: int = 100,
    step: Optional[int] = None,
    feature_cols: Optional[list[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert event dataframe → sequence windows for RNNs.

    Parameters
    ----------
    df : pd.DataFrame
        Combined event dataset.

    window_size : int
        Length of each sliding window.

    step : int, optional
        Step size (default = window_size = non-overlapping).

    feature_cols : list[str], optional
        Which columns to use as features.

    Returns
    -------
    X : np.ndarray
        Shape: (num_windows, window_size, num_features)

    y : np.ndarray
        Window-level labels if available.
    """
    step = step or window_size

    if feature_cols is None:
        feature_cols = ["dx", "dy", "dt", "speed"] 

    X_windows = []
    y_windows = []

    for session_id, sdf in df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)

        if "type" in sdf.columns:
            sdf = sdf[sdf["type"] == "mouse"].reset_index(drop=True)

        sdf["dt"] = sdf["timestamp_ms"].diff().fillna(1).clip(lower=1)
        sdf["dx"] = sdf["x"].diff().fillna(0)
        sdf["dy"] = sdf["y"].diff().fillna(0)
        sdf["speed"] = (sdf["dx"]**2 + sdf["dy"]**2) ** 0.5 / sdf["dt"]

        n = len(sdf)
        if n < window_size:
            continue

        for start in range(0, n - window_size + 1, step):
            w = sdf.iloc[start:start + window_size]

            X_windows.append(w[feature_cols].to_numpy(dtype=np.float32))

            if "label" in w.columns:
                y_windows.append(int(w["label"].iloc[0]))

    if not X_windows:
        raise ValueError("No windows created – try smaller window_size or check data.")

    X = np.stack(X_windows, axis=0)
    y = np.array(y_windows, dtype=np.int64)

    return X, y

def split_and_scale_sequence(X, y, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split RNN sequence data into train/val/test + scale features.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, T, F)
    y : np.ndarray
    val_size : float
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    scaler : StandardScaler
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=random_state)

    n_features = X_train.shape[2]
    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat   = X_val.reshape(-1, n_features)
    X_test_flat  = X_test.reshape(-1, n_features)

    X_train = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val   = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test  = scaler.transform(X_test_flat).reshape(X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
