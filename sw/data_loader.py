import json
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
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
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(parse_custom_event(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue

    return pd.DataFrame(records)

def load_pmc_file(path: Path) -> pd.DataFrame:
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

def df_to_windows(
    df: pd.DataFrame,
    window_size: int = 100,
    step: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    step = step or window_size

    features = []
    labels = []

    for session_id, sdf in df.groupby("session_id"):
        sdf = sdf.reset_index(drop=True)

        for start in range(0, len(sdf) - window_size + 1, step):
            w = sdf.iloc[start:start + window_size]

            feat = {
                "mean_speed": w["speed"].mean(),
                "std_speed": w["speed"].std(),
                "mean_dx": w["dx"].mean(),
                "mean_dy": w["dy"].mean(),
            }
            features.append(feat)

            if "label" in w.columns:
                labels.append(w["label"].iloc[0])

    X = pd.DataFrame(features).fillna(0)
    y = pd.Series(labels) if labels else None

    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), scaler