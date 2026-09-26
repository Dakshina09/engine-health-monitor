"""Loading raw C-MAPSS files and building RUL labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_COLS, RUL_CAP


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(RAW_COLS)]
    df.columns = RAW_COLS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def load_train(subset: str) -> pd.DataFrame:
    """Run-to-failure trajectories with RUL labels (true + capped)."""
    df = _read(DATA_DIR / f"train_{subset}.txt")
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL_true"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL_true"].clip(upper=RUL_CAP)
    df["life_frac"] = df["cycle"] / max_cycle          # 0 -> new, 1 -> failed
    return df


def load_test(subset: str) -> pd.DataFrame:
    """Truncated trajectories (engines still in service) + ground-truth RUL at every cycle."""
    df = _read(DATA_DIR / f"test_{subset}.txt")
    rul_path = DATA_DIR / f"RUL_{subset}.txt"
    if rul_path.exists():
        rul_end = pd.read_csv(rul_path, header=None, sep=r"\s+").iloc[:, 0].to_numpy()
        units = np.sort(df["unit"].unique())
        end_map = dict(zip(units, rul_end))
        last = df.groupby("unit")["cycle"].transform("max")
        df["RUL_true"] = df["unit"].map(end_map) + (last - df["cycle"])
        df["RUL"] = df["RUL_true"].clip(upper=RUL_CAP)
    return df
