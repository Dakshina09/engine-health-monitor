"""Central configuration for the Engine Health Monitoring pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_data_dir() -> Path:
    """$CMAPSS_DIR, else ./data, else the parent folder (project dropped inside CMAPSSData/)."""
    candidates = [os.getenv("CMAPSS_DIR"), ROOT / "data", ROOT.parent]
    for c in candidates:
        if c and (Path(c) / "train_FD001.txt").exists():
            return Path(c)
    return ROOT / "data"


DATA_DIR = _find_data_dir()
MODEL_DIR = ROOT / "models"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Column layout of the raw C-MAPSS text files (26 columns, space separated)
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting1", "setting2", "setting3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Physical meaning of each sensor (Saxena et al., PHM08) -> used to make SHAP explanations readable
SENSOR_INFO = {
    "s1":  ("T2",        "Fan inlet total temperature (°R)"),
    "s2":  ("T24",       "LPC outlet total temperature (°R)"),
    "s3":  ("T30",       "HPC outlet total temperature (°R)"),
    "s4":  ("T50",       "LPT outlet total temperature (°R)"),
    "s5":  ("P2",        "Fan inlet pressure (psia)"),
    "s6":  ("P15",       "Bypass-duct total pressure (psia)"),
    "s7":  ("P30",       "HPC outlet total pressure (psia)"),
    "s8":  ("Nf",        "Physical fan speed (rpm)"),
    "s9":  ("Nc",        "Physical core speed (rpm)"),
    "s10": ("epr",       "Engine pressure ratio (P50/P2)"),
    "s11": ("Ps30",      "HPC outlet static pressure (psia)"),
    "s12": ("phi",       "Fuel flow / Ps30 (pps/psi)"),
    "s13": ("NRf",       "Corrected fan speed (rpm)"),
    "s14": ("NRc",       "Corrected core speed (rpm)"),
    "s15": ("BPR",       "Bypass ratio"),
    "s16": ("farB",      "Burner fuel-air ratio"),
    "s17": ("htBleed",   "Bleed enthalpy"),
    "s18": ("Nf_dmd",    "Demanded fan speed (rpm)"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed (rpm)"),
    "s20": ("W31",       "HPT coolant bleed (lbm/s)"),
    "s21": ("W32",       "LPT coolant bleed (lbm/s)"),
}

# Number of operating regimes (FD002/FD004 fly in six flight conditions)
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

# Piece-wise linear RUL target: engines are "healthy" early on, so cap the label
RUL_CAP = 125

# Feature engineering windows (cycles)
ROLL_WINDOWS = (5, 15, 30)
SLOPE_WINDOW = 20

# ---- Risk & maintenance policy -------------------------------------------------
RISK_HORIZON = 30          # P(failure within this many cycles) drives the risk tier
LEAD_TIME = 10             # cycles needed to plan an inspection (parts, slot, crew)
CONSERVATIVE_Q = 0.10      # quantile of RUL used for scheduling (10th percentile)

# An engine lands in a tier if EITHER rule fires:
#   (tier, min P(fail within RISK_HORIZON), max conservative RUL (P10))
#   Critical is driven only by the conservative (P10) RUL: failure is imminent
RISK_TIERS = [
    ("Critical", None, 15),
    ("High",     0.20, 35),
    ("Medium",   0.05, 60),
]
# anything else -> "Low"
ROUTINE_INTERVAL = 50      # cycles until next routine check for low-risk engines
