"""
Feature Selection (NumPy Edition) — iBudget / Florida APD

Pandas-free pipeline:
- Mutual Information (with permutation baseline enabled by default)
- Spearman ρ (continuous↔continuous)
- Bias-corrected Cramér’s V with chi-square validity screening (categorical↔categorical)
- Bias-adjusted correlation ratio η (epsilon-squared) for categorical→continuous
- Optional VIF screen for multicollinearity (continuous features)
- Redundancy filtering across type pairs (configurable breadth)
- Heatmaps, pairplots, LaTeX table and commands

Outputs
-------
report/logs/FeatureSelection.txt
report/logs/FeatureSelectionSummary.csv
report/logs/TopFeaturesTable.tex
report/logs/FeatureSelectionCommands.tex
report/figures/*.png
"""

from __future__ import annotations

import os
import re
import csv
import math
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression

from models.base_model import BaseiBudgetModel  # project logger


# ------------------------- Defaults / Tunables -------------------------

OUTPUT_ROOT = Path("../report")
FIG_DIR = OUTPUT_ROOT / "figures"   # -> ../report/figures
LOG_DIR = OUTPUT_ROOT / "logs"      # -> ../report/logs
CMD_DIR = LOG_DIR                   # commands alongside logs


VARS: List[str] = [
    "Age", "GENDER", "RACE", "Ethnicity", "County",
    "PrimaryDiagnosis", "SecondaryDiagnosis", "OtherDiagnosis",
    "MentalHealthDiag1", "MentalHealthDiag2", "DevelopmentalDisability",
    "RESIDENCETYPE", "LivingSetting", "AgeGroup",
    "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22",
    "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30",
    "Q31a", "Q31b", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37",
    "Q38", "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45",
    "Q46", "Q47", "Q48", "Q49", "Q50",
    "FSum", "BSum", "PSum", "FLEVEL", "BLEVEL", "PLEVEL", "OLEVEL",
    "LOSRI", "TotalCost"
]

CATEGORICAL: set = {
    "GENDER", "RACE", "Ethnicity", "County",
    "PrimaryDiagnosis", "SecondaryDiagnosis", "OtherDiagnosis",
    "MentalHealthDiag1", "MentalHealthDiag2", "DevelopmentalDisability",
    "RESIDENCETYPE", "LivingSetting", "AgeGroup",
    "FLEVEL", "BLEVEL", "PLEVEL", "OLEVEL"
}

FILTERS = {"LateEntry": 0, "EarlyExit": 0, "MissingQSI": 0, "InsufficientDays": 0}
MISSING = "__MISSING__"
MAX_CARDINALITY_FOR_CRAMERS = 50
MAX_CATEGORICAL_FOR_MATRIX = 25
ETA_PAIR_LIMIT = 500
PAIRPLOT_SAMPLE = 1000
RANDOM_STATE = 42

REDUNDANCY_MAX_FEATURES: Optional[int] = 100  # None = scan all

# Chi-square validity rules
CHI2_MIN_EXPECTED = 1.0
CHI2_MIN_PROP_GE5 = 0.80


# ------------------------- Logger -------------------------

class FeatureSelectionJob(BaseiBudgetModel):
    def prepare_features(self, records):
        return None, []
    def _fit_core(self, X, y):
        return None
    def _predict_core(self, X):
        return None

def get_ibudget_logger():
    job = FeatureSelectionJob(model_id=99, model_name="FeatureSelection", use_outlier_removal=False)
    return job, job.logger



# ------------------------- Utilities -------------------------

def _is_missing(arr: np.ndarray) -> np.ndarray:
    """
    Vectorized missing detector:
    - float: np.isnan
    - int/uint: no missing
    - object: True for None or NaN
    """
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in ("i", "u"):
        return np.zeros(arr.shape, dtype=bool)
    is_none = np.vectorize(lambda x: x is None, otypes=[bool])(arr)
    is_nan = np.vectorize(lambda x: (isinstance(x, float) and math.isnan(x)), otypes=[bool])(arr)
    return np.logical_or(is_none, is_nan)


def _sanitize_suffix(s: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]", "_", s.lower())
    return re.sub(r"_+", "_", safe).strip("_-")


def _latex_escape(text: str) -> str:
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(c, c) for c in text)


# ------------------------- Association Metrics -------------------------

def _collapse_rare(levels: np.ndarray, min_freq: int = 25) -> np.ndarray:
    """
    Normalize missing to sentinel, cast to string labels, then collapse rare levels.
    """
    levels = np.asarray(levels, dtype=object)
    # replace None/NaN with sentinel, then cast to str so np.unique won't compare unlike types
    levels = np.where(_is_missing(levels), MISSING, levels).astype(str)
    uniq, inv, counts = np.unique(levels, return_inverse=True, return_counts=True)
    rare = set(uniq[counts < min_freq])
    if rare:
        mask = np.isin(levels, list(rare))
        levels[mask] = "__OTHER__"
    return levels


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Build contingency table with normalized string labels (no mixed None/str).
    """
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    a = np.where(_is_missing(a), MISSING, a).astype(str)
    b = np.where(_is_missing(b), MISSING, b).astype(str)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    tbl = np.zeros((ua.size, ub.size), dtype=int)
    np.add.at(tbl, (ia, ib), 1)
    return tbl



def _chi2_expected_ok(tbl: np.ndarray) -> bool:
    if tbl.sum() == 0:
        return False
    chi2, p, dof, expected = chi2_contingency(tbl, correction=False)
    if np.any(expected < CHI2_MIN_EXPECTED):
        return False
    prop_ge5 = np.mean(expected >= 5)
    return prop_ge5 >= CHI2_MIN_PROP_GE5


def cramers_v_corrected(a: np.ndarray, b: np.ndarray) -> float:
    a2 = _collapse_rare(a)
    b2 = _collapse_rare(b)
    tbl = _contingency(a2, b2)
    n = tbl.sum()
    if n <= 1:
        return 0.0
    try:
        if not _chi2_expected_ok(tbl):
            return 0.0
        chi2, _, _, _ = chi2_contingency(tbl, correction=False)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0
    r, k = tbl.shape
    phi2 = max(0.0, (chi2 / n) - ((r - 1) * (k - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(r_corr - 1.0, k_corr - 1.0)
    if (not np.isfinite(phi2)) or (not np.isfinite(denom)) or denom <= 0.0:
        return 0.0
    val = math.sqrt(phi2 / denom)
    return 0.0 if not np.isfinite(val) else float(val)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return 0.0
    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else 0.0


def correlation_ratio_eta(cat: np.ndarray, cont: np.ndarray) -> float:
    """
    Returns η = sqrt(ε²), ε² uses epsilon-squared with df correction:
        ε² = (SS_between - (k-1)*MS_within) / SS_total
    Requires n > k and n ≥ 3.
    """
    miss_cat = _is_missing(cat)
    miss_cont = ~np.isfinite(cont)
    mask = ~(miss_cat | miss_cont)
    cat = cat[mask].astype(object)
    cont = cont[mask].astype(float)
    n = cont.size
    if n < 3:
        return 0.0

    cat = _collapse_rare(cat)
    levels = np.unique(cat)
    k = levels.size
    if k < 2 or n <= k:
        return 0.0

    gm = cont.mean()
    ss_total = float(((cont - gm) ** 2).sum())
    if ss_total == 0.0:
        return 0.0

    ss_between = 0.0
    ss_within = 0.0
    for lev in levels:
        sel = (cat == lev)
        if not sel.any():
            continue
        grp = cont[sel]
        m = grp.mean()
        ss_between += grp.size * (m - gm) ** 2
        ss_within += float(((grp - m) ** 2).sum())

    df_within = n - k
    if df_within <= 0:
        return 0.0
    ms_within = ss_within / df_within
    eps2 = (ss_between - (k - 1) * ms_within) / ss_total
    eps2 = max(0.0, eps2)
    val = math.sqrt(eps2)
    return 0.0 if not np.isfinite(val) else float(val)


# ------------------------- Encoding / Imputation -------------------------

def encode_and_impute(arrs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, LabelEncoder]]:
    encoded: Dict[str, np.ndarray] = {}
    encoders: Dict[str, LabelEncoder] = {}

    for name, col in arrs.items():
        col = np.asarray(col)
        if name in CATEGORICAL:
            miss_mask = _is_missing(col)
            col2 = np.where(miss_mask, MISSING, col).astype(str)
            le = LabelEncoder()
            encoded[name] = le.fit_transform(col2).astype(float)
            encoders[name] = le
        else:
            miss_mask = _is_missing(col)
            num = np.empty(col.shape, dtype=float)
            try:
                num[~miss_mask] = col[~miss_mask].astype(float, copy=False)
            except (ValueError, TypeError):
                num[~miss_mask] = np.fromiter((float(x) for x in col[~miss_mask]), dtype=float, count=np.count_nonzero(~miss_mask))
            num[miss_mask] = np.nan
            med = np.nanmedian(num) if np.any(~np.isnan(num)) else 0.0
            num = np.where(np.isnan(num), med, num)
            encoded[name] = num

    return encoded, encoders


def _nonconstant_feature_names(encoded: Dict[str, np.ndarray]) -> List[str]:
    names = []
    for n, col in encoded.items():
        if n == "TotalCost":
            continue
        if n in CATEGORICAL:
            if np.unique(col).size > 1:
                names.append(n)
        else:
            if np.std(col) > 0:
                names.append(n)
    return names


def design_matrix(encoded: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    names = _nonconstant_feature_names(encoded)
    X = np.column_stack([encoded[n] for n in names]).astype(float) if names else np.zeros((len(encoded["TotalCost"]), 0), dtype=float)
    y = encoded["TotalCost"].astype(float)
    discrete_mask = np.array([n in CATEGORICAL for n in names], dtype=bool)
    return X, y, names, discrete_mask


# ------------------------- VIF Screen (continuous) -------------------------

def _vif_screen(encoded: Dict[str, np.ndarray], names: List[str], vif_threshold: float = 10.0) -> List[str]:
    cont = [n for n in names if n not in CATEGORICAL]
    if len(cont) < 2:
        return []

    X = np.column_stack([encoded[n] for n in cont]).astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    keep = list(range(X.shape[1]))
    dropped = []

    while True:
        worst_vif = 0.0
        worst_j = None
        for j in list(keep):
            others = [k for k in keep if k != j]
            if not others:
                continue
            Xj = X[:, j]
            Xo = X[:, others]
            Xo_aug = np.column_stack([np.ones(Xo.shape[0]), Xo])
            try:
                beta, _, _, _ = np.linalg.lstsq(Xo_aug, Xj, rcond=None)
                y_hat = Xo_aug @ beta
                ss_res = float(((Xj - y_hat) ** 2).sum())
                ss_tot = float(((Xj - Xj.mean()) ** 2).sum())
                r2 = 0.0 if ss_tot == 0.0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
                vif = float(1.0 / max(1e-12, (1.0 - r2)))
            except (np.linalg.LinAlgError, ValueError, TypeError, ZeroDivisionError):
                vif = 0.0
            if np.isfinite(vif) and vif > worst_vif:
                worst_vif = vif
                worst_j = j
        if worst_j is not None and worst_vif > vif_threshold:
            dropped.append(cont[worst_j])
            keep.remove(worst_j)
        else:
            break
    return dropped


# ------------------------- MI -------------------------

def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    discrete_mask: np.ndarray,
    *,
    permutation_baseline_reps: int = 10,
    rng: np.random.Generator | None = None
) -> List[Tuple[str, float, str]]:
    if X.shape[1] == 0:
        return []
    base_mi = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=RANDOM_STATE)
    adj_mi = base_mi.copy()
    if permutation_baseline_reps and permutation_baseline_reps > 0:
        rng = rng or np.random.default_rng(RANDOM_STATE)
        baselines = np.zeros_like(base_mi)
        for _ in range(permutation_baseline_reps):
            y_perm = rng.permutation(y)
            baselines += mutual_info_regression(X, y_perm, discrete_features=discrete_mask, random_state=RANDOM_STATE)
        baselines /= float(permutation_baseline_reps)
        adj_mi = np.maximum(0.0, base_mi - baselines)
    rows = [(n, float(s), "Categorical" if d else "Continuous") for n, s, d in zip(names, adj_mi, discrete_mask)]
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows


# ------------------------- Matrices & Plots -------------------------

def continuous_corr_matrix(encoded: Dict[str, np.ndarray], names: List[str]) -> Tuple[np.ndarray | None, List[str]]:
    cont_names = [n for n in names if n not in CATEGORICAL]
    if not cont_names:
        return None, []
    k = len(cont_names)
    M = np.eye(k, dtype=float)
    cols = [encoded[n].astype(float) for n in cont_names]
    for i in range(k):
        for j in range(i + 1, k):
            rho = spearman_corr(cols[i], cols[j])
            M[i, j] = M[j, i] = rho
    return M, cont_names


def categorical_cramers_matrix(arrs_orig: Dict[str, np.ndarray], cat_names: List[str]) -> Tuple[np.ndarray | None, List[str]]:
    # normalize and compute cardinality safely
    def _card(col: np.ndarray) -> int:
        col = np.asarray(col, dtype=object)
        col = np.where(_is_missing(col), MISSING, col).astype(str)
        return np.unique(col).size

    high_card = sum(1 for c in cat_names if _card(arrs_orig[c]) > MAX_CARDINALITY_FOR_CRAMERS)
    if high_card > 3:
        return None, []
    if len(cat_names) > MAX_CATEGORICAL_FOR_MATRIX:
        # pick by non-missing prevalence
        def _present_count(col: np.ndarray) -> int:
            col = np.asarray(col, dtype=object)
            return int(np.count_nonzero(~_is_missing(col)))
        cat_names = sorted(cat_names, key=lambda c: _present_count(arrs_orig[c]), reverse=True)[:MAX_CATEGORICAL_FOR_MATRIX]
    k = len(cat_names)
    if k < 2:
        return None, []
    M = np.eye(k, dtype=float)
    cols = [arrs_orig[c] for c in cat_names]
    for i in range(k):
        for j in range(i + 1, k):
            try:
                v = cramers_v_corrected(cols[i], cols[j])
            except (ValueError, TypeError, ZeroDivisionError):
                v = 0.0
            M[i, j] = M[j, i] = v if np.isfinite(v) else 0.0
    return M, cat_names



def mixed_eta_matrix(arrs_orig: Dict[str, np.ndarray], cat_names: List[str], cont_names: List[str]) -> Tuple[np.ndarray | None, List[str], List[str]]:
    if not cat_names or not cont_names:
        return None, [], []
    if len(cat_names) * len(cont_names) > ETA_PAIR_LIMIT:
        cat_names = sorted(cat_names, key=lambda c: np.count_nonzero(~_is_missing(arrs_orig[c])), reverse=True)[:10]
        cont_scores = []
        cont_keep = []
        for c in cont_names:
            if c == "TotalCost":
                continue
            try:
                rho = spearman_corr(arrs_orig[c].astype(float), arrs_orig["TotalCost"].astype(float))
            except (ValueError, TypeError):
                rho = 0.0
            cont_scores.append(abs(rho))
            cont_keep.append(c)
        idx = np.argsort(cont_scores)[::-1][:10]
        cont_names = [cont_keep[i] for i in idx]
    R, C = len(cat_names), len(cont_names)
    M = np.zeros((R, C), dtype=float)
    for i, cat in enumerate(cat_names):
        for j, cont in enumerate(cont_names):
            try:
                eta = correlation_ratio_eta(arrs_orig[cat], arrs_orig[cont].astype(float))
            except (ValueError, TypeError, ZeroDivisionError):
                eta = 0.0
            M[i, j] = eta if np.isfinite(eta) else 0.0
    return M, cat_names, cont_names


def _plot_matrix(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], fy: int, title_suffix: str, kind: str, logger):
    if matrix is None:
        return
    n_rows, n_cols = matrix.shape
    n_vars = max(n_rows, n_cols)
    if n_vars > 30:
        fig_size = (20, 16)
    elif n_vars > 20:
        fig_size = (16, 14)
    else:
        fig_size = (12, 10)
    plt.figure(figsize=fig_size)
    if kind in ("cramers_v", "eta"):
        vmin, vmax, cmap, cbar_label = 0, 1, "YlOrRd", ("Cramér's V" if kind == "cramers_v" else "Correlation ratio η")
    else:
        vmin, vmax, cmap, cbar_label = -1, 1, "coolwarm", "Spearman ρ"
    mat = matrix
    if n_rows == n_cols:
        mask = np.triu(np.ones_like(mat, dtype=bool))
        mat = np.where(mask, np.nan, mat)
    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f"FY{fy} {title_suffix}", fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=90)
    plt.xticks(range(n_cols), col_labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(n_rows), row_labels, fontsize=8)
    plt.gca().set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    plt.grid(which="minor", color=(0, 0, 0, 0.1), linestyle="-", linewidth=0.5)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_suffix(title_suffix)
    fname = FIG_DIR / (f"fy{fy}_{safe}.png" if safe else f"fy{fy}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {fname}")


# ------------------------- Data IO -------------------------

def _find_year_file(data_dir: Path, fy: int) -> Path:
    for name in (f"fy{fy}.pkl", f"fy{fy}_data.pkl", f"FY{fy}_data.pkl", f"{fy}_data.pkl"):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find data file for FY{fy}")


def _load_year_records(data_dir: Path, fy: int) -> List[Any]:
    """Load FY pickle and normalize to a list of record-like objects."""
    p = _find_year_file(data_dir, fy)
    with open(p, "rb") as f:
        obj = pickle.load(f)

    # If already a list, use it directly
    if isinstance(obj, list):
        return obj

    # If dict: look for a list of rows under common keys
    if isinstance(obj, dict):
        for k in ("records", "rows", "data"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        # last resort: if values look like rows, flatten them
        for v in obj.values():
            if isinstance(v, list):
                return v

    # If it's any other iterable (e.g., numpy array), make a list
    try:
        return list(obj)
    except Exception:
        # Fallback: wrap single object
        return [obj]

def _get_field(rec: Any, key: str, default=None):
    """
    Safely extract a field from various record shapes:
    - dict-like: r.get(key)
    - attribute-style: getattr(r, key)
    - mapping with __getitem__: r[key]
    Falls back to `default` if not found.
    """
    # dict-like
    if isinstance(rec, dict):
        return rec.get(key, default)
    # attribute-style
    try:
        if hasattr(rec, key):
            return getattr(rec, key)
    except Exception:
        pass
    # mapping / row-like
    try:
        return rec[key]
    except Exception:
        return default


def _filter_records(records: List[Any], logger=None) -> List[Any]:
    """
    Apply quality filters if the record structure exposes the expected flags.
    If it does not, skip filtering and log a note (to avoid crashing or dropping all rows).
    """
    if not records:
        return records

    # Check whether at least one record exposes all filter flags
    sample = records[0]
    can_filter = all(_get_field(sample, k, None) is not None for k in FILTERS.keys())

    if not can_filter:
        if logger:
            logger.info("Quality filter fields not found on record; skipping quality filters for this year.")
        return records

    out = []
    for r in records:
        keep = True
        for k, v in FILTERS.items():
            if _get_field(r, k, None) != v:
                keep = False
                break
        if keep:
            out.append(r)
    return out


def _build_column_arrays(records: List[Any]) -> Dict[str, np.ndarray]:
    cols = {v: [] for v in VARS}
    for r in records:
        for v in VARS:
            val = _get_field(r, v, None)
            if v == "TotalCost" and isinstance(val, str):
                try:
                    val = float(val.replace("$", "").replace(",", ""))
                except (ValueError, AttributeError):
                    val = None
            cols[v].append(val)

    arrs = {k: np.array(v, dtype=object) for k, v in cols.items()}

    # Build mask: TotalCost is finite and > 0
    tc_obj = np.asarray(arrs["TotalCost"], dtype=object)
    miss = _is_missing(tc_obj)

    tc_float = np.empty(tc_obj.shape, dtype=float)
    try:
        tc_float[~miss] = tc_obj[~miss].astype(float, copy=False)
    except (ValueError, TypeError):
        tc_float[~miss] = np.fromiter((float(x) for x in tc_obj[~miss]),
                                      dtype=float,
                                      count=np.count_nonzero(~miss))
    tc_float[miss] = np.nan

    keep = np.isfinite(tc_float) & (tc_float > 0.0)

    # Apply mask to every column
    for k in arrs:
        arrs[k] = arrs[k][keep]

    return arrs




# ------------------------- Year analysis -------------------------

def analyze_year(
    logger,
    data_dir: Path,
    fy: int,
    *,
    permutation_baseline_reps: int = 10
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, LabelEncoder], List[Tuple[str, float, str]], Dict[str, Any]]:
    logger.info("=" * 80)
    logger.info(f"FISCAL YEAR {fy}")
    logger.info("=" * 80)

    records = _load_year_records(data_dir, fy)
    total_count = len(records)
    records = _filter_records(records, logger=logger)
    filtered_count = len(records)
    logger.info(f"Total records: {total_count}")
    logger.info(f"After quality filters: {filtered_count}")

    arrs_orig = _build_column_arrays(records)
    final_records = len(arrs_orig["TotalCost"])
    logger.info(f"Final with TotalCost > 0: {final_records}")
    if final_records < 30:
        logger.info(f"Sample size is small (n={final_records}); association estimates may be unstable.")
    if final_records == 0:
        logger.warning("No usable records for this year; skipping year.")
        return arrs_orig, {}, {}, [], {
            "total_records": total_count,
            "filtered_records": filtered_count,
            "final_records": final_records,
            "num_features": 0,
            "num_categorical": 0,
            "num_continuous": 0
        }

    encoded, encoders = encode_and_impute(arrs_orig)
    X, y, feature_names, discrete_mask = design_matrix(encoded)

    # Optional VIF screen on continuous features
    vif_drops = _vif_screen(encoded, feature_names, vif_threshold=10.0)
    if vif_drops:
        feature_names = [n for n in feature_names if n not in vif_drops]
        X = np.column_stack([encoded[n] for n in feature_names]).astype(float) if feature_names else np.zeros((len(encoded["TotalCost"]), 0), dtype=float)
        discrete_mask = np.array([n in CATEGORICAL for n in feature_names], dtype=bool)
        logger.info(f"VIF screen dropped {len(vif_drops)} feature(s): {', '.join(sorted(vif_drops))}")

    hc = [n for n in feature_names if n in {"County", "SecondaryDiagnosis", "OtherDiagnosis"}]
    if hc:
        logger.info(f"High-cardinality categoricals present: {hc}")

    logger.info("")
    logger.info("1. MUTUAL INFORMATION ANALYSIS")
    logger.info("-" * 40)
    mi_rows = compute_mutual_information(X, y, feature_names, discrete_mask,
                                         permutation_baseline_reps=permutation_baseline_reps)
    for n, s, t in mi_rows[:20]:
        logger.info(f"{n:30s}: {s:.4f} ({t})")

    stats = {
        "total_records": total_count,
        "filtered_records": filtered_count,
        "final_records": final_records,
        "num_features": len(feature_names),
        "num_categorical": sum(1 for n in feature_names if n in CATEGORICAL),
        "num_continuous": sum(1 for n in feature_names if n not in CATEGORICAL),
    }

    # Correlation (continuous)
    logger.info("")
    logger.info("2. CORRELATION MATRIX (CONTINUOUS ONLY)")
    logger.info("-" * 40)
    cont_corr, cont_names = continuous_corr_matrix(encoded, feature_names)
    if cont_corr is not None:
        _plot_matrix(cont_corr, cont_names, cont_names, fy, " - Continuous (Spearman)", "correlation", logger)
        if "TotalCost" in cont_names:
            tc = cont_names.index("TotalCost")
            tmp = np.abs(cont_corr[tc, :]).copy()
            tmp[tc] = -np.inf
            j = int(np.argmax(tmp))
            stats["top_correlation"] = float(np.abs(cont_corr[tc, j]))
            stats["top_correlation_feature"] = cont_names[j]

    # Categorical associations
    logger.info("")
    logger.info("3. CATEGORICAL ASSOCIATIONS (BIAS-CORRECTED CRAMÉR'S V)")
    logger.info("-" * 40)
    cat_names = [n for n in feature_names if n in CATEGORICAL]
    cramers, cat_labels = categorical_cramers_matrix(arrs_orig, cat_names)
    if cramers is not None:
        _plot_matrix(cramers, cat_labels, cat_labels, fy, " - Categorical (Cramers V)", "cramers_v", logger)

    # Mixed (η)
    logger.info("")
    logger.info("4. MIXED-TYPE ASSOCIATIONS (CORRELATION RATIO η)")
    logger.info("-" * 40)
    cont_only = [n for n in feature_names if n not in CATEGORICAL]
    eta_mat, eta_rows, eta_cols = mixed_eta_matrix(arrs_orig, cat_names, cont_only)
    if eta_mat is not None:
        _plot_matrix(eta_mat, eta_rows, eta_cols, fy, " - Mixed (Correlation Ratio)", "eta", logger)

    # Pairplot on continuous
    logger.info("")
    logger.info("5. PAIRWISE PLOTS (CONTINUOUS ONLY)")
    logger.info("-" * 40)
    top_features = [n for (n, _, _) in mi_rows[:20]]
    cont_top = [f for f in top_features[:7] if f not in CATEGORICAL]
    if "TotalCost" not in cont_top:
        cont_top.append("TotalCost")
    _pairplot_continuous(encoded, cont_top[:7], fy, logger)

    return arrs_orig, encoded, encoders, mi_rows, stats


def _pairplot_continuous(encoded: Dict[str, np.ndarray], features: List[str], fy: int, logger):
    feats = [f for f in features if f not in CATEGORICAL and f in encoded]
    if len(feats) < 2:
        logger.info("Not enough continuous features for pairplot.")
        return
    n = len(encoded[feats[0]])
    idx = np.arange(n)
    if n > PAIRPLOT_SAMPLE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(idx, size=PAIRPLOT_SAMPLE, replace=False)
    data = [encoded[f].astype(float)[idx] for f in feats]
    k = len(feats)
    fig, axes = plt.subplots(k, k, figsize=(15, 15))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[i], bins=30, alpha=0.7)
            else:
                ax.scatter(data[j], data[i], alpha=0.3, s=2)
            if i == k - 1:
                ax.set_xlabel(feats[j], fontsize=8)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(feats[i], fontsize=8)
            else:
                ax.set_ylabel("")
    plt.suptitle(f"FY{fy} - Continuous Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fname = FIG_DIR / f"fy{fy}_pairplot_top_features.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pairplot: {fname}")


# ------------------------- Redundancy filter -------------------------

def redundancy_filter(
    top_mi_rows: List[Tuple[str, float, str]],
    encoded: Dict[str, np.ndarray],
    arrs_orig: Dict[str, np.ndarray],
    threshold: float = 0.9,
    max_features: int | None = REDUNDANCY_MAX_FEATURES
) -> List[Tuple[str, float, str]]:
    names_scanned = [r[0] for r in (top_mi_rows if max_features is None else top_mi_rows[:max_features])]
    to_drop = set()
    name_to_mi = {n: s for n, s, _ in top_mi_rows}

    for i, f1 in enumerate(names_scanned):
        if f1 in to_drop:
            continue
        for f2 in names_scanned[i + 1:]:
            if f2 in to_drop:
                continue
            if f1 not in encoded or f2 not in encoded:
                continue

            is_cat1, is_cat2 = (f1 in CATEGORICAL), (f2 in CATEGORICAL)
            try:
                if is_cat1 and is_cat2:
                    assoc = cramers_v_corrected(arrs_orig[f1], arrs_orig[f2])
                elif (not is_cat1) and (not is_cat2):
                    assoc = abs(spearman_corr(encoded[f1], encoded[f2]))
                else:
                    assoc = correlation_ratio_eta(arrs_orig[f1], encoded[f2]) if is_cat1 \
                            else correlation_ratio_eta(arrs_orig[f2], encoded[f1])
            except (ValueError, TypeError, ZeroDivisionError):
                assoc = 0.0

            if not np.isfinite(assoc):
                assoc = 0.0

            if assoc > threshold and name_to_mi.get(f2, 0.0) < name_to_mi.get(f1, 0.0):
                to_drop.add(f2)

    return [row for row in top_mi_rows if row[0] not in to_drop]


# ------------------------- Reporting -------------------------

def _write_summary_csv(rows: List[dict], path: Path):
    fields = ["Feature", "Mean_MI", "Std_MI", "Max_MI", "Min_MI", "Years_in_Top10", "Years_in_Top20"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _generate_summary_and_tex(all_mi_by_year: Dict[int, List[Tuple[str, float, str]]],
                              fyears: List[int],
                              logger):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_features = set()
    for rows in all_mi_by_year.values():
        all_features.update([r[0] for r in rows])
    summary_rows = []
    for feat in sorted(all_features):
        scores = []
        y10 = y20 = 0
        for _, rows in all_mi_by_year.items():
            top10 = [r[0] for r in rows[:10]]
            top20 = [r[0] for r in rows[:20]]
            if feat in top10:
                y10 += 1
            if feat in top20:
                y20 += 1
            found = next((s for (n, s, _) in rows if n == feat), None)
            if found is not None:
                scores.append(found)
        mean_mi = float(np.mean(scores)) if scores else 0.0
        std_mi  = float(np.std(scores)) if scores else 0.0
        max_mi  = float(np.max(scores)) if scores else 0.0
        min_mi  = float(np.min(scores)) if scores else 0.0
        summary_rows.append({
            "Feature": feat,
            "Mean_MI": round(mean_mi, 6),
            "Std_MI": round(std_mi, 6),
            "Max_MI": round(max_mi, 6),
            "Min_MI": round(min_mi, 6),
            "Years_in_Top10": y10,
            "Years_in_Top20": y20,
        })
    summary_rows.sort(key=lambda d: d["Mean_MI"], reverse=True)
    csv_path = LOG_DIR / "FeatureSelectionSummary.csv"
    _write_summary_csv(summary_rows, csv_path)
    logger.info(f"Summary saved to: {csv_path}")

    num_years = len(fyears)
    tex_path = LOG_DIR / "TopFeaturesTable.tex"
    top15 = summary_rows[:15]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Top 15 Features by Mean Mutual Information\n")
        f.write("% Automatically generated - do not edit manually\n")
        f.write("% Note: MI values are not directly comparable across distributions/years\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{Top 15 features by mean MI across {num_years} fiscal years. ")
        f.write("Note: MI values are not directly comparable across distributions or years.}\n")
        f.write("\\label{tab:top-features-mi}\n\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Feature} & \\textbf{Mean MI} & \\textbf{Std MI} & \\textbf{Max MI} & \\textbf{Min MI} & \\textbf{Top 10} & \\textbf{Top 20} \\\\\n")
        f.write("\\hline\n")
        for row in top15:
            name = _latex_escape(row["Feature"])
            f.write(f"{name} & {row['Mean_MI']:.4f} & {row['Std_MI']:.4f} & "
                    f"{row['Max_MI']:.4f} & {row['Min_MI']:.4f} & "
                    f"{int(row['Years_in_Top10'])}/{num_years} & {int(row['Years_in_Top20'])}/{num_years} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    logger.info(f"LaTeX table saved to: {tex_path}")
    return summary_rows


def _write_commands_tex(all_mi_by_year: Dict[int, List[Tuple[str, float, str]]],
                        fyears: List[int],
                        stats_by_year: Dict[int, dict],
                        logger):
    CMD_DIR.mkdir(parents=True, exist_ok=True)
    path = CMD_DIR / "FeatureSelectionCommands.tex"
    if not stats_by_year:
        with open(path, "w", encoding="utf-8") as f:
            f.write("% Feature Selection LaTeX Commands - No years processed\n")
            f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\\newcommand{\\FSNumFiscalYears}{0}\n")
        logger.info(f"LaTeX commands saved to: {path}")
        return

    totals = [stats_by_year[fy]["total_records"] for fy in stats_by_year]
    filtered = [stats_by_year[fy]["filtered_records"] for fy in stats_by_year]
    finals = [stats_by_year[fy]["final_records"] for fy in stats_by_year]
    feats = [stats_by_year[fy]["num_features"] for fy in stats_by_year]

    counts = {}
    for rows in all_mi_by_year.values():
        for feat in [r[0] for r in rows[:10]]:
            counts[feat] = counts.get(feat, 0) + 1
    features_all_years = sum(1 for _, c in counts.items() if c == len(fyears))
    features_most_years = sum(1 for _, c in counts.items() if c >= max(1, len(fyears) - 1))

    top_mi_scores = {fy: {"feature": rows[0][0], "score": rows[0][1]} for fy, rows in all_mi_by_year.items() if rows}

    year_words = {
        2020: "TwoThousandTwenty", 2021: "TwoThousandTwentyOne",
        2022: "TwoThousandTwentyTwo", 2023: "TwoThousandTwentyThree",
        2024: "TwoThousandTwentyFour", 2025: "TwoThousandTwentyFive"
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("% Feature Selection LaTeX Commands - iBudget\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("% Include with: \\input{report/logs/FeatureSelectionCommands.tex}\n\n")
        f.write(f"\\newcommand{{\\FSNumFiscalYears}}{{{len(fyears)}}}\n\n")
        f.write("% Methodological note\n")
        f.write("\\newcommand{\\FSNoteMI}{Mutual information is not on a universal scale across years; consider z-scoring per year before averaging.}\n\n")
        f.write("% Dataset sizes\n")
        f.write(f"\\newcommand{{\\FSMinRecordsTotal}}{{{min(totals):,}}}\n")
        f.write(f"\\newcommand{{\\FSMaxRecordsTotal}}{{{max(totals):,}}}\n")
        f.write(f"\\newcommand{{\\FSMeanRecordsTotal}}{{{int(np.mean(totals)):,}}}\n\n")
        f.write(f"\\newcommand{{\\FSMinRecordsFiltered}}{{{min(filtered):,}}}\n")
        f.write(f"\\newcommand{{\\FSMaxRecordsFiltered}}{{{max(filtered):,}}}\n\n")
        f.write(f"\\newcommand{{\\FSMinRecordsFinal}}{{{min(finals):,}}}\n")
        f.write(f"\\newcommand{{\\FSMaxRecordsFinal}}{{{max(finals):,}}}\n\n")
        f.write("% Feature counts\n")
        f.write(f"\\newcommand{{\\FSNumCandidateVariables}}{{{max(feats)}}}\n")
        f.write(f"\\newcommand{{\\FSFeaturesAllYears}}{{{features_all_years}}}\n")
        f.write(f"\\newcommand{{\\FSFeaturesMostYears}}{{{features_most_years}}}\n\n")
        for fy in fyears:
            if fy not in stats_by_year:
                continue
            yrw = year_words.get(fy, str(fy))
            s = stats_by_year[fy]
            f.write(f"% FY{fy}\n")
            f.write(f"\\newcommand{{\\FSRecordsTotalFY{yrw}}}{{{s['total_records']:,}}}\n")
            f.write(f"\\newcommand{{\\FSRecordsFilteredFY{yrw}}}{{{s['filtered_records']:,}}}\n")
            f.write(f"\\newcommand{{\\FSRecordsFinalFY{yrw}}}{{{s['final_records']:,}}}\n")
            if fy in top_mi_scores:
                ftr = top_mi_scores[fy]["feature"]
                sc = top_mi_scores[fy]["score"]
                f.write(f"\\newcommand{{\\FSTopFeatureFY{yrw}}}{{{_latex_escape(ftr)}}}\n")
                f.write(f"\\newcommand{{\\FSTopMIFY{yrw}}}{{{sc:.4f}}}\n\n")
    logger.info(f"LaTeX commands saved to: {path}")


# ------------------------- Orchestration -------------------------

def run_feature_selection(
    *,
    fiscal_years: List[int] | None = None,
    permutation_baseline_reps: int = 10,
    redundancy_max_features: int | None = REDUNDANCY_MAX_FEATURES
):
    base, logger = get_ibudget_logger()
    base.log_section("Feature Selection Analysis (NumPy)", "=")
    logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = Path("models/data/cached")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")


    years = fiscal_years or [2025, 2024, 2023, 2022, 2021, 2020]


    all_mi_by_year: Dict[int, List[Tuple[str, float, str]]] = {}
    stats_by_year: Dict[int, dict] = {}

    for fy in years:
        try:
            arrs_orig, encoded, encoders, mi_rows, stats = analyze_year(
                logger, data_dir, fy, permutation_baseline_reps=permutation_baseline_reps
            )
        except FileNotFoundError as e:
            logger.info(f"Skipping FY{fy}: {e}")
            continue

        if mi_rows:
            before = {n for (n, _, _) in mi_rows}
            mi_rows = redundancy_filter(mi_rows, encoded, arrs_orig, threshold=0.9, max_features=redundancy_max_features)
            after = {n for (n, _, _) in mi_rows}
            dropped = sorted(before - after)
            if dropped:
                logger.info(f"Redundancy filter dropped {len(dropped)} feature(s): {', '.join(dropped)}")

        all_mi_by_year[fy] = mi_rows
        stats_by_year[fy] = stats

    eff_years = list(all_mi_by_year.keys())
    logger.info("=" * 80)
    logger.info("CROSS-YEAR CONSISTENCY ANALYSIS")
    logger.info(f"Processed {len(eff_years)} years: {eff_years}")
    logger.info("=" * 80)
    feature_counts: Dict[str, int] = {}
    for rows in all_mi_by_year.values():
        for feat in [r[0] for r in rows[:10]]:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    for feat, count in sorted(feature_counts.items(), key=lambda kv: kv[1], reverse=True):
        if count > 1:
            logger.info(f"  {feat:30s}: {count}/{len(eff_years)} years")

    if not eff_years:
        logger.info("No fiscal years processed. Skipping reports.")
        return

    logger.info("=" * 80)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 80)
    _generate_summary_and_tex(all_mi_by_year, eff_years, logger)
    _write_commands_tex(all_mi_by_year, eff_years, stats_by_year, logger)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Log file: {LOG_DIR / 'model_99_log.txt'}")
    logger.info(f"Figures:   {FIG_DIR}")
    logger.info(f"Summary:   {LOG_DIR / 'FeatureSelectionSummary.csv'}")
    logger.info(f"LaTeX:     {LOG_DIR / 'TopFeaturesTable.tex'}")
    logger.info(f"Commands:  {CMD_DIR / 'FeatureSelectionCommands.tex'}")


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    run_feature_selection(permutation_baseline_reps=10, redundancy_max_features=100)
