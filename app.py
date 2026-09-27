from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV, Ridge
from sklearn.pipeline import Pipeline
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

try:
    from pygam import LinearGAM, s
    HAS_PYGAM = True
except Exception:
    HAS_PYGAM = False


APP_TITLE = "Model Selection Menggunakan AIC dan CV"
DEFAULT_FILE_CANDIDATES = [
    "Folds5x2_pp.xlsx",
    "Fold5x2_pp.xlsx",
]
TARGET_DEFAULT = "PE"
FEATURE_DEFAULT = ["AT", "V", "AP", "RH"]
RANDOM_STATE = 42


# -----------------------------
# Global styling
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
/* =========================
   FORCE WHITE TEXT
   ========================= */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
  color: #ffffff !important;
}

/* Markdown */
[data-testid="stMarkdownContainer"] * {
  color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] * {
  color: #ffffff !important;
}

/* Metric */
[data-testid="stMetric"] * {
  color: #ffffff !important;
}

/* Selectbox, slider, number input, checkbox */
[data-testid="stWidgetLabel"] * {
  color: #ffffff !important;
}

[data-baseweb="select"] * {
  color: #ffffff !important;
}

[data-baseweb="input"] * {
  color: #ffffff !important;
}

/* Buttons */
.stButton button,
.stButton button * {
  color: #ffffff !important;
}

/* Caption */
.stCaption,
[data-testid="stCaptionContainer"] * {
  color: #ffffff !important;
}

/* Radio / checkbox */
[data-testid="stRadio"] *,
[data-testid="stCheckbox"] *,
[data-testid="stSlider"] * {
  color: #ffffff !important;
}

<style>
:root {
  --bg: #09111f;
  --panel: #111b2d;
  --panel2: #17243a;
  --text: #ffffff;
  --muted: #ffffff;
  --accent: #7c6cff;
  --accent2: #29d3c2;
  --border: rgba(255,255,255,.08);
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 15% 0%, rgba(124,108,255,.16), transparent 28%),
              radial-gradient(circle at 88% 8%, rgba(41,211,194,.11), transparent 25%),
              var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(14,23,40,.98), rgba(9,17,31,.98));
  border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
.hero {
  padding: 1.4rem 1.6rem;
  border: 1px solid var(--border);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(124,108,255,.15), rgba(41,211,194,.07) 60%, rgba(255,255,255,.025));
  box-shadow: 0 24px 70px rgba(0,0,0,.18);
  margin-bottom: 1rem;
}
.hero-title { font-size: 2.15rem; font-weight: 780; letter-spacing: -.035em; margin: 0; }
.hero-sub { color: var(--muted); margin-top: .35rem; font-size: 1rem; }
.section-title { font-size: 1.18rem; font-weight: 720; margin: .4rem 0 .55rem; }
.small-muted { color: var(--muted); font-size: .86rem; }
.metric-card {
  background: linear-gradient(180deg, rgba(23,36,58,.95), rgba(17,27,45,.95));
  border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.05rem;
  min-height: 110px; box-shadow: 0 12px 34px rgba(0,0,0,.14);
}
.metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 1.65rem; font-weight: 760; margin-top: .3rem; }
.metric-note { color: var(--muted); font-size: .76rem; margin-top: .2rem; }
.badge {
  display:inline-block; padding:.26rem .55rem; border-radius:999px; font-size:.76rem; font-weight:650;
  background: rgba(124,108,255,.13); border:1px solid rgba(124,108,255,.26); color:#c7c1ff;
}
.badge-green { background: rgba(41,211,194,.11); border-color: rgba(41,211,194,.24); color:#9af1e9; }
.note {
  border-left: 3px solid var(--accent); padding: .75rem .95rem; background: rgba(124,108,255,.07);
  border-radius: 0 12px 12px 0; color: #cdd6e6; font-size: .9rem;
}
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(23,36,58,.95), rgba(17,27,45,.95));
  border: 1px solid var(--border); border-radius: 16px; padding: .9rem;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
button[kind="secondary"], button[kind="primary"] { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Utility functions
# -----------------------------
def compute_aic(n: int, rss: float, k: float) -> float:
    return float(n * np.log(rss / n) + 2 * k)


def compute_aicc(n: int, rss: float, k: float) -> float:
    aic = compute_aic(n, rss, k)
    if n - k - 1 > 0:
        aic += float((2 * k * (k + 1)) / (n - k - 1))
    return float(aic)


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2_score_local(y_true, y_pred) -> float:
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    denom = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / denom) if denom else float("nan")


def make_train_test_split(n: int, test_size: float = 0.20, seed: int = RANDOM_STATE):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(np.ceil(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def make_folds(n: int, K: int = 10, seed: int = RANDOM_STATE):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n)
    fold_id = np.resize(np.arange(1, K + 1), n)
    folds = [np.sort(shuffled[fold_id == k]) for k in range(1, K + 1)]
    return folds


def make_splits(n: int, K: int, seed: int):
    folds = make_folds(n, K, seed)
    all_idx = np.arange(n)
    return [(np.setdiff1d(all_idx, val_idx), val_idx) for val_idx in folds]


def fit_scaler(X):
    center = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1)
    scale = scale.where(scale.notna() & (scale != 0), 1.0)
    return center, scale


def apply_scaler(X, center, scale):
    return (X - center) / scale


def canonical_array(df: pd.DataFrame) -> np.ndarray:
    a = df.to_numpy(dtype=float)
    idx = np.lexsort(tuple(a[:, j] for j in reversed(range(a.shape[1]))))
    return a[idx]


def find_default_file() -> Path | None:
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    for root in roots:
        for name in DEFAULT_FILE_CANDIDATES:
            p = root / name
            if p.exists():
                return p
    return None


@st.cache_data(show_spinner=False)
def load_excel_bytes(file_bytes: bytes, filename: str) -> Dict[str, pd.DataFrame]:
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    out: Dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        frame = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
        frame.columns = [str(c).strip() for c in frame.columns]
        out[sheet] = frame
    return out


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes, filename: str) -> Dict[str, pd.DataFrame]:
    frame = pd.read_csv(io.BytesIO(file_bytes))
    frame.columns = [str(c).strip() for c in frame.columns]
    return {"Data": frame}


def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_regression_df(df: pd.DataFrame, target: str) -> Tuple[bool, str, pd.DataFrame, list[str]]:
    if df is None or df.empty:
        return False, "Dataset kosong.", df, []
    numeric = coerce_numeric_frame(df)
    if target not in numeric.columns:
        return False, f"Target `{target}` tidak ditemukan.", numeric, []
    features = [c for c in numeric.columns if c != target]
    if len(features) < 1:
        return False, "Minimal satu variabel prediktor diperlukan.", numeric, []
    usable = numeric.dropna(subset=[target] + features).copy()
    if len(usable) < 50:
        return False, "Minimal 50 observasi lengkap disarankan untuk dashboard ini.", usable, features
    return True, "OK", usable, features


# -----------------------------
# GAM: exact pyGAM when available, fast spline fallback otherwise
# -----------------------------
class FastSplineGAM:
    def __init__(self, lam: float, n_splines: int = 20, transformers=None):
        self.lam = float(lam)
        self.n_splines = int(n_splines)
        self.transformers = transformers or []
        self.model = None
        self.edof_ = np.nan

    @staticmethod
    def fit_transformer(X, n_splines=20):
        X = np.asarray(X, dtype=float)
        transformers = []
        Z_parts = []
        for j in range(X.shape[1]):
            stf = SplineTransformer(n_knots=n_splines, degree=3, include_bias=False)
            Z_parts.append(stf.fit_transform(X[:, [j]]))
            transformers.append(stf)
        return np.hstack(Z_parts), transformers

    def fit(self, X, y, transformers=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if transformers is None:
            Z, self.transformers = self.fit_transformer(X, self.n_splines)
        else:
            self.transformers = transformers
            Z = self.transform(X)
        self.model = Ridge(alpha=self.lam).fit(Z, y)

        Xd = np.column_stack([np.ones(len(Z)), Z])
        p = Xd.shape[1]
        penalty = np.eye(p) * self.lam
        penalty[0, 0] = 0.0
        A = Xd.T @ Xd + penalty
        self.edof_ = float(np.trace(np.linalg.solve(A, Xd.T @ Xd)))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([tr.transform(X[:, [j]]) for j, tr in enumerate(self.transformers)])

    def predict(self, X):
        return self.model.predict(self.transform(X))


def build_pygam(X_train, y_train, lam):
    terms = s(0, n_splines=20)
    for j in range(1, X_train.shape[1]):
        terms = terms + s(j, n_splines=20)
    return LinearGAM(terms, lam=lam).fit(X_train, y_train)


def gam_fixed_cv(X_train, y_train, splits, lam, exact=True, fallback_transformers=None):
    folds = []
    if (not exact) or (not HAS_PYGAM):
        if fallback_transformers is None:
            _, fallback_transformers = FastSplineGAM.fit_transformer(X_train, n_splines=20)
        Z = np.hstack([tr.transform(np.asarray(X_train)[:, [j]]) for j, tr in enumerate(fallback_transformers)])
        y_arr = np.asarray(y_train, dtype=float)
        for tr_idx, val_idx in splits:
            m = Ridge(alpha=float(lam)).fit(Z[tr_idx], y_arr[tr_idx])
            pred = m.predict(Z[val_idx])
            folds.append(rmse(y_arr[val_idx], pred))
        return float(np.mean(folds)), folds

    for tr_idx, val_idx in splits:
        m = build_pygam(np.asarray(X_train)[tr_idx], np.asarray(y_train)[tr_idx], lam)
        pred = m.predict(np.asarray(X_train)[val_idx])
        folds.append(rmse(np.asarray(y_train)[val_idx], pred))
    return float(np.mean(folds)), folds


def analyze_gam(X_train, y_train, splits, exact: bool):
    lams_gcv = np.logspace(-3, 3, 25)
    lams_cv = np.logspace(-3, 3, 10)

    if exact and HAS_PYGAM:
        terms = s(0, n_splines=20)
        for j in range(1, X_train.shape[1]):
            terms = terms + s(j, n_splines=20)
        gam = LinearGAM(terms).gridsearch(X_train, y_train, lam=lams_gcv, progress=False)
        best_lam_gcv = float(np.ravel(gam.lam)[0])
        edof = float(gam.statistics_["edof"])
        pred = gam.predict(X_train)
        rss = float(np.sum((y_train - pred) ** 2))
        aicc = compute_aicc(len(y_train), rss, edof)
        fixed_grid = []
        for lam in lams_cv:
            cv_rmse, folds = gam_fixed_cv(X_train, y_train, splits, float(lam), exact=True)
            fixed_grid.append({"lam": float(lam), "CV_RMSE": cv_rmse, "fold_RMSE": folds})
        best_cv = min(fixed_grid, key=lambda x: x["CV_RMSE"])
        return {
            "engine": "pyGAM (exact notebook mode)",
            "gcv_model": gam,
            "best_lam_gcv": best_lam_gcv,
            "edof": edof,
            "AICc_GCV": aicc,
            "gcv_grid": [(float(x), None) for x in lams_gcv],
            "fallback_transformers": None,
            "cv_grid": fixed_grid,
            "best_cv": best_cv,
        }

    # Fast, deterministic fallback: additive cubic splines + ridge penalty.
    Z, fallback_transformers = FastSplineGAM.fit_transformer(X_train, n_splines=20)
    Xd = np.column_stack([np.ones(len(Z)), Z])
    XtX = Xd.T @ Xd
    Xty = Xd.T @ np.asarray(y_train, dtype=float)
    best_gcv = None
    gcv_rows = []
    for lam in lams_gcv:
        pen = np.eye(Xd.shape[1]) * float(lam)
        pen[0, 0] = 0.0
        beta = np.linalg.solve(XtX + pen, Xty)
        pred = Xd @ beta
        rss = float(np.sum((np.asarray(y_train, dtype=float) - pred) ** 2))
        edof = float(np.trace(np.linalg.solve(XtX + pen, XtX)))
        gcv = float(len(y_train) * rss / max((len(y_train) - edof) ** 2, 1e-12))
        row = (float(lam), gcv, edof, rss)
        gcv_rows.append(row)
        if best_gcv is None or gcv < best_gcv[1]:
            best_gcv = row
    best_lam_gcv = best_gcv[0]
    aicc = compute_aicc(len(y_train), best_gcv[3], best_gcv[2])

    fixed_grid = []
    for lam in lams_cv:
        cv_rmse, folds = gam_fixed_cv(X_train, y_train, splits, float(lam), exact=False, fallback_transformers=fallback_transformers)
        fixed_grid.append({"lam": float(lam), "CV_RMSE": cv_rmse, "fold_RMSE": folds})
    best_cv = min(fixed_grid, key=lambda x: x["CV_RMSE"])

    return {
        "engine": "Fast spline-GAM fallback (20 cubic splines/term)",
        "gcv_model": FastSplineGAM(best_lam_gcv).fit(X_train, y_train, transformers=fallback_transformers),
        "fallback_transformers": fallback_transformers,
        "best_lam_gcv": float(best_lam_gcv),
        "edof": float(best_gcv[2]),
        "AICc_GCV": float(aicc),
        "gcv_grid": gcv_rows,
        "cv_grid": fixed_grid,
        "best_cv": best_cv,
    }


# -----------------------------
# Core analysis
# -----------------------------
@st.cache_data(show_spinner=False)
def analyze_dataset(df: pd.DataFrame, target: str, seed: int, test_size: float, cv_folds: int, exact_gam: bool):
    valid, msg, work, features = validate_regression_df(df, target)
    if not valid:
        raise ValueError(msg)

    X = work[features].copy()
    y = work[target].copy()
    train_idx, test_idx = make_train_test_split(len(work), test_size=test_size, seed=seed)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    n_train = len(X_train)
    splits = make_splits(n_train, cv_folds, seed)

    # Polynomial (nested)
    poly_rows = []
    for degree in [1, 2, 3]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp_train = poly.fit_transform(X_train)
        lr = LinearRegression().fit(Xp_train, y_train)
        pred_train = lr.predict(Xp_train)
        rss = float(np.sum((y_train.values - pred_train) ** 2))
        k = Xp_train.shape[1] + 1
        aicc = compute_aicc(n_train, rss, k)
        pipe = Pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)), ("lr", LinearRegression())])
        cv_mse = -cross_val_score(pipe, X_train, y_train, cv=splits, scoring="neg_mean_squared_error", n_jobs=-1)
        poly_rows.append({
            "degree": degree,
            "k": k,
            "AICc": aicc,
            "CV_RMSE": float(np.sqrt(cv_mse.mean())),
        })
    poly_df = pd.DataFrame(poly_rows)

    # Elastic Net
    center, scale = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, center, scale).values
    X_test_scaled = apply_scaler(X_test, center, scale).values
    alphas = np.sort(np.logspace(-4, 1, 20))[::-1]
    l1_ratios = [0.1, 0.5, 0.9, 1.0]

    en_rows = []
    for l1r in l1_ratios:
        for a in alphas:
            m = ElasticNet(alpha=float(a), l1_ratio=float(l1r), max_iter=10000, random_state=seed)
            m.fit(X_train_scaled, y_train)
            pred = m.predict(X_train_scaled)
            rss = float(np.sum((y_train.values - pred) ** 2))
            k = int(np.sum(np.abs(m.coef_) > 1e-12) + 1)
            en_rows.append({"l1_ratio": float(l1r), "alpha": float(a), "k": k, "AICc": compute_aicc(n_train, rss, k)})
    en_df = pd.DataFrame(en_rows)
    en_best_aicc = en_df.loc[en_df["AICc"].idxmin()].to_dict()

    en_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=splits,
        max_iter=10000,
        random_state=seed,
        n_jobs=1,
    ).fit(X_train_scaled, y_train)
    # ElasticNetCV already stores the fold-wise MSE for every alpha × l1_ratio.
    # Reusing mse_path_ avoids the redundant refits of the original notebook loop.
    mse_path = np.asarray(en_cv.mse_path_, dtype=float)
    if mse_path.ndim == 2:
        mse_path = mse_path[None, ...]
    cv_mse_values = []
    for i_l1, l1r in enumerate(l1_ratios):
        path = mse_path[i_l1]
        for j_alpha, a in enumerate(en_cv.alphas_):
            row_mask = (en_df["l1_ratio"] == float(l1r)) & (np.isclose(en_df["alpha"].to_numpy(), float(a)))
            cv_val = float(np.mean(path[j_alpha]))
            en_df.loc[row_mask, "CV_MSE"] = cv_val
    cv_mse_values = en_df["CV_MSE"].to_numpy(dtype=float)
    en_df["CV_RMSE"] = np.sqrt(en_df["CV_MSE"])
    en_cv_idx = int(en_df["CV_RMSE"].idxmin())
    en_best_cv_row = en_df.loc[en_cv_idx].to_dict()
    en_best_aicc_after = en_df.loc[en_df["AICc"].idxmin()].to_dict()
    rho_en, p_en = spearmanr(en_df["AICc"], en_df["CV_MSE"])

    # GAM
    X_train_arr = X_train.values
    y_train_arr = y_train.values
    gam = analyze_gam(X_train_arr, y_train_arr, splits, exact=bool(exact_gam and HAS_PYGAM))
    gam_cv_df = pd.DataFrame(gam["cv_grid"])
    gam_cv_df["lam"] = gam_cv_df["lam"].astype(float)
    gam_cv_df["CV_RMSE"] = gam_cv_df["CV_RMSE"].astype(float)
    best_lam_cv = float(gam["best_cv"]["lam"])
    gam_cv_gcv_rmse, _ = gam_fixed_cv(X_train_arr, y_train_arr, splits, gam["best_lam_gcv"], exact=bool(exact_gam and HAS_PYGAM))

    # Test evaluation for the same choices as notebook
    def evaluate_poly(degree):
        model = Pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)), ("lr", LinearRegression())]).fit(X_train, y_train)
        pred = model.predict(X_test)
        return model, pred, rmse(y_test.values, pred), r2_score_local(y_test.values, pred)

    def evaluate_en(alpha, l1_ratio):
        model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=10000, random_state=seed).fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        return model, pred, rmse(y_test.values, pred), r2_score_local(y_test.values, pred)

    poly_aicc_deg = int(poly_df.loc[poly_df["AICc"].idxmin(), "degree"])
    poly_cv_deg = int(poly_df.loc[poly_df["CV_RMSE"].idxmin(), "degree"])
    poly_a_model, poly_a_pred, poly_a_rmse, poly_a_r2 = evaluate_poly(poly_aicc_deg)
    poly_c_model, poly_c_pred, poly_c_rmse, poly_c_r2 = evaluate_poly(poly_cv_deg)
    en_a_model, en_a_pred, en_a_rmse, en_a_r2 = evaluate_en(en_best_aicc["alpha"], en_best_aicc["l1_ratio"])
    en_c_model, en_c_pred, en_c_rmse, en_c_r2 = evaluate_en(en_cv.alpha_, en_cv.l1_ratio_)

    X_test_arr = X_test.values
    gam_a_model = gam["gcv_model"]
    gam_a_pred = gam_a_model.predict(X_test_arr)
    gam_a_rmse = rmse(y_test.values, gam_a_pred)
    gam_a_r2 = r2_score_local(y_test.values, gam_a_pred)
    gam_c_model = build_pygam(X_train_arr, y_train_arr, best_lam_cv) if (exact_gam and HAS_PYGAM) else FastSplineGAM(best_lam_cv).fit(X_train_arr, y_train_arr)
    gam_c_pred = gam_c_model.predict(X_test_arr)
    gam_c_rmse = rmse(y_test.values, gam_c_pred)
    gam_c_r2 = r2_score_local(y_test.values, gam_c_pred)

    model_compare = pd.DataFrame([
        {"Model": "Polynomial", "Selection": f"degree={poly_aicc_deg} (AICc)", "AICc": float(poly_df.loc[poly_df["degree"] == poly_aicc_deg, "AICc"].iloc[0]), "CV_RMSE": float(poly_df.loc[poly_df["degree"] == poly_aicc_deg, "CV_RMSE"].iloc[0]), "Test_RMSE": poly_a_rmse, "Test_R2": poly_a_r2},
        {"Model": "Polynomial", "Selection": f"degree={poly_cv_deg} (CV)", "AICc": float(poly_df.loc[poly_df["degree"] == poly_cv_deg, "AICc"].iloc[0]), "CV_RMSE": float(poly_df.loc[poly_df["degree"] == poly_cv_deg, "CV_RMSE"].iloc[0]), "Test_RMSE": poly_c_rmse, "Test_R2": poly_c_r2},
        {"Model": "Elastic Net", "Selection": f"alpha={en_best_aicc_after['alpha']:.6g}, l1={en_best_aicc_after['l1_ratio']:.1f} (AICc)", "AICc": float(en_best_aicc_after["AICc"]), "CV_RMSE": float(en_best_aicc_after["CV_RMSE"]), "Test_RMSE": en_a_rmse, "Test_R2": en_a_r2},
        {"Model": "Elastic Net", "Selection": f"alpha={en_cv.alpha_:.6g}, l1={en_cv.l1_ratio_:.1f} (CV)", "AICc": float(en_df.loc[en_cv_idx, "AICc"]), "CV_RMSE": float(np.sqrt(en_df.loc[en_cv_idx, "CV_MSE"])), "Test_RMSE": en_c_rmse, "Test_R2": en_c_r2},
        {"Model": "GAM", "Selection": f"lambda≈{gam['best_lam_gcv']:.6g} (GCV)", "AICc": float(gam["AICc_GCV"]), "CV_RMSE": float(gam_cv_gcv_rmse), "Test_RMSE": gam_a_rmse, "Test_R2": gam_a_r2},
        {"Model": "GAM", "Selection": f"lambda≈{best_lam_cv:.6g} (CV)", "AICc": float(gam_cv_df.loc[gam_cv_df["lam"] == best_lam_cv, "CV_RMSE"].iloc[0] * 0 + gam["AICc_GCV"]), "CV_RMSE": float(gam_cv_df.loc[gam_cv_df["lam"] == best_lam_cv, "CV_RMSE"].iloc[0]), "Test_RMSE": gam_c_rmse, "Test_R2": gam_c_r2},
    ])

    poly_cv_rmse_value = float(poly_df.loc[poly_df["degree"] == poly_cv_deg, "CV_RMSE"].iloc[0])
    en_cv_rmse_value = float(en_df.loc[en_cv_idx, "CV_RMSE"])
    gam_cv_rmse_value = float(gam_cv_df.loc[gam_cv_df["lam"] == best_lam_cv, "CV_RMSE"].iloc[0])
    overall_candidates = pd.DataFrame([
        {"Model": "Polynomial", "Parameter": f"degree={poly_cv_deg}", "AICc": float(poly_df.loc[poly_df["degree"] == poly_cv_deg, "AICc"].iloc[0]), "CV_RMSE": poly_cv_rmse_value, "Test_RMSE": poly_c_rmse, "Test_R2": poly_c_r2, "fit": poly_c_model, "pred": poly_c_pred},
        {"Model": "Elastic Net", "Parameter": f"alpha={en_cv.alpha_:.6g}, l1={en_cv.l1_ratio_:.1f}", "AICc": float(en_df.loc[en_cv_idx, "AICc"]), "CV_RMSE": en_cv_rmse_value, "Test_RMSE": en_c_rmse, "Test_R2": en_c_r2, "fit": en_c_model, "pred": en_c_pred},
        {"Model": "GAM", "Parameter": f"lambda={best_lam_cv:.6g}", "AICc": float(gam["AICc_GCV"]), "CV_RMSE": gam_cv_rmse_value, "Test_RMSE": gam_c_rmse, "Test_R2": gam_c_r2, "fit": gam_c_model, "pred": gam_c_pred},
    ])
    overall_best = overall_candidates.loc[overall_candidates["CV_RMSE"].idxmin()].to_dict()

    return {
        "target": target,
        "features": features,
        "work": work,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "splits": splits,
        "poly_df": poly_df,
        "en_df": en_df,
        "gam_cv_df": gam_cv_df,
        "gam": gam,
        "en_best_aicc": en_best_aicc_after,
        "en_cv": en_cv,
        "en_best_cv_row": en_best_cv_row,
        "rho_en": float(rho_en),
        "p_en": float(p_en),
        "best_deg_aic": poly_aicc_deg,
        "best_deg_cv": poly_cv_deg,
        "best_lam_cv": best_lam_cv,
        "test_results": {
            "poly_aicc": (poly_a_model, poly_a_pred, poly_a_rmse, poly_a_r2),
            "poly_cv": (poly_c_model, poly_c_pred, poly_c_rmse, poly_c_r2),
            "en_aicc": (en_a_model, en_a_pred, en_a_rmse, en_a_r2),
            "en_cv": (en_c_model, en_c_pred, en_c_rmse, en_c_r2),
            "gam_gcv": (gam_a_model, gam_a_pred, gam_a_rmse, gam_a_r2),
            "gam_cv": (gam_c_model, gam_c_pred, gam_c_rmse, gam_c_r2),
        },
        "model_compare": model_compare,
        "overall_candidates": overall_candidates,
        "overall_best": overall_best,
        "n_total": len(work),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "seed": seed,
        "cv_folds": cv_folds,
        "gam_engine": gam["engine"],
    }


# -----------------------------
# UI helpers
# -----------------------------
def hero(page_title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">{page_title}</div>
          <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def df_style_numeric(df: pd.DataFrame, digits: int = 4):
    return df.style.format({c: f"{{:.{digits}f}}" for c in df.select_dtypes(include=[np.number]).columns})


def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5f4", size=12),
        title_font=dict(size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def safe_sample(df: pd.DataFrame, n=2500, seed=42):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def partial_line(model_name: str, model_obj, X_ref: pd.DataFrame, feature: str, grid_points=120):
    X_plot = X_ref.copy()
    medians = X_plot.median(numeric_only=True)
    grid = np.linspace(float(X_plot[feature].min()), float(X_plot[feature].max()), grid_points)
    base = pd.DataFrame(np.tile(medians.values, (grid_points, 1)), columns=medians.index)
    base[feature] = grid
    pred = model_obj.predict(base.values if model_name == "GAM" else base)
    return grid, pred


# -----------------------------
# Load dataset: default workbook or upload
# -----------------------------
def get_data_sources():
    default_path = find_default_file()
    if "source_mode" not in st.session_state:
        st.session_state.source_mode = "Bawaan"
    return default_path


def render_sidebar(default_path: Path | None, sheets: Dict[str, pd.DataFrame]):
    with st.sidebar:
        st.markdown("### ◈ ANALYTICS LAB")
        st.markdown("<span class='small-muted'>AICc • Cross-Validation • Regression</span>", unsafe_allow_html=True)
        st.divider()
        page = st.radio(
            "NAVIGASI",
            [
                "01 · Eksplorasi Data",
                "02 · Analisis Model",
                "03 · Evaluasi & Perbandingan",
                "04 · Simulasi Interaktif",
                "05 · Upload Dataset",
            ],
            index=0,
        )
        st.divider()
        st.caption("Konfigurasi analisis")
        st.session_state.seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        st.session_state.test_size = st.slider("Proporsi test", 0.10, 0.40, 0.20, 0.05)
        st.session_state.cv_folds = st.slider("Jumlah fold CV", 5, 10, 10, 1)
        st.session_state.exact_gam = st.checkbox(
            "Exact GAM (pyGAM)",
            value=HAS_PYGAM,
            disabled=not HAS_PYGAM,
            help="Aktifkan hanya bila pyGAM terpasang. Tanpa pyGAM dashboard memakai fallback spline-GAM yang jauh lebih cepat.",
        )
        if not HAS_PYGAM:
            st.caption("pyGAM tidak tersedia di environment saat ini → fallback GAM aktif.")
        if default_path:
            st.caption(f"Data bawaan: `{default_path.name}`")
        return page


# -----------------------------
# Page 1
# -----------------------------
def page_exploration(sheets: Dict[str, pd.DataFrame]):
    hero("01 · Eksplorasi Data", "Eksplorasi distribusi, kualitas data, korelasi, outlier, dan diagnostik regresi awal.")
    sheet = st.selectbox("Pilih sheet", list(sheets.keys()))
    df_raw = sheets[sheet]
    valid, msg, df, features = validate_regression_df(df_raw, TARGET_DEFAULT)
    if not valid:
        st.error(msg)
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Observasi", f"{len(df):,}", "baris lengkap")
    with c2: metric_card("Prediktor", f"{len(features)}", ", ".join(features))
    with c3: metric_card("Missing", f"{int(df_raw.isna().sum().sum()):,}", "nilai sebelum dropna")
    with c4: metric_card("Duplikat", f"{int(df_raw.duplicated().sum()):,}", "baris duplikat")

    st.markdown("<div class='section-title'>Statistik deskriptif lengkap</div>", unsafe_allow_html=True)
    desc = df.describe().T
    desc["median"] = df.median(numeric_only=True)
    desc["mode"] = df.mode(numeric_only=True).iloc[0]
    desc["IQR"] = df.quantile(0.75) - df.quantile(0.25)
    desc["skewness"] = df.skew(numeric_only=True)
    desc["kurtosis"] = df.kurt(numeric_only=True)
    desc = desc[["count","mean","std","min","25%","median","mode","75%","max","IQR","skewness","kurtosis"]]
    st.dataframe(df_style_numeric(desc, 4), use_container_width=True, height=345)

    st.markdown("<div class='section-title'>Distribusi & boxplot</div>", unsafe_allow_html=True)
    cols = df.columns.tolist()
    fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
    for idx, col in enumerate(cols):
        r = idx // 3 + 1; c = idx % 3 + 1
        fig.add_trace(go.Histogram(x=df[col], nbinsx=45, name=col, showlegend=False, opacity=.84), row=r, col=c)
    chart_layout(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Korelasi dan hubungan prediktor terhadap PE</div>", unsafe_allow_html=True)
    corr = df.corr(numeric_only=True)
    heat = px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriks korelasi")
    chart_layout(heat, 460)
    st.plotly_chart(heat, use_container_width=True)

    cols_to_plot = features
    fig2 = make_subplots(rows=1, cols=len(cols_to_plot), subplot_titles=cols_to_plot)
    sample = safe_sample(df, 2500, int(st.session_state.get("seed", 42)))
    for i, col in enumerate(cols_to_plot, 1):
        fig2.add_trace(go.Scatter(x=sample[col], y=sample[TARGET_DEFAULT], mode="markers", marker=dict(size=5, opacity=.34), name=col, showlegend=False), row=1, col=i)
    chart_layout(fig2, 380)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Outlier IQR</div>", unsafe_allow_html=True)
    out_rows = []
    for col in df.columns:
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3 - q1
        lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
        count = int(((df[col] < lb) | (df[col] > ub)).sum())
        out_rows.append({"Variabel": col, "Q1": q1, "Q3": q3, "IQR": iqr, "Batas bawah": lb, "Batas atas": ub, "Outlier": count})
    st.dataframe(df_style_numeric(pd.DataFrame(out_rows), 4), use_container_width=True)

    st.markdown("<div class='section-title'>Diagnostik regresi linear awal</div>", unsafe_allow_html=True)
    X = df[features]
    y = df[TARGET_DEFAULT]
    Xc = add_constant(X)
    beta = np.linalg.lstsq(Xc.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)[0]
    resid = y.to_numpy(dtype=float) - Xc.to_numpy(dtype=float) @ beta
    # Shapiro limit mengikuti notebook: maksimal 5000 residu acak.
    sample_resid = pd.Series(resid).sample(min(5000, len(resid)), random_state=int(st.session_state.get("seed", 42)))
    shapiro_stat, shapiro_p = stats.shapiro(sample_resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, Xc.to_numpy(dtype=float))
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4g}")
    with c2:
        st.metric("Breusch-Pagan p-value", f"{bp_p:.4g}")
    diag = make_subplots(rows=1, cols=2, subplot_titles=["Distribusi residual", "QQ-Plot residual"])
    diag.add_trace(go.Histogram(x=resid, nbinsx=50, showlegend=False), row=1, col=1)
    osm, osr = stats.probplot(resid, dist="norm", fit=False)
    diag.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(size=4, opacity=.55), showlegend=False), row=1, col=2)
    chart_layout(diag, 390)
    st.plotly_chart(diag, use_container_width=True)
    st.markdown("<div class='note'>Catatan interpretasi mengikuti notebook: p-value kecil pada Breusch–Pagan menunjukkan indikasi heteroskedastisitas; hal tersebut dicatat sebagai diagnostik, bukan alasan otomatis untuk transformasi.</div>", unsafe_allow_html=True)


# -----------------------------
# Page 2
# -----------------------------
def page_analysis(sheets: Dict[str, pd.DataFrame]):
    hero("02 · Analisis Model", "Model selection mengikuti struktur notebook: Polynomial, Elastic Net, dan GAM dengan AICc dan cross-validation.")
    sheet = st.selectbox("Pilih sheet analisis", list(sheets.keys()), key="analysis_sheet")
    df = sheets[sheet]
    with st.spinner("Menghitung pipeline model selection dan menyimpan hasil di cache…"):
        result = analyze_dataset(df, TARGET_DEFAULT, int(st.session_state.seed), float(st.session_state.test_size), int(st.session_state.cv_folds), bool(st.session_state.exact_gam))

    st.markdown(f"<span class='badge'>{result['gam_engine']}</span>", unsafe_allow_html=True)
    st.caption(f"Train = {result['n_train']:,} | Test = {result['n_test']:,} | CV = {result['cv_folds']}-fold | Seed = {result['seed']} | Target = PE")

    tabs = st.tabs(["Polynomial", "Elastic Net", "GAM"])

    with tabs[0]:
        a, b = st.columns(2)
        with a: metric_card("AICc terendah", f"degree={result['best_deg_aic']}", "kriteria informasi")
        with b: metric_card("CV terendah", f"degree={result['best_deg_cv']}", "CV_RMSE")
        poly_df = result["poly_df"].copy()
        st.dataframe(df_style_numeric(poly_df, 4), use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=poly_df["degree"], y=poly_df["AICc"], mode="lines+markers", name="AICc"))
        fig.add_trace(go.Scatter(x=poly_df["degree"], y=poly_df["CV_RMSE"], mode="lines+markers", name="CV_RMSE", yaxis="y2"))
        fig.update_layout(title="Polynomial: AICc dan CV_RMSE menurut degree", yaxis=dict(title="AICc"), yaxis2=dict(title="CV_RMSE", overlaying="y", side="right"))
        chart_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='note'>Pada notebook, CV_RMSE polynomial dihitung sebagai sqrt(mean(CV_MSE) pada fold), dengan fold custom round-robin.</div>", unsafe_allow_html=True)

    with tabs[1]:
        en = result["en_df"].copy()
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("AICc terbaik", f"{result['en_best_aicc']['AICc']:.2f}", f"alpha={result['en_best_aicc']['alpha']:.5g}")
        with c2: metric_card("CV terbaik", f"{math.sqrt(float(result['en_best_cv_row']['CV_MSE'])):.4f}", f"alpha={result['en_best_cv_row']['alpha']:.5g}")
        with c3: metric_card("Spearman", f"ρ={result['rho_en']:.3f}", f"p={result['p_en']:.3g} · AICc vs CV_MSE")
        pivot = en.pivot_table(index="l1_ratio", columns="alpha", values="CV_RMSE", aggfunc="mean")
        heat = px.imshow(pivot, aspect="auto", title="Elastic Net · CV_RMSE", labels={"x":"alpha","y":"l1_ratio","color":"CV_RMSE"})
        chart_layout(heat, 440)
        st.plotly_chart(heat, use_container_width=True)
        st.dataframe(df_style_numeric(en.sort_values("AICc").head(20), 5), use_container_width=True, height=360)
        st.markdown("<div class='note'>Grid notebook: alpha = 20 titik log-spaced dari 10⁻⁴ hingga 10¹; l1_ratio = 0.1, 0.5, 0.9, 1.0. Standardisasi menggunakan mean dan sample SD (ddof=1).</div>", unsafe_allow_html=True)

    with tabs[2]:
        gam = result["gam"]
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Lambda GCV", f"{gam['best_lam_gcv']:.5g}", "25 titik grid log")
        with c2: metric_card("Effective DoF", f"{gam['edof']:.2f}", "dipakai pada AICc")
        with c3: metric_card("Lambda CV", f"{result['best_lam_cv']:.5g}", "10 titik grid manual")
        gamcv = result["gam_cv_df"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gamcv["lam"], y=gamcv["CV_RMSE"], mode="lines+markers", name="CV_RMSE"))
        fig.update_xaxes(type="log", title="lambda")
        fig.update_yaxes(title="CV_RMSE")
        fig.update_layout(title="GAM · CV_RMSE menurut lambda")
        chart_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_style_numeric(gamcv, 5), use_container_width=True)
        st.markdown("<div class='note'>Notebook menggunakan LinearGAM dengan 20 spline pada setiap smooth term dan pencarian lambda. Dashboard memilih pyGAM ketika tersedia; bila tidak, digunakan fallback additive cubic-spline + ridge untuk menjaga interaktivitas.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ringkasan test set</div>", unsafe_allow_html=True)
    compare = result["model_compare"].copy()
    st.dataframe(df_style_numeric(compare, 4), use_container_width=True)

    # Actual vs predicted for model choices
    choices = [
        ("Polynomial · CV", result["test_results"]["poly_cv"]),
        ("Elastic Net · CV", result["test_results"]["en_cv"]),
        ("GAM · CV", result["test_results"]["gam_cv"]),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[x[0] for x in choices])
    for j, (name, (_, pred, _, _)) in enumerate(choices, 1):
        y_test = result["y_test"].to_numpy()
        sample_idx = np.linspace(0, len(y_test)-1, min(1800, len(y_test))).astype(int)
        fig.add_trace(go.Scatter(x=y_test[sample_idx], y=np.asarray(pred)[sample_idx], mode="markers", marker=dict(size=4, opacity=.42), showlegend=False), row=1, col=j)
    chart_layout(fig, 360)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page 3
# -----------------------------
def flatten_sheet_result(sheet, r):
    poly = r["poly_df"]
    en = r["en_df"]
    gam = r["gam"]
    best_poly_a = poly.loc[poly["AICc"].idxmin()]
    best_poly_c = poly.loc[poly["CV_RMSE"].idxmin()]
    best_en_a = en.loc[en["AICc"].idxmin()]
    best_en_c = en.loc[en["CV_RMSE"].idxmin()]
    best_gam_c = r["gam_cv_df"].loc[r["gam_cv_df"]["CV_RMSE"].idxmin()]
    return {
        "Sheet": sheet,
        "Polynomial_AICc": best_poly_a["AICc"],
        "Polynomial_CV_RMSE": best_poly_c["CV_RMSE"],
        "Polynomial_degree_AICc": int(best_poly_a["degree"]),
        "Polynomial_degree_CV": int(best_poly_c["degree"]),
        "ElasticNet_AICc": best_en_a["AICc"],
        "ElasticNet_CV_RMSE": best_en_c["CV_RMSE"],
        "ElasticNet_alpha_AICc": best_en_a["alpha"],
        "ElasticNet_alpha_CV": best_en_c["alpha"],
        "GAM_AICc": gam["AICc_GCV"],
        "GAM_CV_RMSE": best_gam_c["CV_RMSE"],
        "GAM_lambda_GCV": gam["best_lam_gcv"],
        "GAM_lambda_CV": best_gam_c["lam"],
    }


def page_evaluation(sheets: Dict[str, pd.DataFrame]):
    hero("03 · Evaluasi & Perbandingan", "Perbandingan lintas 5 sheet, kestabilan akibat pengacakan baris, dan identifikasi model berdasarkan kriteria yang dipilih.")
    if len(sheets) == 0:
        st.warning("Tidak ada sheet yang tersedia.")
        return

    all_results = {}
    with st.spinner("Menyiapkan hasil lintas sheet (cache aktif)…"):
        for sheet, df in sheets.items():
            all_results[sheet] = analyze_dataset(df, TARGET_DEFAULT, int(st.session_state.seed), float(st.session_state.test_size), int(st.session_state.cv_folds), bool(st.session_state.exact_gam))

    rows = [flatten_sheet_result(s, all_results[s]) for s in sheets]
    cross = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    best_aic_row = cross.loc[cross[["Polynomial_AICc","ElasticNet_AICc","GAM_AICc"]].min(axis=1).idxmin()]
    overall_cv = pd.DataFrame([
        {"Sheet": r["Sheet"], "Model":"Polynomial", "CV_RMSE":r["Polynomial_CV_RMSE"]} for _,r in cross.iterrows()
    ] + [
        {"Sheet": r["Sheet"], "Model":"Elastic Net", "CV_RMSE":r["ElasticNet_CV_RMSE"]} for _,r in cross.iterrows()
    ] + [
        {"Sheet": r["Sheet"], "Model":"GAM", "CV_RMSE":r["GAM_CV_RMSE"]} for _,r in cross.iterrows()
    ])
    best_cv = overall_cv.loc[overall_cv["CV_RMSE"].idxmin()]
    with c1: metric_card("Model CV minimum", f"{best_cv['Model']}", f"{best_cv['Sheet']} · RMSE {best_cv['CV_RMSE']:.4f}")
    with c2: metric_card("AICc minimum global", f"{cross[["Polynomial_AICc","ElasticNet_AICc","GAM_AICc"]].min().min():.2f}", "di antara model & sheet")
    with c3: metric_card("Jumlah sheet", f"{len(cross)}", "seluruh permutasi")
    with c4: metric_card("Data per sheet", f"{len(next(iter(sheets.values()))):,}", "observasi")

    st.markdown("<div class='section-title'>Tabel perbandingan AICc dan CV setiap sheet</div>", unsafe_allow_html=True)
    display_cross = cross.copy()
    st.dataframe(df_style_numeric(display_cross, 5), use_container_width=True, height=390)

    tabs = st.tabs(["AICc", "CV_RMSE", "Pengaruh Pengacakan"])
    with tabs[0]:
        aic_long = cross.melt(id_vars=["Sheet"], value_vars=["Polynomial_AICc","ElasticNet_AICc","GAM_AICc"], var_name="Model", value_name="AICc")
        aic_long["Model"] = aic_long["Model"].str.replace("_AICc", "", regex=False)
        fig = px.line(aic_long, x="Sheet", y="AICc", color="Model", markers=True, title="AICc per sheet")
        chart_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        cv_long = cross.melt(id_vars=["Sheet"], value_vars=["Polynomial_CV_RMSE","ElasticNet_CV_RMSE","GAM_CV_RMSE"], var_name="Model", value_name="CV_RMSE")
        cv_long["Model"] = cv_long["Model"].str.replace("_CV_RMSE", "", regex=False)
        fig = px.line(cv_long, x="Sheet", y="CV_RMSE", color="Model", markers=True, title="CV_RMSE per sheet")
        chart_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        # Verify row permutation property.
        base = canonical_array(sheets[list(sheets.keys())[0]][["AT","V","AP","RH","PE"]].dropna()) if all(c in sheets[list(sheets.keys())[0]].columns for c in ["AT","V","AP","RH","PE"]) else None
        same_multiset = True
        if base is not None:
            for sname, d in sheets.items():
                candidate = d[["AT","V","AP","RH","PE"]].dropna()
                same_multiset = same_multiset and np.array_equal(base, canonical_array(candidate))
        if same_multiset:
            st.markdown("<div class='note'><b>Struktur data:</b> lima sheet teridentifikasi sebagai permutasi baris dari himpunan observasi yang sama. Jadi perbedaan hasil model terutama terkait dengan cara split 80/20 dan fold CV berinteraksi dengan urutan baris.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='note'>Lima sheet tidak terverifikasi sebagai permutasi baris yang identik; interpretasi pengaruh pengacakan harus lebih hati-hati.</div>", unsafe_allow_html=True)

        stability_rows = []
        metric_map = {
            "Polynomial AICc": "Polynomial_AICc",
            "Polynomial CV_RMSE": "Polynomial_CV_RMSE",
            "Elastic Net AICc": "ElasticNet_AICc",
            "Elastic Net CV_RMSE": "ElasticNet_CV_RMSE",
            "GAM AICc": "GAM_AICc",
            "GAM CV_RMSE": "GAM_CV_RMSE",
        }
        for label, col in metric_map.items():
            vals = cross[col].astype(float)
            mean = vals.mean(); sd = vals.std(ddof=1)
            rel_range = (vals.max()-vals.min())/abs(mean)*100 if mean != 0 else np.nan
            stability_rows.append({"Metrik": label, "Mean":mean, "SD":sd, "CV%":sd/abs(mean)*100, "Range%":rel_range, "Min":vals.min(), "Max":vals.max()})
        stab = pd.DataFrame(stability_rows)
        st.dataframe(df_style_numeric(stab, 5), use_container_width=True)

        st.markdown("<div class='note'><b>Kesimpulan deskriptif:</b> pengacakan baris memang dapat mengubah nilai numerik AICc/CV karena notebook melakukan shuffle sebelum split train–test dan pembentukan fold. Namun tabel stabilitas di atas menunjukkan besarnya perubahan antar-sheet. Untuk menyebut perubahan tersebut “signifikan” secara inferensial diperlukan replikasi pengacakan yang lebih banyak dan uji statistik yang dirancang khusus; lima sheet saja tidak cukup kuat sebagai uji signifikansi formal.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model terbaik untuk sheet terpilih</div>", unsafe_allow_html=True)
    sheet_pick = st.selectbox("Sheet untuk visualisasi model terbaik", list(sheets.keys()), key="best_sheet_pick")
    r = all_results[sheet_pick]
    best = r["overall_best"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Model", best["Model"], best["Parameter"])
    with c2: metric_card("CV_RMSE", f"{best['CV_RMSE']:.4f}", "kriteria pemilihan")
    with c3: metric_card("Test_RMSE", f"{best['Test_RMSE']:.4f}", "hold-out 20%")
    with c4: metric_card("Test_R²", f"{best['Test_R2']:.4f}", "test set")

    feature_pick = st.selectbox("Prediktor untuk garis regresi", r["features"], key="best_feature")
    grid, pred_line = partial_line(best["Model"], best["fit"], r["X_train"], feature_pick)
    fig = go.Figure()
    sample = safe_sample(r["work"], 2300, r["seed"])
    fig.add_trace(go.Scatter(x=sample[feature_pick], y=sample[TARGET_DEFAULT], mode="markers", name="Observasi", marker=dict(size=5, opacity=.28)))
    fig.add_trace(go.Scatter(x=grid, y=pred_line, mode="lines", name="Garis prediksi (prediktor lain = median)", line=dict(width=4)))
    fig.update_layout(title=f"Garis regresi model terbaik · {best['Model']} · {feature_pick}", xaxis_title=feature_pick, yaxis_title=TARGET_DEFAULT)
    chart_layout(fig, 470)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page 4
# -----------------------------
def page_simulation(sheets: Dict[str, pd.DataFrame]):
    hero("04 · Simulasi Interaktif", "Eksperimen cepat dengan jumlah data, seed, model, parameter, split, dan fold CV. Grafik dan metrik berubah langsung mengikuti slider.")
    sheet = st.selectbox("Basis data", list(sheets.keys()), key="sim_sheet")
    raw = sheets[sheet]
    valid, msg, df, features = validate_regression_df(raw, TARGET_DEFAULT)
    if not valid:
        st.error(msg); return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_sim = st.slider("Jumlah data", 100, min(2500, len(df)), min(1000, len(df)), 50, key="sim_n")
    with c2:
        sim_test = st.slider("Test size", 0.10, 0.40, 0.20, 0.05, key="sim_test")
    with c3:
        sim_folds = st.slider("Fold CV", 3, 10, 5, 1, key="sim_folds")
    with c4:
        sim_seed = st.slider("Seed", 0, 500, 42, 1, key="sim_seed")

    sample = df.sample(n=n_sim, random_state=sim_seed).reset_index(drop=True)
    model_name = st.selectbox("Tipe model", ["Polynomial", "Elastic Net", "GAM"], key="sim_model")

    degree = None; alpha = None; l1 = None; lam = None
    if model_name == "Polynomial":
        degree = st.slider("Degree", 1, 3, 2, 1, key="sim_degree")
    elif model_name == "Elastic Net":
        e1, e2 = st.columns(2)
        with e1:
            log_alpha = st.slider("log10(alpha)", -4.0, 1.0, -2.0, 0.1, key="sim_log_alpha")
            alpha = 10 ** log_alpha
        with e2:
            l1 = st.slider("l1_ratio", 0.1, 1.0, 0.9, 0.1, key="sim_l1")
    else:
        log_lam = st.slider("log10(lambda)", -3.0, 3.0, -1.0, 0.1, key="sim_log_lam")
        lam = 10 ** log_lam

    valid, msg, simdf, features = validate_regression_df(sample, TARGET_DEFAULT)
    if not valid:
        st.error(msg); return

    X = simdf[features].copy(); y = simdf[TARGET_DEFAULT].copy()
    tr_idx, te_idx = make_train_test_split(len(simdf), sim_test, sim_seed)
    Xtr = X.iloc[tr_idx].reset_index(drop=True); Xte = X.iloc[te_idx].reset_index(drop=True)
    ytr = y.iloc[tr_idx].reset_index(drop=True); yte = y.iloc[te_idx].reset_index(drop=True)
    splits = make_splits(len(Xtr), sim_folds, sim_seed)

    if model_name == "Polynomial":
        model = Pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)), ("lr", LinearRegression())]).fit(Xtr, ytr)
        pred_tr = model.predict(Xtr)
        pred_te = model.predict(Xte)
        Xp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(Xtr)
        k = Xp.shape[1] + 1
        rss = float(np.sum((ytr.values - pred_tr) ** 2))
        aicc = compute_aicc(len(ytr), rss, k)
        cv_mse = -cross_val_score(model, Xtr, ytr, cv=splits, scoring="neg_mean_squared_error", n_jobs=-1)
        cv_rmse = float(np.sqrt(cv_mse.mean()))
        selected_label = f"degree={degree}"
        model_for_line = model
    elif model_name == "Elastic Net":
        center, scale = fit_scaler(Xtr)
        Xtrs = apply_scaler(Xtr, center, scale).values
        Xtes = apply_scaler(Xte, center, scale).values
        model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1), max_iter=10000, random_state=sim_seed).fit(Xtrs, ytr)
        pred_tr = model.predict(Xtrs); pred_te = model.predict(Xtes)
        rss = float(np.sum((ytr.values - pred_tr) ** 2))
        k = int(np.sum(np.abs(model.coef_) > 1e-12) + 1)
        aicc = compute_aicc(len(ytr), rss, k)
        fold_rmse = []
        for trf, valf in splits:
            m = ElasticNet(alpha=float(alpha), l1_ratio=float(l1), max_iter=10000, random_state=sim_seed).fit(Xtrs[trf], ytr.iloc[trf])
            fold_rmse.append(rmse(ytr.iloc[valf], m.predict(Xtrs[valf])))
        cv_rmse = float(np.mean(fold_rmse))
        selected_label = f"alpha={alpha:.5g}, l1={l1:.2f}"
        class WrappedEN:
            def predict(self, d):
                d2 = apply_scaler(pd.DataFrame(d, columns=features), center, scale)
                return model.predict(d2.values)
        model_for_line = WrappedEN()
    else:
        if st.session_state.exact_gam and HAS_PYGAM:
            model = build_pygam(Xtr.values, ytr.values, lam)
        else:
            model = FastSplineGAM(lam).fit(Xtr.values, ytr.values)
        pred_tr = model.predict(Xtr.values); pred_te = model.predict(Xte.values)
        edof = float(model.statistics_["edof"] if hasattr(model, "statistics_") else model.edof_)
        rss = float(np.sum((ytr.values - pred_tr) ** 2))
        aicc = compute_aicc(len(ytr), rss, edof)
        cv_rmse, _ = gam_fixed_cv(Xtr.values, ytr.values, splits, lam, exact=(st.session_state.exact_gam and HAS_PYGAM))
        selected_label = f"lambda={lam:.5g}"
        model_for_line = model

    test_rmse = rmse(yte.values, pred_te)
    test_r2 = r2_score_local(yte.values, pred_te)

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("AICc", f"{aicc:.3f}", selected_label)
    with m2: metric_card("CV_RMSE", f"{cv_rmse:.4f}", f"{sim_folds}-fold")
    with m3: metric_card("Test RMSE", f"{test_rmse:.4f}", f"n_test={len(yte):,}")
    with m4: metric_card("Test R²", f"{test_r2:.4f}", f"n_train={len(ytr):,}")

    feature_line = st.selectbox("Prediktor untuk visualisasi", features, key="sim_feature")
    grid, pred_line = partial_line(model_name, model_for_line, Xtr, feature_line)
    sample_vis = safe_sample(pd.concat([Xte.reset_index(drop=True), yte.reset_index(drop=True)], axis=1), 1800, sim_seed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_vis[feature_line], y=sample_vis[TARGET_DEFAULT], mode="markers", marker=dict(size=5, opacity=.3), name="Test data"))
    fig.add_trace(go.Scatter(x=grid, y=pred_line, mode="lines", line=dict(width=4), name="Garis prediksi"))
    fig.update_layout(title=f"Simulasi real-time · {model_name}", xaxis_title=feature_line, yaxis_title=TARGET_DEFAULT)
    chart_layout(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    actual_sorted = np.argsort(yte.to_numpy())
    show_idx = actual_sorted[:min(800, len(actual_sorted))]
    compare = pd.DataFrame({"Actual": yte.to_numpy()[show_idx], "Predicted": np.asarray(pred_te)[show_idx]})
    line = go.Figure()
    line.add_trace(go.Scatter(y=compare["Actual"], mode="lines", name="Actual"))
    line.add_trace(go.Scatter(y=compare["Predicted"], mode="lines", name="Predicted"))
    line.update_layout(title="Actual vs predicted (test set · urutan berdasarkan actual)")
    chart_layout(line, 360)
    st.plotly_chart(line, use_container_width=True)

    st.markdown("<div class='note'>Slider pada halaman ini sengaja membatasi ukuran simulasi agar respons tetap cepat. Nilai AICc dan CV dihitung ulang pada setiap perubahan parameter, bukan lookup angka statis.</div>", unsafe_allow_html=True)


# -----------------------------
# Page 5
# -----------------------------
def page_upload(sheets: Dict[str, pd.DataFrame]):
    hero("05 · Upload Dataset", "Uji dataset Anda sendiri dengan struktur yang kompatibel dengan pipeline regresi pada notebook.")
    st.markdown("<div class='note'><b>Format yang direkomendasikan:</b> XLSX/CSV dengan satu target numerik dan minimal satu prediktor numerik. Untuk replikasi penuh dataset contoh, gunakan kolom <code>AT</code>, <code>V</code>, <code>AP</code>, <code>RH</code>, dan target <code>PE</code>.</div>", unsafe_allow_html=True)

    st.markdown("### Petunjuk tipe dataset")
    instructions = pd.DataFrame([
        {"Aturan": "Baris", "Ketentuan": "Satu baris = satu observasi."},
        {"Aturan": "Kolom", "Ketentuan": "Semua prediktor harus numerik; target harus numerik untuk regresi."},
        {"Aturan": "Target", "Ketentuan": "Secara default dashboard mencari kolom PE, tetapi target bisa dipilih ulang setelah upload."},
        {"Aturan": "Missing", "Ketentuan": "Baris tidak lengkap pada target/prediktor akan dikeluarkan saat analisis."},
        {"Aturan": "Ukuran", "Ketentuan": "Minimal 50 observasi lengkap disarankan agar CV dan diagnostik lebih bermakna."},
        {"Aturan": "XLSX multi-sheet", "Ketentuan": "Setiap sheet dapat dipilih sebagai dataset terpisah."},
    ])
    st.dataframe(instructions, use_container_width=True, hide_index=True)

    uploaded = st.file_uploader("Upload XLSX atau CSV", type=["xlsx", "xls", "csv"], accept_multiple_files=False)
    if uploaded is None:
        st.info("Belum ada file baru. Dashboard tetap menggunakan file bawaan yang tersedia.")
        return

    file_bytes = uploaded.getvalue()
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        uploaded_sheets = load_excel_bytes(file_bytes, uploaded.name)
    else:
        uploaded_sheets = load_csv_bytes(file_bytes, uploaded.name)

    uploaded_sheet = st.selectbox("Sheet", list(uploaded_sheets.keys()), key="upload_sheet")
    raw = uploaded_sheets[uploaded_sheet]
    st.caption(f"Preview file `{uploaded.name}` · sheet `{uploaded_sheet}`")
    st.dataframe(raw.head(15), use_container_width=True)

    num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    target_options = num_cols if num_cols else list(raw.columns)
    target = st.selectbox("Target", target_options, index=(target_options.index(TARGET_DEFAULT) if TARGET_DEFAULT in target_options else 0))
    valid, msg, cleaned, features = validate_regression_df(raw, target)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Rows", f"{len(raw):,}")
    with c2: metric_card("Kolom numerik", f"{len(num_cols)}")
    with c3: metric_card("Prediktor", f"{len(features)}")
    with c4: metric_card("Missing", f"{int(raw.isna().sum().sum()):,}")

    if valid:
        st.success(f"Dataset siap dianalisis · target `{target}` · prediktor: {', '.join(features)}")
        if st.button("Jalankan model selection pada file upload", type="primary"):
            with st.spinner("Menghitung Polynomial + Elastic Net + GAM…"):
                result = analyze_dataset(cleaned, target, int(st.session_state.seed), float(st.session_state.test_size), int(st.session_state.cv_folds), bool(st.session_state.exact_gam))
            st.markdown("### Ringkasan hasil")
            st.dataframe(df_style_numeric(result["model_compare"], 4), use_container_width=True)
            st.write({
                "best_degree_AICc": result["best_deg_aic"],
                "best_degree_CV": result["best_deg_cv"],
                "ElasticNet_AICc": {"alpha": result["en_best_aicc"]["alpha"], "l1_ratio": result["en_best_aicc"]["l1_ratio"]},
                "ElasticNet_CV": {"alpha": float(result["en_cv"].alpha_), "l1_ratio": float(result["en_cv"].l1_ratio_)},
                "GAM_lambda_GCV": result["gam"]["best_lam_gcv"],
                "GAM_lambda_CV": result["best_lam_cv"],
            })
    else:
        st.error(msg)


# -----------------------------
# Main
# -----------------------------
def main():
    default_path = get_data_sources()

    # Default data. Upload is intentionally kept on page 5 so navigation stays fast.
    if default_path:
        try:
            file_bytes = default_path.read_bytes()
            sheets = load_excel_bytes(file_bytes, default_path.name)
        except Exception as e:
            st.error(f"Gagal membaca file bawaan: {e}")
            sheets = {}
    else:
        sheets = {}

    page = render_sidebar(default_path, sheets)

    st.markdown(
        f"<div class='small-muted' style='margin-bottom:.3rem;'>SOURCE · PROJECT_REGRESI (1).ipynb · DATA · Folds5x2_pp.xlsx</div>",
        unsafe_allow_html=True,
    )

    if not sheets and page != "05 · Upload Dataset":
        st.warning("File Folds5x2_pp.xlsx belum ditemukan di folder dashboard. Letakkan file XLSX di folder yang sama dengan app.py, atau gunakan halaman Upload Dataset.")

    if page == "01 · Eksplorasi Data":
        page_exploration(sheets)
    elif page == "02 · Analisis Model":
        page_analysis(sheets)
    elif page == "03 · Evaluasi & Perbandingan":
        page_evaluation(sheets)
    elif page == "04 · Simulasi Interaktif":
        page_simulation(sheets)
    elif page == "05 · Upload Dataset":
        page_upload(sheets)


if __name__ == "__main__":
    main()
