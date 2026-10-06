"""Core logic: cleaning, breed growth reference, weight estimator, future-weight model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REQUIRED = ["breed", "sex", "age_months", "weight_kg"]
MEASURES = ["height_cm", "chest_girth_cm", "body_length_cm"]
UNDER, OVER = 0.85, 1.15  # weight / reference weight thresholds

# Approximate mature weights (male kg, female kg) and growth rate k per month.
# Starting values only. Replaced by curves fitted on the farmer's own data when enough rows exist.
BUILTIN = {
    "Boer": (110.0, 75.0, 0.11),
    "Sirohi": (50.0, 35.0, 0.12),
    "Shannan": (55.0, 40.0, 0.12),
    "Khari": (30.0, 22.0, 0.13),
    "Black Bengal": (20.0, 15.0, 0.14),
}
ALIASES = {"Sirohiya": "Sirohi", "Shanan": "Shannan", "Bengal": "Black Bengal"}
SEX_MAP = {"m": "male", "male": "male", "buck": "male", "f": "female", "female": "female", "doe": "female"}


def gompertz(t, A, b, k):
    return A * np.exp(-b * np.exp(-k * np.asarray(t, dtype=float)))


def builtin_params(breed: str, sex: str):
    male, female, k = BUILTIN[breed]
    A = male if sex == "male" else female
    birth = 1.5 + 0.025 * A
    return (A, float(np.log(A / birth)), k)


# ---------- data ----------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df["breed"] = df["breed"].astype(str).str.strip().str.title().replace(ALIASES)
    df["sex"] = df["sex"].astype(str).str.strip().str.lower().map(SEX_MAP)
    for c in ["age_months", "weight_kg"] + [m for m in MEASURES if m in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df = df[(df["weight_kg"] > 0) & (df["age_months"] >= 0)]
    return df.reset_index(drop=True)


# ---------- reference growth curve ----------
def _fit(g: pd.DataFrame):
    A0 = max(float(g["weight_kg"].quantile(0.95)), 5.0)
    p, _ = curve_fit(
        gompertz, g["age_months"], g["weight_kg"], p0=[A0, 2.5, 0.12],
        bounds=([3, 0.5, 0.02], [300, 6, 0.6]), maxfev=5000,
    )
    return tuple(float(x) for x in p)


def fit_reference(df: pd.DataFrame) -> dict:
    """Return {(breed, sex): params, '*': generic params}."""
    ref = {}
    try:
        ref["*"] = _fit(df)
    except Exception:
        ref["*"] = (40.0, 3.0, 0.12)
    for (breed, sex), g in df.groupby(["breed", "sex"]):
        params = None
        if len(g) >= 20 and g["age_months"].nunique() >= 6:
            try:
                params = _fit(g)
            except Exception:
                params = None
        if params is None and breed in BUILTIN:
            params = builtin_params(breed, sex)
        if params is not None:
            ref[(breed, sex)] = params
    return ref


def ref_weight(ref: dict, breeds, sexes, ages) -> np.ndarray:
    out = np.empty(len(ages))
    for i, (b, s, a) in enumerate(zip(breeds, sexes, ages)):
        p = ref.get((b, s))
        if p is None and b in BUILTIN and s in ("male", "female"):
            p = builtin_params(b, s)
        if p is None:
            p = ref["*"]
        out[i] = gompertz(a, *p)
    return out


def status_from_ratio(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.where(r < UNDER, "Underweight", np.where(r > OVER, "Overweight", "Normal"))


def add_condition(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    df["ref_weight_kg"] = ref_weight(ref, df["breed"], df["sex"], df["age_months"])
    df["condition_ratio"] = df["weight_kg"] / df["ref_weight_kg"]
    df["status"] = status_from_ratio(df["condition_ratio"])
    return df


# ---------- models ----------
def _pipe(cat, num) -> Pipeline:
    pre = ColumnTransformer(
        [("c", OneHotEncoder(handle_unknown="ignore"), cat), ("n", "passthrough", num)]
    )
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    return Pipeline([("pre", pre), ("m", model)])


def _split(n, groups=None):
    if groups is not None and groups.nunique() >= 10:
        tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(np.zeros(n), groups=groups))
        return tr, te
    return train_test_split(np.arange(n), test_size=0.2, random_state=42)


def train_weight_estimator(df: pd.DataFrame, use_age: bool = True):
    """Estimate current weight from body measurements (+ breed, sex, age)."""
    measures = [m for m in MEASURES if m in df.columns and df[m].notna().mean() >= 0.5]
    if not measures:
        return None
    num = measures + (["age_months"] if use_age else [])
    d = df.dropna(subset=num).reset_index(drop=True)
    if len(d) < 30:
        return None
    cat = ["breed", "sex"]
    X, y = d[cat + num], d["weight_kg"]
    groups = d["goat_id"] if "goat_id" in d.columns else None
    tr, te = _split(len(d), groups)
    pipe = _pipe(cat, num).fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    metrics = {"MAE (kg)": mean_absolute_error(y.iloc[te], pred), "R2": r2_score(y.iloc[te], pred), "Rows": len(d)}
    pipe.fit(X, y)
    return {"pipe": pipe, "cat": cat, "num": num, "metrics": metrics}


FUTURE_FEATS_CAT = ["breed", "sex"]
FUTURE_FEATS_NUM = ["age_months", "weight_kg", "condition_ratio", "horizon", "ref_growth"]


def make_pairs(df: pd.DataFrame, max_h: float = 12.0) -> pd.DataFrame:
    """Pair every earlier record of a goat with each later record (same goat)."""
    if "goat_id" not in df.columns:
        return pd.DataFrame()
    rows = []
    for gid, g in df.sort_values("age_months").groupby("goat_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                h = g.at[j, "age_months"] - g.at[i, "age_months"]
                if 0.5 <= h <= max_h:
                    rows.append({
                        "goat_id": gid, "breed": g.at[i, "breed"], "sex": g.at[i, "sex"],
                        "age_months": g.at[i, "age_months"], "weight_kg": g.at[i, "weight_kg"],
                        "condition_ratio": g.at[i, "condition_ratio"], "horizon": h,
                        "ref_growth": g.at[j, "ref_weight_kg"] / g.at[i, "ref_weight_kg"],
                        "future_weight_kg": g.at[j, "weight_kg"],
                    })
    return pd.DataFrame(rows)


def train_future(df: pd.DataFrame):
    """Future-weight model. Needs repeat measurements per goat (goat_id). Otherwise reference-curve mode."""
    pairs = make_pairs(df)
    if len(pairs) < 60 or pairs["goat_id"].nunique() < 15:
        return {"mode": "reference", "metrics": {}}
    X = pairs[FUTURE_FEATS_CAT + FUTURE_FEATS_NUM]
    y = pairs["future_weight_kg"] / pairs["weight_kg"]  # growth ratio
    tr, te = _split(len(pairs), pairs["goat_id"])
    pipe = _pipe(FUTURE_FEATS_CAT, FUTURE_FEATS_NUM).fit(X.iloc[tr], y.iloc[tr])
    pred_w = pipe.predict(X.iloc[te]) * pairs["weight_kg"].iloc[te].values
    base_w = (pairs["weight_kg"] * pairs["ref_growth"]).iloc[te].values
    true_w = pairs["future_weight_kg"].iloc[te].values
    metrics = {
        "MAE model (kg)": mean_absolute_error(true_w, pred_w),
        "MAE reference-curve baseline (kg)": mean_absolute_error(true_w, base_w),
        "Pairs": len(pairs),
    }
    pipe.fit(X, y)
    return {"mode": "model", "pipe": pipe, "metrics": metrics}


# ---------- bundle ----------
def train_all(raw: pd.DataFrame, use_age_for_estimator: bool = True):
    df = clean(raw)
    if len(df) < 30:
        raise ValueError("Need at least 30 valid rows.")
    ref = fit_reference(df)
    df = add_condition(df, ref)
    bundle = {
        "ref": ref,
        "weight_model": train_weight_estimator(df, use_age_for_estimator),
        "future": train_future(df),
        "breeds": sorted(df["breed"].unique().tolist()),
        "rows": len(df),
    }
    return bundle, df


def predict_weight(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    wm = bundle["weight_model"]
    if wm is None:
        raise ValueError("Weight estimator not available. Data need height/girth/length columns.")
    return wm["pipe"].predict(rows[wm["cat"] + wm["num"]])


def predict_future(bundle: dict, rows: pd.DataFrame, horizon) -> pd.DataFrame:
    """rows: breed, sex, age_months, weight_kg. horizon: months ahead (scalar or per-row)."""
    out = rows.copy().reset_index(drop=True)
    h = np.broadcast_to(np.asarray(horizon, dtype=float), (len(out),)).copy()
    ref = bundle["ref"]
    ref_now = ref_weight(ref, out["breed"], out["sex"], out["age_months"])
    ref_fut = ref_weight(ref, out["breed"], out["sex"], out["age_months"] + h)
    cond = out["weight_kg"].values / ref_now
    growth = ref_fut / ref_now
    fut = bundle["future"]
    if fut["mode"] == "model":
        X = pd.DataFrame({
            "breed": out["breed"], "sex": out["sex"], "age_months": out["age_months"],
            "weight_kg": out["weight_kg"], "condition_ratio": cond, "horizon": h, "ref_growth": growth,
        })
        future_w = fut["pipe"].predict(X) * out["weight_kg"].values
    else:
        future_w = out["weight_kg"].values * growth
    out["ref_weight_now_kg"] = ref_now
    out["status_now"] = status_from_ratio(cond)
    out["future_age_months"] = out["age_months"] + h
    out["future_weight_kg"] = future_w
    out["ref_weight_future_kg"] = ref_fut
    out["status_future"] = status_from_ratio(future_w / ref_fut)
    return out


# ---------- feeding suggestion ----------
# General rule-of-thumb values for goats. Not a ration formulated by a nutritionist.
DM_PCT = {"Underweight": 3.5, "Normal": 3.0, "Overweight": 2.5}  # total dry matter, % body weight
CONC_PCT = {  # concentrate as fed, % body weight
    True: {"Underweight": 1.5, "Normal": 1.0, "Overweight": 0.5},    # growing, 3 to 12 months
    False: {"Underweight": 1.0, "Normal": 0.5, "Overweight": 0.2},   # adult, 12+ months
}


def feeding_plan(weight_kg: float, age_months: float, status: str):
    """Daily feed per goat. Returns None for kids under 3 months (milk + creep feed, ask vet)."""
    if age_months < 3:
        return None
    growing = age_months < 12
    dm_total = weight_kg * DM_PCT[status] / 100
    conc = weight_kg * CONC_PCT[growing][status] / 100
    rough_dm = max(dm_total - conc * 0.9, 0.0)       # concentrate is about 90% dry matter
    dry = 0.5 * rough_dm / 0.9                        # half of roughage from dry fodder (hay, straw, leaves)
    green = 0.5 * rough_dm / 0.25                     # half from green fodder (about 25% dry matter)
    mineral_g = 10 if weight_kg < 20 else 15
    return {
        "Concentrate (grain/pellet)": conc,
        "Dry roughage (hay, straw, tree leaves)": dry,
        "Green fodder": green,
        "Mineral mix (kg)": mineral_g / 1000,
    }


FEED_NOTES = {
    "Underweight": "Check deworming, teeth, and illness first. Raise concentrate step by step. Offer good quality roughage. Weigh every 2 weeks.",
    "Overweight": "Cut concentrate first, keep roughage. Do not starve the goat. Increase grazing or walking. Weigh every 2 weeks.",
    "Normal": "Keep current feeding. Adjust as the goat grows.",
}