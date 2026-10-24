"""
Reproducible evaluation for the RFM / K-Means / Random Forest segmentation article.

Produces every number the revised article needs:
  1. Exact configuration (seeds, hyperparameters, package versions)
  2. Leak-free evaluation: scaler + K-Means + Random Forest fitted on training customers only,
     tested on unseen customers. Also re-runs the original (leaky) setup for comparison.
  3. Cluster validation for K = 2..8: silhouette, Davies-Bouldin, Calinski-Harabasz
  4. Stability: K-Means across random seeds and bootstrap resamples (Adjusted Rand Index)
  5. Alternative methods at K = 4: Gaussian mixture, agglomerative (Ward)

Usage:
  pip install pandas numpy scikit-learn openpyxl joblib
  python evaluate_segmentation.py --data online_retail_II.xlsx
  python evaluate_segmentation.py --data online_retail_II.xlsx --model backend/model/artifacts/rf_model.joblib

Output: results/evaluation_results.json  (paste or upload this file back)
"""
import argparse, json, platform, sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, adjusted_rand_score, calinski_harabasz_score,
                             classification_report, confusion_matrix, davies_bouldin_score,
                             f1_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
K_FINAL = 4
K_RANGE = range(2, 9)
TEST_SIZE = 0.20
N_SEEDS = 20
N_BOOT = 50
RF_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=1,
                 class_weight=None, random_state=SEED, n_jobs=-1)
KM_PARAMS = dict(n_init=10, random_state=SEED)
FEATURES = ["Recency", "Frequency_log", "Monetary_log"]


# ---------- data (same cleaning as train.py) ----------
def load_and_clean(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Data file not found: {path.resolve()}")
    sheets = pd.read_excel(path, sheet_name=["Year 2009-2010", "Year 2010-2011"])
    df = pd.concat(sheets.values(), ignore_index=True)
    n_raw = len(df)
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    return df, n_raw


def build_rfm(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum")).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])   # matches rfm_segmented.csv
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    return rfm


# ---------- helpers ----------
def name_segments(rfm, labels):
    """Map cluster ids to names from their profiles (same rule as the article)."""
    prof = rfm.assign(c=labels).groupby("c")[["Recency", "Frequency", "Monetary"]].mean()
    names = {}
    champ = prof["Monetary"].idxmax(); names[champ] = "Champions"
    rest = prof.drop(champ)
    lost = rest["Recency"].idxmax(); names[lost] = "Hibernating / Lost"
    rest = rest.drop(lost)
    if len(rest) == 2:
        loyal = rest["Frequency"].idxmax(); names[loyal] = "Loyal Customers"
        names[rest.drop(loyal).index[0]] = "At Risk"
    return names


def cluster_scores(X, labels):
    return dict(silhouette=float(silhouette_score(X, labels)),
                davies_bouldin=float(davies_bouldin_score(X, labels)),
                calinski_harabasz=float(calinski_harabasz_score(X, labels)))


def rf_eval(y_true, y_pred):
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro")),
                report=classification_report(y_true, y_pred, output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(y_true, y_pred).tolist())


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).with_name("online_retail_II.xlsx")))
    ap.add_argument("--model", help="optional: saved Random Forest (joblib) from the original app")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = {}

    # 1. configuration
    out["config"] = dict(seed=SEED, k_final=K_FINAL, k_range=list(K_RANGE), test_size=TEST_SIZE,
                         features=FEATURES, scaler="StandardScaler", kmeans=KM_PARAMS,
                         random_forest=RF_PARAMS, n_seeds=N_SEEDS, n_bootstrap=N_BOOT,
                         versions=dict(python=platform.python_version(), numpy=np.__version__,
                                       pandas=pd.__version__, sklearn=sklearn.__version__))
    if a.model:
        import joblib
        m = joblib.load(a.model)
        out["original_model_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k, v in m.get_params().items()}

    df, n_raw = load_and_clean(a.data)
    rfm = build_rfm(df)
    out["data"] = dict(raw_rows=int(n_raw), clean_rows=int(len(df)), customers=int(len(rfm)))
    print(f"Customers: {len(rfm)}")
    X_all = rfm[FEATURES].to_numpy()

    # 3. cluster validation on all customers (descriptive)
    Xs_all = StandardScaler().fit_transform(X_all)
    out["k_validation"] = []
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, **KM_PARAMS).fit_predict(Xs_all)
        out["k_validation"].append(dict(k=k, **cluster_scores(Xs_all, lab)))
        print(f"K={k}: " + ", ".join(f"{m}={v:.3f}" for m, v in out["k_validation"][-1].items() if m != "k"))

    # 4. stability
    ref = KMeans(n_clusters=K_FINAL, **KM_PARAMS).fit(Xs_all)
    ref_lab = ref.labels_
    seed_ari = [adjusted_rand_score(ref_lab, KMeans(n_clusters=K_FINAL, n_init=10, random_state=s)
                                    .fit_predict(Xs_all)) for s in range(N_SEEDS)]
    rng = np.random.default_rng(SEED)
    boot_ari = {}
    for k in K_RANGE:
        base = KMeans(n_clusters=k, **KM_PARAMS).fit_predict(Xs_all)
        scores = []
        for _ in range(N_BOOT):
            idx = rng.choice(len(Xs_all), len(Xs_all), replace=True)
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs_all[idx])
            scores.append(adjusted_rand_score(base, km.predict(Xs_all)))
        boot_ari[k] = dict(mean=float(np.mean(scores)), std=float(np.std(scores)))
    out["stability"] = dict(seed_ari_k4=dict(mean=float(np.mean(seed_ari)), min=float(np.min(seed_ari))),
                            bootstrap_ari_by_k=boot_ari)
    print("Bootstrap ARI by K:", {k: round(v["mean"], 3) for k, v in boot_ari.items()})

    # 5. alternative methods at K = 4
    gmm = GaussianMixture(n_components=K_FINAL, random_state=SEED, n_init=5).fit(Xs_all)
    alt = {"kmeans": ref_lab,
           "gaussian_mixture": gmm.predict(Xs_all),
           "agglomerative_ward": AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward").fit_predict(Xs_all)}
    out["alternatives_k4"] = {n: dict(**cluster_scores(Xs_all, lab), ari_vs_kmeans=float(adjusted_rand_score(ref_lab, lab)))
                              for n, lab in alt.items()}

    names = name_segments(rfm, ref_lab)
    prof = rfm.assign(Segment=[names.get(c, str(c)) for c in ref_lab]).groupby("Segment")[
        ["Recency", "Frequency", "Monetary"]].agg(["mean", "median"])
    prof["Customers"] = rfm.assign(Segment=[names.get(c, str(c)) for c in ref_lab]).groupby("Segment").size()
    out["segment_profiles"] = json.loads(prof.to_json(orient="index"))

    # 2a. ORIGINAL setup (leaky): scaler + K-Means on all customers, then split
    y_all = ref_lab
    tr, te = train_test_split(np.arange(len(rfm)), test_size=TEST_SIZE, stratify=y_all, random_state=SEED)
    rf = RandomForestClassifier(**RF_PARAMS).fit(Xs_all[tr], y_all[tr])
    out["eval_original_leaky"] = rf_eval(y_all[te], rf.predict(Xs_all[te]))

    # 2b. LEAK-FREE: split customers first; scaler, K-Means and RF fitted on training customers only
    tr, te = train_test_split(np.arange(len(rfm)), test_size=TEST_SIZE, random_state=SEED)
    sc = StandardScaler().fit(X_all[tr])
    Xtr, Xte = sc.transform(X_all[tr]), sc.transform(X_all[te])
    km = KMeans(n_clusters=K_FINAL, **KM_PARAMS).fit(Xtr)
    ytr, yte = km.labels_, km.predict(Xte)          # test labels = segment the train-only model assigns
    rf = RandomForestClassifier(**RF_PARAMS).fit(Xtr, ytr)
    out["eval_leak_free"] = rf_eval(yte, rf.predict(Xte))
    out["eval_leak_free"]["train_vs_full_ari"] = float(adjusted_rand_score(y_all[te], yte))
    out["eval_leak_free"]["feature_importances"] = dict(zip(FEATURES, map(float, rf.feature_importances_)))

    # 2c. LEAK-FREE 5-fold cross-validation
    cv = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_all, y_all):
        sc = StandardScaler().fit(X_all[tr])
        km = KMeans(n_clusters=K_FINAL, **KM_PARAMS).fit(sc.transform(X_all[tr]))
        rf = RandomForestClassifier(**RF_PARAMS).fit(sc.transform(X_all[tr]), km.labels_)
        yte = km.predict(sc.transform(X_all[te]))
        p = rf.predict(sc.transform(X_all[te]))
        cv.append(dict(accuracy=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro")))
    out["eval_leak_free_cv5"] = {m: dict(mean=float(np.mean([c[m] for c in cv])),
                                         std=float(np.std([c[m] for c in cv]))) for m in ["accuracy", "macro_f1"]}

    print(f"Original (leaky) macro F1: {out['eval_original_leaky']['macro_f1']:.3f}")
    print(f"Leak-free macro F1:        {out['eval_leak_free']['macro_f1']:.3f}")
    print(f"Leak-free 5-fold macro F1: {out['eval_leak_free_cv5']['macro_f1']['mean']:.3f} "
          f"± {out['eval_leak_free_cv5']['macro_f1']['std']:.3f}")

    Path(a.out).mkdir(exist_ok=True)
    p = Path(a.out) / "evaluation_results.json"
    p.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nSaved {p}")


if __name__ == "__main__":
    sys.exit(main())
