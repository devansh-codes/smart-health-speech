"""
Parkinson's Disease vs. Concussion (TBI) Audio Feature Analysis
Covers:
  Part 1 – Statistical significance (T-test / Mann-Whitney U)
  Part 2 – Dimensionality reduction & visualisation (PCA, LDA, t-SNE)
  Part 3 – Robust ML pipeline with 10-fold stratified CV + leaderboard
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix,
)

# ── paths ────────────────────────────────────────────────────────────────────
BASE = "/Users/devanshchaudhary/Documents/smart-health/smart-health-speech"
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

PD_CSV  = os.path.join(BASE, "features_PD.csv")
TBI_CSV = os.path.join(BASE, "features_TBI.csv")

# ── palette ──────────────────────────────────────────────────────────────────
PD_COLOR  = "#e63946"   # red
TBI_COLOR = "#457b9d"   # blue

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load & merge data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …")
pd_df  = pd.read_csv(PD_CSV)
tbi_df = pd.read_csv(TBI_CSV)

# make sure labels are set
pd_df["label"]  = "PD"
tbi_df["label"] = "TBI"

df = pd.concat([pd_df, tbi_df], ignore_index=True)

# Identify feature columns (numeric, excluding metadata)
NON_FEATURE = {"label", "file_name", "audio_path", "patient_id", "date", "time"}
FEATURE_COLS = [c for c in df.columns
                if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
print(f"  PD rows: {len(pd_df)}  |  TBI rows: {len(tbi_df)}  |  Features: {len(FEATURE_COLS)}")

X = df[FEATURE_COLS].values
y = (df["label"] == "PD").astype(int).values   # PD=1, TBI=0

X_pd  = df.loc[df["label"] == "PD",  FEATURE_COLS].values
X_tbi = df.loc[df["label"] == "TBI", FEATURE_COLS].values

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 – Statistical significance
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part 1: Statistical Analysis ──")

results = []
for i, feat in enumerate(FEATURE_COLS):
    pd_vals  = X_pd[:, i]
    tbi_vals = X_tbi[:, i]

    # Normality check (Shapiro-Wilk) – use Mann-Whitney if either group non-normal
    _, p_norm_pd  = stats.shapiro(pd_vals)
    _, p_norm_tbi = stats.shapiro(tbi_vals)

    if p_norm_pd > 0.05 and p_norm_tbi > 0.05:
        stat, p_val = stats.ttest_ind(pd_vals, tbi_vals, equal_var=False)
        test_used = "t-test"
    else:
        stat, p_val = stats.mannwhitneyu(pd_vals, tbi_vals, alternative="two-sided")
        test_used = "Mann-Whitney U"

    results.append({
        "feature":   feat,
        "test":      test_used,
        "statistic": stat,
        "p_value":   p_val,
        "pd_mean":   pd_vals.mean(),
        "tbi_mean":  tbi_vals.mean(),
        "pd_std":    pd_vals.std(),
        "tbi_std":   tbi_vals.std(),
    })

stats_df = pd.DataFrame(results).sort_values("p_value")

sig_001  = stats_df[stats_df["p_value"] < 0.01].copy()
sig_0005 = stats_df[stats_df["p_value"] < 0.005].copy()

print(f"  Features with p < 0.01 : {len(sig_001)}")
print(f"  Features with p < 0.005: {len(sig_0005)}")

# save
stats_df.to_csv(os.path.join(OUT, "all_features_pvalues.csv"), index=False)
sig_001.to_csv(os.path.join(OUT, "significant_features_p001.csv"), index=False)
sig_0005.to_csv(os.path.join(OUT, "significant_features_p0005.csv"), index=False)
print("  Saved: all_features_pvalues.csv, significant_features_p001.csv, significant_features_p0005.csv")

# ── p-value distribution plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(stats_df["p_value"], bins=50, color="#6a994e", edgecolor="white", linewidth=0.3)
ax.axvline(0.01,  color="#e63946", linestyle="--", linewidth=1.5, label="p = 0.01")
ax.axvline(0.005, color="#f4a261", linestyle="--", linewidth=1.5, label="p = 0.005")
ax.set_xlabel("p-value"); ax.set_ylabel("Count"); ax.set_title("Distribution of p-values across all features")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_pvalue_distribution.png"), dpi=150)
plt.close(fig)

# ── top-20 significant features bar chart ────────────────────────────────
top20 = sig_001.head(20).copy()
top20["-log10(p)"] = -np.log10(top20["p_value"].clip(lower=1e-300))
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20["feature"][::-1], top20["-log10(p)"][::-1], color="#457b9d")
ax.axvline(-np.log10(0.01),  color="#e63946", linestyle="--", linewidth=1.2, label="p=0.01")
ax.axvline(-np.log10(0.005), color="#f4a261", linestyle="--", linewidth=1.2, label="p=0.005")
ax.set_xlabel("-log₁₀(p-value)"); ax.set_title("Top-20 Most Significant Features")
ax.legend(); plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_top20_significant_features.png"), dpi=150)
plt.close(fig)
print("  Saved: plot_pvalue_distribution.png, plot_top20_significant_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 – Dimensionality reduction & visualisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part 2: Dimensionality Reduction ──")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

labels_str = df["label"].values
colors = [PD_COLOR if l == "PD" else TBI_COLOR for l in labels_str]

def scatter_2d(coords, title, filename, labels=labels_str, c1=PD_COLOR, c2=TBI_COLOR):
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, col in [("PD", c1), ("TBI", c2)]:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=col, label=lbl, alpha=0.7, s=40, edgecolors="white", linewidths=0.3)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    ax.legend(title="Class", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

# ── PCA ───────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100
scatter_2d(X_pca,
           f"PCA  (PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  |  PD=red · TBI=blue)",
           "plot_pca.png")

# ── PCA scree plot ────────────────────────────────────────────────────────
pca_full = PCA(n_components=min(30, X_scaled.shape[1]), random_state=42)
pca_full.fit(X_scaled)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(pca_full.explained_variance_ratio_)+1),
       pca_full.explained_variance_ratio_*100, color="#457b9d")
ax.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
        np.cumsum(pca_full.explained_variance_ratio_)*100, "r-o", markersize=3)
ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
ax.set_title("PCA Scree Plot (top 30 components)")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_pca_scree.png"), dpi=150)
plt.close(fig)

# ── LDA ───────────────────────────────────────────────────────────────────
lda_vis = LinearDiscriminantAnalysis(n_components=1)
X_lda_1d = lda_vis.fit_transform(X_scaled, y)

fig, ax = plt.subplots(figsize=(8, 5))
for lbl, val, col in [("PD", 1, PD_COLOR), ("TBI", 0, TBI_COLOR)]:
    mask = y == val
    ax.hist(X_lda_1d[mask, 0], bins=25, alpha=0.65, color=col, label=lbl, edgecolor="white")
ax.set_xlabel("LDA Component 1"); ax.set_ylabel("Count")
ax.set_title("LDA  (1D projection  |  PD=red · TBI=blue)")
ax.legend(title="Class"); plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_lda_1d.png"), dpi=150)
plt.close(fig)
print("  Saved: plot_lda_1d.png")

# LDA 2D via PCA pre-reduction for visualisation
pca_50 = PCA(n_components=min(50, X_scaled.shape[0]-1, X_scaled.shape[1]), random_state=42)
X_pca50 = pca_50.fit_transform(X_scaled)
lda_2d = LinearDiscriminantAnalysis(n_components=1)   # binary → max 1 discriminant
X_lda_proj = lda_2d.fit_transform(X_pca50, y)
# Pair with first PCA component for 2D scatter
X_lda_2d = np.column_stack([X_pca50[:, 0], X_lda_proj[:, 0]])
scatter_2d(X_lda_2d,
           "LDA  (LDA axis vs. PCA-1  |  PD=red · TBI=blue)",
           "plot_lda_2d.png")

# ── t-SNE ─────────────────────────────────────────────────────────────────
print("  Running t-SNE (may take ~30 s) …")
pca_50b = PCA(n_components=min(50, X_scaled.shape[0]-1, X_scaled.shape[1]), random_state=42)
X_tsne_input = pca_50b.fit_transform(X_scaled)
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, learning_rate="auto", init="pca")
X_tsne = tsne.fit_transform(X_tsne_input)
scatter_2d(X_tsne, "t-SNE  (PD=red · TBI=blue)", "plot_tsne.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 – Machine-learning pipeline  (Group-Based Splits, No Data Leakage)
# ─────────────────────────────────────────────────────────────────────────────
# Groups are patient IDs — GroupShuffleSplit guarantees that every recording
# from a given patient lands exclusively in train OR test, never both.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part 3: Machine Learning Pipeline (Group-Based Validation) ──")

PATIENT_ID_COL = "patient_id"
groups = df[PATIENT_ID_COL].values
print(f"  Unique patients  : {df[PATIENT_ID_COL].nunique()}")
print(f"  Total recordings : {len(df)}")

COLORS_ML = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced", random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5,
                                           class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ]),
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
    }

SPLIT_SCENARIOS = [("70:30", 0.30), ("80:20", 0.20)]

all_results = {}   # {split_label: {model_name: {metric: float}}}

for split_label, test_size in SPLIT_SCENARIOS:
    tag = split_label.replace(":", "_")
    print(f"\n{'─'*60}")
    print(f"  Split: {split_label}  "
          f"(train ≈ {1 - test_size:.0%} / test ≈ {test_size:.0%})")
    print(f"{'─'*60}")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    # Strict no-overlap verification — any overlap means data leakage
    train_patients = set(groups[train_idx])
    test_patients  = set(groups[test_idx])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"Data leakage! Overlapping patients: {overlap}"
    print(f"  Train patients: {len(train_patients)}  |  "
          f"Test patients: {len(test_patients)}  |  Overlap: 0 ✓")
    print(f"  Train samples : {len(train_idx)}       |  "
          f"Test samples : {len(test_idx)}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models       = make_models()
    split_results = {}
    roc_data     = {}
    pr_data      = {}

    for name, model in models.items():
        print(f"  Training {name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC":   roc_auc_score(y_test, y_prob),
        }
        split_results[name] = metrics
        print(f"    Acc={metrics['Accuracy']:.3f}  "
              f"F1={metrics['F1-Score']:.3f}  "
              f"AUC={metrics['AUC-ROC']:.3f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics["AUC-ROC"])

        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        pr_data[name] = (rec_arr, prec_arr, ap)

        # ── Confusion matrix ──────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["TBI (pred)", "PD (pred)"],
                    yticklabels=["TBI (true)", "PD (true)"])
        ax_cm.set_title(f"Confusion Matrix – {name} ({split_label})")
        plt.tight_layout()
        fig_cm.savefig(
            os.path.join(OUT, f"plot_cm_{name.replace(' ','_').lower()}_{tag}.png"),
            dpi=150,
        )
        plt.close(fig_cm)

    all_results[split_label] = split_results

    # ── ROC Curves — all models on one graph ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, (fpr, tpr, auc_val)), col in zip(roc_data.items(), COLORS_ML):
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves — {split_label} Group Split",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"plot_roc_curves_{tag}.png"), dpi=200)
    plt.close(fig)

    # ── Precision-Recall Curves — all models on one graph ────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, (rec_arr, prec_arr, ap)), col in zip(pr_data.items(), COLORS_ML):
        ax.plot(rec_arr, prec_arr, color=col, linewidth=2,
                label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {split_label} Group Split",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"plot_pr_curves_{tag}.png"), dpi=200)
    plt.close(fig)

    # ── Random Forest feature importance — top 10 ─────────────────────────
    rf_pipeline = models["Random Forest"]
    importances = rf_pipeline.named_steps["clf"].feature_importances_
    feat_imp_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1],
            color="#2a9d8f")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=12)
    ax.set_title(f"Random Forest — Top 10 Features ({split_label} Split)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"plot_rf_feature_importance_{tag}.png"), dpi=200)
    plt.close(fig)

    print(f"  Saved: ROC, PR, RF feature importance, and confusion matrix "
          f"plots for {split_label}")

# ─────────────────────────────────────────────────────────────────────────────
# Comparison Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Comparison Leaderboard ──")

lb_rows = []
for split_label, split_results in all_results.items():
    for model_name, metrics in split_results.items():
        lb_rows.append({"Split": split_label, "Model": model_name, **metrics})

lb_df = (
    pd.DataFrame(lb_rows)
    .sort_values(["Split", "AUC-ROC"], ascending=[True, False])
    .reset_index(drop=True)
)
print(lb_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
lb_df.to_csv(os.path.join(OUT, "leaderboard.csv"), index=False)

# ── Leaderboard table PNG ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 1 + len(lb_df) * 0.55))
ax.axis("off")
tbl = ax.table(
    cellText=lb_df.round(4).values,
    colLabels=lb_df.columns.tolist(),
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
for col_idx in range(len(lb_df.columns)):
    tbl[0, col_idx].set_facecolor("#2c7bb6")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")
for row_idx in range(1, len(lb_df) + 1):
    bg = "#f0f8ff" if row_idx % 2 == 0 else "white"
    for col_idx in range(len(lb_df.columns)):
        tbl[row_idx, col_idx].set_facecolor(bg)
ax.set_title("Model Comparison Leaderboard (Group-Based Splits: 70:30 & 80:20)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_leaderboard.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Leaderboard heatmap ───────────────────────────────────────────────────
pivot = lb_df.set_index(["Model", "Split"])[
    ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
]
fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.55)))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax,
            vmin=0.5, vmax=1.0, linewidths=0.5, cbar_kws={"label": "Score"})
ax.set_title("Model Comparison Heatmap (70:30 & 80:20 Group Splits)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_leaderboard_heatmap.png"), dpi=200)
plt.close(fig)

print("\n  Saved: leaderboard.csv, plot_leaderboard.png, plot_leaderboard_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n══ Analysis Complete ══")
print(f"All outputs written to: {OUT}/")
output_files = sorted(os.listdir(OUT))
for f in output_files:
    print(f"  {f}")
