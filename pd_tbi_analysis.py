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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
# PART 3 – Machine-learning pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part 3: Machine Learning Pipeline ──")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
}

leaderboard_rows = []
roc_fig, roc_ax   = plt.subplots(figsize=(8, 6))
pr_fig,  pr_ax    = plt.subplots(figsize=(8, 6))
COLORS_ML = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

for (name, model), col in zip(MODELS.items(), COLORS_ML):
    print(f"  Training {name} …")

    # ── collect fold-level metrics ──────────────────────────────────────
    fold_acc, fold_prec, fold_rec, fold_f1, fold_auc = [], [], [], [], []

    y_prob_all = np.zeros(len(y))
    y_pred_all = np.zeros(len(y), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        y_pred_all[val_idx] = y_pred
        y_prob_all[val_idx] = y_prob

        fold_acc.append(accuracy_score(y_val, y_pred))
        fold_prec.append(precision_score(y_val, y_pred, zero_division=0))
        fold_rec.append(recall_score(y_val, y_pred, zero_division=0))
        fold_f1.append(f1_score(y_val, y_pred, zero_division=0))
        fold_auc.append(roc_auc_score(y_val, y_prob))

    # ── aggregate ────────────────────────────────────────────────────────
    row = {
        "Model":         name,
        "Accuracy":      f"{np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}",
        "Precision":     f"{np.mean(fold_prec):.3f} ± {np.std(fold_prec):.3f}",
        "Recall":        f"{np.mean(fold_rec):.3f} ± {np.std(fold_rec):.3f}",
        "F1-Score":      f"{np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}",
        "ROC-AUC":       f"{np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}",
        # raw for sorting
        "_acc":  np.mean(fold_acc),
        "_f1":   np.mean(fold_f1),
        "_auc":  np.mean(fold_auc),
    }
    leaderboard_rows.append(row)
    print(f"    Acc={row['Accuracy']}  F1={row['F1-Score']}  AUC={row['ROC-AUC']}")

    # ── ROC curve (pooled OOF) ────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y, y_prob_all)
    auc_val = roc_auc_score(y, y_prob_all)
    roc_ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{name} (AUC={auc_val:.3f})")

    # ── Precision-Recall curve (pooled OOF) ──────────────────────────────
    prec_c, rec_c, _ = precision_recall_curve(y, y_prob_all)
    ap = average_precision_score(y, y_prob_all)
    pr_ax.plot(rec_c, prec_c, color=col, linewidth=2, label=f"{name} (AP={ap:.3f})")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred_all)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["TBI (pred)", "PD (pred)"],
                yticklabels=["TBI (true)", "PD (true)"])
    ax_cm.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    fig_cm.savefig(os.path.join(OUT, f"plot_cm_{name.replace(' ','_').lower()}.png"), dpi=150)
    plt.close(fig_cm)

# finalise ROC
roc_ax.plot([0,1],[0,1], "k--", linewidth=1)
roc_ax.set_xlabel("False Positive Rate"); roc_ax.set_ylabel("True Positive Rate")
roc_ax.set_title("ROC Curves – All Models (OOF 10-fold CV)")
roc_ax.legend(loc="lower right"); roc_fig.tight_layout()
roc_fig.savefig(os.path.join(OUT, "plot_roc_curves.png"), dpi=150)
plt.close(roc_fig)

# finalise PR
pr_ax.set_xlabel("Recall"); pr_ax.set_ylabel("Precision")
pr_ax.set_title("Precision-Recall Curves – All Models (OOF 10-fold CV)")
pr_ax.legend(loc="lower left"); pr_fig.tight_layout()
pr_fig.savefig(os.path.join(OUT, "plot_pr_curves.png"), dpi=150)
plt.close(pr_fig)
print("  Saved: ROC, PR, and confusion matrix plots")

# ── Feature Importance – Random Forest ───────────────────────────────────
print("  Computing Random Forest feature importances …")
rf_full = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
])
rf_full.fit(X, y)
importances = rf_full.named_steps["clf"].feature_importances_
feat_imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=False)
top10_rf = feat_imp_df.head(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RF top-10
axes[0].barh(top10_rf["feature"][::-1], top10_rf["importance"][::-1], color="#2a9d8f")
axes[0].set_xlabel("Mean Decrease in Impurity")
axes[0].set_title("Top-10 Features – Random Forest")

# Statistical top-10
top10_stat = sig_0005.head(10) if len(sig_0005) >= 10 else sig_001.head(10)
top10_stat = top10_stat.copy()
top10_stat["-log10(p)"] = -np.log10(top10_stat["p_value"].clip(lower=1e-300))
axes[1].barh(top10_stat["feature"][::-1].values,
             top10_stat["-log10(p)"][::-1].values, color="#e9c46a")
axes[1].set_xlabel("-log₁₀(p-value)")
axes[1].set_title("Top-10 Features – Statistical (p-value)")

plt.suptitle("Feature Importance Comparison: Random Forest vs. Statistical Test", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_feature_importance_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved: plot_feature_importance_comparison.png")

# ── Overlap analysis ─────────────────────────────────────────────────────
rf_top10_set   = set(top10_rf["feature"].tolist())
stat_top10_set = set(top10_stat["feature"].tolist())
overlap = rf_top10_set & stat_top10_set
print(f"\n  Overlap between RF top-10 and statistical top-10: {len(overlap)} features")
if overlap:
    print(f"    {', '.join(sorted(overlap))}")

# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Leaderboard ──")
lb_df = pd.DataFrame(leaderboard_rows)
lb_display = lb_df[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]].copy()
lb_display = lb_display.sort_values("ROC-AUC", key=lambda s: lb_df["_auc"], ascending=False)
print(lb_display.to_string(index=False))

lb_display.to_csv(os.path.join(OUT, "leaderboard.csv"), index=False)
print("\n  Saved: leaderboard.csv")

# ── Leaderboard heatmap ───────────────────────────────────────────────────
lb_numeric = lb_df[["Model","_acc","_f1","_auc"]].set_index("Model")
lb_numeric.columns = ["Accuracy","F1-Score","ROC-AUC"]
lb_numeric = lb_numeric.sort_values("ROC-AUC", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(lb_numeric, annot=True, fmt=".3f", cmap="YlGn", ax=ax,
            vmin=0.5, vmax=1.0, linewidths=0.5, cbar_kws={"label": "Score"})
ax.set_title("Model Leaderboard  (10-fold Stratified CV)", fontsize=13, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "plot_leaderboard_heatmap.png"), dpi=150)
plt.close(fig)
print("  Saved: plot_leaderboard_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n══ Analysis Complete ══")
print(f"All outputs written to: {OUT}/")
output_files = sorted(os.listdir(OUT))
for f in output_files:
    print(f"  {f}")
