# =============================================================
# ML TRAINING PIPELINE - PD vs TBI CLASSIFICATION
# =============================================================
# Course: Smart and Connected Health
# Project: PD vs TBI Classification using Speech Analysis
# =============================================================
# This module handles:
#   1. Data loading & preprocessing
#   2. Feature analysis (PD vs TBI comparison)
#   3. Model training (Random Forest, SVM, XGBoost)
#   4. Evaluation (accuracy, precision, recall, F1, confusion matrix)
#   5. Model saving for inference
# =============================================================


# =============================================================
# CELL 1: INSTALL (Run once in Colab)
# =============================================================
# !pip install scikit-learn xgboost matplotlib seaborn joblib


# =============================================================
# CELL 2: IMPORTS
# =============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")


# =============================================================
# CELL 3: LOAD & PREPROCESS DATA
# =============================================================
def load_and_preprocess(csv_path):
    """
    Load feature CSV and prepare for training.

    Steps:
        1. Load CSV
        2. Separate features from labels
        3. Handle NaN/Inf values
        4. Encode labels (PD=1, TBI=0)
        5. Scale features

    Returns:
        X_scaled, y, feature_names, scaler, label_encoder
    """
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    drop_cols = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y_raw = df["label"].copy()

    # Handle NaN / Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # PD=1, TBI=0 (alphabetical)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    return X_scaled, y, feature_cols, scaler, le


# =============================================================
# CELL 4: FEATURE ANALYSIS - PD vs TBI COMPARISON
# =============================================================
def analyze_features(csv_path, top_n=15):
    """
    Compare features between PD and TBI groups.

    Produces:
        - Mean and std for each feature by group
        - Top discriminating features (by F-statistic)
        - Bar chart of top features
    """
    df = pd.read_csv(csv_path)
    drop_cols = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Group statistics
    print("=" * 70)
    print("FEATURE ANALYSIS: PD vs TBI")
    print("=" * 70)

    stats = df.groupby("label")[feature_cols].agg(["mean", "std"])
    print(f"\n{'Feature':<25} {'PD Mean':>12} {'PD Std':>12} {'TBI Mean':>12} {'TBI Std':>12}")
    print("-" * 75)

    for feat in feature_cols:
        pd_mean = stats.loc["PD", (feat, "mean")] if "PD" in stats.index else 0
        pd_std = stats.loc["PD", (feat, "std")] if "PD" in stats.index else 0
        tbi_mean = stats.loc["TBI", (feat, "mean")] if "TBI" in stats.index else 0
        tbi_std = stats.loc["TBI", (feat, "std")] if "TBI" in stats.index else 0
        print(f"  {feat:<23} {pd_mean:>12.4f} {pd_std:>12.4f} {tbi_mean:>12.4f} {tbi_std:>12.4f}")

    # Feature ranking using ANOVA F-statistic
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    selector = SelectKBest(f_classif, k=min(top_n, len(feature_cols)))
    selector.fit(X, y)

    scores = pd.DataFrame({
        "feature": feature_cols,
        "f_score": selector.scores_,
        "p_value": selector.pvalues_,
    }).sort_values("f_score", ascending=False)

    print(f"\n\nTOP {top_n} DISCRIMINATING FEATURES (ANOVA F-test)")
    print("-" * 50)
    for i, row in scores.head(top_n).iterrows():
        print(f"  {row['feature']:<25} F={row['f_score']:>10.4f}  p={row['p_value']:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top = scores.head(top_n)
    ax.barh(top["feature"][::-1], top["f_score"][::-1], color="steelblue")
    ax.set_xlabel("F-Statistic")
    ax.set_title("Top Discriminating Features: PD vs TBI")
    plt.tight_layout()
    plt.savefig("feature_importance_analysis.png", dpi=150)
    plt.show()
    print("\nSaved: feature_importance_analysis.png")

    return scores


# =============================================================
# CELL 5: MODEL TRAINING WITH CROSS-VALIDATION
# =============================================================
def train_and_evaluate(X, y, feature_names, scaler, le, output_dir="."):
    """
    Train multiple classifiers, evaluate with stratified k-fold CV.

    Models:
        1. Random Forest
        2. Support Vector Machine (RBF kernel)
        3. Gradient Boosting (XGBoost-style)
        4. Logistic Regression

    Evaluation:
        - 5-fold stratified cross-validation
        - Accuracy, Precision, Recall, F1
        - Confusion matrix
        - ROC curve (if enough data)

    Returns:
        best_model, results_dict
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }

    # Stratified K-Fold
    n_splits = min(5, min(np.bincount(y)))  # adapt to small datasets
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("=" * 70)
    print(f"MODEL TRAINING ({n_splits}-Fold Stratified Cross-Validation)")
    print("=" * 70)

    results = {}

    for name, model in models.items():
        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")

        results[name] = {
            "accuracy_mean": cv_accuracy.mean(),
            "accuracy_std": cv_accuracy.std(),
            "f1_mean": cv_f1.mean(),
            "f1_std": cv_f1.std(),
        }

        print(f"\n{name}:")
        print(f"  Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
        print(f"  F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

    # Train-test split evaluation for detailed metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n\n" + "=" * 70)
    print("DETAILED EVALUATION (80/20 Train-Test Split)")
    print("=" * 70)

    best_f1 = -1
    best_model = None
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # Retrain best model on full data
    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    best_model_full = models[best_name]
    best_model_full.fit(X, y)

    # Save model artifacts
    model_path = os.path.join(output_dir, "best_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    le_path = os.path.join(output_dir, "label_encoder.joblib")
    features_path = os.path.join(output_dir, "feature_names.joblib")

    joblib.dump(best_model_full, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    joblib.dump(feature_names, features_path)

    print(f"\nModel saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Label encoder saved: {le_path}")
    print(f"Feature names saved: {features_path}")

    # Confusion matrix for best model (on test set)
    best_model_eval = models[best_name]
    best_model_eval.fit(X_train, y_train)
    y_pred_best = best_model_eval.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")

    # ROC Curve (binary)
    if len(le.classes_) == 2 and hasattr(best_model_eval, "predict_proba"):
        y_prob = best_model_eval.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {best_name}")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.show()
        print("Saved: roc_curve.png")

    # Feature importance (for tree-based models)
    if hasattr(best_model_full, "feature_importances_"):
        importances = best_model_full.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = feat_imp.head(20)
        ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="teal")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 20 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=150)
        plt.show()
        print("Saved: feature_importances.png")

    return best_model_full, results


# =============================================================
# CELL 6: MAIN - RUN FULL PIPELINE
# =============================================================
if __name__ == "__main__":
    CSV_PATH = "features.csv"

    if not os.path.exists(CSV_PATH):
        print(f"Features CSV not found at: {CSV_PATH}")
        print("Run feature_extraction_enhanced.py first to generate features.")
        print("\nExample:")
        print("  from feature_extraction_enhanced import process_all_files")
        print('  df = process_all_files("path/to/PD", "path/to/TBI")')
        print('  df.to_csv("features.csv", index=False)')
        exit(1)

    # Step 1: Load & preprocess
    X, y, feature_names, scaler, le = load_and_preprocess(CSV_PATH)

    # Step 2: Feature analysis
    scores = analyze_features(CSV_PATH)

    # Step 3: Train & evaluate
    best_model, results = train_and_evaluate(X, y, feature_names, scaler, le)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
