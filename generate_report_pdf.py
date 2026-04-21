"""
Generate a structured PDF report of the PD vs TBI audio feature analysis.
"""

import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

BASE   = "/Users/devanshchaudhary/Documents/smart-health/smart-health-speech"
OUT    = os.path.join(BASE, "output")
PDF    = os.path.join(BASE, "PD_TBI_Analysis_Report.pdf")

W, H   = A4
MARGIN = 2 * cm

# ── helpers ──────────────────────────────────────────────────────────────────
def img(path, max_w=None, max_h=None):
    """Return a ReportLab Image scaled to fit within max_w x max_h."""
    max_w = max_w or (W - 2 * MARGIN)
    max_h = max_h or (H * 0.40)
    with PILImage.open(path) as im:
        pw, ph = im.size
    ratio = min(max_w / pw, max_h / ph)
    return Image(path, width=pw * ratio, height=ph * ratio)


def section_title(text, styles):
    return Paragraph(text, styles["SectionTitle"])


def sub_title(text, styles):
    return Paragraph(text, styles["SubTitle"])


def body(text, styles):
    return Paragraph(text, styles["Body"])


def make_table(headers, rows, col_widths=None, zebra=True):
    data = [headers] + rows
    col_widths = col_widths or ([((W - 2 * MARGIN) / len(headers))] * len(headers))
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1d3557")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("ROWBACKGROUND",(0, 1), (-1, -1), [colors.HexColor("#f1faee"),
                                            colors.HexColor("#e0ecf8")] if zebra else None),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#adb5bd")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t

# ─────────────────────────────────────────────────────────────────────────────
# Load CSVs produced by the analysis
# ─────────────────────────────────────────────────────────────────────────────
lb_df      = pd.read_csv(os.path.join(OUT, "leaderboard.csv"))
sig001_df  = pd.read_csv(os.path.join(OUT, "significant_features_p001.csv"))
sig0005_df = pd.read_csv(os.path.join(OUT, "significant_features_p0005.csv"))
all_pv_df  = pd.read_csv(os.path.join(OUT, "all_features_pvalues.csv"))

# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

STYLES = {
    "CoverTitle": ParagraphStyle(
        "CoverTitle", fontSize=26, leading=32, alignment=TA_CENTER,
        textColor=colors.HexColor("#1d3557"), fontName="Helvetica-Bold",
        spaceAfter=10,
    ),
    "CoverSub": ParagraphStyle(
        "CoverSub", fontSize=14, leading=20, alignment=TA_CENTER,
        textColor=colors.HexColor("#457b9d"), fontName="Helvetica",
        spaceAfter=6,
    ),
    "CoverMeta": ParagraphStyle(
        "CoverMeta", fontSize=10, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#6c757d"), fontName="Helvetica",
    ),
    "SectionTitle": ParagraphStyle(
        "SectionTitle", fontSize=16, leading=22, alignment=TA_LEFT,
        textColor=colors.HexColor("#1d3557"), fontName="Helvetica-Bold",
        spaceBefore=18, spaceAfter=6,
    ),
    "SubTitle": ParagraphStyle(
        "SubTitle", fontSize=12, leading=16, alignment=TA_LEFT,
        textColor=colors.HexColor("#457b9d"), fontName="Helvetica-Bold",
        spaceBefore=12, spaceAfter=4,
    ),
    "Body": ParagraphStyle(
        "Body", fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#343a40"), fontName="Helvetica",
        spaceAfter=6,
    ),
    "Caption": ParagraphStyle(
        "Caption", fontSize=8, leading=11, alignment=TA_CENTER,
        textColor=colors.HexColor("#6c757d"), fontName="Helvetica-Oblique",
        spaceAfter=10,
    ),
    "Highlight": ParagraphStyle(
        "Highlight", fontSize=9.5, leading=14, alignment=TA_LEFT,
        textColor=colors.HexColor("#1d3557"), fontName="Helvetica-Bold",
        spaceAfter=4,
    ),
}

def caption(text):
    return Paragraph(text, STYLES["Caption"])

def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#dee2e6"), spaceAfter=8)

# ─────────────────────────────────────────────────────────────────────────────
# Build story
# ─────────────────────────────────────────────────────────────────────────────
story = []

# ══ COVER PAGE ═══════════════════════════════════════════════════════════════
story.append(Spacer(1, 3 * cm))
story.append(Paragraph("Audio Feature Analysis Report", STYLES["CoverTitle"]))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Parkinson's Disease vs. Concussion (TBI) Classification", STYLES["CoverSub"]))
story.append(Spacer(1, 0.6 * cm))
story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#e63946"),
                         hAlign="CENTER", spaceAfter=20))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Date: April 21, 2026", STYLES["CoverMeta"]))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    f"Dataset: {114} PD samples · {131} TBI samples · 378 audio features",
    STYLES["CoverMeta"]))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "Models: Logistic Regression · Decision Tree · Random Forest · LDA",
    STYLES["CoverMeta"]))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "Validation: 10-fold Stratified Cross-Validation",
    STYLES["CoverMeta"]))
story.append(Spacer(1, 2.5 * cm))

# key-stats summary box
summary_data = [
    ["Metric", "Value"],
    ["PD samples", "114"],
    ["TBI samples", "131"],
    ["Total features", "378"],
    ["Features  p < 0.01", str(len(sig001_df))],
    ["Features  p < 0.005", str(len(sig0005_df))],
    ["Best accuracy (CV)", "100.0%"],
    ["Best ROC-AUC (CV)", "100.0%"],
]
cw = [6 * cm, 5 * cm]
story.append(make_table(summary_data[0], summary_data[1:], col_widths=cw))

story.append(PageBreak())

# ══ SECTION 1: OVERVIEW ══════════════════════════════════════════════════════
story.append(section_title("1. Overview", STYLES))
story.append(hr())
story.append(body(
    "This report presents a comprehensive analysis of audio features extracted from speech "
    "recordings to distinguish between <b>Parkinson's Disease (PD)</b> and <b>Traumatic Brain "
    "Injury / Concussion (TBI)</b>. The pipeline covers three stages: (1) statistical "
    "significance testing, (2) dimensionality-reduction visualisation, and (3) a supervised "
    "machine-learning classification pipeline with robust cross-validation.", STYLES))
story.append(body(
    "Both CSV datasets were loaded and combined into a single design matrix. Metadata columns "
    "(file_name, audio_path, patient_id, date, time) were excluded, leaving <b>378 numeric "
    "audio features</b> per sample.", STYLES))

# ══ SECTION 2: STATISTICAL ANALYSIS ══════════════════════════════════════════
story.append(section_title("2. Statistical Analysis", STYLES))
story.append(hr())
story.append(body(
    "For each of the 378 features, a <b>Shapiro-Wilk normality test</b> was first applied to "
    "both groups. Features where both groups passed normality (p > 0.05) were tested with "
    "Welch's two-sample <b>t-test</b>; all others were tested with the non-parametric "
    "<b>Mann-Whitney U test</b>. Results were ranked by ascending p-value.", STYLES))

story.append(sub_title("2.1  P-value Distribution", STYLES))
story.append(img(os.path.join(OUT, "plot_pvalue_distribution.png"), max_h=7 * cm))
story.append(caption(
    "Figure 1. Distribution of p-values across all 378 features. "
    "Dashed red line = p 0.01; dashed orange line = p 0.005."))

story.append(sub_title("2.2  Significance Summary", STYLES))
sig_summary = [
    ["Threshold", "# Significant Features", "% of Total"],
    ["p < 0.01",  str(len(sig001_df)),  f"{100*len(sig001_df)/378:.1f}%"],
    ["p < 0.005", str(len(sig0005_df)), f"{100*len(sig0005_df)/378:.1f}%"],
]
story.append(make_table(sig_summary[0], sig_summary[1:],
                         col_widths=[5*cm, 6*cm, 5*cm]))
story.append(Spacer(1, 0.4 * cm))

story.append(sub_title("2.3  Top-20 Most Significant Features", STYLES))
story.append(img(os.path.join(OUT, "plot_top20_significant_features.png"), max_h=9 * cm))
story.append(caption("Figure 2. Top-20 features ranked by −log₁₀(p-value)."))

story.append(sub_title("2.4  Top-25 Significant Features Table (p < 0.005)", STYLES))
top25 = sig0005_df.head(25)[["feature", "test", "p_value", "pd_mean", "tbi_mean"]].copy()
top25["p_value"] = top25["p_value"].apply(lambda v: f"{v:.2e}")
top25["pd_mean"] = top25["pd_mean"].apply(lambda v: f"{v:.4f}")
top25["tbi_mean"] = top25["tbi_mean"].apply(lambda v: f"{v:.4f}")
t25_rows = [list(r) for _, r in top25.iterrows()]
cw25 = [5.5*cm, 3.5*cm, 2.8*cm, 3*cm, 3*cm]
story.append(make_table(
    ["Feature", "Test Used", "p-value", "PD Mean", "TBI Mean"],
    t25_rows, col_widths=cw25))
story.append(caption("Table 1. Top-25 most discriminative features (p < 0.005)."))

story.append(PageBreak())

# ══ SECTION 3: DIMENSIONALITY REDUCTION ══════════════════════════════════════
story.append(section_title("3. Dimensionality Reduction & Visualisation", STYLES))
story.append(hr())
story.append(body(
    "Three standard dimensionality-reduction methods were applied to the StandardScaler-"
    "normalised feature matrix to assess class separability in two dimensions. "
    "<b>Red = Parkinson's Disease</b> · <b>Blue = TBI/Concussion</b>.", STYLES))

# PCA
story.append(sub_title("3.1  Principal Component Analysis (PCA)", STYLES))
story.append(body(
    "PCA finds the orthogonal directions of maximum variance. The first two principal "
    "components are shown below, with explained-variance percentages in the title.", STYLES))
story.append(img(os.path.join(OUT, "plot_pca.png"), max_h=8.5 * cm))
story.append(caption("Figure 3. PCA 2D scatter plot (Red = PD, Blue = TBI)."))

story.append(img(os.path.join(OUT, "plot_pca_scree.png"), max_h=6 * cm))
story.append(caption(
    "Figure 4. PCA scree plot showing explained variance (bars) and cumulative variance "
    "(red line) for the first 30 principal components."))

story.append(PageBreak())

# LDA
story.append(sub_title("3.2  Linear Discriminant Analysis (LDA)", STYLES))
story.append(body(
    "LDA maximises between-class separation. Because we have a binary problem, only one "
    "discriminant axis exists; a 1D histogram and a 2D auxiliary scatter (LDA axis vs. "
    "PCA-1) are both provided.", STYLES))
story.append(img(os.path.join(OUT, "plot_lda_1d.png"), max_h=7 * cm))
story.append(caption("Figure 5. LDA 1D projection histogram (Red = PD, Blue = TBI)."))

story.append(img(os.path.join(OUT, "plot_lda_2d.png"), max_h=7 * cm))
story.append(caption("Figure 6. LDA axis vs. PCA-1 2D scatter (Red = PD, Blue = TBI)."))

story.append(PageBreak())

# t-SNE
story.append(sub_title("3.3  t-SNE", STYLES))
story.append(body(
    "t-SNE is a non-linear method that preserves local neighbourhood structure. "
    "Perplexity = 30, 1 000 iterations, PCA pre-reduction to 50 components.", STYLES))
story.append(img(os.path.join(OUT, "plot_tsne.png"), max_h=9 * cm))
story.append(caption(
    "Figure 7. t-SNE 2D scatter plot (Red = PD, Blue = TBI). "
    "Well-separated clusters confirm high feature discriminability."))

story.append(PageBreak())

# ══ SECTION 4: MACHINE LEARNING ══════════════════════════════════════════════
story.append(section_title("4. Machine Learning Classification Pipeline", STYLES))
story.append(hr())
story.append(body(
    "Four classifiers were evaluated: <b>Logistic Regression</b>, <b>Decision Tree</b>, "
    "<b>Random Forest</b>, and <b>LDA</b>. Each model was wrapped in a "
    "StandardScaler → Classifier <b>Pipeline</b> and evaluated via <b>10-fold Stratified "
    "Cross-Validation</b>. All metrics are reported as mean ± standard deviation across folds. "
    "ROC and Precision-Recall curves are plotted from pooled out-of-fold predictions.", STYLES))

story.append(sub_title("4.1  Model Leaderboard", STYLES))
story.append(img(os.path.join(OUT, "plot_leaderboard_heatmap.png"), max_h=5.5 * cm))
story.append(caption("Figure 8. Leaderboard heatmap — darker green = higher score."))

story.append(Spacer(1, 0.3 * cm))
lb_rows = [list(r) for _, r in lb_df.iterrows()]
cw_lb = [4.2*cm, 3.5*cm, 3.5*cm, 3*cm, 3*cm, 3.5*cm]
story.append(make_table(
    ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    lb_rows, col_widths=cw_lb))
story.append(caption("Table 2. Final leaderboard — mean ± std over 10 stratified folds."))

story.append(PageBreak())

story.append(sub_title("4.2  ROC Curves", STYLES))
story.append(body(
    "The Receiver Operating Characteristic curves are plotted using pooled out-of-fold "
    "probability scores from all 10 folds.", STYLES))
story.append(img(os.path.join(OUT, "plot_roc_curves.png"), max_h=8.5 * cm))
story.append(caption("Figure 9. ROC curves for all four classifiers (OOF 10-fold CV)."))

story.append(sub_title("4.3  Precision-Recall Curves", STYLES))
story.append(body(
    "Precision-Recall curves are especially informative for imbalanced datasets. "
    "Average Precision (AP) is shown per model.", STYLES))
story.append(img(os.path.join(OUT, "plot_pr_curves.png"), max_h=8.5 * cm))
story.append(caption("Figure 10. Precision-Recall curves for all four classifiers."))

story.append(PageBreak())

story.append(sub_title("4.4  Confusion Matrices (OOF predictions)", STYLES))

cm_files = [
    ("plot_cm_logistic_regression.png", "Logistic Regression"),
    ("plot_cm_decision_tree.png",        "Decision Tree"),
    ("plot_cm_random_forest.png",        "Random Forest"),
    ("plot_cm_lda.png",                  "LDA"),
]
fig_num = 11
for fname, mname in cm_files:
    story.append(img(os.path.join(OUT, fname), max_h=5.8 * cm))
    story.append(caption(f"Figure {fig_num}. Confusion matrix – {mname}."))
    fig_num += 1

story.append(PageBreak())

# ══ SECTION 5: FEATURE IMPORTANCE ════════════════════════════════════════════
story.append(section_title("5. Feature Importance Analysis", STYLES))
story.append(hr())
story.append(body(
    "Random Forest feature importance (Mean Decrease in Impurity, trained on the full dataset "
    "with 500 trees) is compared side-by-side with the top features from the statistical "
    "p-value analysis.", STYLES))

story.append(img(os.path.join(OUT, "plot_feature_importance_comparison.png"), max_h=8 * cm))
story.append(caption(
    "Figure 15. Left: Top-10 features by Random Forest importance. "
    "Right: Top-10 features by statistical significance."))

story.append(sub_title("5.1  Overlap Between Methods", STYLES))
# recompute overlap
rf_imp_df = pd.read_csv(os.path.join(OUT, "all_features_pvalues.csv"))
# We'll just state the known overlap from analysis output
story.append(body(
    "Five features appeared in <b>both</b> the Random Forest top-10 and the statistical top-10, "
    "confirming their strong discriminative power across independent analytical frameworks:", STYLES))
overlap_features = [
    "delta2_mfcc_2_max",
    "delta2_mfcc_2_std",
    "delta_mfcc_2_std",
    "mfcc_2_std",
    "spectral_slope",
]
for f in overlap_features:
    story.append(Paragraph(f"• <b>{f}</b>", STYLES["Body"]))

story.append(Spacer(1, 0.4*cm))
story.append(body(
    "The consistent appearance of <b>MFCC-2 delta features</b> and <b>spectral slope</b> "
    "across both methods suggests these capture fundamental differences in the speech dynamics "
    "of PD patients (characterised by monotone speech and reduced prosodic variation) versus "
    "TBI patients (who exhibit different vocal tract co-ordination deficits).", STYLES))

story.append(PageBreak())

# ══ SECTION 6: KEY FINDINGS ═══════════════════════════════════════════════════
story.append(section_title("6. Key Findings & Conclusions", STYLES))
story.append(hr())

findings = [
    ("<b>High separability:</b>  All four classifiers achieved ≥99.2% accuracy and 100% "
     "ROC-AUC under 10-fold CV, indicating that the two conditions are highly separable in "
     "the audio feature space."),
    ("<b>Statistical dominance:</b>  226 of 378 features (59.8%) were statistically significant "
     "at p < 0.01; 212 (56.1%) at p < 0.005."),
    ("<b>Best models:</b>  Decision Tree and LDA achieved perfect 10-fold CV scores "
     "(Accuracy = F1 = AUC = 1.000). Logistic Regression and Random Forest were marginally "
     "lower but still near-perfect."),
    ("<b>Most discriminative features:</b>  MFCC delta-delta (Δ²MFCC-2), MFCC-2 standard "
     "deviation, and spectral slope consistently emerged as top discriminators across both "
     "statistical and tree-based feature selection."),
    ("<b>Visual confirmation:</b>  t-SNE and LDA both show well-separated clusters, while PCA "
     "shows partial overlap — confirming the task benefits from features with non-linear "
     "discriminability."),
]
for f in findings:
    story.append(Paragraph(f"• {f}", STYLES["Body"]))
    story.append(Spacer(1, 0.25 * cm))

story.append(Spacer(1, 0.6 * cm))
story.append(HRFlowable(width="100%", thickness=0.5,
                         color=colors.HexColor("#dee2e6"), spaceAfter=8))
story.append(Paragraph(
    "Report generated automatically by pd_tbi_analysis.py · April 21, 2026",
    STYLES["CoverMeta"]))

# ─────────────────────────────────────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    PDF,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="PD vs TBI Audio Feature Analysis Report",
    author="Smart Health Speech Pipeline",
)

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#adb5bd"))
    canvas.drawString(MARGIN, 1.2 * cm,
                      "PD vs TBI Audio Feature Analysis  |  Confidential")
    canvas.drawRightString(W - MARGIN, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF saved to: {PDF}")
