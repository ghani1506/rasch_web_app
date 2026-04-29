import io
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


st.set_page_config(
    page_title="Rasch Trial Item Analyzer",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Utility functions
# -----------------------------

def detect_item_columns(df: pd.DataFrame):
    """Detect binary item columns. Non-binary or ID columns are ignored."""
    item_cols = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna().unique()
        if len(valid) > 0 and set(valid).issubset({0, 1}):
            item_cols.append(col)
    return item_cols


def prepare_response_matrix(df: pd.DataFrame, item_cols):
    Y_df = df[item_cols].apply(pd.to_numeric, errors="coerce")
    Y_df = Y_df.dropna(axis=0, how="any")
    return Y_df


def fit_rasch_jmle(Y_df: pd.DataFrame, ridge: float = 0.05, maxiter: int = 1000):
    """Fit a simple Rasch-style model with ridge-regularized JMLE.

    This is intended for education and trial-item screening.
    """
    Y = Y_df.values.astype(float)
    N, I = Y.shape

    def unpack(params):
        theta = params[:N]
        b_raw = params[N:N + I]
        b = b_raw - np.mean(b_raw)  # identify scale by centering item difficulty
        return theta, b

    def nll(params):
        theta, b = unpack(params)
        eta = theta[:, None] - b[None, :]
        P = expit(eta)
        eps = 1e-9
        value = -np.sum(Y * np.log(P + eps) + (1 - Y) * np.log(1 - P + eps))
        value += ridge * np.sum(params ** 2)
        return value

    result = minimize(
        nll,
        x0=np.zeros(N + I),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    theta, difficulty = unpack(result.x)
    return theta, difficulty, result


def calculate_fit(Y_df: pd.DataFrame, theta: np.ndarray, difficulty: np.ndarray):
    Y = Y_df.values.astype(float)
    items = list(Y_df.columns)

    eta = theta[:, None] - difficulty[None, :]
    P = expit(eta)
    residual = Y - P
    variance = P * (1 - P)

    outfit = ((residual ** 2) / np.maximum(variance, 1e-9)).mean(axis=0)
    infit = (residual ** 2).sum(axis=0) / np.maximum(variance.sum(axis=0), 1e-9)
    p_correct = Y_df.mean(axis=0).values

    point_measure_corr = []
    for j in range(Y.shape[1]):
        item_response = Y[:, j]
        total_without_item = Y.sum(axis=1) - item_response
        if np.std(item_response) == 0 or np.std(total_without_item) == 0:
            point_measure_corr.append(np.nan)
        else:
            point_measure_corr.append(np.corrcoef(item_response, total_without_item)[0, 1])

    results = pd.DataFrame({
        "item": items,
        "p_correct": p_correct,
        "difficulty_logit": difficulty,
        "infit": infit,
        "outfit": outfit,
        "point_measure_corr": point_measure_corr,
    })

    return results


def decide_item(row):
    problems = []

    if row["p_correct"] >= 0.95:
        problems.append("too easy / ceiling effect")
    elif row["p_correct"] <= 0.05:
        problems.append("too hard / floor effect")

    if row["difficulty_logit"] > 3:
        problems.append("very high difficulty")
    elif row["difficulty_logit"] < -3:
        problems.append("very low difficulty")

    if row["infit"] > 1.5:
        problems.append("high infit / noisy")
    elif row["infit"] < 0.5:
        problems.append("low infit / redundant")

    if row["outfit"] > 1.5:
        problems.append("high outfit / outliers")
    elif row["outfit"] < 0.5:
        problems.append("low outfit / too predictable")

    corr = row["point_measure_corr"]
    if pd.notna(corr):
        if corr < 0:
            problems.append("negative correlation")
        elif corr < 0.20:
            problems.append("weak correlation")

    if len(problems) == 0:
        return pd.Series(["KEEP", "Good item"])

    serious_keywords = [
        "negative correlation",
        "high infit",
        "high outfit",
        "too easy",
        "too hard",
    ]
    serious = any(any(key in p for key in serious_keywords) for p in problems)

    if serious:
        return pd.Series(["REMOVE / MAJOR REVISION", "; ".join(problems)])
    return pd.Series(["REVISE", "; ".join(problems)])


def make_distribution_plot(theta, difficulty):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(theta, bins=30, alpha=0.55, density=True, label="Students / ability")
    ax.hist(difficulty, bins=15, alpha=0.55, density=True, label="Items / difficulty")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Logit scale")
    ax.set_ylabel("Density")
    ax.set_title("Person Ability and Item Difficulty Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def make_person_item_map(theta, difficulty, item_names):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Jitter persons slightly for visibility
    rng = np.random.default_rng(123)
    x_person = rng.normal(0, 0.025, len(theta))
    ax.scatter(x_person, theta, alpha=0.25, s=12, label="Persons")

    ax.scatter(np.ones(len(difficulty)), difficulty, marker="D", s=60, label="Items")
    for y, label in zip(difficulty, item_names):
        ax.text(1.04, y, str(label), va="center", fontsize=8)

    ax.set_xlim(-0.25, 1.65)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Persons\n(ability)", "Items\n(difficulty)"])
    ax.set_ylabel("Logits")
    ax.set_title("Person-Item Map")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig


def make_icc_plot(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    theta_grid = np.linspace(-4, 4, 150)

    # Plot up to 8 representative items sorted by difficulty
    tmp = results.sort_values("difficulty_logit")
    if len(tmp) > 8:
        idx = np.linspace(0, len(tmp) - 1, 8).round().astype(int)
        tmp = tmp.iloc[idx]

    for _, row in tmp.iterrows():
        b = row["difficulty_logit"]
        prob = expit(theta_grid - b)
        ax.plot(theta_grid, prob, label=row["item"])

    ax.set_xlabel("Ability (theta)")
    ax.set_ylabel("Probability correct")
    ax.set_title("Item Characteristic Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf


def generate_pdf_report(results, theta, difficulty, item_names, dataset_name="uploaded_data.csv"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Rasch Trial Item Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Dataset: {dataset_name}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    n_keep = int((results["decision"] == "KEEP").sum())
    n_revise = int((results["decision"] == "REVISE").sum())
    n_remove = int((results["decision"] == "REMOVE / MAJOR REVISION").sum())

    summary_text = (
        f"This report screens trial exam items using a Rasch-style 1PL model. "
        f"Items are classified as KEEP, REVISE, or REMOVE / MAJOR REVISION based on difficulty, "
        f"Infit, Outfit, and point-measure correlation. Summary: KEEP={n_keep}, "
        f"REVISE={n_revise}, REMOVE / MAJOR REVISION={n_remove}."
    )
    story.append(Paragraph(summary_text, styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    # Table
    table_cols = [
        "item", "p_correct", "difficulty_logit", "infit", "outfit",
        "point_measure_corr", "decision", "reason"
    ]
    table_df = results[table_cols].copy()
    for col in ["p_correct", "difficulty_logit", "infit", "outfit", "point_measure_corr"]:
        table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    data = [table_cols] + table_df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("Visual Diagnostics", styles["Heading1"]))
    figs = [
        make_distribution_plot(theta, difficulty),
        make_person_item_map(theta, difficulty, item_names),
        make_icc_plot(results),
    ]

    for fig in figs:
        img_buf = fig_to_png_bytes(fig)
        plt.close(fig)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_buf.getvalue())
            tmp_path = tmp.name
        story.append(Image(tmp_path, width=7.3 * inch, height=4.1 * inch))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Interpretation Guide", styles["Heading1"]))
    guide = (
        "KEEP: item is behaving well statistically. REVISE: item may be useful but should be reviewed. "
        "REMOVE / MAJOR REVISION: item has serious problems such as extreme difficulty, poor fit, outliers, "
        "or negative/weak relationship with the overall test. Always combine these statistics with content review."
    )
    story.append(Paragraph(guide, styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Streamlit interface
# -----------------------------

st.title("📊 Rasch Trial Item Analyzer")
st.write(
    "Upload trial exam response data and generate item-quality decisions with a downloadable PDF report."
)

with st.expander("CSV format example"):
    st.code(
        "person,item_1,item_2,item_3,item_4\n"
        "P1,1,1,0,0\n"
        "P2,1,0,0,0\n"
        "P3,1,1,1,0",
        language="csv",
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

with st.sidebar:
    st.header("Decision thresholds")
    lower_fit = st.number_input("Minimum acceptable Infit/Outfit", value=0.5, step=0.1)
    upper_fit = st.number_input("Maximum acceptable Infit/Outfit", value=1.5, step=0.1)
    min_corr = st.number_input("Minimum point-measure correlation", value=0.20, step=0.05)
    st.caption("Current app logic uses the default beginner rules in the code. Edit `decide_item()` to customize further.")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded data preview")
    st.dataframe(raw_df.head())

    item_cols = detect_item_columns(raw_df)
    if len(item_cols) < 2:
        st.error("Could not detect enough binary item columns. Make sure item columns contain only 0 and 1.")
        st.stop()

    Y_df = prepare_response_matrix(raw_df, item_cols)

    st.info(f"Detected {len(item_cols)} item columns and {len(Y_df)} complete student rows.")

    if st.button("Run Rasch Analysis", type="primary"):
        with st.spinner("Fitting Rasch-style model and checking item quality..."):
            theta, difficulty, result = fit_rasch_jmle(Y_df)
            results = calculate_fit(Y_df, theta, difficulty)
            decisions = results.apply(decide_item, axis=1)
            results["decision"] = decisions[0]
            results["reason"] = decisions[1]
            results = results.sort_values("difficulty_logit").reset_index(drop=True)

        st.success("Analysis complete.")

        c1, c2, c3 = st.columns(3)
        c1.metric("KEEP", int((results["decision"] == "KEEP").sum()))
        c2.metric("REVISE", int((results["decision"] == "REVISE").sum()))
        c3.metric("REMOVE / MAJOR REVISION", int((results["decision"] == "REMOVE / MAJOR REVISION").sum()))

        st.subheader("Item decision table")
        st.dataframe(results, use_container_width=True)

        st.subheader("Visual diagnostics")
        fig1 = make_distribution_plot(theta, difficulty)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = make_person_item_map(theta, difficulty, list(results.sort_values("difficulty_logit")["item"]))
        st.pyplot(fig2)
        plt.close(fig2)

        fig3 = make_icc_plot(results)
        st.pyplot(fig3)
        plt.close(fig3)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        pdf_bytes = generate_pdf_report(
            results=results,
            theta=theta,
            difficulty=difficulty,
            item_names=list(results.sort_values("difficulty_logit")["item"]),
            dataset_name=uploaded_file.name,
        )

        st.download_button(
            "Download item decision CSV",
            data=csv_bytes,
            file_name="rasch_item_decision_results.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download final PDF report",
            data=pdf_bytes,
            file_name="rasch_trial_item_report.pdf",
            mime="application/pdf",
        )
else:
    st.warning("Upload a CSV file to begin.")
