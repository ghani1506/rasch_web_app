import io
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

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
    page_title="Rasch / Partial Credit Trial Item Analyzer",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Data detection and preparation
# -----------------------------

def detect_item_columns(df: pd.DataFrame):
    """
    Detect dichotomous or ordered polytomous item columns.

    Accepts columns with numeric integer categories such as:
    0/1
    0/1/2
    0/1/2/3/4

    Non-numeric, ID, text, and non-integer columns are ignored.
    """
    item_cols = []

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()

        if valid.empty:
            continue

        values = sorted(valid.unique())

        # Must be integer-like
        if not np.allclose(values, np.round(values)):
            continue

        values = [int(v) for v in values]

        # Must start at 0 and be consecutive
        if values[0] != 0:
            continue

        if values != list(range(values[-1] + 1)):
            continue

        # Need at least two categories
        if len(values) >= 2:
            item_cols.append(col)

    return item_cols


def prepare_response_matrix(df: pd.DataFrame, item_cols):
    Y_df = df[item_cols].apply(pd.to_numeric, errors="coerce")
    Y_df = Y_df.dropna(axis=0, how="any")
    Y_df = Y_df.astype(int)
    return Y_df


def is_dichotomous(Y_df: pd.DataFrame):
    values = np.unique(Y_df.values)
    return set(values).issubset({0, 1})


# -----------------------------
# Dichotomous Rasch model
# -----------------------------

def fit_rasch_jmle(Y_df: pd.DataFrame, ridge: float = 0.05, maxiter: int = 1000):
    Y = Y_df.values.astype(float)
    N, I = Y.shape

    def unpack(params):
        theta = params[:N]
        b_raw = params[N:N + I]
        b = b_raw - np.mean(b_raw)
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


def calculate_binary_fit(Y_df, theta, difficulty):
    Y = Y_df.values.astype(float)
    items = list(Y_df.columns)

    eta = theta[:, None] - difficulty[None, :]
    P = expit(eta)

    residual = Y - P
    variance = P * (1 - P)

    outfit = ((residual ** 2) / np.maximum(variance, 1e-9)).mean(axis=0)
    infit = (residual ** 2).sum(axis=0) / np.maximum(variance.sum(axis=0), 1e-9)

    p_score = Y_df.mean(axis=0).values

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
        "max_score": 1,
        "mean_score": p_score,
        "mean_score_pct": p_score,
        "difficulty_logit": difficulty,
        "infit": infit,
        "outfit": outfit,
        "point_measure_corr": point_measure_corr,
    })

    return results


# -----------------------------
# Polytomous Partial Credit Model
# -----------------------------

def fit_pcm_jmle(Y_df: pd.DataFrame, ridge: float = 0.05, maxiter: int = 1500):
    """
    Fits a simple Partial Credit Model style Rasch model using ridge-regularized JMLE.

    P(X_ni = k) proportional to exp(sum_m=1^k(theta_n - delta_im))

    This is suitable for ordered polytomous item screening.
    """
    Y = Y_df.values.astype(int)
    N, I = Y.shape

    max_scores = Y_df.max(axis=0).astype(int).values
    step_counts = max_scores  # item with max score m has m step parameters

    offsets = np.cumsum([0] + list(step_counts))
    total_steps = int(sum(step_counts))

    def unpack(params):
        theta = params[:N]
        raw_steps = params[N:N + total_steps]

        # Center all step parameters for scale identification
        if total_steps > 0:
            raw_steps = raw_steps - np.mean(raw_steps)

        steps = []
        for j in range(I):
            start, end = offsets[j], offsets[j + 1]
            steps.append(raw_steps[start:end])

        return theta, steps

    def item_log_probs(theta_n, item_steps):
        m = len(item_steps)
        scores = np.arange(m + 1)

        # category 0 has exponent 0
        exponents = [0.0]
        cumulative = 0.0

        for k in range(1, m + 1):
            cumulative += theta_n - item_steps[k - 1]
            exponents.append(cumulative)

        exponents = np.array(exponents)
        log_probs = exponents - logsumexp(exponents)
        return log_probs

    def nll(params):
        theta, steps = unpack(params)
        value = 0.0

        for n in range(N):
            for j in range(I):
                y = Y[n, j]
                log_probs = item_log_probs(theta[n], steps[j])
                value -= log_probs[y]

        value += ridge * np.sum(params ** 2)
        return value

    result = minimize(
        nll,
        x0=np.zeros(N + total_steps),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    theta, steps = unpack(result.x)

    # Item location is the average step difficulty
    item_difficulty = np.array([
        np.mean(s) if len(s) > 0 else 0.0
        for s in steps
    ])

    return theta, item_difficulty, steps, result


def pcm_expected_variance(theta, item_steps):
    m = len(item_steps)
    scores = np.arange(m + 1)

    exponents = [0.0]
    cumulative = 0.0

    for k in range(1, m + 1):
        cumulative += theta - item_steps[k - 1]
        exponents.append(cumulative)

    exponents = np.array(exponents)
    probs = np.exp(exponents - logsumexp(exponents))

    expected = np.sum(scores * probs)
    variance = np.sum(((scores - expected) ** 2) * probs)

    return expected, variance


def calculate_pcm_fit(Y_df, theta, difficulty, steps):
    Y = Y_df.values.astype(float)
    items = list(Y_df.columns)
    N, I = Y.shape

    expected = np.zeros_like(Y, dtype=float)
    variance = np.zeros_like(Y, dtype=float)

    for n in range(N):
        for j in range(I):
            expected[n, j], variance[n, j] = pcm_expected_variance(theta[n], steps[j])

    residual = Y - expected

    outfit = ((residual ** 2) / np.maximum(variance, 1e-9)).mean(axis=0)
    infit = (residual ** 2).sum(axis=0) / np.maximum(variance.sum(axis=0), 1e-9)

    max_scores = Y_df.max(axis=0).values
    mean_score = Y_df.mean(axis=0).values
    mean_score_pct = mean_score / np.maximum(max_scores, 1)

    point_measure_corr = []
    for j in range(I):
        item_score = Y[:, j]
        total_without_item = Y.sum(axis=1) - item_score

        if np.std(item_score) == 0 or np.std(total_without_item) == 0:
            point_measure_corr.append(np.nan)
        else:
            point_measure_corr.append(np.corrcoef(item_score, total_without_item)[0, 1])

    results = pd.DataFrame({
        "item": items,
        "max_score": max_scores,
        "mean_score": mean_score,
        "mean_score_pct": mean_score_pct,
        "difficulty_logit": difficulty,
        "infit": infit,
        "outfit": outfit,
        "point_measure_corr": point_measure_corr,
    })

    return results


# -----------------------------
# Decision logic
# -----------------------------

def decide_item(row):
    problems = []

    if row["mean_score_pct"] >= 0.95:
        problems.append("too easy / ceiling effect")
    elif row["mean_score_pct"] <= 0.05:
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


# -----------------------------
# Plots
# -----------------------------

def make_distribution_plot(theta, difficulty):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(theta, bins=30, alpha=0.55, density=True, label="Persons / ability")
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


def make_binary_icc_plot(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    theta_grid = np.linspace(-4, 4, 150)

    tmp = results.sort_values("difficulty_logit")
    if len(tmp) > 8:
        idx = np.linspace(0, len(tmp) - 1, 8).round().astype(int)
        tmp = tmp.iloc[idx]

    for _, row in tmp.iterrows():
        b = row["difficulty_logit"]
        prob = expit(theta_grid - b)
        ax.plot(theta_grid, prob, label=row["item"])

    ax.set_xlabel("Ability")
    ax.set_ylabel("Probability correct")
    ax.set_title("Item Characteristic Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def make_pcm_category_plot(item_name, item_steps):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    theta_grid = np.linspace(-4, 4, 150)

    m = len(item_steps)
    scores = np.arange(m + 1)

    probs_all = []

    for theta in theta_grid:
        exponents = [0.0]
        cumulative = 0.0

        for k in range(1, m + 1):
            cumulative += theta - item_steps[k - 1]
            exponents.append(cumulative)

        exponents = np.array(exponents)
        probs = np.exp(exponents - logsumexp(exponents))
        probs_all.append(probs)

    probs_all = np.array(probs_all)

    for k in scores:
        ax.plot(theta_grid, probs_all[:, k], label=f"Score {k}")

    ax.set_xlabel("Ability")
    ax.set_ylabel("Category probability")
    ax.set_title(f"Category Probability Curves: {item_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf


# -----------------------------
# PDF report
# -----------------------------

def generate_pdf_report(
    results,
    theta,
    difficulty,
    item_names,
    model_type,
    dataset_name="uploaded_data.csv",
):
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

    story.append(Paragraph("Rasch / Partial Credit Trial Item Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Dataset: {dataset_name}", styles["Normal"]))
    story.append(Paragraph(f"Model used: {model_type}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    n_keep = int((results["decision"] == "KEEP").sum())
    n_revise = int((results["decision"] == "REVISE").sum())
    n_remove = int((results["decision"] == "REMOVE / MAJOR REVISION").sum())

    summary_text = (
        f"This report screens trial items using {model_type}. "
        f"Items are classified as KEEP, REVISE, or REMOVE / MAJOR REVISION based on "
        f"difficulty, Infit, Outfit, and point-measure correlation. "
        f"Summary: KEEP={n_keep}, REVISE={n_revise}, "
        f"REMOVE / MAJOR REVISION={n_remove}."
    )

    story.append(Paragraph(summary_text, styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    table_cols = [
        "item",
        "max_score",
        "mean_score",
        "mean_score_pct",
        "difficulty_logit",
        "infit",
        "outfit",
        "point_measure_corr",
        "decision",
        "reason",
    ]

    table_df = results[table_cols].copy()

    for col in [
        "mean_score",
        "mean_score_pct",
        "difficulty_logit",
        "infit",
        "outfit",
        "point_measure_corr",
    ]:
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
        "KEEP: item is behaving well statistically. "
        "REVISE: item may be useful but should be reviewed. "
        "REMOVE / MAJOR REVISION: item has serious problems such as extreme difficulty, "
        "poor fit, outliers, or negative/weak relationship with the overall test. "
        "For polytomous items, mean_score_pct is the average item score divided by the maximum possible score. "
        "Always combine these statistics with content review."
    )
    story.append(Paragraph(guide, styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Streamlit interface
# -----------------------------

st.title("📊 Rasch / Partial Credit Trial Item Analyzer")

st.write(
    "Upload trial exam response data. The app automatically handles dichotomous "
    "items such as 0/1 and ordered polytomous items such as 0/1/2/3."
)

with st.expander("CSV format examples"):
    st.code(
        "person,item_1,item_2,item_3,item_4\n"
        "P1,1,1,0,0\n"
        "P2,1,0,0,0\n"
        "P3,1,1,1,0",
        language="csv",
    )

    st.code(
        "person,item_1,item_2,item_3,item_4\n"
        "P1,2,1,3,0\n"
        "P2,1,0,2,1\n"
        "P3,3,2,3,2",
        language="csv",
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

with st.sidebar:
    st.header("Decision thresholds")
    st.caption(
        "This version still uses the default beginner rules inside decide_item(). "
        "You can customize those thresholds directly in the function."
    )

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded data preview")
    st.dataframe(raw_df.head())

    item_cols = detect_item_columns(raw_df)

    if len(item_cols) < 2:
        st.error(
            "Could not detect enough item columns. "
            "Items must be coded as ordered integers starting at 0, such as 0/1 or 0/1/2/3."
        )
        st.stop()

    Y_df = prepare_response_matrix(raw_df, item_cols)

    if len(Y_df) < 5:
        st.warning("Very small sample size detected. Results may be unstable.")

    dichotomous = is_dichotomous(Y_df)

    if dichotomous:
        model_type = "Dichotomous Rasch 1PL model"
    else:
        model_type = "Polytomous Partial Credit Model"

    st.info(
        f"Detected {len(item_cols)} item columns and {len(Y_df)} complete student rows. "
        f"Model selected: {model_type}."
    )

    st.subheader("Detected item score ranges")
    range_df = pd.DataFrame({
        "item": item_cols,
        "min_score": Y_df.min(axis=0).values,
        "max_score": Y_df.max(axis=0).values,
        "observed_categories": [
            ", ".join(map(str, sorted(Y_df[col].unique())))
            for col in item_cols
        ],
    })
    st.dataframe(range_df, use_container_width=True)

    if st.button("Run Rasch Analysis", type="primary"):
        with st.spinner("Fitting model and checking item quality..."):

            if dichotomous:
                theta, difficulty, result = fit_rasch_jmle(Y_df)
                results = calculate_binary_fit(Y_df, theta, difficulty)
                steps = None
            else:
                theta, difficulty, steps, result = fit_pcm_jmle(Y_df)
                results = calculate_pcm_fit(Y_df, theta, difficulty, steps)

            decisions = results.apply(decide_item, axis=1)
            results["decision"] = decisions[0]
            results["reason"] = decisions[1]

            results = results.sort_values("difficulty_logit").reset_index(drop=True)

        if not result.success:
            st.warning(
                "The optimizer did not fully converge. "
                "Interpret results cautiously or try increasing sample size / checking extreme items."
            )

        st.success("Analysis complete.")

        c1, c2, c3 = st.columns(3)
        c1.metric("KEEP", int((results["decision"] == "KEEP").sum()))
        c2.metric("REVISE", int((results["decision"] == "REVISE").sum()))
        c3.metric(
            "REMOVE / MAJOR REVISION",
            int((results["decision"] == "REMOVE / MAJOR REVISION").sum()),
        )

        st.subheader("Item decision table")
        st.dataframe(results, use_container_width=True)

        st.subheader("Visual diagnostics")

        fig1 = make_distribution_plot(theta, difficulty)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = make_person_item_map(
            theta,
            difficulty,
            list(results.sort_values("difficulty_logit")["item"]),
        )
        st.pyplot(fig2)
        plt.close(fig2)

        if dichotomous:
            fig3 = make_binary_icc_plot(results)
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.subheader("Category probability curves")

            sorted_items = list(results.sort_values("difficulty_logit")["item"])

            selected_item = st.selectbox(
                "Choose a polytomous item to inspect",
                sorted_items,
            )

            item_index = list(Y_df.columns).index(selected_item)
            fig3 = make_pcm_category_plot(selected_item, steps[item_index])
            st.pyplot(fig3)
            plt.close(fig3)

            step_df = []
            for item, item_steps in zip(Y_df.columns, steps):
                for step_number, step_value in enumerate(item_steps, start=1):
                    step_df.append({
                        "item": item,
                        "step": step_number,
                        "step_difficulty_logit": step_value,
                    })

            st.subheader("Step difficulty table")
            st.dataframe(pd.DataFrame(step_df), use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")

        pdf_bytes = generate_pdf_report(
            results=results,
            theta=theta,
            difficulty=difficulty,
            item_names=list(results.sort_values("difficulty_logit")["item"]),
            model_type=model_type,
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
