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
    page_title="Rasch / Partial Credit / Multidimensional Analyzer",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Data detection and preparation
# -----------------------------

def detect_item_columns(df: pd.DataFrame):
    item_cols = []

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()

        if valid.empty:
            continue

        values = sorted(valid.unique())

        if not np.allclose(values, np.round(values)):
            continue

        values = [int(v) for v in values]

        if values[0] != 0:
            continue

        if values != list(range(values[-1] + 1)):
            continue

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
# Partial Credit Model
# -----------------------------

def fit_pcm_jmle(Y_df: pd.DataFrame, ridge: float = 0.05, maxiter: int = 1500):
    Y = Y_df.values.astype(int)
    N, I = Y.shape

    max_scores = Y_df.max(axis=0).astype(int).values
    step_counts = max_scores
    offsets = np.cumsum([0] + list(step_counts))
    total_steps = int(sum(step_counts))

    def unpack(params):
        theta = params[:N]
        raw_steps = params[N:N + total_steps]

        if total_steps > 0:
            raw_steps = raw_steps - np.mean(raw_steps)

        steps = []
        for j in range(I):
            start, end = offsets[j], offsets[j + 1]
            steps.append(raw_steps[start:end])

        return theta, steps

    def item_log_probs(theta_n, item_steps):
        m = len(item_steps)

        exponents = [0.0]
        cumulative = 0.0

        for k in range(1, m + 1):
            cumulative += theta_n - item_steps[k - 1]
            exponents.append(cumulative)

        exponents = np.array(exponents)
        return exponents - logsumexp(exponents)

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
# Single-dimension analysis
# -----------------------------

def run_single_analysis(Y_df):
    if is_dichotomous(Y_df):
        model_type = "Dichotomous Rasch 1PL model"
        theta, difficulty, result = fit_rasch_jmle(Y_df)
        results = calculate_binary_fit(Y_df, theta, difficulty)
        steps = None
    else:
        model_type = "Polytomous Partial Credit Model"
        theta, difficulty, steps, result = fit_pcm_jmle(Y_df)
        results = calculate_pcm_fit(Y_df, theta, difficulty, steps)

    decisions = results.apply(decide_item, axis=1)
    results["decision"] = decisions[0]
    results["reason"] = decisions[1]

    results = results.sort_values("difficulty_logit").reset_index(drop=True)

    return {
        "model_type": model_type,
        "theta": theta,
        "difficulty": difficulty,
        "steps": steps,
        "result": result,
        "results": results,
    }


# -----------------------------
# Multidimensional analysis
# -----------------------------

def run_multidimensional_analysis(Y_df, dimension_map_df):
    required_cols = {"item", "dimension"}

    if not required_cols.issubset(dimension_map_df.columns):
        raise ValueError("Dimension map must contain columns: item, dimension")

    dimension_map_df = dimension_map_df.copy()
    dimension_map_df["item"] = dimension_map_df["item"].astype(str)
    dimension_map_df["dimension"] = dimension_map_df["dimension"].astype(str)

    all_item_results = []
    person_abilities = pd.DataFrame(index=Y_df.index)
    dimension_summary = []

    for dimension in sorted(dimension_map_df["dimension"].unique()):
        dim_items = dimension_map_df.loc[
            dimension_map_df["dimension"] == dimension,
            "item"
        ].tolist()

        dim_items = [item for item in dim_items if item in Y_df.columns]

        if len(dim_items) < 2:
            dimension_summary.append({
                "dimension": dimension,
                "number_of_items": len(dim_items),
                "model_used": "Not fitted",
                "status": "Skipped: needs at least 2 valid items",
            })
            continue

        Y_dim = Y_df[dim_items]

        analysis = run_single_analysis(Y_dim)

        dim_results = analysis["results"].copy()
        dim_results["dimension"] = dimension
        dim_results["model_used"] = analysis["model_type"]
        dim_results["converged"] = analysis["result"].success

        all_item_results.append(dim_results)

        person_abilities[f"{dimension}_ability_logit"] = analysis["theta"]

        dimension_summary.append({
            "dimension": dimension,
            "number_of_items": len(dim_items),
            "model_used": analysis["model_type"],
            "status": "Converged" if analysis["result"].success else "Not fully converged",
        })

    if len(all_item_results) == 0:
        return None, person_abilities, pd.DataFrame(dimension_summary)

    item_results = pd.concat(all_item_results, ignore_index=True)
    summary_df = pd.DataFrame(dimension_summary)

    return item_results, person_abilities, summary_df


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
    """
    Improved person-item map with non-overlapping item labels.
    Labels are automatically spread vertically when items have similar logits.
    """

    fig_height = max(7, 0.35 * len(item_names))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    rng = np.random.default_rng(123)
    x_person = rng.normal(0, 0.035, len(theta))

    ax.scatter(
        x_person,
        theta,
        alpha=0.25,
        s=14,
        label="Persons"
    )

    ax.scatter(
        np.ones(len(difficulty)),
        difficulty,
        marker="D",
        s=65,
        label="Items"
    )

    # Sort items by difficulty
    item_data = sorted(
        zip(difficulty, item_names),
        key=lambda x: x[0]
    )

    sorted_difficulty = np.array([x[0] for x in item_data])
    sorted_names = [x[1] for x in item_data]

    # Create adjusted label positions to avoid overlap
    min_gap = 0.18
    adjusted_y = sorted_difficulty.copy()

    for i in range(1, len(adjusted_y)):
        if adjusted_y[i] - adjusted_y[i - 1] < min_gap:
            adjusted_y[i] = adjusted_y[i - 1] + min_gap

    # Draw labels with connector lines
    for original_y, label_y, label in zip(sorted_difficulty, adjusted_y, sorted_names):
        ax.plot(
            [1.02, 1.12],


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


def make_dimension_ability_boxplot(person_abilities):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    person_abilities.boxplot(ax=ax)
    ax.set_title("Ability Distribution by Dimension")
    ax.set_ylabel("Ability logit")
    ax.tick_params(axis="x", rotation=45)
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
# Streamlit Interface
# -----------------------------

st.title("📊 Rasch / Partial Credit / Multidimensional Trial Item Analyzer")

st.write(
    "Upload trial exam response data. The app supports dichotomous items, "
    "polytomous partial-credit items, and practical multidimensional analysis "
    "by fitting separate Rasch/PCM models for each dimension."
)

with st.expander("CSV format examples"):
    st.write("Main response data example:")
    st.code(
        "person,item_1,item_2,item_3,item_4,item_5,item_6\n"
        "P1,1,1,0,0,2,3\n"
        "P2,1,0,0,0,1,2\n"
        "P3,1,1,1,0,3,3",
        language="csv",
    )

    st.write("Optional dimension map example:")
    st.code(
        "item,dimension\n"
        "item_1,AO1\n"
        "item_2,AO1\n"
        "item_3,AO2\n"
        "item_4,AO2\n"
        "item_5,AO3\n"
        "item_6,AO3",
        language="csv",
    )

uploaded_file = st.file_uploader("Upload main response CSV file", type=["csv"])

dimension_file = st.file_uploader(
    "Optional: Upload dimension map CSV",
    type=["csv"],
    help="CSV must contain two columns: item and dimension."
)

with st.sidebar:
    st.header("About this app")
    st.info(
        "This is a Rasch-style screening tool using ridge-regularized JMLE. "
        "For official high-stakes calibration, compare results with Winsteps, ConQuest, TAM, or mirt."
    )

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    detected_item_cols = detect_item_columns(raw_df)

    if len(detected_item_cols) < 2:
        st.error(
            "Could not detect enough item columns. "
            "Items must be coded as ordered integers starting at 0, such as 0/1 or 0/1/2/3."
        )
        st.stop()

    st.subheader("Item Column Selection")

    selected_item_cols = st.multiselect(
        "Select item columns to include in the Rasch analysis",
        options=list(raw_df.columns),
        default=detected_item_cols,
    )

    if len(selected_item_cols) < 2:
        st.error("Please select at least 2 item columns.")
        st.stop()

    Y_df = prepare_response_matrix(raw_df, selected_item_cols)

    if len(Y_df) < 5:
        st.warning("Very small sample size detected. Results may be unstable.")

    st.info(
        f"Using {len(selected_item_cols)} item columns and {len(Y_df)} complete student rows."
    )

    st.subheader("Detected Item Score Ranges")
    range_df = pd.DataFrame({
        "item": selected_item_cols,
        "min_score": Y_df.min(axis=0).values,
        "max_score": Y_df.max(axis=0).values,
        "observed_categories": [
            ", ".join(map(str, sorted(Y_df[col].unique())))
            for col in selected_item_cols
        ],
    })
    st.dataframe(range_df, use_container_width=True)

    dimension_map_df = None
    use_multidimensional = False

    if dimension_file is not None:
        dimension_map_df = pd.read_csv(dimension_file)

        st.subheader("Dimension Map Preview")
        st.dataframe(dimension_map_df, use_container_width=True)

        if {"item", "dimension"}.issubset(dimension_map_df.columns):
            matched_items = [
                item for item in dimension_map_df["item"].astype(str).tolist()
                if item in selected_item_cols
            ]

            st.success(
                f"Dimension map detected. {len(matched_items)} mapped items match selected item columns."
            )

            use_multidimensional = st.checkbox(
                "Run multidimensional analysis by dimension",
                value=True
            )
        else:
            st.error("Dimension map must contain columns: item and dimension.")

    if st.button("Run Rasch Analysis", type="primary"):
        with st.spinner("Fitting model and checking item quality..."):
            analysis = run_single_analysis(Y_df)

            st.session_state["analysis"] = analysis
            st.session_state["Y_df"] = Y_df
            st.session_state["uploaded_name"] = uploaded_file.name

            if use_multidimensional and dimension_map_df is not None:
                md_item_results, md_person_abilities, md_summary = run_multidimensional_analysis(
                    Y_df,
                    dimension_map_df
                )

                st.session_state["md_item_results"] = md_item_results
                st.session_state["md_person_abilities"] = md_person_abilities
                st.session_state["md_summary"] = md_summary
            else:
                st.session_state["md_item_results"] = None
                st.session_state["md_person_abilities"] = None
                st.session_state["md_summary"] = None

    if "analysis" in st.session_state:
        analysis = st.session_state["analysis"]
        results = analysis["results"]
        theta = analysis["theta"]
        difficulty = analysis["difficulty"]
        steps = analysis["steps"]
        result = analysis["result"]
        model_type = analysis["model_type"]

        if not result.success:
            st.warning(
                "The optimizer did not fully converge. "
                "Interpret results cautiously or check sample size and extreme items."
            )

        st.success("Analysis complete.")

        c1, c2, c3 = st.columns(3)
        c1.metric("KEEP", int((results["decision"] == "KEEP").sum()))
        c2.metric("REVISE", int((results["decision"] == "REVISE").sum()))
        c3.metric(
            "REMOVE / MAJOR REVISION",
            int((results["decision"] == "REMOVE / MAJOR REVISION").sum()),
        )

        st.subheader("Overall Item Decision Table")
        st.dataframe(results, use_container_width=True)

        st.subheader("Overall Visual Diagnostics")

        sorted_results = results.sort_values("difficulty_logit")

        fig1 = make_distribution_plot(theta, sorted_results["difficulty_logit"].values)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = make_person_item_map(
            theta,
            sorted_results["difficulty_logit"].values,
            list(sorted_results["item"]),
        )
        st.pyplot(fig2)
        plt.close(fig2)

        if is_dichotomous(st.session_state["Y_df"]):
            fig3 = make_binary_icc_plot(results)
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.subheader("Category Probability Curves")

            sorted_items = list(results.sort_values("difficulty_logit")["item"])

            selected_item = st.selectbox(
                "Choose a polytomous item to inspect",
                sorted_items,
            )

            item_index = list(st.session_state["Y_df"].columns).index(selected_item)
            fig3 = make_pcm_category_plot(selected_item, steps[item_index])
            st.pyplot(fig3)
            plt.close(fig3)

            step_df = []
            for item, item_steps in zip(st.session_state["Y_df"].columns, steps):
                for step_number, step_value in enumerate(item_steps, start=1):
                    step_df.append({
                        "item": item,
                        "step": step_number,
                        "step_difficulty_logit": step_value,
                    })

            st.subheader("Step Difficulty Table")
            st.dataframe(pd.DataFrame(step_df), use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")

        pdf_bytes = generate_pdf_report(
            results=results,
            theta=theta,
            difficulty=sorted_results["difficulty_logit"].values,
            item_names=list(sorted_results["item"]),
            model_type=model_type,
            dataset_name=st.session_state["uploaded_name"],
        )

        st.download_button(
            "Download overall item decision CSV",
            data=csv_bytes,
            file_name="rasch_item_decision_results.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download overall PDF report",
            data=pdf_bytes,
            file_name="rasch_trial_item_report.pdf",
            mime="application/pdf",
        )

        md_item_results = st.session_state.get("md_item_results")
        md_person_abilities = st.session_state.get("md_person_abilities")
        md_summary = st.session_state.get("md_summary")

        if md_summary is not None:
            st.divider()
            st.subheader("Multidimensional Rasch / PCM Results")

            st.write("### Dimension Summary")
            st.dataframe(md_summary, use_container_width=True)

            if md_item_results is None:
                st.warning(
                    "No multidimensional model was fitted. "
                    "Each dimension needs at least 2 valid item columns."
                )
            else:
                st.write("### Item Results by Dimension")
                st.dataframe(md_item_results, use_container_width=True)

                st.write("### Person Ability by Dimension")
                st.dataframe(md_person_abilities, use_container_width=True)

                if md_person_abilities.shape[1] >= 2:
                    fig_md = make_dimension_ability_boxplot(md_person_abilities)
                    st.pyplot(fig_md)
                    plt.close(fig_md)

                md_item_csv = md_item_results.to_csv(index=False).encode("utf-8")
                md_person_csv = md_person_abilities.to_csv(index=True).encode("utf-8")
                md_summary_csv = md_summary.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download multidimensional item results CSV",
                    data=md_item_csv,
                    file_name="multidimensional_item_results.csv",
                    mime="text/csv",
                )

                st.download_button(
                    "Download multidimensional person abilities CSV",
                    data=md_person_csv,
                    file_name="multidimensional_person_abilities.csv",
                    mime="text/csv",
                )

                st.download_button(
                    "Download multidimensional summary CSV",
                    data=md_summary_csv,
                    file_name="multidimensional_summary.csv",
                    mime="text/csv",
                )

else:
    st.warning("Upload a CSV file to begin.")
