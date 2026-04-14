
# ============================================================
# app.py — Student At-Risk Predictor
# Author: Adewale Samson Adeagbo
# Purpose: Help teachers identify at-risk students early
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Student At-Risk Predictor",
    page_icon  = "🎓",
    layout     = "wide"
)

# ── MODEL LOADING ─────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """
    Load model from disk if it exists.
    If not, train it fresh from scratch.
    @st.cache_resource means this only runs ONCE per session —
    not every time a user interacts with the app.
    """
    model_path   = "model/best_rf_model.pkl"
    scaler_path  = "model/scaler.pkl"
    imputer_path = "model/imputer.pkl"

    if (os.path.exists(model_path) and
        os.path.exists(scaler_path) and
        os.path.exists(imputer_path)):
        model   = joblib.load(model_path)
        scaler  = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
    else:
        st.info("Setting up model for first time. Please wait 20-30 seconds...")
        from train_model import train_and_save
        model, scaler, imputer = train_and_save()

    return model, scaler, imputer

model, scaler, imputer = load_or_train_model()

# Feature column order must match training exactly
FEATURE_COLS = [
    "class_level", "maths_score", "english_score",
    "science_score", "social_studies", "attendance_rate",
    "assignment_rate", "prev_term_avg"
]

LEVEL_ORDER = {"JSS1":0,"JSS2":1,"JSS3":2,"SSS1":3,"SSS2":4,"SSS3":5}

# ── HELPER FUNCTIONS ──────────────────────────────────────────
def preprocess_input(df_input):
    """Apply the same preprocessing steps used during training."""
    df = df_input.copy()
    df["class_level"] = df["class_level"].map(LEVEL_ORDER)
    df = pd.DataFrame(
        imputer.transform(df[FEATURE_COLS]),
        columns=FEATURE_COLS
    )
    df = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_COLS
    )
    return df


def predict_student(df_scaled):
    """Return label, probability, and SHAP values."""
    prob        = model.predict_proba(df_scaled)[0][1]
    label       = "🔴 At Risk" if prob >= 0.5 else "🟢 Not At Risk"
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scaled)
    # shap_values[1] = contributions toward at-risk class
    return label, prob, shap_values[1][0]


def shap_bar_chart(shap_vals, feature_names):
    """Render a clean horizontal SHAP bar chart."""
    shap_df = pd.DataFrame({
        "Feature"    : feature_names,
        "SHAP Value" : shap_vals
    }).sort_values("SHAP Value")

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Value"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(shap_df["Feature"], shap_df["SHAP Value"],
                   color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title("What drove this prediction?", fontsize=13, fontweight="bold")
    ax.set_xlabel("SHAP Value  (red = increases risk | green = reduces risk)")

    for bar, val in zip(bars, shap_df["SHAP Value"]):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9
        )
    plt.tight_layout()
    return fig


def plain_english_explanation(shap_vals, feature_names, raw_input):
    """
    Convert SHAP values into a sentence a teacher can read.
    Shows the top 2 risk factors and top 1 protective factor.
    """
    shap_df = pd.DataFrame({
        "feature" : feature_names,
        "shap"    : shap_vals
    })

    risk_factors = shap_df[shap_df["shap"] > 0].sort_values(
        "shap", ascending=False
    )
    protective   = shap_df[shap_df["shap"] < 0].sort_values("shap")

    # Friendly feature name mapping
    name_map = {
        "maths_score"    : "Maths score",
        "english_score"  : "English score",
        "science_score"  : "Science score",
        "social_studies" : "Social Studies score",
        "attendance_rate": "attendance rate",
        "assignment_rate": "assignment submission rate",
        "prev_term_avg"  : "previous term average",
        "class_level"    : "class level"
    }

    lines = []

    if not risk_factors.empty:
        top_risks = [name_map.get(f, f) for f in risk_factors["feature"].head(2)]
        lines.append(f"**Main risk factors:** {' and '.join(top_risks)}")

    if not protective.empty:
        top_protect = name_map.get(protective.iloc[0]["feature"], "")
        if top_protect:
            lines.append(f"**Protective factor:** {top_protect}")

    return "  
".join(lines)

# ── UI LAYOUT ─────────────────────────────────────────────────
st.title("🎓 Student At-Risk Predictor")
st.markdown(
    "Predict which students are likely to fail **before the term ends** "
    "so you can intervene early."
)
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Single Student", "📂 Upload CSV"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Single Student Input
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Student Information**")
        student_name = st.text_input("Student Name (optional)", placeholder="e.g. Chukwuemeka Obi")
        class_level  = st.selectbox("Class Level", ["JSS1","JSS2","JSS3","SSS1","SSS2","SSS3"])

        st.markdown("**Behavioural Data**")
        attendance_rate = st.slider("Attendance Rate (%)", 40, 100, 75)
        assignment_rate = st.slider("Assignment Submission Rate (%)", 30, 100, 70)
        prev_term_avg   = st.slider(
            "Previous Term Average (leave at 0 if unknown)", 0, 100, 0
        )

    with col2:
        st.markdown("**Subject Scores (Current Term)**")
        maths_score    = st.slider("Mathematics", 20, 100, 55)
        english_score  = st.slider("English Language", 20, 100, 58)
        science_score  = st.slider("Science", 20, 100, 52)
        social_studies = st.slider("Social Studies", 20, 100, 60)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    if predict_btn:
        # Handle unknown previous term average
        prev_avg_val = np.nan if prev_term_avg == 0 else float(prev_term_avg)

        # Build input dataframe
        input_df = pd.DataFrame([{
            "class_level"    : class_level,
            "maths_score"    : float(maths_score),
            "english_score"  : float(english_score),
            "science_score"  : float(science_score),
            "social_studies" : float(social_studies),
            "attendance_rate": float(attendance_rate),
            "assignment_rate": float(assignment_rate),
            "prev_term_avg"  : prev_avg_val
        }])

        # Preprocess and predict
        scaled_input        = preprocess_input(input_df)
        label, prob, shap_v = predict_student(scaled_input)

        # ── Display Results ───────────────────────────────────
        name_display = student_name if student_name else "This student"
        st.markdown("---")
        st.subheader(f"Prediction for {name_display}")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prob >= 0.5:
                st.error(f"### {label}")
                st.metric(
                    label = "Probability of Failing",
                    value = f"{prob*100:.1f}%"
                )
                st.warning(
                    "⚠️ This student is flagged for early intervention. "
                    "Consider scheduling a one-on-one meeting."
                )
            else:
                st.success(f"### {label}")
                st.metric(
                    label = "Probability of Failing",
                    value = f"{prob*100:.1f}%"
                )
                st.info("✅ This student is currently on track.")

        with res_col2:
            st.markdown("**Why this prediction?**")
            explanation = plain_english_explanation(
                shap_v, FEATURE_COLS, input_df
            )
            st.markdown(explanation)

        # SHAP chart below the columns
        st.markdown("**Factor breakdown:**")
        fig = shap_bar_chart(shap_v, FEATURE_COLS)
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — CSV Upload
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload a CSV File")
    st.markdown(
        "Upload a CSV with these columns: "
        "`class_level, maths_score, english_score, science_score, "
        "social_studies, attendance_rate, assignment_rate, prev_term_avg`"
    )
    st.markdown(
        "Set `prev_term_avg` to `0` for students without a previous record. "
        "The app will treat 0 as missing."
    )

    # Download template button
    template_df = pd.DataFrame([{
        "student_name"   : "Chukwuemeka Obi",
        "class_level"    : "JSS2",
        "maths_score"    : 42,
        "english_score"  : 55,
        "science_score"  : 38,
        "social_studies" : 61,
        "attendance_rate": 58,
        "assignment_rate": 45,
        "prev_term_avg"  : 47
    },{
        "student_name"   : "Amara Nwosu",
        "class_level"    : "SSS1",
        "maths_score"    : 71,
        "english_score"  : 68,
        "science_score"  : 65,
        "social_studies" : 74,
        "attendance_rate": 88,
        "assignment_rate": 82,
        "prev_term_avg"  : 70
    }])

    st.download_button(
        label     = "📥 Download CSV Template",
        data      = template_df.to_csv(index=False),
        file_name = "student_template.csv",
        mime      = "text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**{len(df_upload)} students loaded.**")
        st.dataframe(df_upload, use_container_width=True)

        if st.button("🔍 Predict for All Students", type="primary"):
            results = []

            # Handle prev_term_avg = 0 as missing
            df_upload["prev_term_avg"] = df_upload["prev_term_avg"].replace(0, np.nan)

            for _, row in df_upload.iterrows():
                input_row = pd.DataFrame([{
                    "class_level"    : row["class_level"],
                    "maths_score"    : row["maths_score"],
                    "english_score"  : row["english_score"],
                    "science_score"  : row["science_score"],
                    "social_studies" : row["social_studies"],
                    "attendance_rate": row["attendance_rate"],
                    "assignment_rate": row["assignment_rate"],
                    "prev_term_avg"  : row.get("prev_term_avg", np.nan)
                }])

                scaled = preprocess_input(input_row)
                prob   = model.predict_proba(scaled)[0][1]
                label  = "🔴 At Risk" if prob >= 0.5 else "🟢 Not At Risk"

                results.append({
                    "Student"        : row.get("student_name", f"Student {_+1}"),
                    "Class"          : row["class_level"],
                    "Prediction"     : label,
                    "Risk Probability": f"{prob*100:.1f}%",
                    "Maths"          : row["maths_score"],
                    "Attendance"     : row["attendance_rate"],
                    "Assignments"    : row["assignment_rate"]
                })

            results_df = pd.DataFrame(results)

            # Sort: at-risk students first, highest probability first
            results_df["_prob_num"] = results_df["Risk Probability"].str.replace("%","").astype(float)
            results_df = results_df.sort_values("_prob_num", ascending=False).drop(columns=["_prob_num"])

            st.markdown("---")
            st.subheader("Results — sorted by risk level")
            st.dataframe(results_df, use_container_width=True)

            # Download results
            st.download_button(
                label     = "📥 Download Results CSV",
                data      = results_df.to_csv(index=False),
                file_name = "at_risk_predictions.csv",
                mime      = "text/csv"
            )

            # Summary metrics
            n_at_risk = results_df["Prediction"].str.contains("At Risk").sum()
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Students", len(results_df))
            m2.metric("At Risk", n_at_risk)
            m3.metric("Not At Risk", len(results_df) - n_at_risk)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by Adewale Samson Adeagbo | "
    "Random Forest Classifier | "
    "Trained on synthetic Nigerian secondary school data"
)
