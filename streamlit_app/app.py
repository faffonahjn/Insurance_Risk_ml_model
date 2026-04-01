"""
Medical Insurance Risk Classifier — Streamlit Dashboard
Tabs: Single Prediction | Batch Prediction | EDA | Model Info
Communicates with FastAPI backend at API_URL.
"""
import io
import os

import httpx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
sns.set_theme(style="whitegrid", palette="muted")

st.set_page_config(
    page_title="Insurance Risk Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=72)
    st.title("Insurance Risk\nClassifier")
    st.markdown("---")
    st.markdown("**Model:** XGBoost v2.0")
    st.markdown("**Target:** Charges > P75 ($16,658)")
    st.markdown("**Threshold:** 0.35 (clinical recall)")
    st.markdown("**Test AUC:** 0.899")
    st.markdown("---")

    # API health check
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        info = r.json()
        st.success(f"API: Online ✅")
        st.caption(f"Threshold: {info.get('decision_threshold', 0.35)}")
    except Exception:
        st.error("API: Offline ❌")
        st.caption(f"Expected at: {API_URL}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📋 Batch Prediction",
    "📊 EDA Dashboard",
    "ℹ️ Model Info",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Patient Risk Assessment")
    st.markdown("Enter patient details to predict high-cost insurance risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 100, 35)
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])

    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.5, step=0.1)
        children = st.slider("Number of Children", 0, 10, 1)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    with col3:
        bmi_age_interaction = round(bmi * age, 2)
        st.metric("BMI × Age Interaction", f"{bmi_age_interaction:.1f}")
        st.markdown("*(auto-calculated)*")

        # BMI category display
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25:
            bmi_cat = "Normal Weight"
        elif bmi < 30:
            bmi_cat = "Overweight"
        elif bmi < 35:
            bmi_cat = "Obese Class I"
        elif bmi < 40:
            bmi_cat = "Obese Class II"
        else:
            bmi_cat = "Obese Class III"
        st.info(f"BMI Category: **{bmi_cat}**")

    st.markdown("---")

    if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
        payload = {
            "age": age, "sex": sex, "bmi": bmi,
            "children": children, "smoker": smoker,
            "region": region, "bmi_age_interaction": bmi_age_interaction,
        }
        try:
            with st.spinner("Running inference..."):
                r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
                result = r.json()

            col_a, col_b, col_c = st.columns(3)
            prob = result["risk_probability"]
            label = result["risk_label"]
            latency = result["latency_ms"]

            with col_a:
                if result["is_high_risk"]:
                    st.error(f"### ⚠️ {label}")
                else:
                    st.success(f"### ✅ {label}")

            with col_b:
                st.metric("Risk Probability", f"{prob:.1%}")

            with col_c:
                st.metric("Inference Latency", f"{latency} ms")

            # Probability gauge
            st.markdown("#### Risk Probability")
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh(["Risk"], [prob], color="#DD8452" if prob >= 0.35 else "#4C72B0", height=0.5)
            ax.barh(["Risk"], [1 - prob], left=[prob], color="#e0e0e0", height=0.5)
            ax.axvline(0.35, color="red", linestyle="--", lw=1.5, label="Threshold (0.35)")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"Risk Score: {prob:.1%}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"API error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch Risk Assessment")
    st.markdown("Upload a CSV file with patient records to score in bulk.")

    st.markdown("**Required columns:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`")
    st.caption("`bmi_age_interaction` will be auto-calculated if not present.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df):,} records")
        st.dataframe(df.head(5), use_container_width=True)

        required = ["age", "sex", "bmi", "children", "smoker", "region"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if "bmi_age_interaction" not in df.columns:
                df["bmi_age_interaction"] = df["bmi"] * df["age"]

            if st.button("🚀 Score All Records", type="primary", use_container_width=True):
                records = df[required + ["bmi_age_interaction"]].to_dict(orient="records")

                if len(records) > 500:
                    st.warning("Batch limit is 500 records. Scoring first 500.")
                    records = records[:500]

                try:
                    with st.spinner(f"Scoring {len(records)} records..."):
                        r = httpx.post(
                            f"{API_URL}/predict/batch",
                            json=records, timeout=60
                        )
                        result = r.json()

                    preds = result["predictions"]
                    df_out = df.head(len(preds)).copy()
                    df_out["risk_probability"] = [p["risk_probability"] for p in preds]
                    df_out["is_high_risk"] = [p["is_high_risk"] for p in preds]
                    df_out["risk_label"] = [p["risk_label"] for p in preds]

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Scored", len(df_out))
                    col2.metric("High Risk", result["high_risk_count"])
                    col3.metric("High Risk Rate", f"{result['high_risk_count']/len(df_out):.1%}")

                    # Results table
                    st.markdown("#### Predictions")
                    st.dataframe(
                        df_out[["age", "sex", "bmi", "smoker", "region",
                                "risk_probability", "risk_label"]].style.apply(
                            lambda x: ["background-color: #ffe6e6" if v == "High Risk"
                                       else "background-color: #e6ffe6"
                                       for v in x], subset=["risk_label"]
                        ),
                        use_container_width=True,
                    )

                    # Risk distribution plot
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    df_out["risk_label"].value_counts().plot(
                        kind="bar", ax=axes[0],
                        color=["#DD8452", "#4C72B0"], edgecolor="white"
                    )
                    axes[0].set_title("Risk Distribution")
                    axes[0].tick_params(axis="x", rotation=0)

                    axes[1].hist(df_out["risk_probability"], bins=20,
                                 color="#4C72B0", edgecolor="white", alpha=0.85)
                    axes[1].axvline(0.35, color="red", linestyle="--", label="Threshold")
                    axes[1].set_xlabel("Risk Probability")
                    axes[1].set_title("Probability Distribution")
                    axes[1].legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Download
                    csv = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        csv, "predictions.csv", "text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"API error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Exploratory Data Analysis")

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("data/raw/medical_insurance.csv")
        except FileNotFoundError:
            return None

    df_eda = load_data()

    if df_eda is None:
        st.warning("Dataset not found at `data/raw/medical_insurance.csv`.")
    else:
        st.markdown(f"**Dataset:** {len(df_eda):,} records · {df_eda.shape[1]} features")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df_eda):,}")
        col2.metric("High Risk", f"{df_eda['is_high_risk'].sum():,}")
        col3.metric("High Risk Rate", f"{df_eda['is_high_risk'].mean():.1%}")
        col4.metric("Avg Charges", f"${df_eda['charges'].mean():,.0f}")

        st.markdown("---")

        # Row 1 — Target + Charges distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Target Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df_eda["is_high_risk"].value_counts().rename({0: "Low Risk", 1: "High Risk"}).plot(
                kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="white"
            )
            ax.set_title("High Risk vs Low Risk")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("#### Charges Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df_eda["charges"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.axvline(df_eda["charges"].quantile(0.75), color="red",
                       linestyle="--", label="P75 threshold ($16,658)")
            ax.set_xlabel("Annual Charges ($)")
            ax.set_title("Charges Distribution")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Row 2 — Risk by category
        st.markdown("#### High Risk Rate by Feature")
        cat_col = st.selectbox("Select feature", ["smoker", "region", "sex", "bmi_category", "age_group"])
        fig, ax = plt.subplots(figsize=(10, 4))
        rate = df_eda.groupby(cat_col)["is_high_risk"].mean().sort_values(ascending=False)
        rate.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="white")
        ax.set_ylabel("High Risk Rate")
        ax.set_title(f"High Risk Rate by {cat_col}")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Row 3 — BMI vs Charges scatter
        st.markdown("#### Charges vs BMI by Smoker Status")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = {"yes": "#DD8452", "no": "#4C72B0"}
        for s, g in df_eda.groupby("smoker"):
            ax.scatter(g["bmi"], g["charges"], alpha=0.4,
                       label=f"Smoker: {s}", color=colors[s], s=15)
        ax.axhline(16658, color="red", linestyle="--", label="P75 threshold")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Charges ($)")
        ax.set_title("Charges vs BMI")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Row 4 — Correlation heatmap
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        num_cols = ["age", "bmi", "children", "charges", "smoker_flag",
                    "bmi_age_interaction", "is_high_risk"]
        corr = df_eda[num_cols].corr()
        import numpy as np
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, square=True, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Model Architecture")
        st.markdown("""
        | Component | Detail |
        |---|---|
        | Algorithm | XGBoost Classifier |
        | Preprocessing | sklearn Pipeline |
        | Categorical | OneHotEncoder (drop=first) |
        | Numeric | StandardScaler |
        | Features | 7 (age, bmi, children, sex, smoker, region, bmi×age) |
        | Target | charges > P75 ($16,658) |
        """)

        st.markdown("#### Performance Metrics")
        st.markdown("""
        | Metric | Value |
        |---|---|
        | CV AUC (5-fold) | 0.879 ± 0.036 |
        | Test AUC | **0.899** |
        | Avg Precision | 0.872 |
        | Sensitivity | 76.1% |
        | Specificity | 95.5% |
        | Decision Threshold | **0.35** |
        """)

    with col2:
        st.markdown("#### Leakage Audit")
        st.markdown("""
        **Dropped features (leakage):**
        - `charges` — direct target source
        - `risk_score`, `insurance_tier`, `monthly_premium_est` — derived from charges
        - `bmi_category` — deterministic bin of bmi
        - `smoker_flag`, `sex_female`, `region_*` — duplicate encodings
        - `age_group` — bin of age

        **Target reconstruction:**
        Original `is_high_risk` was a 100% deterministic rule
        (`BMI ≥ 30 OR smoker = yes`) — perfect AUC was a red flag.
        Rebuilt as `charges > P75` — realistic AUC 0.899.
        """)

        st.markdown("#### Clinical Threshold Rationale")
        st.markdown("""
        At threshold **0.35**:
        - Recall: **77.6%** — catches 3 in 4 high-risk patients
        - Precision: **76.5%** — 3 in 4 flagged patients are truly high-risk
        - Specificity: **95.5%** — low false alarm rate

        Chosen over default 0.5 because in clinical risk stratification,
        **missing a high-risk patient is more costly than a false alarm.**
        """)

    st.markdown("---")
    st.markdown("#### Feature Importance")
    try:
        fi_img = plt.imread("artifacts/plots/feature_importance.png")
        st.image(fi_img, caption="XGBoost Feature Importance", use_container_width=True)
    except FileNotFoundError:
        st.info("Feature importance plot not found. Run training pipeline first.")
