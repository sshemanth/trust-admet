import json
import streamlit as st

from trust_admet.trust.trust_engine import predict_with_trust


st.set_page_config(
    page_title="TRUST-ADMET",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 TRUST-ADMET")
st.subheader("Trustworthy AI Framework for ADMET Prediction")

st.markdown(
    """
TRUST-ADMET combines prediction, calibration, applicability domain,
uncertainty, and explainability into a single trust-aware report.
"""
)

with st.sidebar:
    st.header("Model Settings")
    dataset = st.selectbox("Dataset", ["BBBP"])
    split = st.selectbox("Split", ["scaffold", "random"])
    model = st.selectbox("Model", ["random_forest", "xgboost"])
    seed = st.text_input("Seed", "42")

st.header("Single Molecule Prediction")

smiles = st.text_input("Enter SMILES", value="CCO")

if st.button("Predict"):
    try:
        result = predict_with_trust(
            smiles=smiles,
            dataset=dataset,
            split=split,
            model_name=model,
            seed=seed,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", result["label"])
            st.metric("Probability BBB+", f"{result['probability_positive']:.3f}")

        with col2:
            st.metric("TRUST Score", f"{result['trust_score']:.1f}/100")
            st.metric("Trust Level", result["trust_level"])

        with col3:
            st.metric("Applicability Domain", result["applicability_domain"])
            st.metric("Nearest Similarity", f"{result['nearest_similarity']:.3f}")

        st.divider()

        st.subheader("TRUST Score Breakdown")

        breakdown = result["score_breakdown"]

        st.progress(min(result["trust_score"] / 100, 1.0))

        st.write(f"Prediction Confidence: **{breakdown['prediction_confidence']:.1f}/25**")
        st.write(f"Calibration: **{breakdown['calibration']:.1f}/25**")
        st.write(f"Applicability Domain: **{breakdown['applicability_domain']:.1f}/25**")
        st.write(f"Uncertainty: **{breakdown['uncertainty']:.1f}/25**")

        st.divider()

        st.subheader("Recommendation")
        st.success(result["recommendation"])

        with st.expander("Raw JSON"):
            st.json(result)

    except Exception as e:
        st.error(str(e))
