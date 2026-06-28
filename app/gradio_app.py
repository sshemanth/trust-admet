import json
import gradio as gr

from trust_admet.trust.trust_engine import predict_with_trust


def predict(smiles, dataset, split, model, seed):
    try:
        result = predict_with_trust(
            smiles=smiles,
            dataset=dataset,
            split=split,
            model_name=model,
            seed=str(seed),
        )

        breakdown = result["score_breakdown"]

        report = f"""
# TRUST-ADMET Report

## Input Molecule
**SMILES:** `{result['smiles']}`

## Prediction
**Class:** {result['label']}  
**Probability BBB+:** {result['probability_positive']:.3f}  
**Prediction Confidence:** {result['prediction_confidence']:.3f}

## TRUST Score
**Total:** {result['trust_score']:.1f} / 100  
**Level:** {result['trust_level']}

## Breakdown
- Prediction Confidence: {breakdown['prediction_confidence']:.1f} / 25
- Calibration: {breakdown['calibration']:.1f} / 25
- Applicability Domain: {breakdown['applicability_domain']:.1f} / 25
- Uncertainty: {breakdown['uncertainty']:.1f} / 25

## Trust Signals
- Applicability Domain: {result['applicability_domain']}
- Nearest Similarity: {result['nearest_similarity']:.3f}
- Uncertainty: {result['uncertainty']:.3f}
- ECE: {result['ece']:.3f}

## Recommendation
**{result['recommendation']}**
"""

        return (
            result["label"],
            f"{result['probability_positive']:.3f}",
            f"{result['trust_score']:.1f} / 100",
            result["trust_level"],
            result["applicability_domain"],
            f"{result['nearest_similarity']:.3f}",
            report,
            json.dumps(result, indent=2),
        )

    except Exception as e:
        return "ERROR", "", "", "", "", "", f"Error: {e}", "{}"


with gr.Blocks(title="TRUST-ADMET") as demo:
    gr.Markdown(
        """
# 🧬 TRUST-ADMET

### Trustworthy AI Framework for ADMET Prediction

Enter a SMILES string and get a prediction with a TRUST Score, applicability-domain status, calibration signal, and recommendation.
"""
    )

    with gr.Row():
        smiles = gr.Textbox(label="SMILES", value="CCO")
        dataset = gr.Dropdown(["BBBP"], value="BBBP", label="Dataset")
        split = gr.Dropdown(["scaffold", "random"], value="scaffold", label="Split")
        model = gr.Dropdown(["random_forest", "xgboost"], value="random_forest", label="Model")
        seed = gr.Textbox(label="Seed", value="42")

    predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        out_label = gr.Textbox(label="Prediction")
        out_prob = gr.Textbox(label="Probability BBB+")
        out_score = gr.Textbox(label="TRUST Score")
        out_level = gr.Textbox(label="Trust Level")
        out_ad = gr.Textbox(label="Applicability Domain")
        out_sim = gr.Textbox(label="Nearest Similarity")

    report = gr.Markdown(label="TRUST Report")

    with gr.Accordion("Raw JSON", open=False):
        raw_json = gr.Code(label="JSON", language="json")

    predict_btn.click(
        fn=predict,
        inputs=[smiles, dataset, split, model, seed],
        outputs=[
            out_label,
            out_prob,
            out_score,
            out_level,
            out_ad,
            out_sim,
            report,
            raw_json,
        ],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

