import json
import gradio as gr

from trust_admet.trust.trust_engine import predict_with_trust


def fmt(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def render_refused(result):
    report = f"""
# ❌ Prediction Refused

## Input Molecule
**SMILES:** `{result['smiles']}`

## Reason
{result['recommendation']}

## Applicability Domain
**Status:** {result['applicability_domain']}

## Violations
"""

    violations = result.get("physchem_ad", {}).get("violations", [])

    if violations:
        for v in violations:
            report += (
                f"\n- **{v['descriptor']}**: "
                f"{v['value']:.2f} "
                f"(allowed {v['min']:.2f} to {v['max']:.2f})"
            )
    else:
        report += "\nNo detailed violations available."

    return (
        "REFUSED",
        "N/A",
        "0 / 100",
        "REFUSED",
        result.get("applicability_domain", "Outside"),
        "N/A",
        report,
        json.dumps(result, indent=2),
    )


def render_accepted(result):
    breakdown = result["score_breakdown"]
    ensemble = result.get("ensemble", {})

    report = f"""
# TRUST-ADMET Report

## Input Molecule
**SMILES:** `{result['smiles']}`

## Prediction
**Class:** {result['label']}  
**Probability BBB+:** {fmt(result['probability_positive'])}  
**Prediction Confidence:** {fmt(result['prediction_confidence'])}  
**Conformal Set:** {result.get('conformal_prediction_set', 'N/A')}

## Ensemble
**Mean Probability BBB+:** {fmt(ensemble.get('ensemble_probability_positive'))}  
**Disagreement Std:** {fmt(ensemble.get('ensemble_std'))}  
**Model Agreement:** {fmt(ensemble.get('ensemble_agreement'), 2)}

## TRUST Score
**Total:** {fmt(result['trust_score'], 1)} / 100  
**Level:** {result['trust_level']}

## Breakdown
- Prediction Confidence: {fmt(breakdown['prediction_confidence'], 1)} / 25
- Calibration: {fmt(breakdown['calibration'], 1)} / 25
- Applicability Domain: {fmt(breakdown['applicability_domain'], 1)} / 25
- Uncertainty: {fmt(breakdown['uncertainty'], 1)} / 25

## Trust Signals
- Applicability Domain: {result['applicability_domain']}
- Nearest Similarity: {fmt(result['nearest_similarity'])}
- Uncertainty: {fmt(result['uncertainty'])}
- ECE: {fmt(result['ece'])}
- Conformal qhat: {fmt(result.get('conformal_qhat'))}

## Recommendation
**{result['recommendation']}**
"""

    return (
        result["label"],
        fmt(result["probability_positive"]),
        f"{fmt(result['trust_score'], 1)} / 100",
        result["trust_level"],
        result["applicability_domain"],
        fmt(result["nearest_similarity"]),
        report,
        json.dumps(result, indent=2),
    )


def predict(smiles, dataset, split, model, seed):
    try:
        result = predict_with_trust(
            smiles=smiles,
            dataset=dataset,
            split=split,
            model_name=model,
            seed=str(seed),
        )

        if result["prediction"] is None:
            return render_refused(result)

        return render_accepted(result)

    except Exception as e:
        return (
            "ERROR",
            "N/A",
            "N/A",
            "ERROR",
            "N/A",
            "N/A",
            f"# Error\n\n{str(e)}",
            "{}",
        )


with gr.Blocks(title="TRUST-ADMET") as demo:
    gr.Markdown(
        """
# 🧬 TRUST-ADMET

### Trustworthy AI Framework for ADMET Prediction

Enter a SMILES string and get a prediction with TRUST decision logic, conformal prediction, ensemble agreement, applicability-domain checks, and calibration-aware reporting.
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
