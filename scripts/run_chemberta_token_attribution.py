from pathlib import Path
import argparse

import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path("outputs/models") / args.dataset / args.split / "chemberta" / f"seed{args.seed}" / "best_model"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    test_df = pd.read_csv(Path("data/splits") / args.dataset / args.split / "test.csv")
    smiles = str(test_df.iloc[args.index]["canonical_smiles"])
    label = test_df.iloc[args.index]["Y"]

    enc = tokenizer(
        smiles,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1]

    lig = LayerIntegratedGradients(
        forward_func,
        model.roberta.embeddings,
    )

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True,
    )

    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu()
    token_attr = token_attr / (torch.norm(token_attr) + 1e-8)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu())

    rows = []
    for tok, score, mask in zip(tokens, token_attr.tolist(), attention_mask.squeeze(0).detach().cpu().tolist()):
        if mask == 0:
            continue
        rows.append({
            "token": tok,
            "attribution": score,
        })

    out_dir = Path("outputs/reports/explainability") / args.dataset / args.split / "chemberta"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"seed{args.seed}_idx{args.index}_token_attributions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    paper_dir = Path("paper/tables")
    paper_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        paper_dir / "table_chemberta_token_attributions_example.csv",
        index=False,
    )

    print("SMILES:", smiles)
    print("Label:", label)
    print("Saved:", out_path)
    print(pd.DataFrame(rows).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
