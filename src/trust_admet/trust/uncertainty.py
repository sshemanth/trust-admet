import numpy as np
import torch


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict_mlp(model, loader, device, n_samples=30):
    model.eval()
    enable_dropout(model)

    all_samples = []

    for _ in range(n_samples):
        preds = []

        for x, _ in loader:
            x = x.to(device)
            pred = model.predict(x)
            preds.append(pred.detach().cpu().numpy())

        all_samples.append(np.concatenate(preds))

    all_samples = np.stack(all_samples, axis=0)

    return {
        "mean": all_samples.mean(axis=0),
        "variance": all_samples.var(axis=0),
        "std": all_samples.std(axis=0),
    }
