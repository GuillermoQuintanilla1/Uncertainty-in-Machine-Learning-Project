import torch
import torch.nn as nn

def enable_dropout(model):

    for m in model.modules():

        if isinstance(
            m,
            nn.Dropout
        ):
            m.train()

def mc_dropout_predict(
        model,
        images,
        T=10):

    model.eval()

    enable_dropout(model)

    predictions = []

    with torch.no_grad():

        for _ in range(T):

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1
            )

            predictions.append(probs)

    predictions = torch.stack(
        predictions
    )

    return predictions

def compute_uncertainties(
        predictions):

    mean_probs = predictions.mean(0)

    epistemic = (
        predictions.var(0)
        .mean(1)
    )

    aleatoric = (
        predictions *
        (1 - predictions)
    ).mean(0).mean(1)

    entropy = (
        -(mean_probs *
          torch.log(mean_probs + 1e-8))
        .sum(1)
    )

    return (
        mean_probs,
        epistemic,
        aleatoric,
        entropy
    )