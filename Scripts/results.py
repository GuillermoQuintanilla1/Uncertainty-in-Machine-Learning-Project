import pandas as pd
import torch

from mcdropout import (
    mc_dropout_predict,
    compute_uncertainties
)

def create_results_dataframe(
        model,
        test_loader,
        device,
        T=10):

    model.eval()

    rows = []

    for images, labels, image_ids in test_loader:

        images = images.to(device)

        predictions = mc_dropout_predict(
            model,
            images,
            T=T
        )

        (
            mean_probs,
            epistemic,
            aleatoric,
            entropy
        ) = compute_uncertainties(
            predictions
        )

        predicted_labels = mean_probs.argmax(
            dim=1
        )

        confidence = mean_probs.max(
            dim=1
        )[0]

        for i in range(len(labels)):

            rows.append({

                "image_id":
                    image_ids[i],

                "true_label":
                    labels[i].item(),

                "predicted_label":
                    predicted_labels[i].item(),

                "confidence":
                    confidence[i].item(),

                "epistemic":
                    epistemic[i].item(),

                "aleatoric":
                    aleatoric[i].item(),

                "entropy":
                    entropy[i].item(),

                "correct":
                    labels[i].item()
                    ==
                    predicted_labels[i].item()
            })

    return pd.DataFrame(rows)