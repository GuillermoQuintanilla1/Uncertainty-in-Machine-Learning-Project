import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from preprocessing import preprocess_data
from dataset import create_dataloaders
from model import build_model
from training import train_one_epoch, validate
from evaluation import evaluate_model
from mcdropout import mc_dropout_predict, compute_uncertainties
from results import create_results_dataframe

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_df, val_df, test_df = preprocess_data()

train_loader, val_loader, test_loader = \
    create_dataloaders(
        train_df,
        val_df,
        test_df
    )

model = build_model().to(device)

classes = np.unique(train_df['cell_type_idx'])

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_df['cell_type_idx']
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)


print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print(train_df['cell_type'].value_counts())

print(device)
print(next(model.parameters()).device)

images, labels, image_ids = next(iter(train_loader))
print(images.shape, labels.shape)

num_epochs = 10
best_val_loss = float('inf')

'''
for epoch in range(num_epochs):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss = validate(
        model,
        val_loader,
        criterion,
        device
    )

    print(
        f"Epoch {epoch+1}: "
        f"Train={train_loss:.4f} "
        f"Val={val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved.")

'''

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

results_df = create_results_dataframe(
    model,
    test_loader,
    device,
    T=10
)

print(results_df.head())

results_df.to_csv(
    "test_results.csv",
    index=False
)

'''
metrics = evaluate_model(
    model,
    test_loader,
    device
)

print(metrics)


images, labels = next(
    iter(test_loader)
)

images = images.to(device)

predictions = mc_dropout_predict(
    model,
    images,
    T=10
)

mean_probs, epistemic, aleatoric, entropy = (
    compute_uncertainties(
        predictions
    )
)

print("Predicted class:",
      mean_probs[0].argmax().item())

print("Image root:",
      mean_probs[0])

print("Confidence:",
      mean_probs[0].max().item())

print("Epistemic:",
      epistemic[0].item())

print("Aleatoric:",
      aleatoric[0].item())

print("Entropy:",
      entropy[0].item())
'''