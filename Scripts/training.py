import torch

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device):

    model.train()

    running_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return (
        running_loss /
        len(loader)
    )

def validate(
        model,
        loader,
        criterion,
        device):

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

    return (
        running_loss /
        len(loader)
    )