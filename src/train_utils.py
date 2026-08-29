import torch
from sklearn.metrics import accuracy_score, f1_score


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    losses, targets, predictions = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item() * labels.size(0))
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": sum(losses) / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(targets, predictions),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
    }
