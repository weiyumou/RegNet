import torch.nn as nn
import tqdm
import torch


def eval_mnist(device, model, dataloader):
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    corrects = 0
    total = 0
    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)

    accuracy = corrects / total
    total_loss = running_loss / total
    return accuracy, total_loss
