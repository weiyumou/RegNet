import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

import evaluation
import utils
import torch.nn.functional as F


def train_mnist(device, model, dataloaders, dataset_sizes, num_epochs, patience=5, threshold=1e-4):
    writer = SummaryWriter()

    model = model.to(device)

    layer_optimisers = dict()
    layer_names = []
    for name, layer in model.named_children():
        layer_optimisers[name] = optim.Adam(layer.parameters())
        layer_names.append(name)

    final_loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in tqdm.tqdm(dataloaders["train"], desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                best_outputs = model(inputs)
                loss = final_loss_fn(best_outputs[layer_names[-1]], labels)
            running_loss += loss.item() * inputs.size(0)

            # Learn the best output for the last layer
            curr_layer = layer_names[-1]
            best_outputs[curr_layer].requires_grad_()
            output_optim = optim.Adam([best_outputs[curr_layer]], lr=1)

            no_improve, last_loss = 0, 0
            while no_improve < patience:
                loss = final_loss_fn(best_outputs[curr_layer], labels)
                loss.backward()
                output_optim.step()
                output_optim.zero_grad()
                if abs(loss.item() - last_loss) < threshold:
                    no_improve += 1
                else:
                    no_improve = 0
                last_loss = loss.item()

            best_outputs[curr_layer].requires_grad_(False)

            # Learn all layers by regressing to the target output
            # Adjust the weight for the current layer and generates
            # target for the previous layer

            layer_idx = len(layer_names) - 1
            while layer_idx > 0:
                curr_layer = layer_names[layer_idx]
                prev_layer = layer_names[layer_idx - 1]

                best_outputs[prev_layer].requires_grad_()
                output_optim = optim.Adam([best_outputs[prev_layer]], lr=1e-3)
                layer_optimisers[curr_layer].zero_grad()

                no_improve, last_loss = 0, 0
                while no_improve < patience:
                    layer_out = getattr(model, curr_layer)(F.relu(best_outputs[prev_layer]))
                    loss = loss_fn(layer_out, best_outputs[curr_layer])
                    loss.backward()
                    output_optim.step()
                    layer_optimisers[curr_layer].step()
                    output_optim.zero_grad()
                    layer_optimisers[curr_layer].zero_grad()
                    if abs(loss.item() - last_loss) < threshold:
                        no_improve += 1
                    else:
                        no_improve = 0
                    last_loss = loss.item()

                with torch.no_grad():
                    best_outputs[prev_layer] = F.relu(best_outputs[prev_layer])
                layer_idx -= 1

            curr_layer = layer_names[layer_idx]
            layer_optimisers[curr_layer].zero_grad()
            no_improve, last_loss = 0, 0
            while no_improve < patience:
                layer_out = getattr(model, curr_layer)(inputs)
                loss = loss_fn(layer_out, best_outputs[curr_layer])
                loss.backward()
                layer_optimisers[curr_layer].step()
                layer_optimisers[curr_layer].zero_grad()
                if abs(loss.item() - last_loss) < threshold:
                    no_improve += 1
                else:
                    no_improve = 0
                last_loss = loss.item()

        epoch_loss = running_loss / dataset_sizes["train"]
        utils.logger.info(f"Epoch {epoch}: Train Loss = {epoch_loss}")

        val_accuracy, val_loss = evaluation.eval_mnist(device, model, dataloaders["val"], dataset_sizes["val"])
        utils.logger.info(f"Epoch {epoch}: Val Accuracy = {val_accuracy}, Val Loss = {val_loss}")

        test_accuracy, test_loss = evaluation.eval_mnist(device, model, dataloaders["test"], dataset_sizes["test"])
        utils.logger.info(f"Epoch {epoch}: Test Accuracy = {test_accuracy}, Test Loss = {test_loss}")

        writer.add_scalars("Loss", {"Train_Loss": epoch_loss, "Val_Loss": val_loss, "Test_Loss": test_loss}, epoch)
        writer.add_scalars("Accuracy", {"Val_Accuracy": val_accuracy, "Test_Accuracy": test_accuracy}, epoch)
