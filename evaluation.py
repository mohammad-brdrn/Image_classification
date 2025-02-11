"""
This script contains different evaluation functions, such as:
model validation, test, metric calculation, training process information plot.
"""

import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from tqdm import tqdm


def plot_losses(model_name: str) -> None:
    """
    This script plots the loss diagrams for training and validation.
    Plot images are saved in the logs directory for the related model_name.
    losses are loaded from the checkpoint related to the model name.
    :param model_name: The name of the model.
    :return: None Image are saved in the logs directory.
    """
    checkpoint = torch.load(
        "checkpoints/checkpoint_" + model_name + ".pth", weights_only=True
    )
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    os.makedirs("logs/loss_plots", exist_ok=True)
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o", linestyle="-")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and validation losses over epochs")
    save_dir = os.path.join("logs/loss_plots", model_name)
    plt.savefig(save_dir + ".png")
    print("Saving loss diagram to", save_dir)
    plt.clf()


def test(
    data_loader: DataLoader,
    model: nn.Module,
    model_name: str,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
) -> tuple[list[float], list[int], list[int]]:
    """
    This function tests the model on the test set.
    :param data_loader: test data loader.
    :param model: trained model.
    :param model_name: model name.
    :param criterion: loss criterion.
    :param device: device to use.
    :return:
    """
    test_losses = []
    targets = []
    predictions = []
    with torch.no_grad():
        state_dict = torch.load(
            "checkpoints/best_" + model_name + ".pth", weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        for data, labels in tqdm(data_loader, desc="Testing process"):
            if len(labels) > 1:  # check if size of the output batch is not 1.
                raise ValueError("Batch-size for the test dataset should be set to 1.")
            targets.append(labels.numpy().item())
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            output = F.log_softmax(output, dim=1)
            predictions.append(output.argmax(dim=1).item())
            test_losses.append(loss)

    return test_losses, targets, predictions


def validation(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
    epoch: int,
) -> float:
    """
    This function evaluates the model on the validation set.
    :param val_loader: validation data loader.
    :param model: model name.
    :param criterion: loss criterion.
    :param device: device to use.
    :param epoch: current epoch.
    :return:
    """
    val_loss = 0.0
    with torch.no_grad():
        model.eval()
        for data, labels in tqdm(val_loader, desc=f"Validation in epoch {epoch}"):
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
    return val_loss


def metrics(targets: list[int], predictions: list[int], model_name: str) -> None:
    """
    This function computes the metrics between targets and predictions.
    :param targets: list of ground-truth labels.
    :param predictions: list of predictions.
    :param model_name: model name.
    :return:
    """
    accuracy = accuracy_score(targets, predictions)
    recall = recall_score(targets, predictions, average="macro")
    precision = precision_score(targets, predictions, average="macro")
    f1 = f1_score(targets, predictions, average="macro")
    conf_matrix = confusion_matrix(targets, predictions)
    with open(
        f"logs/confusion_matrix_{model_name}.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Accuracy", "Recall", "Precision", "F1", "Confusion Matrix"])
        writer.writerow([accuracy, recall, precision, f1, conf_matrix])
    print(f"Metrics extracted and saved to logs/confusion_matrix_{model_name}.csv")
