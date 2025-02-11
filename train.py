"""
This script take cares of the training process and registers the model
at two stages: checkpoints of all epochs and parameters of the best model.
"""

import os

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import validation


def register_checkpoints(
    epoch: int,
    train_losses: list[float],
    val_losses: list[float],
    model: torch.nn.Module,
    optimizer: Optimizer,
    model_name: str = 'checkpoints/best.pth',
) -> None:
    """
    This function registers a checkpoint for each epoch.
    :param epoch: current training epoch
    :param train_losses: list of training losses
    :param val_losses: list of validation losses
    :param model: model to register
    :param optimizer: optimizer to register
    :param model_name:
    :return:
    """
    check_point = {
        "epoch": epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    os.makedirs("checkpoints/", exist_ok=True)
    torch.save(check_point, f"checkpoints/checkpoint_{model_name}.pth")


def train(
    epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.modules.loss._Loss,
    model_name: str,
    device: torch.device = torch.device("cuda"),
    scheduler=None,
    patience: int = 5,
) -> None:
    """
    This function trains the model for epochs.
    :param epochs: current training epoch
    :param model: model to train
    :param train_loader: loader of training data
    :param val_loader: loader of validation data
    :param optimizer: optimizer to train
    :param criterion: loss function
    :param device: device to train on
    :param model_name:
    :param scheduler: learning rate scheduler
    :param patience: number of epochs to wait before early stopping
    :return:
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, labels in tqdm(train_loader, desc=f"Training in epoch {epoch}"):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = validation(val_loader, model, criterion, device, epoch)
        val_losses.append(val_loss)
        scheduler.step()
        print(f"epoch{epoch}, train_loss: {train_loss}, val_loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/best_{model_name}.pth")
            print(f"saving the best model in best_{model_name}.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

        register_checkpoints(
            epoch, train_losses, val_losses, model, optimizer, model_name =model_name
        )
