import csv

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, confusion_matrix

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F



def plot_losses (model_name:str) -> None:
    """
    This script plots the loss diagrams for training and validation.
    Plot images are saved in the logs directory for the related model_name.
    losses are loaded from the checkpoint related to the model name.
    :param model_name: The name of the model.
    :return: None Image are saved in the logs directory.
    """
    checkpoint = torch.load('checkpoints/checkpoint_' + model_name + '.pth',weights_only=True)
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    os.makedirs("logs/loss_plots", exist_ok=True)
    epochs = list(range(len(train_losses)))
    plt.plot (epochs, train_losses, label="Training Loss", marker='o', linestyle='-')
    plt.plot (epochs, val_losses, label="Validation Loss", marker='s', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and validation losses over epochs')
    save_dir = os.path.join("logs/loss_plots", model_name)
    plt.savefig(save_dir+ '.png')
    print('Saving loss diagram to', save_dir)
    plt.clf()


def test (data_loader:DataLoader, model:nn.Module, model_name:str, criterion , device ) -> tuple[list[float],list[int],list[int]]:
    test_losses =[]
    targets = []
    output_labels = []
    with torch.no_grad():
        state_dict = torch.load('checkpoints/best_' + model_name + '.pth',weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        for data, labels in tqdm(data_loader, desc=f'Testing process'):
            if len(labels)>1: # check if size of the output batch is not 1.
                raise ValueError('Batch-size for the test dataset should be set to 1.')
            targets.append(labels.numpy().item())
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            output = F.log_softmax(output, dim=1)
            output_labels.append(output.argmax(dim=1).item())
            test_losses.append(loss)

    return test_losses, targets, output_labels


def validation (val_loader:DataLoader, model:nn.Module, criterion , device, epoch ) ->float:
    val_loss = 0
    with torch.no_grad():
        model.eval()
        for data, labels in tqdm(val_loader, desc=f'Validation in epoch {epoch}'):
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
    return val_loss


def metrics (targets: list[int], output_labels:list[int], model_name) ->None:
    accuracy = accuracy_score(targets, output_labels)
    recall = recall_score(targets, output_labels, average='macro')
    precision = precision_score(targets, output_labels, average='macro')
    f1 = f1_score(targets, output_labels, average='macro')
    conf_matrix = confusion_matrix(targets, output_labels)
    with open (f'logs/confusion_matrix_{model_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Accuracy', 'Recall', 'Precision', 'F1', 'Confusion Matrix'])
        writer.writerow([accuracy, recall, precision, f1, conf_matrix])
    print(f'Metircs extracted and saved to logs/confusion_matrix_{model_name}.csv')




if __name__ == '__main__':
    targets = [1,2,1,0,1,1,1,0,1]
    output_labels = [0,2,1,0,1,0,1,0,1]
    model_name = 'test_be_deleted'
    metrics(targets, output_labels, model_name)


