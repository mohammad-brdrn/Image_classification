"""This Python script is developed for training and evaluation
of Image classification models. External arguments:
-m --mode: shows the training mode,can be either 'onetime' or 'GS' (greedy search).
-e --epochs: the number of epochs to train the model.

Instructions:

- you can change and define your own dataset for training and evaluation.
- You can change some of the predefine hyperparameters in the HPs section
in the main.py file. Other hyperparameters are fetched from "hyperparameters.yaml" file.

To run the code in terminal, use the following command:
python main.py --mode onetime --epochs 10

you can modify each argument mentioned above.


"""

import argparse
from itertools import product

import torch
import yaml
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import (
    CIFAR10,
)  # will be deleted after testing the performance of the script

from Data.transforms import Image_classification_transform
from evaluation import metrics, plot_losses, test
from models.cnn_models import CNN
from train import train
from utils.functions import set_seed


def main(training_mode: str, num_epochs: int) -> None:
    """
    Main function for training and evaluation.
    :param training_mode: Can be either 'onetime' or 'GS' (greedy search).
    :param num_epochs: number of epochs to train the model.
    :return:
    """

    # HPs (Hyperparameters to define)
    project = "CIFAR10_classification"
    loss_function = torch.nn.CrossEntropyLoss()

    weight_decay = 1e-4
    image_size = (32, 32)
    epochs = num_epochs
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    train_size = 0.80
    val_size = 0.10
    num_workers = 4
    seed = 1234
    patience = 5

    set_seed(seed)  # set a fix seed to ensure reproducibility.
    device = torch.device(
        "mps" if torch.mps.is_available() else "cpu"
    )  # mps in Mac, CUDA in Linux and Windows
    print(f"Current device is {device}")

    with open("hyperparameters.yaml", "r", encoding='utf-8') as F:
        config = yaml.safe_load(F)
    learning_rates = config["learning_rate"]
    batch_sizes = config["batch_size"]
    optimizers = config["optimizer"]
    network = config["network"]
    if training_mode == "onetime":
        learning_rates = [learning_rates[0]]
        batch_sizes = [batch_sizes[0]]
        optimizers = [optimizers[0]]
        network = [network[0]]

    transform = Image_classification_transform(
        image_size=image_size, augmentation=True, mean=mean, std=std
    )
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_size = int(train_size * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - (val_size + train_size)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    for learning_rate, batch_size, optimizer_name, network_name in product(
        learning_rates, batch_sizes, optimizers, network
    ):

        model = CNN(model_name=network_name, pretrained=True, out_channels=10).to(
            device
        )

        if optimizer_name == "AdamW":
            optimizer = AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "Adam":
            optimizer = Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            optimizer = SGD(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        model_name = f"model_{project}_{learning_rate}_{batch_size}_{optimizer_name}_{network_name}"
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )
        print(f"train size {train_size} | val size {val_size} | test size {test_size}")
        train(
            epochs,
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_function,
            model_name,
            device=device,
            scheduler=scheduler,
            patience=patience,
        )
        plot_losses(model_name)  # plots and saves the diagram of train and val losses
        _, targets, predictions = test(
            test_loader, model, model_name, loss_function, device
        )
        metrics(targets, predictions, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External arguments for the script")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="onetime",
        choices=("onetime", "GS"),
        help="training mode: one time train or train in a greedy search",
    )
    parser.add_argument("-e", "--epochs", type=int, default=30, help="number of epochs")
    args = parser.parse_args()
    mode = args.mode
    epochs = args.epochs
    main(mode, epochs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
