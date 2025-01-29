# This Python script is developed for training and evaluation
# of Image classification models.

#import os
#import sys
import argparse
import yaml
import torch
from itertools import product


from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, random_split


from torchvision.datasets import CIFAR10 # will be deleted after testing the performance of the script



from models.CNNs import CNN
from train import train
from Data.transforms import Image_classification_transform
from evaluation import plot_losses , test, metrics




def main(training_mode: str, num_epochs: int) -> None:
    project = 'CIFAR10_classification'
    loss_function = torch.nn.CrossEntropyLoss()
    # seed
    weight_decay = 1e-4
    image_size = (32, 32)
    epochs = 30
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    train_size = 0.80
    val_size = 0.10
    num_workers = 4





    device = torch.device('mps' if torch.mps.is_available() else 'cpu')  # mps in Mac, CUDA in Linux and Windows
    print(f'Current device is {device}')


    with open('hyperparameters.yaml', 'r') as F:
        config = yaml.safe_load(F)
    learning_rates = config['learning_rate']
    batch_sizes = config['batch_size']
    optimizers = config['optimizer']
    network = config['network']
    if training_mode == 'onetime':
        learning_rates = [learning_rates[0]]
        batch_sizes = [batch_sizes[0]]
        optimizers = [optimizers[0]]
        network = [network[0]]

    for learning_rate, batch_size, optimizer_name, network_name in product(learning_rates, batch_sizes, optimizers, network):

        model = CNN(model_name=network_name, pretrained=True, out_channels=10).to(device)

        if optimizer_name == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        model_name = f'model_{project}_{learning_rate}_{batch_size}_{optimizer_name}_{network_name}'
        transform = Image_classification_transform(image_size = image_size, augmentation=True, mean = mean, std = std)
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_size = int(train_size *len(dataset))
        val_size = int(val_size * len(dataset))
        test_size = len(dataset) - (val_size+train_size)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        print(f'train size {train_size} | val size {val_size} | test size {test_size}')
        train(epochs, model, train_loader, val_loader, optimizer, loss_function, device, model_name, scheduler = scheduler)
        plot_losses(model_name)  # plots and saves the diagram of train and val losses
        _,targets, output_labels = test(test_loader, model, model_name, loss_function, device)
        metrics(targets,output_labels, model_name)








    # reading hyperparameters
    # reading data
    # create dataset and data loader
    #define model, loss, optimizer
    # class train
    #test and metrics




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='External arguments for the script')
    parser.add_argument('-m', '--mode', type=str, default='onetime',choices=('onetime', 'GS'), help = 'training mode: one time train or train in a greedy search')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of epochs')
    args = parser.parse_args()
    mode = args.mode
    epochs = args.epochs
    main(mode, epochs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
