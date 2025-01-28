# This Python script is developed for training and evaluation
# of Image classification models.

#import os
#import sys
import argparse
import yaml
import torch

from models.CNNs import CNN






def main(training_mode: str, num_epochs: int) -> None:

    with open('hyperparameters.yaml', 'r') as F:
        config = yaml.safe_load(F)
    learning_rates = config['learning_rate']
    batch_sizes = config['batch_size']
    optimizers = config['optimizer']
    loss_function = torch.nn.CrossEntropyLoss()


    device = torch.device('mps' if torch.mps.is_available() else 'cpu')  # mps in Mac, CUDA in Linux and Windows
    print( f'Current device is {device}')


    model = CNN(model_name = 'resnet18', pretrained=True, out_channels=10).to(device)








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
