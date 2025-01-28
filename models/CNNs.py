import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    """
    This class replicates the CNN architecture for SOTA networks.
    arguments:
        -model_name: string  / example: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        -pretrained: boolean
        -out_channels: integer  / describing the number of output channels.
    """
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True, out_channels: int = 256):
        super(CNN, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT if pretrained else None)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet152':
            self.model = models.resnet152(weights = models.ResNet152_Weights.DEFAULT if pretrained else None)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        else:
            raise NotImplementedError('This model is not implemented. Try other models.')

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    model = CNN('resnet18', True)
    print(model)
