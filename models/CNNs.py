import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    '''
    This class replicates the CNN architecture for SOTA networks.
    arguments:
        -model_name: string  / example: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        -pretrained: boolean
        -out_channels: integer  / describing the number of output channels.
    '''
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True, out_channels: int = 256):
        super(CNN, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        elif model_name == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        else:
            raise NotImplementedError('This model is not implemented. Try other models.')

    def forward(self, x):
        return self.model(x)
