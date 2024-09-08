import torch
import torch.nn as nn
import torch.nn.functional as F
from .badge import BayesianMLP_EBM, BayesianMNISTCNN_EBM

    
class ConvNet(nn.Module):
    '''
    Convolutional neural network with two convolutional layers and two fully connected layers,
    compatible with BackPACK.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.dim_out = 32
                
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=self.dim_out, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(4*4*self.dim_out, self.dim_out),
            nn.ReLU(),
            nn.Linear(self.dim_out, 10)
        )

        self._last_layer = self.classifier[2]

        # move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_last_layer_parameters(self):
        return self._last_layer.parameters()


class MLP(nn.Module):
    '''
    Multi-layer perceptron with two hidden layers.

    Made for Imagenet embeddings, so the input is 784 dimensional.
    '''

    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self._last_layer = self.fc3
        
        # move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    '''
    Convolutional neural network with two convolutional layers and two fully connected layers.
    '''

    def __init__(self):
        super(CNN, self).__init__()
        self.dim_out = 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=self.dim_out, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*self.dim_out, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.dim_out)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def get_model_constructor(method, dataset):
    if dataset == 'imagenet':
        # MLP models only
        if method == 'badge':
            return BayesianMLP_EBM
        else:
            return MLP
    else:
        # MNIST CNN models
        if method == 'badge':
            return BayesianMNISTCNN_EBM
        else:
            return ConvNet