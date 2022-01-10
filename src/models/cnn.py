
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        #self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        #self.backbone.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        self.backbone = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        #self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
        #self.backbone.fc = torch.nn.Linear(in_features=1000, out_features=10, bias=True)


    def forward(self, inputs):
        x = self.backbone(inputs)
        return x