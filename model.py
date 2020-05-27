import torch
import torch.nn as nn
import torch.nn.functional as F

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()        

        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=2):
        super(SimpleNet,self).__init__()
        
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.avgpool)
        
        self.fc1 = nn.Linear(in_features=64*14*14,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=2)

    def forward(self, input):
        output = self.net(input)
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output