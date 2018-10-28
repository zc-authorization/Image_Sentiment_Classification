import torch
import torch.optim.optimizer as optimizer
import torch.nn.functional as F
import torchvision.transforms as transforms

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1=torch.nn.Conv2d(1,6,3,1)
        self.pooling1=torch.nn.MaxPool2d(2)
        self.conv2=torch.nn.Conv2d(6,1,3,1)
        self.fc1=torch.nn.Linear(21*21,100)
        self.fc2=torch.nn.Linear(100,10)
        self.fc3=torch.nn.Linear(10,10)
        self.fc4=torch.nn.Linear(10,7)

    def forward(self, x):
        x=F.relu(self.conv1(x.float()))
        x=self.pooling1(x)
        x=F.relu(self.conv2(x))
        x=x.view(-1,1*21*21)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        #x=F.softmax(x)
        return x


