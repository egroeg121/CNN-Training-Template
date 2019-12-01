import torch
from torch import nn,optim
from torch.nn import functional
from torchvision import datasets,transforms


class trainer:
    def __init__(self):

        self.train_loader,self.test_loader = dataloaders(batch_size=32)

        self.model = basicConv()

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.001, momentum=0.9)

        self.epoch = 0
        self.train_loss_history = []
        pass

    def train_iter(self,data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.model(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def validate(self):
        pass

    def train_epoch(self):
        self.epoch +=1
        running_loss = 0
        for i,data in enumerate(self.train_loader):
            iter_loss = self.train_iter(data)
            running_loss += iter_loss
        self.train_loss_history.append(running_loss)
        return running_loss





def dataloaders(loc='../data',batch_size=1,shuffle=True):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(loc, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=shuffle,)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(loc, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False,)
    return train_loader,test_loader


class basicConv(nn.Module):
    def __init__(self):
        super(basicConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.conv2(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = functional.log_softmax(x, dim=1)
        return output