import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



class MoonBoardDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        from load_moonboard import load_moonboard
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_moonboard()
        self.x_train = self.x_train.reshape(-1,18,11).astype(float)
        self.x_test = self.x_test.reshape(-1,18,11).astype(float)
        self.y_train = self.y_train.reshape(-1,1).astype(int)
        self.y_test = self.y_test.reshape(-1,1).astype(int)
        #self.y_train = np.eye(17)[self.y_train]
        #self.y_test = np.eye(17)[self.y_test]

    def __len__(self):
        if self.train:
            return len(self.x_train)
        else:
            return len(self.x_test)

    def __getitem__(self, idx):
        if self.train:
            return self.x_train[idx], self.y_train[idx]
        else:
            return self.x_test[idx], self.y_test[idx]


    # prepare data

dataset = MoonBoardDataset(train = True)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128, shuffle=True)
    
test_loader = torch.utils.data.DataLoader(
    MoonBoardDataset(train = False),
    batch_size=32, shuffle=True)




    #np.unique(y_test)
    #Out[20]: array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16], dtype=uint8)

    #np.unique(y_train)
    #Out[21]: array([ 0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16], dtype=uint8)

    # to one-hot rep
n_cls = 17
    #Y_train = np.eye(n_cls)[y_train]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 2, 1)
        self.conv2 = nn.Conv2d(20, 50, 2, 1)
        self.fc1 = nn.Linear(11*18, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 17)

    def forward(self, x):
        x = x.view(-1, 11*18)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x




torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def acc(model, test=True):
    correct = 0
    total = 0
    with torch.no_grad():
        if test:
            loader = test_loader
        else:
            loader = train_loader
        for data in loader:
            images, labels = data
            #print(images[0])
            outputs = model(images.to(device, dtype=torch.float32))

            outputs = outputs.view((-1,17))            
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)

            
            labels = labels.view((-1))
            total += labels.size(0)
            correct += (predicted == labels.to(device, dtype=torch.long)).sum().item()


    print('Accuracy of the network on the 10000 test images: %f' % (
        100 * correct / total))

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        target = target.view((-1))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        

for i in range(100):
    train(model, device, train_loader, optimizer, i)
    acc(model)

class_correct = list(0. for i in range(18))
class_total = list(0. for i in range(18))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images.to(device, dtype=torch.float32))
        outputs = outputs.view((-1,17))

        _, predicted = torch.max(outputs.data, 1)

        labels = labels.view((-1))
        c = (predicted == labels.to(device, dtype=torch.long))
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(1,18):
    if class_total[i] != 0:
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))



