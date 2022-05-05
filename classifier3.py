import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# eigen imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import random
import torch.backends.cudnn as cudnn
from train_GAN_4_control import plot_img
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(123)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# defining classifier class
class classifier(nn.Module):

    def __init__(self, inp, out):
        super(classifier, self).__init__()

        """self.net = nn.Sequential(
            #nn.Linear(inp, 600),
            nn.Linear(inp, 7688),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            #nn.Linear(600, 600),
            nn.Linear(7688, 7688),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            #nn.Linear(600, 400),
            nn.Linear(7688, 7688),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            #nn.Linear(400, out),
            nn.Linear(7688, out),
            nn.Sigmoid()
        )"""

        self.conv1 = nn.Conv2d(4, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 24, 7)
        self.conv4 = nn.Conv2d(24, 32, 9)
        self.net = nn.Sequential(
            nn.Linear(3872, 121),
            #nn.Linear(484, 121),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(121, 25),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )
        self.pool = nn.MaxPool2d(2, 1)


    def forward(self, x):
        #x = self.net(x)
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.net(x)
        return x


# eigen dataset class
class MyDataset(Dataset):
    def __init__(self, file_name):
        read_in = pd.read_csv(file_name, header=None)
        eigen_data = np.asarray(read_in.values, dtype=np.float32)
        self.data = eigen_data
        #eigen_data_als_tensor = torch.from_numpy(eigen_data)
        #self.data = eigen_data_als_tensor

    def __getitem__(self, index):
        """# x = torch.from_numpy(self.data[index])
        x = self.data[index]
        # convert to one-hot encoding
        x = x.long()  # cast naar torch.int64
        x = torch.nn.functional.one_hot(x, num_classes=4)
        x = x.reshape((3844))
        x = x.float()  # cast naar torch.float32"""
        level = self.data[index][0:-1]
        level = torch.tensor(level)
        target = self.data[index][-1]
        target = torch.tensor(target)
        return level, target

    def __len__(self):
        return len(self.data)


def validate(cl, test_loader):
    val_loss = 0
    batch_size = 40
    criterion = nn.BCELoss()
    for levels, targets in test_loader:
        #levels_reshaped = torch.reshape(levels, (batch_size, 31, 31, 4))
        levels_reshaped = torch.reshape(levels, (batch_size, 961, 4))
        levels_reshaped = torch.transpose(levels_reshaped, 1, 2)
        levels_reshaped = torch.reshape(levels_reshaped, (batch_size, 4, 31, 31))
        output = cl(levels_reshaped)
        val_loss += criterion(output, targets.unsqueeze(1)).data.item()
        #print(criterion(output, targets.unsqueeze(1)).data.item())
    val_loss /= len(test_loader)

    print('\nValidation set: Average loss over all mazes of testdata: {:.4f}'.format(
        val_loss))
    print("(0 is best)\n")

def validate_v(cl, test_loader):
    val_loss = 0
    batch_size = 40
    criterion = nn.BCELoss()
    for levels, targets in test_loader:
        #levels_reshaped = torch.reshape(levels, (batch_size, 31, 31, 4))
        levels_reshaped = torch.reshape(levels, (batch_size, 961, 4))
        levels_reshaped = torch.transpose(levels_reshaped, 1, 2)
        levels_reshaped = torch.reshape(levels_reshaped, (batch_size, 4, 31, 31))
        output = cl(levels_reshaped)
        val_loss += abs(torch.mean(torch.abs(output - targets)))
        #print(criterion(output, targets.unsqueeze(1)).data.item())
    val_loss /= len(test_loader)

    print('\nValidation set: Average loss over all mazes of testdata: {:.4f}'.format(
        val_loss))
    print("(0 is best)\n")


if __name__=='__main__':
    batch_size = 40


    # load data
    train_dataset = MyDataset("trainingsdata_classifier.csv")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               worker_init_fn=seed_worker
                                               )

    # choose optimization algorithm and loss function, make classifier network
    cl = classifier(3844, 1)
    #optimizer = torch.optim.SGD(cl.parameters(), lr=0.005, momentum=0.5) #0.005 normaal?
    optimizer = torch.optim.Adam(cl.parameters(), lr=0.0005, betas=(0.9, 0.999))
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    epochs = 500


    # training loop
    print("Training loop 3...")
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 1):
            levels, targets = data
            #plot_img(levels[3])
            #plot_img2(targets[3])
            # Zero gradient buffers
            optimizer.zero_grad()
            # test
            #levels_reshaped = torch.reshape(levels, (batch_size, 31, 31, 4))
            levels_reshaped = torch.reshape(levels, (batch_size, 961, 4))
            levels_reshaped = torch.transpose(levels_reshaped, 1, 2)
            levels_reshaped = torch.reshape(levels_reshaped, (batch_size, 4, 31, 31))
            #print("normaal level")
            #print(levels[3].size())
            #reshaped_level = levels_reshaped[3]
            #print("reshaped level")
            #print(reshaped_level.size())
            #print(reshaped_level)
            #print("flatten")
            #reshaped_level = torch.flatten(reshaped_level)
            #print(reshaped_level.size())
            #print(reshaped_level)
            # Pass training levels through the network
            output = cl(levels_reshaped)
            #print("output")
            #print(output[0].size())
            #plot_img2(output[2].detach())
            # Calculate loss
            #print("output size: ", output.size())
            #print("targets size: ", targets.size())
            loss = criterion(output, targets.unsqueeze(1))
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()

            # check progress
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tloss: %.4f\t'
                      % (epoch, epochs, i, len(train_loader), loss.data.item()))


        # save weights after every 5 epochs
        if ((epoch+1) % 5) == 0:
            torch.save(cl.state_dict(), '1class3_weights_epoch_' + str(epoch + 1) + '.pth')
            print("saving weights")


