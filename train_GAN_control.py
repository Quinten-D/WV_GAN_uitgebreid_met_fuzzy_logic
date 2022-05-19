### GEBRUIKTE MODEL VOOR HET CONTROLE GAN ###


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


def one_hot_to_index(array):
    array = array.reshape((961, 4))
    index_values = torch.tensor([[0], [1], [2], [3]], dtype=torch.float32)
    array = torch.matmul(array, index_values)
    array = array.reshape((961))
    return array


# om resultaten te plotten
def plot_img(array, number=None):
    array = one_hot_to_index(array)
    array = array.detach()
    array = array.reshape(31, 31)
    plt.imshow(array, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if number:
        plt.xlabel(number, fontsize='x-large')
    plt.show()


# random noise als input voor generator netwerk
def make_some_noise():
    #return torch.rand(batch_size, 100)
    return torch.randn(batch_size, 100)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


# defining generator class
class generator(nn.Module):

    def __init__(self, inp, out):
        super(generator, self).__init__()

        self.net = nn.Sequential(
            # nn.Linear(inp, 300),
            # nn.Linear(inp, 1200),
            nn.Linear(inp, 600),
            nn.ReLU(inplace=True),
            # nn.Linear(300, 1000),
            # nn.Linear(1200, 4000),
            nn.Linear(600, 2000),
            nn.ReLU(inplace=True),
            # nn.Linear(1000, 800),
            # nn.Linear(4000, 3200),
            nn.Linear(2000, 1600),
            nn.ReLU(inplace=True),
            # nn.Linear(800, out)
            # nn.Linear(3200, out)
            nn.Linear(1600, out)
        )

    def forward(self, x):
        x = self.net(x)
        y = x.reshape((40, 961, 4))
        y = torch.nn.functional.log_softmax(y, dim=2)
        z = torch.nn.functional.gumbel_softmax(y, tau=0.1, hard=True)
        # index_values = torch.tensor([[0], [1], [2], [3]], dtype=torch.float32)
        # u = torch.matmul(z,index_values)
        # v = u.reshape((4,961))
        w = z.reshape((40, 3844))
        return w


# defining discriminator class
class discriminator(nn.Module):

    def __init__(self, inp, out):
        super(discriminator, self).__init__()

        self.net = nn.Sequential(
            # nn.Linear(inp, 300),
            # nn.Linear(inp, 1200),
            nn.Linear(inp, 600),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # nn.Linear(300, 300),
            # nn.Linear(1200, 1200),
            nn.Linear(600, 600),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # nn.Linear(300, 200),
            # nn.Linear(1200, 800),
            nn.Linear(600, 400),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            # nn.Linear(200, out),
            # nn.Linear(800, out),
            nn.Linear(400, out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x


# eigen dataset class
class MyDataset(Dataset):
    def __init__(self, file_name):
        read_in = pd.read_csv(file_name, header=None)
        eigen_data = np.asarray(read_in.values, dtype=np.float32)
        eigen_data_als_tensor = torch.from_numpy(eigen_data)
        self.data = eigen_data_als_tensor
        # eigen_data = torch.tensor(read_in.values, dtype=torch.float32)
        # self.data = torch.tensor(eigen_data, dtype=torch.float32)

    def __getitem__(self, index):
        # x = torch.from_numpy(self.data[index])
        x = self.data[index]
        # convert to one-hot encoding
        x = x.long()  # cast naar torch.int64
        x = torch.nn.functional.one_hot(x, num_classes=4)
        x = x.reshape((3844))
        x = x.float()  # cast naar torch.float32
        return x

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = MyDataset('levels_corrected.csv')
    loader = DataLoader(
        dataset,
        batch_size=40,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker
    )


    # maak generator
    gen = generator(100, 3844)
    gen.apply(weights_init)
    # maak discriminator
    dis = discriminator(3844, 1)
    dis.apply(weights_init)


    # training variabelen, en stochastic gradient descent SGD, loss function BCE
    # optimizerd is het proces van SGD, criteriond is de gebruikte loss functie
    batch_size = 40
    #d_steps = 100
    d_steps = 1
    #g_steps = 100
    g_steps = 1

    criteriond1 = nn.BCELoss()  # loss functie voor de discriminator
    #optimizerd1 = optim.SGD(dis.parameters(), lr=0.002, momentum=0.9)
    optimizerd1 = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.05, 0.999))
    criteriond2 = nn.BCELoss()
    #optimizerd2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
    optimizerd2 = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.05, 0.999))

    epochs = 50 # 15

    # training
    print("Training loop: ...")
    # save untrained model
    torch.save(gen.state_dict(), 'control_weights_50_batches_' + str(0) + '.pth')
    print("saving weights 50")
    # start training loop
    for epoch in range(epochs):
        for i, inp_real_x in enumerate(loader, 1):

            # training discriminator on real data (inp_real_x)
            dis.zero_grad()  # reset de gradientvector naar allemaal nullen
            dis_real_out = dis(inp_real_x)
            D_x = (dis_real_out.view(-1)).mean().item()
            # print("input disc: "+ str(inp_real_x.size()))
            # print("output disc: "+str(dis_real_out.size()))
            dis_real_loss = criteriond1(dis_real_out, Variable(torch.ones(batch_size, 1)))  # bereken loss, !ones
            dis_real_loss.backward()  # bereken de gradient met backprop (1)
            # training discriminator on data produced by generator
            inp_fake_x_gen = make_some_noise()
            # output from generator is generated
            dis_inp_fake_x = gen(
                inp_fake_x_gen).detach()  # detach zodat je niet de afgeleiden naar gewichten van de generator berekent?
            dis_fake_out = dis(dis_inp_fake_x)
            D_G_z1 = (dis_fake_out.view(-1)).mean().item()
            dis_fake_loss = criteriond1(dis_fake_out, Variable(torch.zeros(batch_size, 1)))  # bereken loss, !zeros
            dis_fake_loss.backward()  # bereken de gradient met backprop (2)
            optimizerd1.step()  # update de gewichten van het netwerk met de gradienten uit (1) en (2) volgens het SGD algoritme

            # training generator for g_steps after dicriminator has been trained for d_steps
            if i % d_steps == 0:
                for g_step in range(g_steps):
                    gen.zero_grad()
                    # generating data for input for generator
                    gen_inp = make_some_noise()
                    gen_out = gen(gen_inp)
                    dis_out_gen_training = dis(gen_out)
                    gen_loss = criteriond2(dis_out_gen_training, Variable(torch.ones(batch_size, 1)))
                    gen_loss.backward()
                    optimizerd2.step()

            # print training progression
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tD(x): %.4f\tD(G(z)): %.4f'
                      % (epoch, epochs, i, len(loader), D_x, D_G_z1))

            # save every 50 batches
            if (i % 50 == 0):
                b = epoch * 250 + i
                torch.save(gen.state_dict(), 'control_weights_50_batches_' + str(b) + '.pth')
                print("saving weights 50")

            # save every 100 batches when epoch >= 1
            """if epoch >= 1:
                if i % 100 == 0:
                    torch.save(gen.state_dict(), 'generator_weights_epoch_' + str(epoch + 1) + '_batches_' + str(i) + '.pth')"""

            # save every 100 batches when epoch == 10 or 12 or 13 or 14 or 15
            """if epoch >= 10 and epoch != 11:
                if i % 100 == 0:
                    torch.save(gen.state_dict(), 'generator_weights_epoch_' + str(epoch + 1) + '_batches_' + str(i) + '.pth')"""

        # save every epoch
        """if epoch > 7:
            print("saving weights")
            torch.save(gen.state_dict(), 'generator_weights_epoch_' + str(epoch+1) + '.pth')"""

    # save model parameters when training is done
    print("saving weights")
    torch.save(gen.state_dict(), 'generator_weights.pth')
