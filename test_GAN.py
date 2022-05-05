import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from train_GAN_4 import generator, one_hot_to_index, plot_img


#generate some noise
manualSeed = np.random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
noise = torch.rand(40, 100)


#make generator and load learned weights
gen = generator(100,3844)
gen.load_state_dict(torch.load('b4/b4_weights_batches_2500.pth'))  #best verkregen model is b4/2500batches
#gen.load_state_dict(torch.load('gb_weights_batches_2250.pth'))
#gen.load_state_dict(torch.load('control_batches/control_weights_control_batches_2250.pth'))
#gen.load_state_dict(torch.load('constraints1&2_0.1/generator_weights_batches_2750.pth'))
#gen.load_state_dict(torch.load('constraint2_0.1/c2g_weights_batches_1500.pth'))
#gen.load_state_dict(torch.load('c1g_weights_batches_1200.pth'))
#gen.load_state_dict(torch.load('control_model/generator_weights_epoch_9.pth'))     #dit is de controle GAN
#gen.load_state_dict(torch.load('trained_model_weights/generator_weights_150_epoch.pth'))
#gen.load_state_dict(torch.load('nieuwe opmaak 33 epochs/generator_weights_epoch_19.pth'))
#gen.load_state_dict(torch.load('3.4_corrected/generator_weights_280.pth'))
gen.eval()


#print some generated levels
samples = gen(noise)
for i in range(20):
    plot_img(samples[i])


"""torch.set_printoptions(threshold=10_000)
print(samples[0])
print(samples[1])
print(samples[2])"""



