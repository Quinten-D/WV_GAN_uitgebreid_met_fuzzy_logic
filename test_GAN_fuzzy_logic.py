import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from train_and_save_fl_3 import generator, one_hot_to_index, plot_img


#generate some noise
manualSeed = np.random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
noise = torch.rand(4, 100)


#make generator and load learned weights
gen = generator(100,3844)
#gen.load_state_dict(torch.load('control_batches/control_weights_control_batches_700.pth'))
#gen.load_state_dict(torch.load('constraint1&2_0.09/g0.09_weights_batches_00.pth'))
#gen.load_state_dict(torch.load('trained_model_weights/generator_weights_150_epoch.pth'))
#gen.load_state_dict(torch.load('nieuwe opmaak 33 epochs/generator_weights_epoch_19.pth'))
#gen.load_state_dict(torch.load('3.4_corrected/generator_weights_280.pth'))
gen.eval()


#print some generated levels
samples = gen(noise)
plot_img(samples[0])
plot_img(samples[1])
plot_img(samples[2])
plot_img(samples[3])



