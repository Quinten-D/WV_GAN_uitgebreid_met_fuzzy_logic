import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

import solvability_algorithm as sa
from train_GAN_4_fl import *
import fuzzy_logic_2 as fl


def L1_norm(generated_batch):
    """
    used as an indication for the variety of generated levels, as an indication for mode collapse
    L1 norm used to measure the difference between each pair of mazes in a generated batch
    higher value would indicate larger variety
    :return: mean L1 norm of difference of a pair of generated levels in generated_batch
    """
    variety = 0
    batch_size = 40
    for i in range(batch_size):
        a = generated_batch[i]
        #for j in range(batch_size):
        for j in range(i):
            b = generated_batch[j]
            variety += torch.sum(torch.abs(torch.subtract(a, b))).item()
    #variety /= (batch_size ** 2)
    variety /= ((batch_size ** 2) - batch_size) / 2
    return variety


def percent_constraint1(generated_batch):
    satisfied = 0
    batch_size = 40
    for i in range(batch_size):
        if fl.constraint1(generated_batch[i], 0) == 1:
            satisfied += 1
    return satisfied / batch_size


def percent_constraint2(generated_batch):
    satisfied = 0
    batch_size = 40
    for i in range(batch_size):
        if fl.constraint2(generated_batch[i], 0) == 1:
            satisfied += 1
    return satisfied / batch_size


def percent_constraint1_and_2(generated_batch):
    satisfied = 0
    batch_size = 40
    for i in range(batch_size):
        if (fl.constraint1(generated_batch[i], 0) == 1) and (fl.constraint2(generated_batch[i], 0) == 1):
            satisfied += 1
    return satisfied / batch_size


def percent_constraint1_and_2_and_3(generated_batch):
    satisfied = 0
    batch_size = 40

    for i in range(batch_size):
        if (fl.constraint1(generated_batch[i], 0) == 1) and (fl.constraint2(generated_batch[i], 0) == 1) and sa.solvable(generated_batch[i]):
            satisfied += 1
    return satisfied / batch_size


def percent_constraint3(generated_batch):
    satisfied = 0
    batch_size = 40
    for i in range(batch_size):
        if sa.solvable(generated_batch[i]):
            satisfied += 1
    return satisfied / batch_size


def correct_for_constraint_1_and_2(level):
    """
    corrects a maze for constraint 2 by randomly changing all faulty 2 & 3 tiles to 0 or 1 tiles
    :param level: maze in one-hot encoding
    :return: corrected maze
    """
    corrected_level = []
    # constraint 2
    for i in range(0, 3844, 4):
        tile = level[i:i+4].detach()
        entry_tile = torch.tensor([0.,0.,1.,0.])
        exit_tile = torch.tensor([0.,0.,0.,1.])
        if (torch.equal(tile, entry_tile) or torch.equal(tile, exit_tile)) and i!=4 and i!=3836:
            random_number = random.uniform(0, 1)
            if random_number <= 0.5:
                corrected_level += [1., 0., 0., 0.]
            if random_number > 0.5:
                corrected_level += [0., 1., 0., 0.]
        else:
            if torch.equal(tile, torch.tensor([1.,0.,0.,0.])):
                corrected_level += [1.,0.,0.,0.]
            if torch.equal(tile, torch.tensor([0.,1.,0.,0.])):
                corrected_level += [0.,1.,0.,0.]
            if torch.equal(tile, torch.tensor([0.,0.,1.,0.])):
                corrected_level += [0.,0.,1.,0.]
            if torch.equal(tile, torch.tensor([0.,0.,0.,1.])):
                corrected_level += [0.,0.,0.,1.]
    # put the empty tiles before the entry tile and exit tile in
    corrected_level[128:132] = [1., 0., 0., 0.]
    corrected_level[3712:3716] = [1., 0., 0., 0.]
    # constraint 1
    for i in range(0, 3844, 4):
        # all tiles in first row are walls
        if i < 124:
            corrected_level[i:i+4] = [0.,1.,0.,0.]
        # all tiles in last row are walls
        if i >= 3720:
            corrected_level[i:i+4] = [0.,1.,0.,0.]
        """
        # all tiles in first column are walls
        if i % 124 == 0:
            pass
            #corrected_level[i:i + 4] = [0., 1., 0., 0.]
        # all tiles in last column are walls
        if i % 120 == 0:
            #pass
            corrected_level[i:i + 4] = [0., 1., 0., 0.]
        """
    for i in range(0, 31):
        for j in range(0, 31):
            index = (124 * i) + (4 * j)
            if j == 0:
                corrected_level[index:index + 4] = [0., 1., 0., 0.]
            if j == 30:
                corrected_level[index:index + 4] = [0., 1., 0., 0.]
    # put the entry tile and exit tile back in
    corrected_level[4:8] = [0., 0., 1., 0.]
    corrected_level[3836:3840] = [0., 0., 0., 1.]
    return torch.tensor(corrected_level)


def correct_batch(batch):
    corr = []
    for maze in batch:
        corr.append(correct_for_constraint_1_and_2(maze))
    return corr


def aantal_fouten(level):
    """
    aantal fouten volgens constraint 1 & 2
    :param level:
    :return:
    """
    aantal = 0
    empty_tile = torch.tensor([1., 0., 0., 0.])
    wall_tile = torch.tensor([0., 1., 0., 0.])
    entry_tile = torch.tensor([0., 0., 1., 0.])
    exit_tile = torch.tensor([0., 0., 0., 1.])
    for i in range(0, 31):
        for j in range(0, 31):
            index = (124 * i) + (4 * j)
            tile = level[index:index+4].detach()
            if i==0 and j==1:
                if not torch.equal(tile, entry_tile):
                    aantal += 1
            elif i==30 and j==29:
                if not torch.equal(tile, exit_tile):
                    aantal += 1
            elif i==1 and j==1:
                if not torch.equal(tile, empty_tile):
                    aantal += 1
            elif i==29 and j==29:
                if not torch.equal(tile, empty_tile):
                    aantal += 1
            elif i==0:
                if not torch.equal(tile, wall_tile):
                    aantal += 1
            elif i==30:
                if not torch.equal(tile, wall_tile):
                    aantal += 1
            elif j==0:
                if not torch.equal(tile, wall_tile):
                    aantal += 1
            elif j==30:
                if not torch.equal(tile, wall_tile):
                    aantal += 1
            elif (not(torch.equal(tile, empty_tile))) and (not(torch.equal(tile, wall_tile))):
                aantal += 1
    return aantal


def gemiddeld_aantal_fouten(batch):
    batch_size = 40
    fouten = 0
    for maze in batch:
        fouten += aantal_fouten(maze)
    return fouten / batch_size



if __name__=='__main__':
    #bat = torch.tensor([[1,2,3], [0,0,0], [0,0,0]])
    #print("var; ", L1_norm(bat))
    # noise settings
    #manualSeed = np.random.randint(1, 10000)
    #random.seed(manualSeed)
    #torch.manual_seed(manualSeed)


    #make generator and load learned weights
    gen = generator(100,3844)
    #gen.load_state_dict(torch.load('generator_weights_epoch_9.pth'))
    #gen.load_state_dict(torch.load('trained_model_weights/generator_weights_150_epoch.pth'))
    #gen.load_state_dict(torch.load('nieuwe opmaak 33 epochs/generator_weights_epoch_19.pth'))
    #gen.load_state_dict(torch.load('3.4_corrected/generator_weights_280.pth'))
    #gen.load_state_dict(torch.load('3.2_corrected/generator_weights_' + str(375) + '.pth'))
    gen.eval()

    # test corrected for constraint 2
    """gen.load_state_dict(torch.load('control_batches/control_weights_control_batches_' + str(800) + '.pth'))
    noise = torch.rand(40, 100)
    samples = gen(noise)
    corr = correct_for_constraint_1_and_2(samples[1])
    print("fouten: ", aantal_fouten(samples[1]))
    plot_img(samples[1])
    #plot_img(corr)
    print("end")
    corrected_samples = []
    for maze in samples:
        corrected_samples.append(correct_for_constraint_1_and_2(maze))

    score = L1_norm(corrected_samples)
    print("L1: ", L1_norm(samples))
    print("L1 corrected: ", score)
    print("con2: ", percent_constraint1_and_2(samples))
    print("con2 corrected: ", percent_constraint1_and_2(corrected_samples))"""

    #verzamel data gemiddelde van 400 levels (40 batch size en 10 batches)
    datalist = []
    list = []
    for q in range(0,3800,50):    #3800 voor 15 epoch
        list.append(q)
    print("batches x-as: ", list)
    for j in range(0,3800,50):
        #gen.load_state_dict(torch.load('b5/b5_weights_batches_' + str(j) + '.pth'))
        #gen.load_state_dict(torch.load('constraint1&2_0.03/g0.03_weights_batches_' + str(j) + '.pth'))
        gen.load_state_dict(torch.load('control_batches/control_weights_control_batches_' + str(j) + '.pth'))
        #gen.load_state_dict(torch.load('all4/all4_weights_batches_' + str(j) + '.pth'))
        #gen.load_state_dict(torch.load('t1_weights_batches_' + str(j) + '.pth'))
        #gen.load_state_dict(torch.load('control_50/control_weights_50_batches_' + str(j) + '.pth'))
        data = 0
        for i in range(1):
            noise = torch.rand(40, 100)
            samples = gen(noise)
            corrected_samples = correct_batch(samples)
            #score = fl.test_constraint2(samples, 0)
            #score = L1_norm(corrected_samples)
            score = gemiddeld_aantal_fouten(samples)
            #score = percent_constraint1_and_2(samples)
            #score = percent_constraint3(samples)
            #data += score * 0.1
            data += score
        #print("enkel sample: " + str(data.item()))
        #datalist.append(data.item())   # uncomment voor constraint
        datalist.append(data)
    print("gemiddelde correctheid constraint y-as:", datalist)



