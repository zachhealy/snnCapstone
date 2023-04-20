#Imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils

import tonic

import torch
dtype = torch.float
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

# Neuron Creation Function
def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
  spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
  mem = beta * mem + x*w - spk*threshold
  return spk, mem

# set neuronal parameters
delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau)

# Neuron Constraint Loop and Generation
num_steps = 200
data = np.random.choice([0, 1], size=(10, 10), p=[.1, .9])
mem = torch.rand((10, 10), dtype=dtype) * 0.5

w = 0.5
w2 = 0
x = torch.cat((torch.zeros(5), torch.ones(num_steps - 5) * 0.5), 0)
beta = 0.819
spk = mem 
mem_rec = []
spk_rec = []
uniquePairs = []
pair = []
same = False

for step in range(num_steps):
    for i in range(len(data)):
        for j in range(len(data[i])):
            hasPair = False
            #upper pair
            if i > 0 and hasPair == False:
                if data[i][j] == 1 and data[i-1][j] == 1:
                    hasPair = True

            #lower pair
            if i < len(data)-1 and hasPair == False:
                if data[i][j] == 1 and data[i+1][j] == 1:
                    hasPair = True

            #left pair
            if j > 0 and hasPair == False:
                if data[i][j] == 1 and data[i][j-1] == 1:
                    hasPair = True

            #right pair
            if j < len(data[i])-1 and hasPair == False:
                if data[i][j] == 1 and data[i][j+1] == 1:
                    hasPair = True

            #Neuron Generation
            #Adjacent Pair found so Weight is normal
            if(hasPair == True):
                spk[i], mem[i] = leaky_integrate_and_fire(mem[i], x[step], w=w, beta=beta)
                mem_rec.append(mem[i])
                spk_rec.append(spk[i])

            #Adjacent Pair not Found so Weight is set to 0
            else:
                spk[i], mem[i] = leaky_integrate_and_fire(mem[i], x[step], w=w2, beta=beta)
                mem_rec.append(mem[i])
                spk_rec.append(spk[i])

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

memString = ""
spkString = ""    


for i in mem_rec:
    memString += str(i) + ", "  

for i in spk_rec:
    spkString += str(i) + ", "

print(memString)
print("")
print("------------------------------------------------------------------------------------------------------------")
print("")
print(spkString)

# fig, ax= plt.subplots()

# #  s: size of scatter points; c: color of scatter points
# ax.imshow(mem_rec, cmap="tab20c", interpolation="nearest", aspect="auto")
# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

