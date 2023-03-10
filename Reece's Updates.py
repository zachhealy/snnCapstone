#Imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
dtype = torch.float
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

#@title Plotting Settings
def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, ylim_max1])
  ax[0].set_xlim([0, 200])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  if title:
    ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, ylim_max2]) 
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  if thr_line:
    ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([]) 

  plt.show()

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,7), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input spikes
  splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
  ax[0].set_ylabel("Input Spikes")
  ax[0].set_title(title)

  # Plot hidden layer spikes
  splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
  ax[1].set_ylabel("Hidden Layer")

  # Plot output spikes
  splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
  ax[2].set_ylabel("Output Spikes")
  ax[2].set_ylim([0, 10])

  plt.show()

# Neuron Creation Function
def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
  spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
  mem = beta * mem + w*x - spk*threshold
  return spk, mem

# set neuronal parameters
delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau)

# Neuron Constraint Loop and Generation
num_steps = 100
data = np.random.choice([0, 1], size=(10, 10), p=[.1, .9])
# ------------ do we need num_steps here? can we just make it 10x10 and update it after each step?
# ------------ ORIGINAL CODE mem = torch.rand((num_steps, 10, 10), dtype=dtype) * 0.5
mem = torch.rand((10, 10), dtype=dtype) * 0.5

w = 0.5
w2 = 0
x = torch.cat((torch.zeros(5), torch.ones(95)*0.5), 0)
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
	    # ------------- Change from boolean to int so that multiple surrounding neurons can affect it?
            # ------------- ORIGINAL CODE hasPair = False
	    pairs = 0
            #upper pair
            if i > 0:
		# -------------- do we need the current neuron to be spiked in order to pair it? do we need to check if it's already paired?
                # -------------- ORIGINAL CODE if data[i][j] == 1 and data[i-1][j] == 1 and hasPair == False:
		    # hasPair = True
		if data[i-1][j] == 1:
                    pairs += 1
		# -------------- edit the other checks if you like this

            #lower pair
            if i < len(data)-1:
                if data[i][j] == 1 and data[i+1][j] == 1 and hasPair == False:
                    hasPair = True

            #left pair
            if j > 0:
                if data[i][j] == 1 and data[i][j-1] == 1 and hasPair == False:
                    hasPair = True

            #right pair
            if j < len(data[i])-1:
                if data[i][j] == 1 and data[i][j+1] == 1 and hasPair == False:
                    hasPair = True

	    # -------------- do we want 8 possible pairs instead of 4? i.e. upper right diagonal pairing possibility ([i-1][j+1])?

            #Neuron Generation
            #Adjacent Pair found so Weight is normal
	    # -------------- if pairs is used in place of hasPair, multiply weight by number of paired spiking neurons
	    # -------------- there is no way of updating the grid value for this neuron, so a spiked neuron will still be a 0 on the grid. add an update to the grid
	    # -------------- should we use SPECIFIC spk and mem values, so that they can be stored and updated for EACH SPECIFIC neuron instead of using them from previous neurons?
            # -------------- ORIGINAL CODE if(hasPair == True):
	    if(pairs > 0):
		# -------------------- ORIGINAL CODE spk[i], mem[i] = leaky_integrate_and_fire(mem[i], x[step], w=w, beta=beta)
                spk[i][j], mem[i][j] = leaky_integrate_and_fire(mem[i][j], x[step], w=w*pairs, beta=beta)
                mem_rec.append(mem[i][j])
                spk_rec.append(spk[i][j])
		# -------------------- added update to grid
		data[i][j] = spk[i][j]

            #Adjacent Pair not Found so Weight is set to 0
            else:
                spk[i], mem[i] = leaky_integrate_and_fire(mem[i], x[step], w=w2, beta=beta)
                mem_rec.append(mem[i])
                spk_rec.append(spk[i])
		# -------------------- if updating grid, do so here as above

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