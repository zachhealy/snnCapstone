import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

class SpikingNeuron:
    def __init__(self, a = 0.02, b = 0.2, c = -65, d = 2, p=0, initial_v_t=-65):
        '''
        Parameters in the model
        a: the time scale of the recovery variable. Smaller values result in slower recovery (typical value=0.02)
        b: the sensitivity of the recovery variable to the subthreshold fluctuations of the membrane potential (typical value=0.2)
        c: the after-spike reset value of the membrane potential caused by the fast high-threshold K+ conductances (typical value=-65)
        d: after-spike reset of the recovery variable caused by slow high-threshold Na+ and K+ conductances (typical value=2)
        I: Synaptic currents or injected dc-currents,  Increasing the strength of the injected dc-current increases the interspike frequency
        p: Internal Pacemaker increase in potential.
        
        Variables in the model
        v_t: the membrane potential of the neuron at time t
        u_t: the membrane recovery variable,it provides negative feedback to v
        Based on the neuronal model found in https://courses.cs.washington.edu/courses/cse528/07sp/izhi1.pdf
        '''
        # get the static parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.I = 0
        self.p = p
        self.initial_v_t = initial_v_t

        # reset the dynamic parameters
        self.reset()


    def reset(self):
        '''
        Reset the neuron to its default state
        '''
        self.v_t = self.initial_v_t
        self.u_t = self.b*self.v_t
        self.v_history = [self.v_t]


    def set_input(self, i):
        '''
        Set the inputs to this neuron to i.
        '''
        self.I = i


    def add_input(self, i):
        '''
        Add i to the set of existing inputs (for cumulative input).
        '''
        self.I += i

  
    def simulate(self):
        '''
        simulate the defined model of a single neuron over a 1ms time step, adding to the potential history
        v_history.
        Returns the potential of the cell.
        '''

        # if the membrane potential is less than 30mV (before-spike stage), 
        # adjust the membrane potential v_t according to the two differential equations defined in Izhikevich's simple Model of Spiking Neurons
        if (self.v_t<30):
            self.v_t = self.v_t + 0.5*(0.04*self.v_t**2 + 5*self.v_t + 140 - self.u_t + self.I + self.p)
            self.v_t = self.v_t + 0.5*(0.04*self.v_t**2 + 5*self.v_t + 140 - self.u_t + self.I + self.p)

            self.u_t = self.u_t + self.a*(self.b*self.v_t - self.u_t)
        
        # if the membrane potential reach its apex at 30mv (after-spike stage), 
        # reset the membrane potential and recovery variable using parameters c and d
        else:
            self.v_t = self.c
            self.u_t = self.u_t+self.d

        # add the calculated membrane potential at the time interval t
        #self.v_history.append(self.v_t)

        # reset inputs
        self.I = 0
               
        return self.v_t


class RandomIzSNN:
    """
    This constructs a random SNN with a given number of neurons. The network replicates the random
    network in Izhikevich 2003.
    Parameters:
      - ne: Number of excitatory neurons.
      - ni: Number of inhibatory neurons.
    """
    def __init__(self, ne=800, ni=200, rng=default_rng()):
        # get the parameters
        self.rng = rng
        self.ne = ne
        self.ni = ni

        #generate the neurons
        ae,be,ce,de = self.random_excitatory(ne)
        ai,bi,ci,di = self.random_inhibitory(ni)
        self.A = np.concatenate((ae,ai))
        self.B = np.concatenate((be,bi))
        self.C = np.concatenate((ce,ci))
        self.D = np.concatenate((de,di))

        # generate the neuron synaptic weights
        n = self.ne + self.ni
        se = 0.5*self.rng.random((n, self.ne))
        si = -self.rng.random((n, self.ni))
        self.S = np.concatenate((se, si), axis=1)

        # initially, we have no history
        self.history = None


    def random_excitatory(self, n):
        """
        Return a random excitatory neuron using biologically plausible parameters.
        """
        re = self.rng.random(n)
        a = 0.02 * np.ones(n)
        b = 0.2 * np.ones(n)
        c = -65 + 15 * re**2
        d = 8 - 6 * re**2

        return (a,b,c,d)


    def random_inhibitory(self,n):
        """
        Return a random inhibitory neuron using biologically plausible parameters.
        """
        ri = self.rng.random(n)
        a = 0.02 + 0.08*ri
        b = 0.25-0.05*ri
        c = -65*np.ones(n)
        d = 2*np.ones(n)

        return (a,b,c,d)



    def compute_input(self, t, fired):
        """
        Compute the neural inputs and return as an array.
        For time t.
        """
        #start with thalemic noise
        I = np.concatenate((5*self.rng.normal(size=self.ne), 2*self.rng.normal(size=self.ni)))
        if (len(fired.shape)==0): # in case fired is an np scalar, thus fired.shape = ()
          fired = np.array([fired]) 

        I += np.sum(self.S[:,fired], axis=1)
        return I


    def compute_initial(self):
        """
        Compute initial state for each neuron
        """
        return -65*np.ones(len(self.A))


    def compute_fired(self, t):
        """
        Return a list of neurons that fired at time t.
        """
        return np.argwhere(self.history[:,t-1] >= 30).squeeze()


    def simulate(self, ms=1000, show_updates=True):
        """
        Run the network simulation and return the v_t history of the neurons.
         - ms = The number of milliseconds to simulate.
        """

        # create the history matrix
        self.history = np.zeros((len(self.A), ms))

        # set the initial state
        self.history[:,0] = self.compute_initial()
        u = self.B * self.history[:,0]

        # run the simulation
        for t in range(1,ms):
            # compute the fired neurons
            fired = self.compute_fired(t)
            
            # compute neural state
            v = np.copy(self.history[:,t-1])
            v[fired] = self.C[fired]
            u[fired] = u[fired] + self.D[fired]
            I = self.compute_input(t, fired)
            v = v + 0.5*(0.04*v**2 + 5 * v + 140 - u+I)
            v = v + 0.5*(0.04*v**2 + 5 * v + 140 - u+I)
            u = u + self.A*(self.B*v - u)

            self.history[:, t] = v
            if show_updates and t % 1000 == 0:
                print("T=%d ms out of %d ms" % (t, ms))

        return self.history

class RandomGritsun (RandomIzSNN):
    def __init__(self, ne=800, ni=200, npm=0, jmax=1000, rng=default_rng()):
        '''
        Construct a random SNN with a given number of neruons. The network replicates
        the random networks in Gritsun 2010.
          - ne: Number of excitatory neurons.
          - ni: Number of inhibatory neurons.
          - npm: Number of the neurons which are inherently bursting "pacemakers"
          - jmax: Maximum frequency of thalamic noise in Hz. 
          - rng: The random number generator to use.
        '''
        super().__init__(ne, ni, rng)
        self.npm = npm
        self.jmax = jmax

        # compute the noise frequencies
        n = self.ne + self.ni
        self.J = self.jmax * abs(self.rng.normal(size=n))

        # compute the noise periods
        jnzs = np.argwhere(self.J != 0).squeeze()
        p = np.zeros(self.J.shape) 
        p[jnzs] = 1000/self.J[jnzs]
        self.jpd = p.astype(int)
        self.jnzs = np.argwhere(self.jpd !=0).squeeze()

        # compute the pacemaker frequencies
        self.pmf = np.zeros(n)
        pmnzs = self.rng.choice(n, size=npm, replace=False)
        self.pmf[pmnzs] = 0.26 * abs(self.rng.normal(size=self.npm))
        p = np.zeros(self.J.shape)
        p[pmnzs] = 1000/self.pmf[pmnzs]
        self.pmpd = p.astype(int)
        self.pmnzs = np.argwhere(self.pmpd !=0).squeeze()


    def compute_input(self, t, fired):
        """
        Compute the neural inputs and return as an array.
        For time t.
        """
        #start with thalemic noise
        #I = np.concatenate((5*self.rng.normal(size=self.ne), 2*self.rng.normal(size=self.ni)))
        I = np.zeros(self.ne + self.ni)

        if (len(fired.shape)==0): # in case fired is an np scalar, thus fired.shape = ()
          fired = np.array([fired]) 

        I += np.sum(self.S[:,fired], axis=1)
        return I


    def compute_fired(self, t):
        """
        Return a list of neurons that fired at time t.
        """

        # handle the pacemaker firings by counting the period
        p = np.ones(self.pmpd.shape)
        p[self.pmnzs] = t % self.pmpd[self.pmnzs]
        pmfired = np.argwhere(p == 0).squeeze()
        self.history[pmfired, t-1] = 30

        return super().compute_fired(t)


def save_fig(plt, name):
    plt.savefig(name)
    print("Saved %s"%(name,))


def draw_spiking_pattern(times, membrane_potentials, save=True):
    '''
    draw the spiking pattern
    times: is a time series e.g. [1...1000] - used as the x variable in the plot
    membrane_potentials: is a list of membrane potentials corresponding to times - used as the y variable in the plot
    '''
    fig, ax = plt.subplots()
    ax.plot(times, membrane_potentials)
    ax.set(xlim=(0, 1000), ylim=(-80, 50))

    plt.gcf().set_size_inches(18, 5)
    if save:
        save_fig(plt, 'spiking-pattern.png')
    else:
        plt.show()



def draw_spike_raster(history, firing_level = 30, save=True):
    img_data = (history >= firing_level) * 1

    fig,ax = plt.subplots()
    ax.imshow(img_data, cmap="Greys", interpolation="nearest", aspect="auto")
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron Number')
    if save:
        save_fig(plt, 'spike-raster.png')
    else:
        plt.show()


def draw_spike_scatter(history, firing_level = 30, save=True):
    x=np.empty(0)
    y=np.empty(0)
    for i in range(history.shape[1]):
        tfire = np.argwhere(history[:,i] >= firing_level).squeeze()
        if len(tfire.shape):
            x = np.concatenate( (x, i * np.ones(tfire.shape[0])))
            y = np.concatenate( (y, tfire) )

    fig,ax = plt.subplots()
    ax.scatter(x=x, y=y, s=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron Number')
    if save:
        save_fig(plt, 'spike-scatter.png')
    else:
        plt.show()


def compare_patterns(models, compare_against='a'):
    '''
    class method, draw and compare different model spiking patterns 
    '''
    fig, axs = plt.subplots(ncols=2, 
                            nrows=int((len(models)+1)/2), 
                            sharex=True, 
                            sharey=True, 
                            figsize= [15, 10],
                            gridspec_kw = {'wspace': 0.05}
                            )
    fig.text(0.5, 0.04, 'At time t(ms)', ha='center')
    fig.text(0.04, 0.5, 'Membrane Potential, variable v (mV)', va='center', rotation='vertical')
    
    for i in range(len(models)):
        model = models[i]
        x = int(i/2)
        y = int(i%2)
        print(axs)
        axs[x,y].plot(model[0], model[1])
        axs[x,y].set(xlim=(0, 1000), 
                     ylim=(-80, 30), 
                     title=f'{compare_against}={model[2]}',
                    )
        
    plt.show()


if __name__ == '__main__':
    net = RandomGritsun(jmax=0, npm=400)
    draw_spike_scatter(net.simulate(60000))
