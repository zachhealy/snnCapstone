"""
Experiments in Cayley Tree implementation.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

def rotation_angle_offsets(z, alpha):
    """
    Parameters
        z     - number of branches
        alpha - total sweep angle
    Returns 
        A list of heading offsets.
    """
    # compute basic parameters
    nleft = math.ceil(z/2)
    nright = math.floor(z/2)
    ha = alpha/2
    la = alpha/nleft/2
    ra = alpha/nright/2

    # start the result
    result = []
    
    #compute the left hand side
    for i in range(0,nleft):
        # an = alpha/2 - (n-1) alpha/nleft
        an = ha - i * la
        result.append(an)
    
    # compute the right hand side
    for i in range(0,nright):
        # an = -alpha/2 + (n-1) alpha/nright
        an = -ha + i * ra
        result.append(an)
    
    return result


def growth_step(sites, grid):
    """
    Perform a growth step according to the list of sites
    """
    pass


def compute_grid_step(a):
    """
    Computes the grid-wise dx,dy for the given angle.
    Parameter
        a - The angle in radians
    
    Returns
        dx - Step size in the x direction
        dy - Step size in the y direction
    """
    sa = round(math.sin(a), 10)
    ca = round(math.cos(a), 10)

    if sa == 0:
        ds = math.inf
    else:
        ds = 1/sa
    
    if ca == 0:
        dc = math.inf
    else:
        dc = 1/ca

    d = min(abs(dc), abs(ds))


    return d * ca, d*sa


def vacant_neighbors(grid, x, y):
    """
    Returns a list of tuples representing the coordinates of vacant
    squares on the grid. This computes the 9 square neighborhood of
    the x,y coordinate.
          ***
          *O*
          ***
    Of course, items on extreme edges may not have all these
    neighbors.
    Parameters
        grid - The numpy array containing the grid
        x    - The x coordinate of the grid square.
        y    - The y coordinate of the grid square.
    Returns
        A list of tuples (x,y) of vacant squares.
    """
    # compute the extremes of the potential neighborhood
    xmin = max(0, x-1)
    ymin = max(0, y-1)
    xmax = min(grid.shape[0]-1, x+1)
    ymax = min(grid.shape[1]-1, y+1)

    # find the vacant squares
    result = []
    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            if grid[y,x] == 0:
                result.append((x,y))
    return result


class GrowthSite:
    def __init__(self, grid, x, y, a, z, l, ba, tn=1):
        """
        Create a growth site to grow a Cayley tree.
        Parameters
            grid - The numpy array containing the grid
            x    - The x coordinate of the growth site
            y    - The y coordinate of the growth site
            a    - The heading of the growth site in radians
            z    - The number of branches to create at split
            ba   - Total angle swept by the extreme branches
            tn   - The unique identifier for this tree and its 
                   branches.
        """
        # the straight up copies
        self.grid = grid
        self.x = x
        self.y = y
        self.a = a
        self.z = z
        self.l = l
        self.ba = ba
        self.tn = tn

        # the slope factors
        self.dx, self.dy = compute_grid_step(self.a)

        # the count down, and checking for trapped condition
        self.counter = l
        self.trapped = False


    def growth_step(self):
        """
        Grow unless we have exhausted our counter.
        """

        # if we have hit our growth limit, we do nothing
        if not self.trapped and self.counter <= 0:
            return
        
        # decrease our step counter and move
        self.counter -= 1
        x = self.x + self.dx
        y = self.y + self.dy

        # adjust for the edge of the grid
        x = max(0, x)
        y = max(0, y)
        x = min(self.grid.shape[0]-1, x)
        y = min(self.grid.shape[1]-1, y)

        # check for an occupied square
        gx = int(x)
        gy = int(y)
        if self.grid[gy,gx] == 0:
            self.x = x
            self.y = y
        else:
            available = vacant_neighbors(self.grid, gx, gy)
            if len(available) == 0:
                # end of the line!
                self.trapped = True
                return

            # move to a random element
            self.x, self.y = random.choice(available)

    

    def branch(self):
        """
        If we have exhausted our counter, return the next set of branches.
        Otherwise do nothing.
        """
        if not self.trapped and self.counter > 0:
            return []

        # build the branches
        result = []
        for da in rotation_angle_offsets(self.z, self.ba):
            result.append(GrowthSite(self.grid, 
                                     self.x, self.y,
                                     self.a + da, 
                                     self.z, self.l, self.ba, self.tn))
        
        return result


def grow(grid, sites, gen):
    """
    Grow on the grid from the list of sites.
    
    Parameters
        grid  - A numpy array where we are growing.
        sites - A list of growth sites
        gen   - The number of branching generations to run.
    """
    counter = gen

    while counter > 0:
        branched = False
        nextgen = []
        for site in sites:
            # grow and then add to the grid
            site.growth_step()
            x,y = int(site.x), int(site.y)
            if not site.trapped:
                grid[y,x] = site.tn 

            # get ready fo the next generation
            if site.counter == 0:
                branched = True
                nextgen += site.branch() 
            else:
                nextgen.append(site)
        if branched:
            counter -= 1
        sites = nextgen


def main():
    if len(sys.argv) != 6:
        sys.stderr.write(f'\nUsage: {sys.argv[0]} distance branches branch_angle iterations neuron_count\n\n')
        return
    grid=np.zeros(shape=(1000,1000))
    l = int(sys.argv[1])
    z = float(sys.argv[2])
    ba = math.radians(float(sys.argv[3]))
    gen = int(sys.argv[4])
    nn = int(sys.argv[5])

    # generate the neurons
    sites = []
    for i in range(nn):
        x = random.randrange(0, grid.shape[0])
        y = random.randrange(0, grid.shape[1])
        a = math.radians(random.random() * 365)
        sites.append(GrowthSite(grid, x, y, a, z, l, ba, i+1))

    grow(grid, sites, gen)
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="tab20c", interpolation="nearest", aspect="auto")
    plt.show()

    np.save("D:\outfile.npy", grid)

if __name__ == '__main__':
    main()
