"""
A simple implementation of Lindenmayer Systems. Along with some
functions for filling a grid using a simple graphical language.
"""
import numpy as np
import matplotlib.pyplot as plt

def expand(rules, axiom, n=10):
    '''
    Expand an l-system from axiom for a fixed number of iterations.

    Parameters:
        rules - A dictionary where A->B is represented as rules[A] -> B
                Where A is the predacessor and B is the successor.
                Note that this implementation assumes a single-character predacessor

        axiom - The initial string to expand.

        n - The number of iterations to expand

    Returns:
        The string after n expansions.
    '''
    for i in range(n):
        result = []
        for c in axiom:
            # either replace or keep
            if c in rules:
                result.append(rules[c])
            else:
                result.append(c)
        # end for
        axiom = ''.join(result)
    # end for

    return axiom



def fill_grid(cmd, shape=(100,100), pos=(0,0), angle=0): 
    '''
    This function creates a two dimensional array filled with 0s. of
    size dim. It then executes the drawing commands beginning at the
    start position.

    The language recognized by the drawing commands is as follows:

        F - Draw Forward
        + - Turn right 45 degrees
        - - Turn left 45 degrees
        [ - Push position and angle on a stack
        ] - Pop position and angle from the stack

    All other symbols are simply ignored.

    The drawing angles follow a grid pattern and are given:

         
       135   90   45
           +---+
       180 |   |  0
           +---+
       225  270  315

    These are the only angles at which the drawing can move.
    Negative angles are corrected to go in the opposite direction:
       -45 = 315
       -90 = 270
    And so on.

    Parameters:
        cmd   - The command string
        shape - The shape of the resultant array (rows, cols)
        pos   - Starting position
        angle - Starting angle
    '''

    # initialize
    grid=np.zeros(shape=shape)
    pos = np.array(pos)
    shape = np.array(shape)
    lbound = np.array((0,0))
    stack = []

    # set up movements
    move = {   0: np.array((0,1)),
              45: np.array((-1,1)),
              90: np.array((-1,0)),
             135: np.array((-1,-1)),
             180: np.array((0,-1)),
             225: np.array((1,-1)),
             270: np.array((1,0)),
             315: np.array((1,1)) }

    # go through each command
    for c in cmd:
        # correct angle
        if angle < 0:
            angle = angle % -360 + 360
        else:
            angle = angle % 360

        if c == 'F':
            # draw forward
            pos2 = pos+move[angle]
            for p in (pos, pos2):
                if (p >= lbound).all() and (p < shape).all():
                    grid[p[0], p[1]] = 1
            pos = pos2
        elif c == '+':
            # turn right 45 degrees
            angle -= 45
        elif c == '-':
            # turn left 45 degrees
            angle += 45
        elif c == '[':
            # push current position and angle
            stack.append((np.array(pos), angle))
        elif c == ']':
            # pop current position and angle
            pos, angle = stack.pop()
    return grid


if __name__ == '__main__':
    rules = {'B': 'F[-B]+B',
             'F': 'FF' }
    cmd = expand(rules, 'B', 6)
    grid = fill_grid(cmd, shape=(100,100), pos=(99,0), angle=45)
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="Greys", interpolation="nearest", aspect="auto")
    plt.show()
