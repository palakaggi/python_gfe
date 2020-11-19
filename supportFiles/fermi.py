import numpy as np

def fermi(xdim, cutoff, trans_width, offset):
    x = [i for i in range(((-xdim) / 2 + offset),(xdim) / 2  + offset)]
    y = [i for i in range((-xdim) / 2,(xdim)/ 2 )]
    (X, Y) = np.meshgrid(x, y)
    radius = (X * X + Y * Y) ** 0.5
    ffilter = (1.0/(1+(np.exp((radius - cutoff)/trans_width))))
    return ffilter

#     FERMI creates a 2D Fermi filter.
# FFILTER = FERMI(XDIM,CUTOFF,TRANS_WIDTH) calculates a 2D
#   Fermi filter on a grid with dimensions XDIM * XDIM.
#  The cutoff frequency is defined by CUTOFF and represents
#   the radius (in pixels) of the circular symmetric function
#   at which the amplitude drops below 0.5
#  TRANS_WIDTH defines the width of the transition.
#  Author: Wally Block, UW-Madison  02/23/01.
#  Call: ffilter = fermi(xdim, cutoff, trans_width);
