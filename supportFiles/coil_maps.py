import numpy as np
import math
from supportFiles.map_code import map_code
import time

def coil_maps(I):
    start = time.time()
    start = time.time()
    rr = 32
    cc = rr
    ncalib = [rr, cc]
    dim1 = []
    for x in range(int(math.floor(0.5 * np.shape(I)[0]) + math.ceil(-ncalib[0]/2)), int(math.floor(0.5 * np.shape(I)[0]) + math.ceil(ncalib[0]/2))):
        dim1.append(x)
    dim2 = []
    for x in range(int(math.floor(0.5 * np.shape(I)[1]) + math.ceil(-ncalib[1]/2)), int(math.floor(0.5 * np.shape(I)[1]) + math.ceil(ncalib[1]/2))):
        dim2.append(x)
    calib_data = np.zeros((len(dim1), len(dim2), np.shape(I)[2]), dtype=complex)

    for i in range(len(dim1)):
        for j in range(len(dim2)):
            calib_data[i][j] = I[dim1[i]][dim2[j]][:]

    calib_data = np.squeeze(calib_data)
    row = np.shape(I)[1]
    # col = np.shape(I)[1]
    print(time.time()-start)

    coil_maps = map_code(calib_data, row, row)
    print(time.time() - start)
    import sys
    sys.exit()
    return coil_maps