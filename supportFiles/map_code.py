import numpy as np
from supportFiles import calibMatrix
from supportFiles import kernelEig
import numpy.matlib.repmat

def map_code(ref,rsize,csize):

    kernel = [6,6]
    eigvalThresh = 0.02
    eigValThresh2 = 0.95

    A = calibMatrix.calibMatrix(ref,kernel)
    [row,col,nc] = np.shape(A)
    A = np.reshape(A,[row,col*nc],order='F')

    [U,S,V] = np.linalg.svd(A,full_matrices=False)
    V = np.conj(np.transpose(V))
    # print(V[156,182])

    k = np.reshape(V, (kernel[0],kernel[1],nc,(np.shape(V)[1])),order = 'F')
    cutoff = int(max(np.argwhere(S>=S[0]*eigvalThresh)))

    k = k[:,:,:,0:cutoff+1]
    # print(k[0,1,2,3])

    [M, W] = kernelEig.kernelEig(k, [rsize, csize])
    # print(W[44,200,:])
    # import sys
    # sys.exit()
    # maps = M[:,:,:,-1]*matlib.repmat()

    # print (np.argwhere(S>=S[0]*eigvalThresh))

