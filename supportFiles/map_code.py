import numpy as np
from supportFiles import calibMatrix
import kernelEig
def map_code(ref,rsize,csize):

    kernel = [6,6]
    eigvalThresh = 0.02
    eigValThresh2 = 0.95

    A = calibMatrix.calibMatrix(ref,kernel)
    [row,col,nc] = np.shape(A)
    A = np.reshape(A,[row,col*nc],order='F')
    # print A[3,4]
    # import sys
    # sys.exit()
    [U,S,V] = np.linalg.svd(A,full_matrices=False)
    # V = np.conj(np.transpose(V))
    print V[2,0]
    import sys
    sys.exit()
    k = np.reshape(V, (kernel[0],kernel[1],nc,(np.shape(V)[1])),order = 'F')
    cutoff = int(max(np.argwhere(S>=S[0]*eigvalThresh)))
    k = k[:,:,:,0:cutoff+1]
    print np.shape(k)
    import sys
    sys.exit()
    [M, W] = kernelEig.kernelEig(k, [rsize, csize])

    # print (np.argwhere(S>=S[0]*eigvalThresh))
    import sys
    sys.exit()
    # return