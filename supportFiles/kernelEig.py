import numpy as np
from numpy.matlib import repmat

def zpad(arr, size):
    # ans = np.zeros(size,dtype=complex)
    padDim1 = int((size[0] - np.shape(arr)[0]) / 2)
    # print(padDim1)
    padDim2 = int((size[1] - np.shape(arr)[1]) / 2)
    # print(padDim1)
    padDim3 = int((size[2] - np.shape(arr)[2]) / 2)
    # print(padDim3)
    ans = np.pad(arr, ((padDim1, padDim1), (padDim2, padDim2), (padDim3, padDim3)), 'constant')
    return ans


def kernelEig(kernel, imSize):
    nc = np.shape(kernel)[2]
    nv = np.shape(kernel)[3]
    kSize = [np.shape(kernel)[0], np.shape(kernel)[1]]
    # "rotate kernel to order by maximum variance"
    k = np.moveaxis(kernel, (0, 1, 2, 3), (0, 1, 3, 2))
    k = np.reshape(k, (np.prod(kSize) * nv, nc), order='F')
    if np.shape(k)[0] < np.shape(k)[1]:
        [u, s, v] = np.linalg.svd(k)
    else:
        [u, s, v] = np.linalg.svd(k, full_matrices=False)
    # v = np.conj(np.transpose(v))
    k = np.dot(k, v)
    kernel = np.reshape(k, (kSize[0], kSize[1], nv, nc), order='F')
    kernel = np.moveaxis(kernel, (0, 1, 2, 3), (0, 1, 3, 2))
    KERNEL = np.zeros((imSize[0], imSize[1], np.shape(kernel)[2], np.shape(kernel)[3]), dtype=complex)
    p = np.zeros((256, 256, 8), dtype=complex)
    for n in range(np.shape(kernel)[3]):
        Temp = zpad(np.conj(kernel[::-1, ::-1, :, n]), (imSize[0], imSize[1], np.shape(kernel)[2]))
        Temp = np.fft.fftshift(Temp)
        for i in range(np.shape(Temp)[2]):
            p[:, :, i] = np.fft.fft2(Temp[:, :, i])
        KERNEL[:, :, :, n] = np.fft.fftshift(p)
    EigenVecs = np.zeros((imSize[0], imSize[1], nc, min(nc, nv)), dtype=complex)
    EigenVals = np.zeros((imSize[0], imSize[1], min(nc, nv)), dtype=complex)

    for i in range(0,np.prod(imSize)):
        [x,y] = np.unravel_index(i,[imSize[0],imSize[1]],'F')
        mtx = np.squeeze(KERNEL[x,y,:,:])
        [C,D,V] = np.linalg.svd(mtx,full_matrices=False)
        # V= np.conj(np.transpose(V))
        # correction = [1,-1,1,1,-1,-1,1,1]
        for n in range(0,np.shape(C)[1]):
            C[n, :] = [C[n, j] for j in range(np.shape(C)[0])]
            # C[n,:] = np.round([C[n,j]*correction[j] for j in range(np.shape(C)[0])],4)
        cmplx=complex(0,-1)
        ph = repmat(np.exp(cmplx*np.angle(C[0,:])),np.shape(C)[0],1)
        C = np.dot(v,(C*ph))
        EigenVals[x,y,:] = D[::-1]
        EigenVecs[x, y,:,:] = C[:,::-1]
    return EigenVecs, EigenVals
