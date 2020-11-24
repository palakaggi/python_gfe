import numpy as np
from scipy import fftpack

def zpad(arr,size):
    # ans = np.zeros(size,dtype=complex)
    padDim1 = int((size[0]-np.shape(arr)[0])/2)
    # print(padDim1)
    padDim2 = int((size[1] - np.shape(arr)[1])/2)
    # print(padDim1)
    padDim3 = int((size[2] - np.shape(arr)[2])/2)
    # print(padDim3)
    ans = np.pad(arr,((padDim1,padDim1),(padDim2,padDim2),(padDim3,padDim3)),'constant')
    return ans

def kernelEig(kernel,imSize):

    nc = np.shape(kernel)[2]
    nv = np.shape(kernel)[3]
    kSize = [np.shape(kernel)[0],np.shape(kernel)[1]]

    # "rotate kernel to order by maximum variance"
    k = np.moveaxis(kernel, (0,1,2,3),(0, 1, 3, 2))
    k = np.reshape(k,(np.prod(kSize)*nv,nc),order='F')

    if np.shape(k)[0] < np.shape(k)[1]:
        [u, s, v] = np.linalg.svd(k)
    else:
        [u,s,v] = np.linalg.svd(k,full_matrices=False)
    v= np.conj(np.transpose(v))
    k = np.dot(k,v)

    kernel = np.reshape(k,(kSize[0],kSize[1],nv,nc), order= 'F')
    kernel = np.moveaxis(kernel,(0,1,2,3),(0,1,3,2))

    KERNEL = np.zeros((imSize[0],imSize[1], np.shape(kernel)[2],np.shape(kernel)[3]),dtype=complex)

    for n in range(np.shape(kernel)[3]):
        Temp = zpad(np.conj(kernel[::-1, ::-1, :, n]), (imSize[0], imSize[1], np.shape(kernel)[2]))
        print(Temp[125,125,:])
        # import sys
        # sys.exit()
        KERNEL[:,:,:,n] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Temp)))
        # Temp = (Temp)
        print(np.shape(KERNEL))
        print (KERNEL[0,0,0,0])
        import sys
        sys.exit()
        Temp = np.fft.fftshift(Temp)
        print (Temp[125][125][0])

        import sys
        sys.exit()
        KERNEL[:, :, :, n] = np.fft.fftshift(temp)
        # KERNEL[:,:,:,n] = np.fft.fftshift(np.fft.fft(np.fft.fftshift()))
        print (KERNEL[0,0,0,0])
        import sys
        sys.exit()


    EigenVecs = np.zeros((imSize[0], imSize[1], nc, min(nc, nv)),dtype=complex)
    EigenVals = np.zeros((imSize[0], imSize[1], min(nc, nv)),dtype=complex)

    return EigenVecs, EigenVals


