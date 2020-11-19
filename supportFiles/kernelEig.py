import numpy as np
from scipy import fftpack

def zpad(arr,size):
    # ans = np.zeros(size,dtype=complex)
    padDim1 = (size[0]-np.shape(arr)[0])/2
    padDim2 = (size[1] - np.shape(arr)[1])/2
    padDim3 = (size[2] - np.shape(arr)[2])/2
    ans = np.pad(arr,((padDim1,padDim1),(padDim2,padDim2),(padDim3,padDim3)),'constant')
    return ans

def kernelEig(kernel,imSize):

    nc = np.shape(kernel)[2]
    nv = np.shape(kernel)[3]
    kSize = [np.shape(kernel)[0],np.shape(kernel)[1]]
    print(np.shape(kernel))
    import sys
    sys.exit()
    # "rotate kernel to order by maximum variance"
    k = np.moveaxis(kernel, (0,1,2,3),(0, 1, 3, 2))
    # k = np.transpose(k)
    k = np.reshape(k,(np.prod(kSize)*nv,nc),order='F')
    print np.shape(k)
    # print k[1789][7]
    import sys
    sys.exit()
    if np.shape(k)[0] < np.shape(k)[1]:
        [u, s, v] = np.linalg.svd(k)
    else:
        [u,s,v] = np.linalg.svd(k,full_matrices=False)

    v= np.conj(np.transpose(v))
    print(np.shape(k))
    print(np.shape(v))
    import sys
    sys.exit()
    k = np.dot(k,v)
    print(np.shape(k))
    import sys
    sys.exit()
    kernel = np.reshape(k,(kSize[0],kSize[1],nv,nc), order= 'F')
    kernel = np.moveaxis(kernel,(0,1,2,3),(0,1,3,2))
    print(np.shape(kernel))
    import sys
    sys.exit()
    for n in range(np.shape(kernel)[3]):
        p = np.conj(kernel[::-1, ::-1, :, n])
        print(p)
        import sys
        sys.exit()
        Temp = zpad(np.conj(kernel[::-1, ::-1, :, n]), (imSize[0], imSize[1], np.shape(kernel)[2]))
        print np.shape(Temp)
        print Temp[125][125][0]

        Temp = np.fft.fft2(np.fft.fftshift(Temp))
        # Temp = (Temp)
        print Temp[0,0,0]
        import sys
        sys.exit()
        Temp = np.fft.fftshift(Temp)
        print Temp[125][125][0]

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


