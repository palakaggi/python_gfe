import numpy as np



def imageRecombination(cmap, im):

    nCoil = np.shape(cmap)[2]
    # Image recombination with intensity correction included.
    cIm = np.zeros((np.shape(im)[0],np.shape(im)[1]),dtype=complex)
    I = np.zeros((np.shape(im)[0],np.shape(im)[1]),dtype=int)

    for count in range(0,nCoil):
        cIm = cIm+(np.conj(cmap[:,:,count])*im[:,:,count])
        I = I+abs(cmap[:,:,count])
    # print(np.shape(cIm))
    # print(cIm[23,198])
    # print(I[23,198])
    I = I*I
    cIm = abs(I*cIm)
    # print(np.shape(abs(im)))
    # print(np.sum(abs(im) * abs(im),2))
    SOS_im = np.sqrt(np.sum((abs(im) * abs(im)), 2))

    SOS_im = I*SOS_im

    # print(SOS_im[201,4])
    # import sys
    # sys.exit()
    return SOS_im,cIm