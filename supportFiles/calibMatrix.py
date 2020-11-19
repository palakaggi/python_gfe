import numpy as np

def calibMatrix(Data,kSize):

    [sx,sy,sz] = np.shape(Data)
    # print (sx,sy,sz)
    # import sys
    # sys.exit()
    A = np.zeros(((sx-kSize[0]+1)*(sy-kSize[1]+1),np.prod(kSize),sz),dtype=complex)
    # print(np.shape(A))
    count = 0
    for y in range(kSize[1]):
        for x in range(kSize[0]):
            dim1 = []
            dim2 = []
            for i in range(x,sx-kSize[0]+x+1):
                dim1.append(i)
            for i in range(y, sy-kSize[1]+y+1):
                dim2.append(i)
            to_be = np.zeros((len(dim1), len(dim2), np.shape(Data)[2]), dtype=complex)
            for i in range(len(dim1)):
                for j in range(len(dim2)):
                    to_be[i][j] = Data[dim1[i]][dim2[j]]
            A[:,count, :] = np.reshape(to_be,((sx-kSize[0]+1)*(sy-kSize[1]+1),sz), order= 'F')
            count += 1

    return A