import numpy as np
from scipy import interpolate

def fix_aspect_ratio(UI,SOS_im):
    ypixel = UI['fovY']/np.shape(SOS_im)[0]
    xpixel = UI['fovX']/np.shape(SOS_im)[1]
    R = xpixel / ypixel

    if R<1:
        y = list(np.transpose(x for x in range(0,np.shape(SOS_im)[0])))
        yi = list(np.transpose(n for n in range(0,np.shape(SOS_im)[0],R)))
        x = list(np.transpose(i for i in range(0, np.shape(SOS_im)[1])))
        xi = x
    else:
        x = np.transpose(list(m for m in range(0,np.shape(SOS_im)[1])))
        xi = np.transpose(list(m for m in range(0,np.shape(SOS_im)[1],int(1/R))))
        y = np.transpose(list(x for x in range(0,np.shape(SOS_im)[0])))
        yi = y

    [X,Y] = np.meshgrid(x,y)
    [Xq,Yq] = np.meshgrid(xi,yi)
    print(Xq[19,75])
    fixed_image = interpolate.interp2d(X,Y,SOS_im,kind= 'linear')
    # Zi=[]
    # Zi = fixed_image(Xq,Yq)
    # for i,j in zip(Xq,Yq):
    #     Zi.append(fixed_image(i,j))

    # print(type(fixed_image))
    # print(np.shape(Zi))

