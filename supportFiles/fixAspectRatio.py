import numpy as np
from scipy import interpolate

def fix_aspect_ratio(UI, SOS_im):
    ypixel = UI['fovY'] / np.shape(SOS_im)[0]
    xpixel = UI['fovX'] / np.shape(SOS_im)[1]
    R = xpixel / ypixel

    if R < 1:
        y = list(np.transpose(x for x in range(0, np.shape(SOS_im)[0])))
        yi = list(np.transpose(n for n in range(0, np.shape(SOS_im)[0], R)))
        x = list(np.transpose(i for i in range(0, np.shape(SOS_im)[1])))
        xi = x
    else:
        x = np.transpose(list(m for m in range(0, np.shape(SOS_im)[1])))
        xi = np.transpose(list(m for m in range(0, np.shape(SOS_im)[1], int(1 / R))))
        y = np.transpose(list(x for x in range(0, np.shape(SOS_im)[0])))
        yi = y
    # print(np.shape(x))
    [X, Y] = np.meshgrid(x, y)
    # print(Y[19,75])
    # [Xq, Yq] = np.meshgrid(xi, yi)
    fixed_image = interpolate.RectBivariateSpline(x, y, SOS_im)
    # print(np.shape(X))

    return np.transpose(fixed_image.ev(X, Y))