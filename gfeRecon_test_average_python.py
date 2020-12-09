from mat4py import loadmat
import datetime
import numpy as np
import math
import cv2
from supportFiles import fermi, coil_maps, image_recombination, fixAspectRatio
from scipy import interpolate
from skimage import *

from matplotlib import pylab
from pylab import *

now = datetime.datetime.now()

# DEFINING VARIABLES AND LOADING DATA
globalParameters = loadmat('globalParameters.mat')
data = loadmat('recon_data.mat')
UI = data['UI']
yres = UI['resolutionY']
xres = float(float(UI['resolutionX']) / float(UI['BW'])) * float(globalParameters['sampling_rate'])

oversample_factor = float(globalParameters['ONE_MHz']) / float(UI['BW'])
num_channels_per_card = 8  # Spine array
chassis_cards = 3  # Each Rx card in the chassis generates an independent file
total_channels = chassis_cards * num_channels_per_card
samples_per_pt = 2  # I and Q
sample_skip = num_channels_per_card * samples_per_pt
number_of_slices = data['number_of_slices']
temp_val = xres * yres * number_of_slices
file_size = int(data['recon_file_size'] / 4)
bdata = np.zeros((chassis_cards, file_size), int)
sig = np.zeros((int(total_channels), int(temp_val)), dtype='float')
sig_complex = np.zeros((int(total_channels), int(temp_val)), dtype='float')
size_1 = np.shape(sig_complex)
# print(sig[0][0])
# ASSIGNING VALUES TO BDATA
Recon_resolution = 1024
handle = open('rx_data_0.bin', "rb")
bdata[0, :] = np.fromfile(handle, dtype='<i4')
handle.close()

handle = open('rx_data_1.bin', "rb")
bdata[1, :] = np.fromfile(handle, dtype='<i4')
handle.close()

handle = open('rx_data_2.bin', "rb")
bdata[2, :] = np.fromfile(handle, dtype='<i4')
handle.close()

# FILLING UP SIG AND PROCESSING SIG
for rx_card in range(chassis_cards):
    for num_channels in range(0, num_channels_per_card):
        sig[num_channels + rx_card * num_channels_per_card, :] = bdata[
            rx_card, np.arange(0 + num_channels * 2, bdata[rx_card, :].size, sample_skip)]
        sig_complex[num_channels + rx_card * num_channels_per_card, :] = (
        bdata[rx_card, np.arange(1 + num_channels * 2, bdata[rx_card, :].size, sample_skip)])

sig_complex1 = 1j * sig_complex

# TO FIX IMAGE ORIENTATION IN UI- WE CHANGE THE DIMENSIONS OF THE SIG ARRAY
sig = np.reshape(sig, (int(total_channels), int(xres), int(number_of_slices), int(yres)), order='F')
sig = np.moveaxis(sig, (0, 1, 2, 3), (-2, -3, -1, -4))
sig_complex1 = np.reshape(sig_complex1, (int(total_channels), int(xres), int(number_of_slices), int(yres)), order='F')
sig_complex1 = np.moveaxis(sig_complex1, (0, 1, 2, 3), (-2, -3, -1, -4))


# print(sig[16][105][18][0])
# print(sig_complex1[16][105][18][0])
# import sys
# sys.exit()

def gfeSpoilPhase(nr_of_pe):
    # print (nr_of_pe)
    spoiler_angle = 55
    spoiler_angle = spoiler_angle * math.pi / 180
    angle_array = np.zeros((nr_of_pe, 1), float)
    maximum_angle = 2 * math.pi
    angle_array[0, 0] = spoiler_angle
    for counter in range(1, nr_of_pe):
        angle_array[counter, 0] = angle_array[counter - 1, 0] + (counter) * spoiler_angle
        angle_array[counter, 0] = angle_array[counter, 0] % maximum_angle
    return angle_array


# DEFINING SPOILER PHASE BASED ON VALUE OF gfeVariant:
if UI['gfeVariants'] == 'Spoilt':
    spoiler_phase = gfeSpoilPhase(yres)
elif (UI['gfeVariants'] == 'SSFP'):
    spoiler_phase = np.zeros(yres, 1)
    spoiler_phase[1::2] = math.pi
elif (UI['gfeVariants'] == 'Default'):
    spoiler_phase = np.zeros(yres, 1)

# DEFINING AVG_SIGNAL BASED ON UI.SIGNALAVERAGES:

if UI['signalAverages'] > 1:
    avg_sig_real = np.zeros((sig.shape[0], sig.shape[1], sig.shape[2], number_of_slices, UI['signalAverages']), float)
    avg_sig_real[:, :, :, :, 0] = sig
    avg_sig_complex = np.zeros(
        (sig_complex1.shape[0], sig_complex1.shape[1], sig_complex1.shape[2], number_of_slices, UI['signalAverages']),
        float)
    avg_sig_complex[:, :, :, :, 0] = sig_complex1

    for i in range(0, UI['signalAverages'] - 1):
        f = str('C:\\MRI\\scanData\\avgScanData\\' + str(i))
        handle = open((f + '\\rx_data_0.bin'))
        bdata[0, :] = np.fromfile(handle, dtype='<i4')
        handle.close()

        handle = open((f + '\\rx_data_1.bin'))
        bdata[1, :] = np.fromfile(handle, dtype='<i4')
        handle.close()

        handle = open((f + '\\rx_data_2.bin'))
        bdata[2, :] = np.fromfile(handle, dtype='<i4')
        handle.close()

        sig = np.zeros((int(total_channels), int(temp_val)), dtype=float)
        for rx_card in range(chassis_cards):
            for num_channels in range(0, num_channels_per_card):
                sig[num_channels + (rx_card) * num_channels_per_card] = bdata[
                    rx_card, np.arange(0 + num_channels * 2, bdata[rx_card].size, sample_skip)]
                sig_complex[num_channels + rx_card * num_channels_per_card, :] = (
                bdata[rx_card, np.arange(1 + num_channels * 2, bdata[rx_card, :].size, sample_skip)])

        sig = np.reshape(sig, (int(total_channels), int(xres), int(number_of_slices), int(yres)), order='F')
        sig = np.moveaxis(sig, (0, 1, 2, 3), (-2, -3, -1, -4))
        sig_complex1 = np.reshape(sig_complex1, (int(total_channels), int(xres), int(number_of_slices), int(yres)),
                                  order='F')
        sig_complex1 = np.moveaxis(sig_complex1, (0, 1, 2, 3), (-2, -3, -1, -4))

        avg_sig_real[:, :, :, :, i + 1] = sig
        avg_sig_complex[:, :, :, :, i + 1] = sig_complex1

    sig = np.squeeze(np.sum(avg_sig_real[:, :, :, :, :]) / UI['signalAverages'])
    sig_complex1 = np.squeeze(np.sum(avg_sig_complex[:, :, :, :, :]) / UI['signalAverages'])

# ADC in the Rx card introduces a DC offset that needs to be subtracted.
# After a noise scan that acquired 1000 * 256 matrix, the Dc point was
# found to be the value given next. This value must be scaled by the
# xres*yres value of each scan before its used in the foor loop below.
# DC_offset = (-7.17e3 - 1i* 7.72e3);
DC_offset = complex(-6.0726e+03, -6.3107e+03)  # Works for HNA
f_filter = fermi.fermi(np.shape(sig)[0], round(0.8 * float(np.shape(sig)[0]), 0),
                       round(0.1 * float(np.shape(sig)[0]), 0), 0)

if UI['coilType'] == 'HNA':
    if UI['sliceOrientation'] == 'Axial':
        Rx = [i for i in range(0, 8)]
        Rx.append(j for j in range(16, 24))
    elif UI['sliceOrientation'] == 'Sagittal':
        Rx = [16, 1, 2, 19, 5, 20, 23, 6]
    else:
        Rx = [i for i in range(0, 8)]
        Rx.append(j for j in range(16, 24))
else:
    if UI['sliceOrientation'] == 'Axial':
        Rx = [i for i in range(2, 6)]
        for j in range(18, 22):
            Rx.append(j)
    elif UI['sliceOrientation'] == 'Sagittal':
        Rx = [1, 3, 5, 16, 18, 20]
    else:
        Rx = [i for i in range(1, 9, 2)]
        for j in range(16, 24, 2):
            Rx.append(j)
# print(Rx)
pixvalX = -round(UI['offCenterX'] / (UI['fovX'] / UI['resolutionX']))
pixvalY = -round(UI['offCenterX'] / (UI['fovY'] / UI['resolutionY']))
if (UI['sliceOrientation'] == 'Coronal'):
    pixvalY = 0
if (UI['sliceOrientation'] == 'Sagittal'):
    pixvalX = 0

im = np.zeros((np.shape(sig)[0], np.shape(sig)[0], len(Rx)), dtype=complex)
Temp_coil_maps_input = np.zeros((np.shape(sig)[0], np.shape(sig)[0], len(Rx)), dtype=complex)
output = np.zeros((Recon_resolution, Recon_resolution, number_of_slices), float)

sig = sig[:, :, Rx, :]
sig_complex1 = sig_complex1[:, :, Rx, :]

for slice_count in range(0, number_of_slices):
    for j in range(len(Rx)):
        Temp = np.squeeze(sig[:, :, j, slice_count])
        Temp_comp = np.squeeze(sig_complex1[:, :, j, slice_count])
        Temp = np.fft.ifftshift(Temp) + np.fft.ifftshift(Temp_comp)
        Temp = np.fft.ifft2(Temp)
        Temp = np.fft.ifftshift(Temp)

        for i in range(len(Temp)):
            Temp[i] = np.roll(Temp[i], [0, 30])

        center_v = 0.5 * (Temp[int(yres / 2) - 1, int(xres / 2) - 1] + Temp[int(yres / 2) + 1, int(xres / 2) + 1])
        Temp[int(yres / 2), :] = 0.5 * (Temp[int(yres / 2), :] + Temp[int(yres / 2) - 2, :])
        Temp[int(yres / 2), int(xres / 2)] = center_v
        check = []
        for x in range(int(round(xres / 2) - round(0.5 * xres / oversample_factor)),
                       int(round(xres / 2) + round(0.5 * xres / oversample_factor))):
            check.append(x)
        Temp = Temp[:, check]

        Temp = np.fft.fftshift(Temp)
        # print(Temp[0,0])
        Temp = np.fft.fft2(Temp)
        # print(Temp[0, 0])

        Temp = np.fft.fftshift(Temp)
        Temp = f_filter * Temp
        # print(Temp[0,0])

        diag = np.array(np.squeeze(np.exp(complex(0, -1) * spoiler_phase)))

        Temp_diag = np.diag(diag)
        Temp[:, :] = np.dot(Temp_diag, Temp)

        Temp_coil_maps_input[:, :, j] = Temp
        # print Temp_coil_maps_input[112][112]

        im[:, :, j] = np.fft.ifftshift(Temp[:, :])
        im[:, :, j] = np.fft.ifft2(im[:, :, j])
        im[:, :, j] = np.fft.ifftshift(im[:, :, j])
# ===================================================================================================

cmap = coil_maps.coil_maps(Temp_coil_maps_input)

[SOS_im, C_im] = image_recombination.imageRecombination(cmap, im)

C_im = C_im / np.max(C_im)
# print(C_im[87,67])

centerX = np.round(0.5 * float(np.shape(C_im)[0])) + 1
centerY = np.round(0.5 * float(np.shape(C_im)[1])) + 1

dim1 = np.round(0.5 * np.shape(C_im)[0])
dim2 = np.round(0.5 * np.shape(C_im)[1])
C_im = C_im[int(centerX - dim1) - 1: int(centerX + dim1 - 1), int(centerY - dim2) - 1: int(centerY + dim2 - 1)]
# print(C_im[86, 202])
C_im_cv = np.float32(C_im)
# print(C_im_cv.dtype)
# print(C_im_cv[86, 202])
C_im = cv2.bilateralFilter(C_im_cv, sigmaSpace=0.5,sigmaColor=100,d=9)
# print(np.shape(C_im))
# print(C_im[86, 202])
C_im_cv = C_im.astype(np.float64)
# print(C_im_cv[86, 202])
# imshow(C_im_cv)

C_im = fixAspectRatio.fix_aspect_ratio(UI,C_im)
print(C_im[86, 202])

x = np.transpose([x for x in range(0, np.shape(C_im)[0])])
y = np.transpose([y for y in range(0, np.shape(C_im)[1])])
xq = np.transpose([x for x in np.arange(0, np.shape(C_im)[0], np.shape(C_im)[0] / Recon_resolution)])
yq = np.transpose([y for y in np.arange(0, np.shape(C_im)[1], np.shape(C_im)[1] / Recon_resolution)])
[X, Y] = np.meshgrid(x, y)
[Xq, Yq] = np.meshgrid(xq, yq)

# eng = matlab.engine.start_matlab()
# corrected_image = eng.interp2(X,Y,C_im,Xq,Yq,'spline')
# print(np.shape(corrected_image))
# import sys
# sys.exit()
# corrected_image = interpolate.interp2d(X,Y,C_im,'spline')


# WRITING DICOM FILE:
for slice_count in range(0, number_of_slices):
    filepath = 'C:\dicomImages\\'
    T = str(slice_count)
    filepath = filepath + T
    filepath = filepath + '.dcm'
    print(filepath)
    import sys

    sys.exit()

    # eng.dicomwrite(uint16(round(output(:,:,slice_count))),str)

end = datetime.datetime.now()

print(end - now)

# import numpy as np
# A=np.array([[1,2,3],[4,5,6]])
# for i in range(len(A)):
#     A[i] = np.roll(A[i],[0,1])
# print A
