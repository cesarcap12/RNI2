#!/usr/bin/env python

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import csv
import time
import sys
import glob
import os
from scipy import ndimage as ndi
from skimage import feature
import argparse
import cv2
from PIL import Image
import PIL
from sklearn.decomposition import PCA
from skimage.measure import compare_ssim
import scipy.ndimage


def load_ct(ct_dir):
    files = []
    slices = []
    scan_ct_dir = os.listdir(ct_dir)
    skipcount = 0
    for file in scan_ct_dir:
        print("loading: {}".format(file))
        files.append(pydicom.dcmread(ct_dir+file))
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount += 1
    print("file count: {}".format(len(files)))
    print("skipped, no SliceLocation: {}".format(skipcount))
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    return slices


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_3d_array(slices):
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3D = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3D[:, :, i] = img2d

    return img3D


def get_ct_aspects(slices):
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    return ax_aspect, sag_aspect,cor_aspect


def get_3d_array(slices):
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3D = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3D[:, :, i] = img2d

    return img3D


def get_img_shape(slices):
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    return img_shape


def get_cor_mid_slice(slices):
    shape_img = get_img_shape(slices)
    img = get_3d_array(slices)
    img_c1 = img[:, shape_img[0] // 2, :]  ##
    img_c2 = img[:, :, shape_img[2]//2]
    img_c1_gray = cv2.normalize(src=img_c1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img_c2_gray = cv2.normalize(src=img_c2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    i1_img_c = np.column_stack(np.where(img_c1_gray > 60))
    i1_img_c = i1_img_c[:, 0]
    ix1 = i1_img_c[0]
    i2_mean_col = np.column_stack(np.mean(img_c2_gray, axis=0))
    i2_img_c = np.where(i2_mean_col > 35)
    ix2_img_c2 = i2_img_c[-1]
    ix2 = ix2_img_c2[-1]

    cor_mid_slice = img[246, :, :]   ##(ix1+ix2)//2
    cor_mid_slice_gray = cv2.normalize(src=cor_mid_slice, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return cor_mid_slice_gray


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing


def search_feasible_path(img_r,origin_l,o):
# check if origin is in right or left half of patient to define search space
    pos = origin_l #origin_position(img_r,origin_l)
    feasible_paths = []
    dict_feasible_paths = defaultdict(dict)
    x_vxl = o[2]
    z_vxl = o[0]
    y_vxl = o[1]
    i=0

    if pos == 'Left':

        for theta in range(360):
            for alpha in range(45, 180):
                valid_path = True
                current_path_len = 1
                current_feasible_path = []
                while valid_path:
                    x_loc = math.floor(x_vxl + np.cos(np.deg2rad(theta))*current_path_len)
                    z_loc = math.floor(z_vxl + np.sin(np.deg2rad(theta))*current_path_len)
                    y_loc = math.floor(y_vxl + np.sin(np.deg2rad(alpha))*current_path_len)
                    print('x,z,y: ', x_loc, z_loc, y_loc, '/n', "theta: ", theta, "alpha: ", alpha, 'HU: ', img_r[z_loc, x_loc, y_loc])
                    if -800 < img_r[z_loc, x_loc, y_loc] < 300:
                        current_feasible_path.append([x_loc, y_loc, z_loc])
                        current_path_len += 1
                    elif img_r[z_loc, x_loc, y_loc] < -800:
                        i+=1
                        dict_feasible_paths['key'+str(i)] = current_feasible_path
                        #feasible_paths.append(current_feasible_path)

                        valid_path = False
                    else:
                        valid_path = False

    elif pos == 'Right':

        for theta in range(360):
            for alpha in range(0, 135):
                valid_path = True
                current_path_len = 1
                current_feasible_path = []
                while valid_path:
                    x_loc = math.floor(x_vxl + np.cos(np.deg2rad(theta))*current_path_len)
                    z_loc = math.floor(z_vxl + np.sin(np.deg2rad(theta))*current_path_len)
                    y_loc = math.floor(y_vxl + np.sin(np.deg2rad(alpha))*current_path_len)
                    print('x,z,y: ', x_loc, z_loc, y_loc, '/n', "theta: ", theta, "alpha: ", alpha, 'HU: ', img_r[z_loc, x_loc, y_loc])
                    if -800 < img_r[z_loc, x_loc, y_loc] < 300:
                        current_feasible_path.append([x_loc, y_loc, z_loc])
                        current_path_len += 1
                    elif -800 < img_r[z_loc, x_loc, y_loc]:
                        #feasible_paths.append(current_feasible_path)
                        i+=1
                        dict_feasible_paths['key'+str(i)] = current_feasible_path
                    else:
                        valid_path = False

    return dict_feasible_paths


time_a = time.clock()
# CT scan directory
CT_F73J = '/home/cesarpuga/CT-Scans/212418_GA403_F_73J/17300/3/'
output_path = working_path = "/home/cesarpuga/PycharmProjects/RNI/outputData/"
output_path2 = "/home/cesarpuga/PycharmProjects/RNI/outputData/"

# execute inital functions to convert dicom to np array
ct_f73j = load_ct(CT_F73J)
np_ct_f73j = get_3d_array(ct_f73j)
shape_ct_f73j = get_img_shape(ct_f73j)

# Get ref file to extract dicom properties
RefDs = pydicom.read_file('/home/cesarpuga/CT-Scans/212418_GA403_F_73J/17300/3/00010001')

# get dicom properties from reference file
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), shape_ct_f73j[2])
print('voxel count in  X, Y, slices (Z): ', ConstPixelDims)

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
print('voxel spacing in mm: ', ConstPixelSpacing)

pixel_spacing = ct_f73j[0].PixelSpacing
slice_spacing = ct_f73j[0].SliceThickness

print('shape ctF73J: ', shape_ct_f73j)

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
print("x,y,z vector shape: ", len(x), len(y), len(z))

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

id=1
patient = load_scan(CT_F73J)
imgs = get_pixels_hu(patient)

#save, uncomment to save if changes are performed
#np.save(output_path + "fullimages_%d.npy" % (id), imgs)

file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64)

# Plot histogram of HU units, relevant values: Bone>700 , Air(Skin border)=-1000 [HU's], normally Max/Min values in CTs
# plt.hist(imgs_to_process.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

# Resample voxel array to match actual CT scan Dimensions in [mm]
print("Shape before resampling\t", imgs_to_process.shape)
# imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
# print("Shape after resampling\t", imgs_after_resamp.shape)
# print("image after resampling\t", imgs_after_resamp)
# print("image after resampling Dtype\t", type(imgs_after_resamp))

id=2
# np.save(output_path + "fullimages_%d.npy" % (id), imgs_after_resamp)
file_used= output_path+"fullimages_%d.npy" % id
imgs_after_resamp = np.load(file_used).astype(np.float64)
print("Shape after resampling\t", imgs_after_resamp.shape)
# print("image after resampling\t", imgs_after_resamp)
print("image after resampling Dtype\t", type(imgs_after_resamp))

# get and display coronal slice of section of interest (origin of mapping algorithm)
# img_cor_roi = get_cor_mid_slice(imgs_after_resamp)  #ct_f73j
# cv2.imshow("coronal slice ROI", img_cor_roi)
img_cor_roi = imgs_after_resamp[:, 240, :]
img_cor_roi_gray = cv2.normalize(src=img_cor_roi, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# cv2.imshow("coronal slice ROI", img_cor_roi_gray)
# cv2.waitKey(0)


# feasible path brute force search begins
origin_hu = imgs_after_resamp[525,240,145]
origin_loc = [525,240,145]
print("HU intensity of Origin= ", origin_hu)


def origin_pos(img_r, origin_l):

    if origin_l[2] > len(img_r[2]):
        position = 'Right'
    else:
        position = 'Left'

    return position


#origin_position = origin_pos(origin_loc,imgs_after_resamp)
#print("origin position: ", origin_position)





t0 = time.clock()
paths = search_feasible_path(imgs_after_resamp, 'Left', [525,240,145])
t1 = time.clock() - t0
time_b = time.clock()
print("Time elapsed to search 1 path: ", t1) # CPU seconds elapsed (floating point)
print('time elapsed in total:', time_b-time_a)
print('paths ', len(paths))
print('paths ', paths)

#with open('Paths', 'w') as f:
#    write = csv.writer(f)
#    write.writerows(paths)
