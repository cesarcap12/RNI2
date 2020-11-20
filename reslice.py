import pydicom
import numpy as np
import matplotlib.pyplot as plt
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


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)  # Set outside-of-scan pixels to 0
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


def load_CT(ct_dir,files,slices):
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


def get_ct_aspects(slices):
    ps = sliceS[0].PixelSpacing
    ss = sliceS[0].SliceThickness
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


# load the DICOM files
FILES = []
FILES2 = []
SLICES = []
SLICES2 = []
CT_DIR  = '/home/cesarpuga/CT-Scans/212418_GA403_F_73J/17300/3/'
CT_DIR2 = '/home/cesarpuga/CT-Scans/416818_GA413_M_49J/18030/12/'

# load specific CT Scan
sliceS  = load_CT(CT_DIR, FILES, SLICES)
sliceS2 = load_CT(CT_DIR2, FILES2, SLICES2)

# get pixel aspects, assuming all slices are the same
ax_aspect, sag_aspect, cor_aspect = get_ct_aspects(sliceS)
ax_aspect2, sag_aspect2, cor_aspect2 = get_ct_aspects(sliceS2)

# get 3D array and img_shape
img3d = get_3d_array(sliceS)
img_shape = get_img_shape(sliceS)
img3d2 = get_3d_array(sliceS2)
img_shape2 = get_img_shape(sliceS2)

imgA = img3d[img_shape[0]//2, :, :]
print(img_shape)
print(img_shape2)
imgB = img3d2[img_shape2[0]//2, :, :]

im = np.array(imgA*1).astype('uint8')
imA_gray = cv2.normalize(src=imgA, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
imB_gray = cv2.normalize(src=imgB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

crop_imgA = imA_gray[:, 200:800]
crop_imgB = imB_gray[:, 100:700]

# 5. Compute the Structural Similarity Index (SSIM) between the two
#    images, ensuring that the difference image is returned
(score, diff) = compare_ssim(crop_imgA, crop_imgB, full=True)
diff = (diff * 255).astype("uint8")

# 6. You can print only the score if you want
print("SSIM: {}".format(score))

cv2.imshow("cropped", crop_imgB)
cv2.imshow("cropped", crop_imgA)
cv2.waitKey(0)

#window_name = 'imageA'

# Using cv2.imshow() method
# Displaying the image
#cv2.imshow(window_name, imB_gray)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
#cv2.waitKey(0)

# closing all open windows

# #get edges of image
# edges1 = img3d[img_shape[0]//2, :, :]
# edges2 = feature.canny(img3d[img_shape[0]//2, :, :], sigma=4)
#
# #plot 3 orthogonal slices
# a1 = plt.subplot(2, 2, 1)
# plt.imshow(img3d[:, :, img_shape[2]//2],cmap=plt.cm.bone)
# a1.set_aspect(ax_aspect)
#
# a2 = plt.subplot(2, 2, 2)
# plt.imshow(img3d[:, img_shape[1]//2, :],cmap=plt.cm.bone)
# a2.set_aspect(sag_aspect)
#
# a3 = plt.subplot(2, 2, 3)
# plt.imshow(img3d[img_shape[0]//2, :, :], cmap=plt.cm.bone)
# a3.set_aspect(cor_aspect)
#
# a3 = plt.subplot(2, 2, 4)
# plt.imshow(edges2, cmap=plt.cm.bone)
# a3.set_aspect(cor_aspect)
#
# plt.show()

# ## TODO Circle detection for future Rib Segmentation
#
# MidCorSlice = img3d[img_shape[0]//2, :, :]
#
# #n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
# faces_pca = PCA(n_components=0.8)
# faces_pca.fit(MidCorSlice)
#
# #plt.imshow(faces_pca, cmap='gray')