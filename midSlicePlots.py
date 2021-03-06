import pydicom
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import glob
import os
from scipy.stats import mode
from scipy import ndimage as ndi
from skimage import feature
from skimage import measure
import argparse
import cv2
from PIL import Image
import PIL
from sklearn.decomposition import PCA
from skimage.measure import compare_ssim
import imutils
#testGit


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
    print("loading: CT Scan")

    for file in scan_ct_dir:
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
    img_c1 = img[:, shape_img[0] // 2, :]
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

    cor_mid_slice = img[(ix1+ix2)//2, :, :]
    cor_mid_slice_gray = cv2.normalize(src=cor_mid_slice, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return cor_mid_slice_gray


def get_skin_cords(img_3d, slices_3d):
    # get array with skin coordinates
    contours_max = []
    img_3d[int(get_img_shape(slices_3d)[1] - slices_3d[0].TableHeight):, :] = 0

    for layer in range(get_img_shape(slices_F73J)[2]):
        contours_slice = measure.find_contours(img3d[:, :, layer], 200)
        x = max(contours_slice, key=len)
        x = list(np.round(x * pxl_spacing[0]).astype(int))
        contours_max.append(x)

    return contours_max


# load the DICOM files
FILES = []
FILES2 = []
FILES3 = []
SLICES = []
SLICES2 = []
SLICES3 = []
CT_F73J = '/home/cesarpuga/CT-Scans/212418_GA403_F_73J/17300/3/'
# CT_M49J = '/home/cesarpuga/CT-Scans/416818_GA413_M_49J/18030/12/'
# CT_F79J = '/home/cesarpuga/CT-Scans/704619_GA427_M_70J/18344/3/'

# load specific CT Scan
slices_F73J = load_CT(CT_F73J, FILES, SLICES)
# slices_M49J = load_CT(CT_M49J, FILES2, SLICES2)
# slices_F79J = load_CT(CT_F79J, FILES3, SLICES3)

# get pixel aspects, assuming all slices are the same
ax_aspect, sag_aspect, cor_aspect = get_ct_aspects(slices_F73J)
# ax_aspect2, sag_aspect2, cor_aspect2 = get_ct_aspects(slices_M49J)
# ax_aspect3, sag_aspect3, cor_aspect3 = get_ct_aspects(slices_F79J)

# # get 3D array and img_shape
img3d = get_3d_array(slices_F73J)
img_shape = get_img_shape(slices_F73J)
print('img shape:', img_shape)
hu_patient = get_pixels_hu(slices_F73J)
print("hu patient shape: ", hu_patient.shape)
#img3d2 = get_3d_array(sliceS2)
#img_shape2 = get_img_shape(sliceS2)
#
# # img acquisition
# imgA = img3d[img_shape[0]//2, :, :]
# imgB = img3d2[img_shape2[0]//2, :, :]
# imgC = img3d[:, img_shape[0]//2, :]


# get TABLE HEIGHT
table_height = slices_F73J[0].TableHeight
print('Table Heigth', table_height)
imgC = img3d[:, :, 361] #img_shape[2]//2
table_height2 = int(img_shape[1] - table_height)
print('table height 2 ', table_height2)
hu_patient[:, table_height2:, :] = -1000
imgC[table_height2:, :] = 0
hu_patient = np.flip(hu_patient, axis=1)
#print('shape imC: ', imgC.shape[1])


# # np arrays to grayscale cv2 images
# im = np.array(imgA*1).astype('uint8')
# imA_gray = cv2.normalize(src=imgA, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# imB_gray = cv2.normalize(src=imgB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# imA_gray_thorax = cv2.normalize(src=imgA, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
imC_gray = cv2.normalize(src=imgC, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# imC2_gray = cv2.normalize(src=imgC2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# get coronal mid slices
#test_mid_cor_slice_A = get_cor_mid_slice(slices_M49J)
test_mid_cor_slice_B = get_cor_mid_slice(slices_F73J)
# test_mid_cor_slice_C = get_cor_mid_slice(slices_F79J)

# SSIM score
# crop_imgA = test_mid_cor_slice_A[:, 100:700]
# crop_imgB = test_mid_cor_slice_B[:, 150:750]
# crop_imgC = test_mid_cor_slice_C[:, 150:750]

# CA comparison
# (scoreCA, diffCA) = compare_ssim(crop_imgA, crop_imgB, full=True)
# diff = (diffCA * 255).astype("uint8")
# print("SSIM A-B: {}".format(scoreCA))
# # CB comparison
# (scoreCB, diffCB) = compare_ssim(crop_imgA, crop_imgC, full=True)
# diffCB = (diffCB * 255).astype("uint8")
# print("SSIM A-C: {}".format(scoreCB))

# Plot mid coronal slice
# cv2.imshow("cropped", test_mid_cor_slice_B)
# cv2.waitKey(0)

#window_name = 'imageA'

# Using cv2.imshow() method
# Displaying the image
# cv2.imshow("imgA", crop_imgA)
# cv2.imshow("imgB", crop_imgB)
# cv2.imshow("imgC", crop_imgC)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
#cv2.waitKey(0)

# closing all open windows

# #get edges of image
#edges1 = img3d[img_shape[0]//3, :, :]
#edges2 = feature.canny(img3d[img_shape[0]//2, :, :], sigma=4)

#plot 3 orthogonal slices
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
# a4 = plt.subplot(2, 2, 4)
# plt.imshow(imC_gray, cmap=plt.cm.bone)
# a4.set_aspect(cor_aspect)
# #
# plt.show()

## Contour Finding: Skin

contours = measure.find_contours(hu_patient[361, :, :], -750)
#print('contours', contours)

# remove table ALWAYS biggest contour
#imax = (max((len(l), i) for i, l in enumerate(contours))[1])
#print('imax= ', imax)
#contours.pop(imax)
# next biggest contour is skin in xy plane
pxl_spacing = slices_F73J[0].PixelSpacing
print("pxl_spacing", pxl_spacing)
contourMax = max(contours, key=len)
print(type(contourMax))
contourMax_r = np.round(contourMax*pxl_spacing[0]).astype(int)
contourMax_X = contourMax_r[:, 0]
contourMax_Y = contourMax_r[:, 1]
print("contourMax", contourMax_r)
print("1st", contourMax_r[0, 0])
# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(hu_patient[361, :, :], cmap=plt.cm.gray)

maxContours = get_skin_cords(hu_patient, slices_F73J)  #img3d
print('ct max len', len(maxContours))

# for layer in range(img_shape[2]):
#     img3d[table_height:, :, layer] = 0
#     contours_slice = measure.find_contours(img3d[:, :, layer], 200)
#     x = max(contours_slice, key=len)
#     #contours_max = np.append(contours_max, max(contours_slice, key=len))
#     x = list(np.round(x * pxl_spacing[0]).astype(int))
#     contours_max.append(x)
#
# print('ct max len', len(contours_max))

# PLOT CONTOURS
ax.plot(contourMax[:, 1], contourMax[:, 0], linewidth=2)
#ax.axis('image Gray')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for z, data in enumerate(contours_max):
#     if z == 10:
#         break
#     for point in data:
#         x = point[0]
#         y = point[1]
#         ax.scatter(x, y, z, marker='o')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
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