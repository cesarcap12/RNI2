import os
import pydicom
import dicom_numpy
import numpy as np
from matplotlib import pyplot as plt
import glob


def extract_voxel_data(list_of_dicom_files):
    datasets = [pydicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray


CT_FOLDER = '/home/cesarpuga/CT-Scans/104119_GA_F_79J/18556/13/'
SCANS_CT_DIR = os.listdir(CT_FOLDER)

LIST_CT_DIR = [CT_FOLDER + el for el in SCANS_CT_DIR]
RefDCM = pydicom.read_file(LIST_CT_DIR[0])

ctScan = extract_voxel_data(LIST_CT_DIR)
PXL_DIMS = (int(RefDCM.Rows), int(RefDCM.Columns), len(LIST_CT_DIR))
PXL_SPACING = (float(RefDCM.PixelSpacing[0]), float(RefDCM.PixelSpacing[1]), float(RefDCM.SliceThickness))

ArrayDicom = np.zeros(PXL_DIMS, dtype=RefDCM.pixel_array.dtype)

for filenameDCM in LIST_CT_DIR:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, LIST_CT_DIR.index(filenameDCM)] = ds.pixel_array

ps_x = PXL_SPACING[0]
ps_y = PXL_SPACING[1]
ss_z = PXL_SPACING[2]

print(ps_x,ps_y,ss_z)

ax_aspect = ps_y/ps_x
sag_aspect = ps_y/ss_z
cor_aspect = ss_z/ps_x
print(cor_aspect)
print(sag_aspect)

# create 3D array
img_shape = list(RefDCM.pixel_array.shape)
img_shape.append(len(LIST_CT_DIR))
img3d = np.zeros(img_shape)

for i, s in enumerate(LIST_CT_DIR):
    img2d = pydicom.read_file(s).pixel_array
    img3d[:, :, i] = img2d

# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2]//2])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1]//2, :])
a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0]//2, :, :].T)
a3.set_aspect(cor_aspect)

plt.show()

