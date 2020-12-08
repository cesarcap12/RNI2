#!/usr/bin/env python

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import scipy.ndimage
from plotly.offline import iplot
from plotly.tools import FigureFactory as FF
import time
import os
import cv2
import scipy.ndimage


def load_ct(ct_dir):
    files = []
    slices = []
    scan_ct_dir = os.listdir(ct_dir)
    skipcount = 0
    print("loading CT scan")
    for file in scan_ct_dir:
        # print("loading: {}".format(file))
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
    slices.sort(key=lambda x: int(x.SliceLocation))
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


def search_feasible_path(img_r, o, skin_array_xyz):
    # check if origin is in right or left half of patient to define search space
    dict_feasible_paths = defaultdict(dict)
    x_vxl = o[2]
    y_vxl = o[1]
    z_vxl = o[0]
    i = 0

    for theta in range(360):
        for alpha in range(0,145):
            valid_path = True
            current_path_len = 1
            current_feasible_path = []
            too_much_air = 0
            too_much_bone =0
            while valid_path:

                x_loc = math.floor(x_vxl + np.cos(np.deg2rad(theta))*current_path_len)
                z_loc = math.floor(z_vxl + np.sin(np.deg2rad(theta))*current_path_len)
                y_loc = math.floor(y_vxl + np.sin(np.deg2rad(alpha))*current_path_len)

                print('x,z,y: ', x_loc, z_loc, y_loc, "theta: ", theta, "alpha: ", alpha, 'HU: ',
                      img_r[z_loc, y_loc, x_loc])

                #print(skin_array_xyz[z_loc])

                #skin_search = np.array([[x_loc, y_loc], [x_loc-1, y_loc], [x_loc+1, y_loc], [x_loc, y_loc+1],
                #                        [x_loc-1, y_loc+1], [x_loc+1, y_loc+1], [x_loc, y_loc-1], [x_loc-1, y_loc-1],
                #                        [x_loc+1, y_loc-1]])

                skin_search = np.array([x_loc,y_loc])

                if img_r[z_loc, y_loc, x_loc] > 300:
                    too_much_bone += 1
                    valid_path = False
                    print('found bone')
                elif img_r[z_loc, y_loc, x_loc] < -800:
                    too_much_air += 1
                    print(too_much_air)
                    if too_much_air > 20:
                        print('too much air')
                        valid_path = False
                elif img_r[z_loc, y_loc, x_loc] < 300:
                    current_feasible_path.append([x_loc, y_loc, z_loc])
                    current_path_len += 1
                    print("path len ", current_path_len)
                if all(skin_search in subl for subl in skin_array_xyz[int(z_loc)]):
                    i += 1
                    dict_feasible_paths[str(i)] = current_feasible_path
                    print('skin found')
                    valid_path = False


    return dict_feasible_paths


def get_n_shortest_paths(x):
    return [k for k in x.keys() if len(x.get(k))==min([len(n) for n in x.values()])]

# def get_n_shortest_paths(paths,n):
#     shortest_paths = defaultdict(dict)
#     i = 0
#     while i < n:
#         min_list = min([len(ls) for ls in paths.values()])
#         min_key = min(len(paths), key=paths.get)
#         paths.pop(min_key, None)
#         shortest_paths['path'+str(i)] = min_list
#         n += 1
#     return shortest_paths


def get_skin_cords(img_3d_after_reshape, slices_3d):
    # get array with skin coordinates
    contours_max = []
    pixelspacing = slices_3d[0].PixelSpacing
    img_3d_after_reshape[:, int(len(img_3d_after_reshape[1]) - slices_3d[0].TableHeight):, :] = -1000

    for layer in range(img_3d_after_reshape.shape[0]):
        contours_slice = measure.find_contours(img_3d_after_reshape[layer, :, :], -750)
        x = max(contours_slice, key=len)
        x = list(np.round(x * pixelspacing[0]).astype(int))
        contours_max.append(x)

    return contours_max


def make_mesh(image, threshold=-300, step_size=1):
    print("Transposing surface")
    p = image.transpose(2, 1, 0)

    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print ("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)


def plt_3d(verts, faces):
    print ("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()

# Execution starts
time_a = time.clock()

# CT scan directory
CT_F73J = '/home/cesarpuga/CT-Scans/212418_GA403_F_73J/17300/3/'
output_path = working_path = "/home/cesarpuga/PycharmProjects/RNI/"

# execute initial functions to convert dicom to np array
ct_f73j = load_ct(CT_F73J)
np_ct_f73j = get_3d_array(ct_f73j)
shape_ct_f73j = get_img_shape(ct_f73j)

# Get ref file to extract dicom properties
RefDs = pydicom.read_file('/home/cesarpuga/CT-Scans/212418_GA403_F_73J/17300/3/00010001')

# get dicom properties from reference file
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), shape_ct_f73j[2])
#print('voxel count in  X, Y, slices (Z): ', ConstPixelDims)

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
print('voxel spacing in mm: ', ConstPixelSpacing)

pixel_spacing = ct_f73j[0].PixelSpacing
slice_spacing = ct_f73j[0].SliceThickness
print('shape ctF73J: ', shape_ct_f73j)

# x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
# y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
# z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
# print("x,y,z vector shape: ", len(x), len(y), len(z))

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

id1 = 1
patient = load_scan(CT_F73J)
imgs = get_pixels_hu(patient)


# save, uncomment to save if changes are performed (New CT is loaded)
# np.save(output_path + "fullimages_%d.npy" % (id1), imgs)
file_used=output_path+"fullimages_%d.npy" % id1
imgs_to_process = np.load(file_used).astype(np.float64)

# Plot histogram of HU units, relevant values: Bone>700 , Air(Skin border)=-1000 [HU's], normally Max/Min values in CTs
# plt.hist(imgs_to_process.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

# Resample voxel array to match actual CT scan Dimensions in [mm]
print("Shape before resampling\t", imgs_to_process.shape)
print("image before resampling Dtype\t", type(imgs_to_process))
# imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])


id2 = 2
# save, uncomment to save if changes are performed (New CT is loaded)
# np.save(output_path + "fullimages_%d.npy" % id2, imgs_after_resamp)
file_used= output_path+"fullimages_%d.npy" % id2
imgs_after_resamp = np.load(file_used).astype(np.float64)
print("Shape after resampling\t", imgs_after_resamp.shape)
print("image after resampling Dtype\t", type(imgs_after_resamp))

# get and display coronal slice of section of interest (origin of mapping algorithm)
# img_cor_roi = get_cor_mid_slice(imgs_after_resamp)  #ct_f73j
# cv2.imshow("coronal slice ROI", img_cor_roi)
#imgs_after_resamp = np.flip(imgs_after_resamp, axis=1)
print("Shape after resampling2\t", imgs_after_resamp.shape)

# Z Y X Array
img_cor_roi = imgs_after_resamp[:, 240, :]
# img_cor_roi_gray = cv2.normalize(src=img_cor_roi, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Plot region of interest ROI where origin of path search exists
# plt.imshow(img_cor_roi, cmap=plt.cm.bone)
# plt.show()




print('Getting Skin Surface Coordinates')
skin_coords = np.array(get_skin_cords(imgs_after_resamp, ct_f73j))
# print("skin_coords: ", skin_coords)
#print('ct max len', len(skin_coords))
#print('Dtype Skin coords', type(skin_coords))
#skin_coords= list(zip(*skin_coords))
#list_skin_coords = [[i[0][-1], i[1][-1]] for i in skin_coords]
#print('ct max len list', len(list_skin_coords))


# COORDS TEST
#print("skin cords Z slice 500: ", skin_coords[500][10])  #[slice][row][2darray]
coords = np.array([[351, 341],[497, 498],[498, 496]])
#cord_test = skin_coords[540].any(coords, axis=0)
cord_test = all(coords in subl for subl in skin_coords[524])
print(skin_coords[524])

# if cord_test:
#     print('ok')

# [range(326, 326+1), range(385-1, 385+1)]
print('cord_test', cord_test)

# plt.imshow(img_cor_roi, cmap=plt.cm.bone)
# plt.show()


#print("skin cords Z slice 500: ", skin_coords[500])

#skin_coords_xyz= [[]]
#for el in range(imgs_after_resamp.shape[0]):
    #skin_coords_xyz = [skin_coords[:, 0], skin_coords[:, 1], el]

#print('skin coords XYZ= ', skin_coords_xyz)

# feasible path brute force search begins
origin_hu = imgs_after_resamp[524, 240, 145]
origin_loc = [524, 240, 145]
print("HU intensity of Origin= ", origin_hu)
print('Calculating Paths / Program Finished...')


# TIME CALCULATIONS / RESULTS
t0 = time.clock()
paths = search_feasible_path(imgs_after_resamp, origin_loc, skin_coords)
t1 = time.clock() - t0
time_b = time.clock()
#
print("Time elapsed to search 1 path: ", t1) # CPU seconds elapsed (floating point)
print('time elapsed in total:', time_b-time_a)
print('paths ', len(paths))
# # print('paths ', paths)
# print('paths type: ', type(paths))
# shortest_paths = get_shortest_paths(paths)
# print(len(shortest_paths))

# PLOT SKIN SURFACE
contours = measure.find_contours(imgs_after_resamp[524, :, :], -750)
contourMax = max(contours, key=len)
fig, ax = plt.subplots()
ax.imshow(imgs_after_resamp[524, :, :], cmap=plt.cm.bone)
ax.plot(contourMax[:, 1], contourMax[:, 0], linewidth=2)
#ax.axis('image Gray')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
plt.show()


# target_entry_pts = []
#
# for key in paths.keys():
#     target_entry_pts[key] = paths.get(key[-1])
#     target_entry_pts[key+1] = paths.get(key[0])
#
# print(target_entry_pts)

# for i in range(len(shortest_paths)):
#    print(paths[str(shortest_paths[i])])

# with open('Paths', 'w') as f:
#    write = csv.writer(f)
#    write.writerows(paths)

# v, f = make_mesh(imgs_after_resamp, 250)
# plt_3d(v, f)
