import numpy as np
import os
import glob
import os.path
import pydicom


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


def get_rescale_parameters(scans):
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

    return intercept, slope


scans = [[]]
slopes = [[]]
intercepts = [[]]
filesDepth3 = glob.glob('/home/cesarpuga/CT-Scans/*/*/*/')
dirsDepth3 = filter(lambda f: os.path.isdir(f), filesDepth3)

scan1 = load_scan(filesDepth3[0])
slope1,intercept1= get_rescale_parameters(scan1)

scan2 = load_scan(filesDepth3[1])
slope2, intercept2 = get_rescale_parameters(scan2)

scan3 = load_scan(filesDepth3[2])
slope3, intercept3 = get_rescale_parameters(scan3)

scan4 = load_scan(filesDepth3[3])
slope4, intercept4 = get_rescale_parameters(scan4)

scan5 = load_scan(filesDepth3[4])
slope5, intercept5 = get_rescale_parameters(scan5)

scan6 = load_scan(filesDepth3[5])
slope6, intercept6 = get_rescale_parameters(scan6)

scan7 = load_scan(filesDepth3[6])
slope7, intercept7 = get_rescale_parameters(scan7)

scan8 = load_scan(filesDepth3[7])
slope8, intercept8 = get_rescale_parameters(scan8)

scan9 = load_scan(filesDepth3[8])
slope9, intercept9 = get_rescale_parameters(scan9)

scan10 = load_scan(filesDepth3[9])
slope10, intercept10 = get_rescale_parameters(scan10)

#print dicom parameters
print("1: ", slope1, intercept1)
print("2: ", slope2, intercept2)
print("3: ", slope3, intercept3)
print("4: ", slope4, intercept4)
print("5: ", slope5, intercept5)
print("6: ", slope6, intercept6)
print("7: ", slope7, intercept7)
print("8: ", slope8, intercept8)
print("9: ", slope9, intercept9)
print("10: ", slope10, intercept10)

#print("slopes= ", slopes)
#print("intercepts= ", intercepts)




