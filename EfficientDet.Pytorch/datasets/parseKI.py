from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

label_paths = [
    'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_1_1',
    'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_1',
    'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_2',
    'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_3_1',
    'KI-Dataset/For KTH/Helena/N10/N10_1_1',
    'KI-Dataset/For KTH/Helena/N10/N10_1_2',
    'KI-Dataset/For KTH/Helena/N10/N10_1_3',
    'KI-Dataset/For KTH/Helena/N10/N10_2_1',
    'KI-Dataset/For KTH/Helena/N10/N10_1_1',
    'KI-Dataset/For KTH/Helena/N10/N10_2_2',
    'KI-Dataset/For KTH/Nikolce/N10_1_1',
    'KI-Dataset/For KTH/Nikolce/N10_1_2',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_1_1',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_2_1',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_2_2',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_3_1',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_3_2',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_4_1',
    'KI-Dataset/For KTH/Rachael/Rach_P9/P9_4_2',
    'KI-Dataset/For KTH/Rachael/Rach_P13/P13_1_1',
    'KI-Dataset/For KTH/Rachael/Rach_P13/P13_1_2',
    'KI-Dataset/For KTH/Rachael/Rach_P13/P13_2_1',
    'KI-Dataset/For KTH/Rachael/Rach_P13/P13_2_2',
    'KI-Dataset/For KTH/Rachael/Rach_P19/P19_1_1',
    'KI-Dataset/For KTH/Rachael/Rach_P19/P19_1_2',
    'KI-Dataset/For KTH/Rachael/Rach_P19/P19_2_1',
    'KI-Dataset/For KTH/Rachael/Rach_P19/P19_2_2',
    'KI-Dataset/For KTH/Rachael/Rach_P19/P19_3_1',
    'KI-Dataset/For KTH/Rachael/Rach_P19/P19_3_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_1_3',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_1_4',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_2_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_2_3',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_2_4',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_3_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_3_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_3_3',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_4_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_4_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_4_3',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_5_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_5_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_6_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_6_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_7_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_7_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_8_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_8_2',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_9_1',
    'KI-Dataset/For KTH/Rachael/Rach_P20/P20_9_2',
    'KI-Dataset/For KTH/Rachael/Rach_P25/P25_2_1',
    'KI-Dataset/For KTH/Rachael/Rach_P25/P25_3_1',
    'KI-Dataset/For KTH/Rachael/Rach_P25/P25_3_2',
    'KI-Dataset/For KTH/Rachael/Rach_P25/P25_4_2',
    'KI-Dataset/For KTH/Rachael/Rach_P25/P25_5_1',
    'KI-Dataset/For KTH/Rachael/Rach_P25/P25_8_2',
    'KI-Dataset/For KTH/Rachael/Rach_P28/P28_10_4',
    'KI-Dataset/For KTH/Rachael/Rach_P28/P28_10_5',
] # Len 58

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']


def parseKI(basePath="", fileCount=len(label_paths)):
    labels = [] # List of lists of targets
    images = [] # List of 512x512x3 numpy arrays

    for path in range(fileCount):
        # Parse xml tree
        tree = ET.parse(basePath+label_paths[path]+'.xml')
        root = tree.getroot()

        # Parse full image to nparray
        image = basePath+label_paths[path]+'.tif'
        im = Image.open(image)
        imarray = np.array(im, dtype=np.double)/256

        # Pad image to 2000x2000x3
        padded_array = np.zeros((2000, 2000, 3))
        shape = np.shape(imarray)
        padded_array[:shape[0], :shape[1]] = imarray
        imarray = padded_array


        targets = []
        slices = []
        for i in range(4):
            for j in range(4):
                xmin = min(i*512, 2000-512)
                xmax = min((i+1)*512, 2000)
                ymin = min(j*512, 2000-512)
                ymax = min((j+1)*512, 2000)
                #print(xmin, xmax, ymin, ymax, i*4+j)
                targets.append([])
                slices.append(imarray[ymin:ymax, xmin:xmax, :])

                # Loop through crops
                for child in root.iter('object'):
                    # Parse label
                    target = [] # [xmin, ymin, xmax, ymax, label]
                    label = ""
                    for name in child.iter('name'):
                        label = name.text

                    # Parse matching image data
                    for box in child.iter('bndbox'):
                        boundaries = []
                        for val in box.iter():
                            boundaries.append(val.text)

                        # Get center of crop and make sure the box fits inside of image
                        meanX = int((int(boundaries[1]) + int(boundaries[3])) / 2)
                        meanY = int((int(boundaries[2]) + int(boundaries[4])) / 2)

                        meanX = max(meanX, 8)
                        meanX = min(meanX, imarray.shape[1]-8)

                        meanY = max(meanY, 8)
                        meanY = min(meanY, imarray.shape[0]-8)

                        if meanX > xmin and meanX <= xmax and meanY > ymin and meanY <= ymax and class_names.index(label) != 4:
                            #print(meanX, xmin, xmax, meanY, ymin, ymax, i*4+j)
                            target.append(max(meanX-8-xmin, 0))
                            target.append(max(meanY-8-ymin, 0))
                            target.append(min(meanX+8-xmin, 512))
                            target.append(min(meanY+8-ymin, 512))
                            target.append(class_names.index(label))
                            targets[i*4+j].append(target)

        for i in range(16):
            if len(targets[i]) > 0:
                labels.append(targets[i])
                images.append(slices[i])

    return images, labels
