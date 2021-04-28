import torch.utils.data as data
import torch
from collections import OrderedDict

from efficientnet_pytorch import EfficientNet

from models.efficientdet import EfficientDet
from utils import EFFICIENTDET, get_state_dict

import numpy as np
from torchvision import transforms

import PIL
from PIL import Image

import xml.etree.ElementTree as ET
from os.path import basename
from PIL import Image
import json
import xml.dom.minidom as md
import argparse
import os

class Args:
    def __init__(self,
                 arg_dataset_root,
                 arg_filepath,
                 arg_threshold,
                 arg_iou_threshold,
                 arg_weight,
                 save_to_neo4j,
                 generate_edges,
                 export_images
                 ):
        self.dataset_root = arg_dataset_root
        self.filepath = arg_filepath
        self.threshold = arg_threshold
        self.iou_threshold = arg_iou_threshold
        self.weight = arg_weight
        self.save_to_neo4j = save_to_neo4j
        self.generate_edges = generate_edges
        self.export_images = export_images

args = Args(
    arg_dataset_root='datasets/',
    # arg_filepath='KI-dataset/For KTH/Rachael/Rach_P13/P13_2_2',  # <- this is NOT being used
    arg_filepath='',
    arg_threshold=0.25,
    arg_iou_threshold=0.5,
    arg_weight='./saved/weights/kebnekaise/checkpoint_54.pth',
    save_to_neo4j=True,
    generate_edges=True,
    export_images=True
)

KI_CLASSES = ('inflammatory', 'lymphocyte', 'fibroblast and endothelial',
              'epithelial')

class KiDataset(data.Dataset):
    """KI Detection Dataset Object for single slide
    input is image, target is annotation
    Arguments:
        root (string): filepath to KIdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
    """

    def __init__(self, root="", filePath="", transform=None):

        print("root: {}".format(root))
        print("filePath: {}".format(filePath))
        images, target, original_image, original_labels, normal_image = parseOneKI(basePath=root, filePath=filePath)

        self.image_set = images  # 57*16 images
        self.target_set = target
        self.image = original_image
        self.targets = original_labels
        self.filePath = root+filePath
        self.filename = filePath.rsplit('/', 1)[-1]
        self.normal_image = normal_image

        self.transform = transform

    def __getitem__(self, index):
        target = self.target_set[index]
        img = self.image_set[index]

        target = np.array(target)

        sample = {'img': img, 'annot': target}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.image_set)

    def num_classes(self):
        return len(KI_CLASSES)

    def get_original_image(self, index):
        return self.image_set[index]

    def label_to_name(self, label):
        return KI_CLASSES[label]

    def load_annotations(self, index):
        return np.array(self.target_set[index])


mins = [0, 496, 992, 1488]
maxs = [512, 1008, 1504, 2000]

def translate_boxes(arr):
    #print(arr)
    labels = []
    for i in range(4):
        for j in range(4):
            index = i*4+j
            xmin = mins[i]
            ymin = mins[j]
            for k in range(len(arr[index])):
                label = []
                if len(arr[index]) > 0:
                    # print("i = {}, j = {}, k = {}".format(i, j, k))
                    label.append(arr[index][k][0]+xmin)
                    label.append(arr[index][k][1]+ymin)
                    label.append(arr[index][k][2]+xmin)
                    label.append(arr[index][k][3]+ymin)
                    labels.append(label)

    # print(labels)
    return labels

def parseOneKI(basePath="", filePath='KI-Dataset/For KTH/Rachael/Rach_P28/P28_10_5'):

    images = []
    labels = []
    original_labels = []

    # Parse full image to nparray
    image = basePath+filePath+'.tif'
    im = Image.open(image)
    imarray = np.array(im, dtype=np.double)/255

    # Pad image to 2000x2000x3
    padded_array = np.ones((2000, 2000, 3))
    shape = np.shape(imarray)
    padded_array[:shape[0], :shape[1]] = imarray
    imarray = padded_array

    normal = imarray.copy()
    B = imarray.copy()
    means = B.mean(axis=2)
    B[means > 0.98,:] = np.nan
    mean = np.nanmean(B, axis=(0,1))
    std = np.nanstd(B, axis=(0,1))
    imarray = (imarray - mean) / std

    # Parse xml tree if labels exist
    try:
        tree = ET.parse(basePath+filePath+'.xml')
        root = tree.getroot()
    except:
        targets = []
        slices = []
        for i in range(4):
            for j in range(4):
                xmin = mins[i]
                xmax = maxs[i]
                ymin = mins[j]
                ymax = maxs[j]
                #print(xmin, xmax, ymin, ymax, i*4+j)
                targets.append([])
                slices.append(imarray[ymin:ymax, xmin:xmax, :])
        for i in range(16):
            labels.append(targets[i])
            images.append(slices[i])
        return images, labels, imarray, original_labels, normal

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

            meanX = max(meanX, 16)
            meanX = min(meanX, imarray.shape[1]-16)

            meanY = max(meanY, 16)
            meanY = min(meanY, imarray.shape[0]-16)

            # Check if full box is inside image
            if class_names.index(label) != 4:
                #print(meanX, xmin, xmax, meanY, ymin, ymax, i*4+j)
                target.append(max(meanX-16, 0))
                target.append(max(meanY-16, 0))
                target.append(min(meanX+16, 2000))
                target.append(min(meanY+16, 2000))
                target.append(class_names.index(label))
                original_labels.append(target)

    # Split into 512x512 slices of full image
    targets = []
    slices = []
    for i in range(4):
        for j in range(4):
            xmin = mins[i]
            xmax = maxs[i]
            ymin = mins[j]
            ymax = maxs[j]
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

                    meanX = max(meanX, 16)
                    meanX = min(meanX, imarray.shape[1]-16)

                    meanY = max(meanY, 16)
                    meanY = min(meanY, imarray.shape[0]-16)

                    # Check if full box is inside image
                    if meanX > xmin + 16 and meanX <= xmax - 16 and meanY > ymin + 16 and meanY <= ymax - 16 and class_names.index(label) != 4:
                        #print(meanX, xmin, xmax, meanY, ymin, ymax, i*4+j)
                        target.append(max(meanX-16-xmin, 0))
                        target.append(max(meanY-16-ymin, 0))
                        target.append(min(meanX+16-xmin, 512))
                        target.append(min(meanY+16-ymin, 512))
                        target.append(class_names.index(label))
                        targets[i*4+j].append(target)

    for i in range(16):
        labels.append(targets[i])
        images.append(slices[i])

    return images, labels, imarray, original_labels, normal


label_paths = [
                                                  # Overall Histological Grade
    # # P9
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_1_1',  # GIII
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_2_1',  # GIII
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_2_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_3_1',  # GIII
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_3_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_4_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P9/P9_4_2', #
    # # P13
    # 'KI-dataset/For KTH/Rachael/Rach_P13/P13_1_1',  # GIII # BM test
    # 'KI-dataset/For KTH/Rachael/Rach_P13/P13_1_2', # # BM test
    # 'KI-dataset/For KTH/Rachael/Rach_P13/P13_2_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P13/P13_2_2',  # GIII # BM test
    # # P19
    # 'KI-dataset/For KTH/Rachael/Rach_P19/P19_1_1', # ->
    # 'KI-dataset/For KTH/Rachael/Rach_P19/P19_1_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P19/P19_2_1',  # GIV  ->
    # 'KI-dataset/For KTH/Rachael/Rach_P19/P19_2_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P19/P19_3_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P19/P19_3_2', #
    # # P20
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_1_3', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_1_4', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_2_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_2_3',  # GO  ->
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_2_4', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_3_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_3_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_3_3', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_4_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_4_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_4_3', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_5_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_5_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_6_1',  # GO  ->
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_6_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_7_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_7_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_8_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_8_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_9_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P20/P20_9_2', #
    # # P25
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_1_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_1_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_2_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_2_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_3_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_3_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_4_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_4_2', #
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_5_1', #
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_5_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_6_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_6_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_7_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_7_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_8_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P25/P25_8_2', #
    # # P28
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_1_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_1_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_2_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_2_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_3_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_3_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_3_6',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_4_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_4_6',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_5_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_5_6',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_6_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_6_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_6_6',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_7_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_8_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_8_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_9_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_9_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_10_4', #
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_10_5', # <-
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_11_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_11_5',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_12_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_12_3',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_12_4',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_13_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_13_3',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_14_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_14_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_15_1',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_15_2',
    # 'KI-dataset/For KTH/Rachael/Rach_P28/P28_16_1',
    # # P7
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_1_1',  # GIII # BM validation
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_1', # # BM validation
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_2', # # BM validation
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_3_1', # # BM validation
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_3_2', #
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_4_1',
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_4_2', #
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_5_1',
    # 'KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_5_2', #
    # # N10
    # 'KI-dataset/For KTH/Helena/N10/N10_1_1', #
    # 'KI-dataset/For KTH/Helena/N10/N10_1_2', #
    'KI-dataset/For KTH/Helena/N10/N10_1_3' #
    # 'KI-dataset/For KTH/Helena/N10/N10_2_1', #
    # 'KI-dataset/For KTH/Helena/N10/N10_2_2', #
    # 'KI-dataset/For KTH/Helena/N10/N10_3_1',  # GO
    # 'KI-dataset/For KTH/Helena/N10/N10_3_2', #
    # 'KI-dataset/For KTH/Helena/N10/N10_4_1', #
    # 'KI-dataset/For KTH/Helena/N10/N10_4_2',  # GO
    # 'KI-dataset/For KTH/Helena/N10/N10_5_1', #
    # 'KI-dataset/For KTH/Helena/N10/N10_5_2', #
    # 'KI-dataset/For KTH/Helena/N10/N10_6_2',  # GO
    # 'KI-dataset/For KTH/Helena/N10/N10_7_2', #
    # 'KI-dataset/For KTH/Helena/N10/N10_7_3', #
    # 'KI-dataset/For KTH/Helena/N10/N10_7_4', #
    # 'KI-dataset/For KTH/Helena/N10/N10_8_2', #
    # 'KI-dataset/For KTH/Helena/N10/N10_8_3', #
    # 'KI-dataset/For KTH/Helena/N10/N10_8_4',
    # # P11
    # 'KI-dataset/For KTH/Helena/P11/P11_1_1',
    # 'KI-dataset/For KTH/Helena/P11/P11_1_2',
    # 'KI-dataset/For KTH/Helena/P11/P11_2_1',
    # 'KI-dataset/For KTH/Helena/P11/P11_2_2',
    # 'KI-dataset/For KTH/Helena/P11/P11_3_1',
    # 'KI-dataset/For KTH/Helena/P11/P11_3_2',
    # 'KI-dataset/For KTH/Helena/P11/P11_4_1',
    # # P16
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_1_1',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_1_2',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_2_1',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_2_2',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_3_1',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_3_2',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_4_1',
    # 'KI-dataset/For KTH/Helena/P16/P16_HE_Default_Extended_4_2',

    # 'KI-dataset/Extras/HE_T12193_90_Default_Extended_1_1',
    # 'KI-dataset/Extras/HE_T12193_90_Default_Extended_1_2',
    # 'KI-dataset/Extras/HE_T12193_90_Default_Extended_1_3',
    # 'KI-dataset/Extras/P01_1_1',
    # 'KI-dataset/Extras/P01_1_2',
    # 'KI-dataset/Extras/P01_2_1',
    # 'KI-dataset/Extras/P01_2_2',
    # 'KI-dataset/Extras/P01_3_1',
    # 'KI-dataset/Extras/P01_3_2',

    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_1_1',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_1_2',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_2_1',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_2_2',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_2_3',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_3_1',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_3_2',
    # 'KI-dataset/For KTH/Helena/P14/P14_HE_Default_Extended_3_3',

    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_1_1',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_1_2',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_2_1',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_2_2',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_2_3',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_3_1',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_3_2',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_3_3',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_4_2',
    # 'KI-dataset/For KTH/Helena/P22/P22_HE_Default_Extended_4_3',
    #
    # 'KI-dataset/For KTH/Helena/P26/P26_HE_Default_Extended_1_1',
    # 'KI-dataset/For KTH/Helena/P26/P26_HE_Default_Extended_1_2',
    # 'KI-dataset/For KTH/Helena/P26/P26_HE_Default_Extended_2_1',
    # 'KI-dataset/For KTH/Helena/P26/P26_HE_Default_Extended_2_2',
    # 'KI-dataset/For KTH/Helena/P26/P26_HE_Default_Extended_3_1',
    #
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_1_1',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_1_2',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_2_1',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_2_2',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_3_1',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_3_2',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_3_3',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_4_1',
    # 'KI-dataset/For KTH/Helena/P29/P29_HE_Default_Extended_4_2',
    #
    # 'KI-dataset/Extras/P30_HE_Default_Extended_1_5',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_1_6',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_2_4',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_2_5',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_2_6',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_3_3',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_3_4',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_3_5',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_4_2',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_4_3',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_4_4',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_5_1',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_5_2',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_5_3',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_6_1',
    # 'KI-dataset/Extras/P30_HE_Default_Extended_6_2',


] # Len 5

class Cell:
    def __init__(self, type, fib, epi, inf, lym, x, y, id=None, ):
        self.id = id
        self.type = type
        self.fib = fib
        self.epi = epi
        self.inf = inf
        self.lym = lym
        self.x = x
        self.y = y

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class CellController:
    def __init__(self):
        self.cells = []

    def __repr__(self):
        # return json.dumps(self)
        return json.dumps(self.cells, default=lambda o: o.__dict__, indent=4)

    def add_cell(self, new_cell):
        self.cells.append(new_cell)
        return new_cell

    def to_json(self):
        return json.dumps(self.cells, default=lambda o: o.__dict__, indent=4)

    def save_to_file(self, path):
        with open(path, 'w') as f:
            f.write(str(self.cells))


class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']


def export_to_json(boxes, labels, image_name, label_scores):

    cells_to_export = CellController()

    for i in range(len(boxes)):
        xCen = (boxes[i][0] + boxes[i][2])//2
        yCen = (boxes[i][1] + boxes[i][3])//2
        label = labels[i]
        cells_to_export.add_cell(Cell(class_names[int(label)], label_scores[i][2], label_scores[i][3], label_scores[i][0], label_scores[i][1], xCen, yCen, i))



    with open(f"json/{image_name}.json", 'w') as f:
            f.write(str(cells_to_export.to_json()))



def _get_detections(dataset, retinanet, effNet, score_threshold=0.05, max_detections=1000, save_config=None, eval_threshold=0.25):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())]
                      for j in range(len(dataset))]
    retinanet.eval() # EfficientDet extends torch.nn.Module who has the eval method.
    effNet.eval() # same here


    all_boxes = [] #
    all_labels = [] #
    all_label_scores = [] #


    # Efficientnet
    mean = np.mean(dataset.image)
    std = np.std(dataset.image)
    normalize = transforms.Normalize(mean=[0.72482513, 0.59128926, 0.76370454],
                                     std=[0.18745105, 0.2514997,  0.15264913])
    #normalize = transforms.Lambda(lambdaTransform) # advprop

    image_size = 64
    val_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Disabling gradient calculation is useful for inference, when you are sure
    # that you will not call :meth:`Tensor.backward()`. It will reduce memory
    # consumption for computations that would otherwise have `requires_grad=True`.
    with torch.no_grad():

        # for each item in the KiDataset()
        for index in range(len(dataset)):
            data = dataset[index]

            # run network
            data['img'] = torch.from_numpy(data['img'])
            # print(data['img'])
            if torch.cuda.is_available():
                scores, labels, boxes, all_scores = retinanet(data['img'].permute( # basically calling EfficientDet.forward()
                    2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes, all_scores = retinanet(data['img'].permute(
                    2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy() # cpu() -> move tensor to the cpu
            labels = labels.cpu().numpy() # numpy() -> convert tensor to numpy: a = torch.ones(5) -> tensor([1., 1., 1., 1., 1.]) ### b = a.numpy() -> [1. 1. 1. 1. 1.]
            boxes = boxes.cpu().numpy()
            all_scores = all_scores.cpu().numpy()

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0] # threshold = 0.05
            eval_indices = np.where((scores > eval_threshold) & (scores < score_threshold))[0]
            if indices.shape[0] > 0 or eval_indices.shape[0] > 0:
                # select those scores
                cert_scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-cert_scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = cert_scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_label_scores = all_scores[indices[scores_sort], :]
                image_detections = np.concatenate([image_boxes, np.expand_dims(
                    image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)


                if eval_indices.shape[0] > 0:
                    # run EfficientNet to decide uncertain scores
                    eval_boxes = boxes[eval_indices, :]
                    cell_imgs = np.zeros((eval_boxes.shape[0], 32, 32, 3))
                    for i in range(eval_boxes.shape[0]):
                        xmin = int(max(min((eval_boxes[i][0]+eval_boxes[i][2])/2-16, 512-32),0))
                        xmax = int(max(min((eval_boxes[i][0]+eval_boxes[i][2])/2+16, 512),32))
                        ymin = int(max(min((eval_boxes[i][1]+eval_boxes[i][3])/2-16, 512-32),0))
                        ymax = int(max(min((eval_boxes[i][1]+eval_boxes[i][3])/2+16, 512),32))
                        cell_imgs[i,:,:,:] = dataset.image[xmin:xmax, ymin:ymax, :]

                    tensor_train_x = torch.from_numpy(cell_imgs).float().to('cpu')
                    tensor_train_x = tensor_train_x.permute(0, 3, 1, 2)
                    input_tensor = torch.empty(eval_boxes.shape[0], 3, image_size, image_size)
                    for i in range(tensor_train_x.size(0)):
                        input_tensor[i,:,:,:] = val_tsfm(tensor_train_x[i,:,:,:])

                    out = effNet(input_tensor)
                    #m = torch.nn.Sigmoid()
                    #out = m(out)


                    eval_scores = out.cpu().numpy()
                    eval_label_scores = eval_scores

                    eval_labels = np.argmax(eval_scores, axis=1)
                    eval_scores = np.amax(eval_scores, axis=1)
                    eval_detections = np.concatenate([eval_boxes, np.expand_dims(
                        eval_scores, axis=1), np.expand_dims(eval_labels, axis=1)], axis=1)
                    all_labels.extend(eval_labels.tolist())
                    all_boxes.append(np.vstack((image_boxes, eval_boxes)).tolist())
                    all_label_scores.extend(np.vstack((image_label_scores, eval_label_scores)).tolist())
                    # copy detections to all_detections
                    for label in range(dataset.num_classes()):
                        all_detections[index][label] = np.vstack((eval_detections[eval_detections[:, -1] == label, :-1], image_detections[image_detections[:, -1] == label, :-1]))

                else:
                    all_boxes.append(image_boxes.tolist())
                    all_label_scores.extend(image_label_scores.tolist())
                    # copy detections to all_detections
                    for label in range(dataset.num_classes()):
                        all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]

                all_labels.extend(image_labels.tolist())


            else:
                # copy detections to all_detections
                all_boxes.append([])
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    translated_boxes = translate_boxes(all_boxes)


    if save_config["export_json"]:
        export_to_json(translated_boxes, all_labels, save_config["export_prefix"]+dataset.filename, all_label_scores)


    return all_detections

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(
        a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(
        a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(
        generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        if len(annotations) > 0:
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        else:
            for label in range(generator.num_classes()):
                all_annotations[i][label] = np.empty((0, 4), dtype=np.int64)

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate(
    generator,
    retinanet,
    effNet,
    config,
    iou_threshold=0.5,
    score_threshold=0.45,
    max_detections=1000,
    # save_path=None,
    eval_threshold=0.45,

):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet       : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(
        generator, retinanet, effNet, score_threshold=score_threshold,
        max_detections=max_detections, save_config=config, eval_threshold=eval_threshold)
    all_annotations = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
        #print('Label {}: {}'.format(label, scores))

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue
        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations



    print('\nmAP:')
    avg_mAP = []
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        avg_mAP.append(average_precisions[label][0])
    print('avg mAP: {}'.format(np.mean(avg_mAP)))
    return np.mean(avg_mAP), average_precisions



config = {
    "enabled" : True,
    "export_prefix": "CNN_keb_54_",
    "export_json": True,
}

def main():
    if args.weight is not None:
        resume_path = str(args.weight)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(args.weight, map_location=lambda storage, loc: storage)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network

        model = EfficientDet( # loading model with checkpoint params [JORGE] aka RetinaNet
            num_classes=args.num_class,
            network=args.network,
            W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
            D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
            D_class=EFFICIENTDET[args.network]['D_class'],
            is_training=False,
            threshold=args.threshold,
            iou_threshold=args.iou_threshold)
        model.load_state_dict(checkpoint['state_dict'])

        effNet = EfficientNet.from_pretrained('efficientnet-b0', advprop=False, num_classes=4)
        checkpoint = torch.load('./models/model_kebnekaise.pth.tar', map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
            #     print('k = {}'.format(k))
                k = k[7:]
            #     print('k = {}'.format(k))
            new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict

        effNet.load_state_dict(new_state_dict)
        print("State loaded")

    if torch.cuda.is_available():  # this is for better performance if supported [JORGE]
        print("cuda available = True")
        model = model.cuda()
    else:
        print("cuda available = False")


    if config["enabled"]:

        '''
        test_dataset = KiDataset(
            root=args.dataset_root,
            filePath=args.filepath)
        evaluate(test_dataset, model, effNet)
        '''
        for i in range(len(label_paths)):  # for each image in the KI Dataset [JORGE], right now Rach_P19 to Rach_P25
            # for i in range(110, 128, 1): # for each image in the KI Dataset [JORGE], right now Rach_P19 to Rach_P25
            print(label_paths[i])
            test_dataset = KiDataset(
                root=args.dataset_root,
                filePath=label_paths[i])
            evaluate(test_dataset, model, effNet, config)

if __name__ == '__main__':
    main()

