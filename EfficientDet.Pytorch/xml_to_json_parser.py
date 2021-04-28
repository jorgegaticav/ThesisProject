# Load
import xml.etree.ElementTree as ET

import json

xml_paths = {
    # ANNOTATIONS

    # P9 #####################################################################################################

    # Daniela
    # 'reannotations/DN/P9/P9_1_1_DN.xml'
    # 'reannotations/DN/P9/P9_2_1_DN.xml'
    # 'reannotations/DN/P9/P9_2_2_DN.xml'

    # Helena
    # "P9_1_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P9_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P9_1_1_reannot_HA.xml',
    #           'datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_1_1.tif'],
    # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P9_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P9_2_1_reannot_HA.xml',
    # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P9_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P9_2_2_reannot_HA.xml',
    # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P9_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P9_3_1_reannot_HA.xml',
    # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P9_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P9_4_1_reannot_HA.xml',

    # "N10_1_1":['labelIMG/reannotations/Ground truth Gen 0/N10/N10_1_1_HA.xml',
    #           'datasets/KI-dataset/For KTH/Helena/Helena_P7/N10_1_1.tif'],
    #
    # "N10_1_2":['labelIMG/reannotations/Ground truth Gen 0/N10/N10_1_2_HA.xml',
    #           'datasets/KI-dataset/For KTH/Helena/Helena_P7/N10_1_2.tif'],
    #
    # "N10_1_3":['labelIMG/reannotations/Ground truth Gen 0/N10/N10_1_3_HA.xml',
    #           'datasets/KI-dataset/For KTH/Helena/Helena_P7/N10_1_3.tif'],
    #
    # "N10_2_1":['labelIMG/reannotations/Ground truth Gen 0/N10/N10_2_1_HA.xml',
    #           'datasets/KI-dataset/For KTH/Helena/Helena_P7/N10_2_1.tif'],
    #
    # "N10_2_2":['labelIMG/reannotations/Ground truth Gen 0/N10/N10_2_2_HA.xml',
    #           'datasets/KI-dataset/For KTH/Helena/Helena_P7/N10_2_2.tif'],


    # Rachael
    # "P9_1_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_1_1_RVS.xml',
    #           'datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_1_1.tif'],
    # "P9_2_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_2_1_RVS.xml',
    #           'datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_2_1.tif'],
    # "P9_2_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_2_2_RVS.xml',
    #           'datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_2_2.tif'],
    # "P9_3_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_3_1_RVS.xml',
    #           'datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_3_1.tif'],
    # "P9_3_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_3_2_RVS.xml',
    #           'datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_3_2.tif'],
    # "P9_4_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_4_1_RVS.xml',
    #           "datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_4_1.tif"],
    # "P9_4_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P9_reannotated_XLM_files/gen_1_VanillaExpNew_bs_4_epochs_500_P9_4_2_RVS.xml',
    #           "datasets/KI-dataset/For KTH/Rachael/Rach_P9/P9_4_2.tif"],
    #
    # # P13 #####################################################################################################
    #
    # # P19 #####################################################################################################
    # # Daniela
    # # 'labelIMG/reannotations/Reannotations Gen 1/DN/P19/P19_1_1_DN.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/DN/P19/P19_2_1_DN.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/DN/P19/P19_2_2_DN.xml',
    #
    # # Helena
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P19_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P19_1_1_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P19_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P19_2_1_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P19_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P19_2_2_ reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P19_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P19_3_1_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P19_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P19_3_2_reannot_HA.xml',
    #
    # # Rachael
    # "P19_1_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P19_reannotated_XML_files/gen_1_VanillaExpNew_bs_4_epochs_500_P19_1_1_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P19/P19_1_1.tif'],
    # "P19_1_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P19_reannotated_XML_files/gen_1_VanillaExpNew_bs_4_epochs_500_P19_1_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P19/P19_1_2.tif'],
    # "P19_2_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P19_reannotated_XML_files/gen_1_VanillaExpNew_bs_4_epochs_500_P19_2_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P19/P19_2_2.tif'],
    # "P19_2_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P19_reannotated_XML_files/gen_1_VanillaExpNew_bs_4_epochs_500_P19_2_1_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P19/P19_2_1.tif'],
    # "P19_3_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P19_reannotated_XML_files/gen_1_VanillaExpNew_bs_4_epochs_500_P19_3_1_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P19/P19_3_1.tif'],
    # "P19_3_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P19_reannotated_XML_files/gen_1_VanillaExpNew_bs_4_epochs_500_P19_3_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P19/P19_3_2.tif'],
    #
    # # P20 #####################################################################################################
    #
    # # Daniela
    # #'labelIMG/reannotations/Reannotations Gen 1/DN/P20/'
    #
    # # Helena
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P20_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P20_3_2_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P20_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P20_4_1_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P20_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P20_4_2_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P20_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P20_5_1_reannot_HA.xml',
    # # 'labelIMG/reannotations/Reannotations Gen 1/Reannotations_HA/P20_gen_1_reannot_HA/gen_1_VanillaExpNew_bs_4_epochs_500_P20_6_1_reannot_HA.xml',
    # #
    # # Rachael
    # "P20_1_3":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_1_3_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_1_3.tif'],
    # "P20_2_3":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_2_3_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_2_3.tif'],
    # "P20_3_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_3_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_3_2.tif'],
    # "P20_3_3":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_3_3_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_3_3.tif'],
    # "P20_4_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_4_1_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_4_1.tif'],
    # "P20_4_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_4_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_4_2.tif'],
    # "P20_5_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_5_1_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_5_1.tif'],
    # "P20_5_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_5_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_5_2.tif'],
    # "P20_6_1":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_6_1_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_6_1.tif'],
    # "P20_6_2":['labelIMG/reannotations/Reannotations Gen 1/Reannotations_RVS/P20_reannotated_xml_files/gen_1_VanillaExpNew_bs_4_epochs_500_P20_6_2_RVS.xml',
    #            'datasets/KI-dataset/For KTH/Rachael/Rach_P20/P20_6_2.tif'],


    # Gen 0
    "12193_90_Default_Extended_1_1": ['labelIMG/reannotations/Ground truth Gen 0/12193_90/HE_T12193_90_Default_Extended_1_1_HA.xml'],
    "12193_90_Default_Extended_1_2": ['labelIMG/reannotations/Ground truth Gen 0/12193_90/HE_T12193_90_Default_Extended_1_2_HA.xml'],

    "N10_1_1": ['labelIMG/reannotations/Ground truth Gen 0/N10/N10_1_1_HA.xml'],
    "N10_1_2": ['labelIMG/reannotations/Ground truth Gen 0/N10/N10_1_2_HA.xml'],
    "N10_1_3": ['labelIMG/reannotations/Ground truth Gen 0/N10/N10_1_3_HA.xml'],
    "N10_2_1": ['labelIMG/reannotations/Ground truth Gen 0/N10/N10_2_1_HA.xml'],
    "N10_2_2": ['labelIMG/reannotations/Ground truth Gen 0/N10/N10_2_2_HA.xml'],

    "P7_HE_Default_Extended_1_1": ['labelIMG/reannotations/Ground truth Gen 0/P7/P7_HE_Default_Extended_1_1.xml'],
    "P7_HE_Default_Extended_2_1": ['labelIMG/reannotations/Ground truth Gen 0/P7/P7_HE_Default_Extended_2_1.xml'],
    "P7_HE_Default_Extended_2_2": ['labelIMG/reannotations/Ground truth Gen 0/P7/P7_HE_Default_Extended_2_2.xml'],
    "P7_HE_Default_Extended_3_1": ['labelIMG/reannotations/Ground truth Gen 0/P7/P7_HE_Default_Extended_3_1.xml'],

    "P9_1_1": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_1_1.xml'],
    "P9_2_1": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_2_1.xml'],
    "P9_2_2": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_2_2.xml'],
    "P9_3_1": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_3_1.xml'],
    "P9_3_2": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_3_2.xml'],
    "P9_4_1": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_4_1.xml'],
    "P9_4_2": ['labelIMG/reannotations/Ground truth Gen 0/P9/P9_4_2.xml'],

    "P11_1_1": ['labelIMG/reannotations/Ground truth Gen 0/P11/P11_1_1_HA.xml'],
    "P11_1_2": ['labelIMG/reannotations/Ground truth Gen 0/P11/P11_1_2_HA.xml'],
    "P11_2_1": ['labelIMG/reannotations/Ground truth Gen 0/P11/P11_2_1_HA.xml'],
    "P11_2_2": ['labelIMG/reannotations/Ground truth Gen 0/P11/P11_2_2_HA.xml'],
    "P11_3_1": ['labelIMG/reannotations/Ground truth Gen 0/P11/P11_3_1_HA.xml'],

    "P13_1_1": ['labelIMG/reannotations/Ground truth Gen 0/P13/P13_1_1_fully labelled.xml'],
    "P13_1_2": ['labelIMG/reannotations/Ground truth Gen 0/P13/P13_1_2.xml'],
    "P13_2_1": ['labelIMG/reannotations/Ground truth Gen 0/P13/P13_2_1.xml'],
    "P13_2_2": ['labelIMG/reannotations/Ground truth Gen 0/P13/P13_2_2.xml'],

    "P19_1_1": ['labelIMG/reannotations/Ground truth Gen 0/P19/P19_1_1.xml'],
    "P19_1_2": ['labelIMG/reannotations/Ground truth Gen 0/P19/P19_1_2.xml'],
    "P19_2_1": ['labelIMG/reannotations/Ground truth Gen 0/P19/P19_2_1.xml'],
    "P19_2_2": ['labelIMG/reannotations/Ground truth Gen 0/P19/P19_2_2.xml'],
    "P19_3_1": ['labelIMG/reannotations/Ground truth Gen 0/P19/P19_3_1.xml'],
    "P19_3_2": ['labelIMG/reannotations/Ground truth Gen 0/P19/P19_3_2.xml'],

    "P20_1_3": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_1_3.xml'],
    "P20_1_4": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_1_4.xml'],
    "P20_2_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_2_2.xml'],
    "P20_2_3": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_2_3.xml'],
    "P20_2_4": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_2_4.xml'],
    "P20_3_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_3_1.xml'],
    "P20_3_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_3_2.xml'],
    "P20_3_3": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_3_3.xml'],
    "P20_4_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_4_1.xml'],
    "P20_4_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_4_2.xml'],
    "P20_4_3": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_4_3.xml'],
    "P20_5_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_5_1.xml'],
    "P20_5_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_5_2.xml'],
    "P20_6_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_6_1.xml'],
    "P20_6_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_6_2.xml'],
    "P20_7_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_7_1.xml'],
    "P20_7_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_7_2.xml'],
    "P20_8_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_8_1.xml'],
    "P20_8_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_8_2.xml'],
    "P20_9_1": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_9_1.xml'],
    "P20_9_2": ['labelIMG/reannotations/Ground truth Gen 0/P20/P20_9_2.xml'],

    "P25_2_1": ['labelIMG/reannotations/Ground truth Gen 0/P25/P25_2_1.xml'],
    "P25_3_1": ['labelIMG/reannotations/Ground truth Gen 0/P25/P25_3_1.xml'],
    "P25_3_2": ['labelIMG/reannotations/Ground truth Gen 0/P25/P25_3_2.xml'],
    "P25_4_2": ['labelIMG/reannotations/Ground truth Gen 0/P25/P25_4_2.xml'],
    "P25_5_1": ['labelIMG/reannotations/Ground truth Gen 0/P25/P25_5_1.xml'],
    "P25_8_2": ['labelIMG/reannotations/Ground truth Gen 0/P25/P25_8_2.xml'],

    "P28_7_5": ['labelIMG/reannotations/Ground truth Gen 0/P28/P28_7_5_HA_fully labelled.xml'],
    "P28_8_5": ['labelIMG/reannotations/Ground truth Gen 0/P28/P28_8_5_RVS_fully labelled.xml'],
    "P28_10_4": ['labelIMG/reannotations/Ground truth Gen 0/P28/P28_10_4_RVS.xml'],
    "P28_10_5": ['labelIMG/reannotations/Ground truth Gen 0/P28/P28_10_5_RVS.xml'],

}

cell_label = {
    'inflammatory' : 0,
    'lymphocyte' : 1,
    'fibroblast and endothelial' : 2,
    'epithelial': 3,
    'apoptosis / civiatte body': 4
}

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']

def read_labelIMG_xml(xml_file_path: str):

    print("xml_file_path = {}".format(xml_file_path))
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    all_boxes = []
    all_labels = []
    all_label_scores = []

    filename = root.find('filename').text

    for boxes in root.iter('object'):

        # print("filename: {}".format(filename))

        label = cell_label[boxes.find("name").text] # cell type
        # label = cell_label[boxes.find("name")] # cell type
        # print("label: {}".format(label))

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        position_array = [xmin, ymin, xmax, ymax]
        all_boxes.append(position_array)
        all_labels.append(label)
        # generate label scores
        features = []

        features.append(1 if label == 0 else 0) # inflammatory
        features.append(1 if label == 1 else 0) # lymphocyte
        features.append(1 if label == 2 else 0) # fibroblast and endothelial
        features.append(1 if label == 3 else 0) # epithelial
        # features = [1 if label == "inflammatory" else 0, 1 if label == "lymphocyte" else 0, 1 if label == "fibroblast and endothelial" else 0, 1 if label == "epithelial" else 0]
        all_label_scores.append(features)

    return all_boxes, all_labels, filename, all_label_scores

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


def export_to_json(boxes, labels, image_name, label_scores, prefix):

    cells_to_export = CellController()

    for i in range(len(boxes)):
        xCen = (boxes[i][0] + boxes[i][2])//2
        yCen = (boxes[i][1] + boxes[i][3])//2
        label = labels[i]
        cells_to_export.add_cell(Cell(class_names[int(label)], label_scores[i][2], label_scores[i][3], label_scores[i][0], label_scores[i][1], xCen, yCen, i))

    with open(f"json/{prefix}{image_name}.json", 'w') as f:
            f.write(str(cells_to_export.to_json()))


visualize_many = {
    "enabled": True,
    # "save": False,
    # "save_suffix": "",
    "export_json": True,
    # "export": False,
    # "export_prefix": "BM_K4"
}


def main():
    if visualize_many["enabled"]:
        # for xml in xml_paths:
        for key, value in xml_paths.items():
            # print("value[1]: {}".format(value[1]))
            xml_all_boxes, xml_all_labels, xml_filename, all_label_scores = read_labelIMG_xml(value[0])

            if visualize_many["export_json"]:
                export_to_json(xml_all_boxes, xml_all_labels, xml_filename[:-4], all_label_scores, "gen0_")


if __name__ == "__main__":
    main()
