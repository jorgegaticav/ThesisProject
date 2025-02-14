import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from collections import OrderedDict

import PIL
from PIL import Image


from efficientnet_pytorch import EfficientNet

from datasets.visual_aug import visualize, compare
from datasets import (Augmenter, Normalizer,
                      Resizer, collater, detection_collate,
                      get_augumentation)
from datasets.parseOneKI import KiDataset, translate_boxes
from models.efficientdet import EfficientDet
from utils import EFFICIENTDET, get_state_dict
from neoExport import save_results, generate_graph
from create_labelimg_xml import create_labelimg

label_paths = [
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_1_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_2_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_2_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_3_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_3_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_4_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_4_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P13/P13_1_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P13/P13_1_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P13/P13_2_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P13/P13_2_2',
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
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_1_1',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_1',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_2',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_3_1',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_3_2',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_4_2',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_5_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_1_1',
    #'KI-Dataset/For KTH/Helena/N10/N10_1_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_2_1',
    #'KI-Dataset/For KTH/Helena/N10/N10_2_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_3_1',
    #'KI-Dataset/For KTH/Helena/N10/N10_3_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_4_1',
    #'KI-Dataset/For KTH/Helena/N10/N10_4_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_5_1',
    #'KI-Dataset/For KTH/Helena/N10/N10_5_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_6_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_7_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_7_3',
    #'KI-Dataset/For KTH/Helena/N10/N10_7_4',
    #'KI-Dataset/For KTH/Helena/N10/N10_8_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_8_3',
] # Len 58

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

def lambdaTransform(image):
    return image * 2.0 - 1.0

def _get_detections(dataset, retinanet, effNet, score_threshold=0.05, max_detections=1000, save_path=None, eval_threshold=0.25):
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
    all_detections = [[None for i in range(
        dataset.num_classes())] for j in range(len(dataset))]
    retinanet.eval()
    effNet.eval()


    all_boxes = []
    all_labels = []
    all_label_scores = []


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

    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]

            # run network
            data['img'] = torch.from_numpy(data['img'])
            #print(data['img'])
            if torch.cuda.is_available():
                scores, labels, boxes, all_scores = retinanet(data['img'].permute(
                    2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes, all_scores = retinanet(data['img'].permute(
                    2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()
            all_scores = all_scores.cpu().numpy()

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
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

    # Export result to Neo4j and generate neighbors
    #save_results(translate_boxes(all_boxes), all_labels, dataset.filename+"_final", all_label_scores)
    #generate_graph(dataset.filename+"_final")

    # Visualize results and save images
    imarray = dataset.normal_image
    imarray[:,:,[0,2]] = imarray[:,:,[2,0]]
    all_boxes = translate_boxes(all_boxes)
    vis = visualize(imarray, all_boxes, all_labels)
    #create_labelimg(all_boxes, all_labels, dataset.filePath+'.tif')
    #vis2 = compare(dataset.image, translate_boxes(all_boxes), dataset.targets)

    cv2.imshow('image', vis)
    cv2.waitKey(0)
    #cv2.imshow('image', vis2)
    #cv2.waitKey(0)
    #cv2.imwrite('visualize_final_{}.png'.format(dataset.filename),vis*255)
    #cv2.imwrite('compare_{}.png'.format(dataset.filename),vis2*255)
    cv2.destroyAllWindows()

    return all_detections


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


def evaluate(
    generator,
    retinanet,
    effNet,
    iou_threshold=0.5,
    score_threshold=0.45,
    max_detections=1000,
    save_path=None,
    eval_threshold=0.45
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(
        generator, retinanet, effNet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path, eval_threshold=eval_threshold)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EfficientDet Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset_root', default='datasets/',
                        help='Dataset root directory path')
    parser.add_argument('--filepath', default='KI-dataset/For KTH/Rachael/Rach_P13/P13_2_2',
                        help='Dataset root directory path')
    parser.add_argument('-t', '--threshold', default=0.25,
                        type=float, help='Visualization threshold')
    parser.add_argument('-it', '--iou_threshold', default=0.5,
                        type=float, help='Visualization threshold')
    parser.add_argument('--weight', default='./saved/weights/kebnekaise/checkpoint_54.pth', type=str,
                        help='Checkpoint state_dict file to resume training from')
    args = parser.parse_args()
    # N10 and P19 and P7
    # KI-dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_1_1
    # KI-dataset/For KTH/Nikolce/N10_1_2
    # KI-dataset/For KTH/Rachael/Rach_P19/P19_2_1



    if(args.weight is not None):
        resume_path = str(args.weight)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(
            args.weight, map_location=lambda storage, loc: storage)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        model = EfficientDet(
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
                k = k[7:]
            new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict

        effNet.load_state_dict(new_state_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    '''
    test_dataset = KiDataset(
        root=args.dataset_root,
        filePath=args.filepath)
    evaluate(test_dataset, model, effNet)

    '''
    for i in range(len(label_paths)):
        test_dataset = KiDataset(
            root=args.dataset_root,
            filePath=label_paths[i])
        evaluate(test_dataset, model, effNet)
