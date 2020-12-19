"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os
import time

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

from PIL import Image, ImageDraw, ImageFilter

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


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_inferences = [None for i in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # Grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        inference_time = time.time() - start

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            __raw_image = generator.load_image(i, mode='rgb') ####

            draw_annotations(__raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name, color=(255,0,0)) ####
            draw_detections(__raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold, color=(0,255,0)) ####

            __orig_name = generator.image_names[i] ####
            cv2.imwrite(os.path.join(save_path, '{:03d}-{}.png'.format(i, __orig_name)), __raw_image) ####

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        all_inferences[i] = inference_time

    return all_detections, all_inferences


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    mode=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_inferences = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        __tp = __fp = __tn = __fn = 0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            __scores = np.zeros((0,)) ####
            __false_positives = np.zeros((0,)) ####
            __true_positives  = np.zeros((0,)) ####

            __image_path = generator.image_path(i) ####

            for j, d in enumerate(detections):
                scores = np.append(scores, d[4])
                __scores = np.append(__scores, d[4]) ####

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    __false_positives = np.append(__false_positives, 1) ####
                    __true_positives  = np.append(__true_positives, 0) ####
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    __false_positives = np.append(__false_positives, 0) ####
                    __true_positives  = np.append(__true_positives, 1) ####
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    __false_positives = np.append(__false_positives, 1) ####
                    __true_positives  = np.append(__true_positives, 0) ####                
            
            __indices = np.argsort(-__scores) ####
            __false_positives = __false_positives[__indices] ####
            __true_positives  = __true_positives[__indices] ####
            __num_tp = np.sum(__true_positives)
            __num_fp = np.sum(__false_positives)
            __num_anno = annotations.shape[0]
            __num_pos = __num_fp + __num_tp
            if __num_anno and __num_tp >= 1:
                __res = 'TP'
                __tp += 1
            elif __num_anno and __num_tp == 0:
                __res = 'FN'
                __fn += 1
            elif __num_pos > 0:
                __res = 'FP'
                __fp += 1
            else:
                __res = 'TN'
                __tn += 1

            if mode == 'TEST':
                print("{:03d}\t{}\t{}\t{}\t{}\t{}".format(i, __num_tp, __num_fp, __num_tp + __num_fp, __num_anno, __res))
            if mode == 'SCORE':
                print("{:03d}\t{}\t{}\t{}".format(i, __image_path, __num_anno, __scores))

        if mode == 'SCORE':
            exit()
        if mode == 'TEST':
            print("{}\t{}\t{}\t{}\t{}\t{}".format('idx', 'tp', 'fp', 'tp+fp', 'anno', 'FP/FP/TN/FN'))
            print('TP\t{}'.format(__tp))
            print('FP\t{}'.format(__fp))
            print('TN\t{}'.format(__tn))
            print('FN\t{}'.format(__fn))
            print('感度\t{}'.format(__tp / (__tp + __fn)))
            print('特異度\t{}'.format(__tn / (__tn + __fp)))
            print('精度\t{}'.format(__tp / (__tp + __fp)))
            print('陽性適中率\t{}'.format(__tp / (__tp + __fp)))
            print('陰性適中率\t{}'.format(__tn / (__tn + __fn)))
            print("iou_threshold : {}".format(iou_threshold))
            print("score_threshold : {}".format(score_threshold))
            exit()


        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)

        if mode == 'FROC':
            num_images = generator.size()
            print("iou_threshold : {}".format(iou_threshold))
            print("score_threshold : {}".format(score_threshold))
            print("num_annotations : {}".format(num_annotations))
            print("num_images : {}".format(num_images))
            cum_true_positive = 0
            cum_false_positive = 0
            print("======== FROC ========")
            print("score\tTP\tFP\tcum.TP\tcum.FP\tTPF\tFPI")
            for idx in indices:
                cum_true_positive += true_positives[idx]
                cum_false_positive += false_positives[idx]
                print("{:.4f}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}".format(
                    scores[idx],
                    true_positives[idx],
                    false_positives[idx],
                    cum_true_positive,
                    cum_false_positive,
                    cum_true_positive / num_annotations,
                    cum_false_positive / num_images,
                ))
            continue

        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]
        
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    if mode == 'FROC':
        exit(0)
        
    # inference time
    inference_time = np.sum(all_inferences) / generator.size()

    return average_precisions, inference_time
