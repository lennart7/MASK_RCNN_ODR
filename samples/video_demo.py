import cv2
import numpy as np
import os
import sys
import coco
import argparse
import pickle
import pandas as pd
from collections import defaultdict
from functools import partial
import tensorflow as tf
ROOT_DIR = os.path.abspath("../")
os.environ['KERAS_BACKEND'] = 'tensorflow'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib


sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def parse_args():

    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='MAskRCNN object detection and segmentation')

    parser.add_argument('--path', dest='path', help='provide the path of the image directory',
                        default=0, type=str)
    parser.add_argument('--tsv', dest='tsv', help='tsv eye tracking data',
                        default=0, type=str)

    args = parser.parse_args()

    return args



def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format('monitor', score) if score else 'monitor' 
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':

    
    args = parse_args()
    Path = str(args.path)

    tsv = str(args.tsv)
    
    capture = cv2.VideoCapture(Path)

    cols = ['Recording timestamp [μs]', 'Gaze point X [MCS px]', 'Gaze point Y [MCS px]']

    tsv_data = pd.read_csv(tsv, sep='\t')[cols]

    tsv_data = tsv_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False).reset_index(drop=True)

    ts, gaze_x, gaze_y = cols

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (frame_width, frame_height))
    print("Video opened:", out.isOpened())

    MIN_FRAME = 200
    MAX_FRAME = 2000
    SKIP_FRAME = 2
    frame_idx = 0
    while frame_idx < MAX_FRAME:
        frame_idx += 1
        ret, frame = capture.read()
        if frame_idx < MIN_FRAME or frame_idx % SKIP_FRAME != 0:
            print("skipping frame: ", frame_idx)
            continue
        print("Writing frame: ", frame_idx)

        results = model.detect([frame], verbose=0)
        all_detections = results[0]

        tvs = defaultdict(list)

        for i, class_id in enumerate(all_detections['class_ids']):
            if class_id != 63:
                continue
            tvs['rois'].append(all_detections['rois'][i])
            tvs['class_ids'].append(all_detections['class_ids'][i])
            tvs['scores'].append(all_detections['scores'][i])
            tvs['masks'].append(all_detections['masks'][:, :, i])

        tvs['rois'] = np.array(tvs['rois'])
        tvs['class_ids'] = np.array(tvs['class_ids'])
        tvs['scores'] = np.array(tvs['scores'])

        tmp = tvs['masks']

        tvs['masks'] = np.zeros((frame_height, frame_width, len(tvs['class_ids'])))

        for i in range(len(tvs['class_ids'])):
            # hacky way of arranging numpy masks.. not ideal
            tvs['masks'][:, :, i] = tmp[i]

        frame = display_instances(
            frame, tvs['rois'], tvs['masks'], tvs['class_ids'], class_names, tvs['scores']
        )

        ref_time = capture.get(cv2.CAP_PROP_POS_MSEC) * 1000

        min_idx = ((tsv_data[ts] - ref_time) ** 2).argmin()

        eyegaze = tsv_data.loc[[min_idx], [gaze_x, gaze_y]]
        center = (int(eyegaze.values[0][0]), int(eyegaze.values[0][1]))

        cv2.circle(frame, center, 5, (0, 255, 0), 2);

        def is_inside(gaze, box):
            y1, x1, y2, x2 = box
            cond1 = x1 <= gaze[0] and gaze[0] <= x2
            cond2 = y1 <= gaze[1] and gaze[1] <= y2
            return  cond1 and cond2

        if len(tvs['rois']) == 2:
            box_left = tvs['rois'][1]
            box_right = tvs['rois'][0]

            if (is_inside(center, box_left)):
                cv2.putText(frame, "Gazing left monitor", (10, frame_height-100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

            if (is_inside(center, box_right)):
                cv2.putText(frame, "Gazing right monitor", (10, frame_height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

        out.write(frame)

        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

    out.release()
    capture.release()
    cv2.destroyAllWindows()
