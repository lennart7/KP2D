import os
import argparse
import os
import pickle
import random
import subprocess
from itertools import combinations

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.evaluation.evaluate import evaluate_keypoint_net
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2d.utils.image import to_color_normalized, to_gray_normalized
from kp2d.datasets.augmentations import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample)
DETECTION_THRESH = 0.999

def make_matching_plot_fast(image0, image1, mkpts0,
                            mkpts1, color=None, text=None, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0, R = image0.shape
    H1, W1, R = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, R), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1
    # out = np.stack([out]*3, -1)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    if color is not None:
        color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    else:
        color = np.array([np.uint8([255, 255, 255])] * len(mkpts0))

    for (y0, x0), (y1, x1), c in zip(mkpts0, mkpts1, color):
        assert(x0 <= H0 and y0 <= W0)
        c = tuple(c)
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, (255, 255, 255), -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    if text:
        Ht = int(30 * sc)  # text height
        txt_color_fg = (255, 255, 255)
        txt_color_bg = (0, 0, 0)
        for i, t in enumerate(text):
            cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    if small_text:
        Ht = int(18 * sc)  # text height
        for i, t in enumerate(reversed(small_text)):
            cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                        0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                        0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']
    files = os.listdir(args.input_dir)

    params = dict(use_color=True, do_upsample=True, do_cross=True)

    model = KeypointNetwithIOLoss(**params)

    model.load_state_dict(checkpoint['state_dict'])

    keypoint_net = model.keypoint_net
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    keypoint_net.training = False

    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    eval_params = [{'res': (320, 240), 'top_k': 300, }]
    eval_params += [{'res': (640, 480), 'top_k': 1000, }]

    for index, (f1, f2) in enumerate(combinations(files, 2)):
        with torch.no_grad():
            for params in eval_params:
                image_1, raw_image_1 = read_rgb_file(os.path.join(args.input_dir, f1), params['res'])
                image_2, raw_image_2 = read_rgb_file(os.path.join(args.input_dir, f2), params['res'])

                sample = torch.stack([image_1, image_2])

                score, coord, desc = keypoint_net(sample)
                score = score.cpu().numpy()
                coord = coord.cpu().numpy()
                desc = desc.cpu().numpy()
                image_1 = image_1.cpu().numpy()
                image_2 = image_2.cpu().numpy()

                B, C, Hc, Wc = desc.shape
                keypoint_matches = get_matched_keypoints(desc, score)

                print(f"{len(keypoint_matches)} keypoints detected")
                if len(keypoint_matches) == 0:
                    continue
                [kpts0, scores_0, kpts1, scores_1] = zip(*keypoint_matches)
                raw_image_1 = cv2.cvtColor(np.array(raw_image_1), cv2.COLOR_BGR2RGB)
                raw_image_2 = cv2.cvtColor(np.array(raw_image_2), cv2.COLOR_BGR2RGB)
                make_matching_plot_fast(cv2.resize(raw_image_1, params['res']), cv2.resize(raw_image_2, params['res']), kpts0, kpts1,
                                        path=f"./results_{params['res'][0]}x{params['res'][1]}_{index}.jpg")

def get_matched_keypoints(desc, score):
    _, C, Hc, Wc = desc.shape
    matches = []
    keypoints_a = desc[0, :, :, :]
    keypoints_b = desc[1, :, :, :]
    for i in range(Hc):
        for j in range(Wc):
            if score[0, 0, i, j] > DETECTION_THRESH:
                kp, kp_desc, sc = nearest_neighbor(desc[0, :, i, j], desc[1, :, :, :], score[1, :, :, :])
                if kp is None:
                    continue
                b_kp, b_kp_desc, b_sc = nearest_neighbor(kp_desc, desc[0, :, :, :], score[0, :, :, :])
                if i == b_kp[0] and j == b_kp[1]:
                    matches.append([np.array([i*8, j*8]), score[0, 0, i, j], np.array([kp[0]*8, kp[1]*8]), sc])

    return matches

def nearest_neighbor(kp_desc, keypoints, scores):
    # match a descripter to the nearest_neighbor in
    # another set of keypoints
    C, Hc, Wc = keypoints.shape
    min_dist = np.inf
    min_keypoint = [None, None, None]
    for i in range(Hc):
        for j in range(Wc):
            dist = np.linalg.norm(kp_desc - keypoints[:, i, j])
            if dist < min_dist and scores[0, i, j] > DETECTION_THRESH:
                min_dist = dist
                min_keypoint = (np.array([i, j]), keypoints[:, i, j], scores[0, i, j])
    return min_keypoint

def read_rgb_file(fpath, shape):
    image = Image.open(fpath)
    sample = dict(image=image)
    sample = resize_sample(sample, image_shape=shape)
    sample = to_tensor_sample(sample)

    return sample['image'].cuda(), image


if __name__ == "__main__":
    main()
