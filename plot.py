import os
import argparse
from itertools import combinations

import cv2
import numpy as np
import torch
from PIL import Image

from kp2d.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2d.datasets.augmentations import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample)
DETECTION_THRESH = 0.5


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

    # eval_params = [{'res': (320, 240), 'top_k': 300, }]
    params = {'res': (480, 640), 'top_k': 1000, }

    # for index, (f1, f2) in enumerate(combinations(files, 2)):
    f1 = "00000000_rgb.png"
    f2 = "00000000_rgb_L.png"
    print(f"Detecting keypoints for: {f1} - {f2}")
    with torch.no_grad():
        # for params in eval_params:
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
        desc1 = desc[0, :, :, :]
        desc2 = desc[1, :, :, :]
        desc1 = desc1.reshape(C, Hc * Wc)
        desc2 = desc2.reshape(C, Hc * Wc)

        coord1 = coord[0].reshape(2, Hc * Wc)
        coord2 = coord[1].reshape(2, Hc * Wc)
        score1 = score[0].reshape(Hc * Wc)
        score2 = score[1].reshape(Hc * Wc)

        # transform into cv2.keypoints
        kp1 = []
        kp2 = []
        for i in range(Hc * Wc):
            x, y = coord1[:, i]
            kp = cv2.KeyPoint(x, y, 5)
            kp1.append(kp)
            x, y = coord2[:, i]
            kp = cv2.KeyPoint(x, y, 5)
            kp2.append(kp)

        # Initiate SIFT detector
        # sift = cv2.SIFT_create()
        # # # find the keypoints and descriptors with SIFT
        # C, W, H = raw_image_1.shape
        # raw_image_1_gray = cv2.cvtColor(raw_image_1, cv2.COLOR_RGB2GRAY)
        # raw_image_2_gray = cv2.cvtColor(raw_image_2, cv2.COLOR_RGB2GRAY)

        # kp1, des1 = sift.detectAndCompute(raw_image_1_gray, None)
        # kp2, des2 = sift.detectAndCompute(raw_image_2_gray, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1.T, desc2.T, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # Apply ratio test
        good = []
        for idx, (m, n) in enumerate(matches):
            if m.distance < 0.75 * n.distance and \
                    score1[idx] > DETECTION_THRESH and score2[idx] > DETECTION_THRESH:
                dist = np.linalg.norm(desc1.T[m.queryIdx] - desc2.T[m.trainIdx])
                print(f"Keypoint detected {m.queryIdx} - {m.trainIdx} with {dist:.2f}")
                good.append([m])

        print(len(good), " matches detected")
        # cv.drawMatchesKnn expects list of lists as matches.

        img1 = cv2.drawKeypoints(raw_image_1, kp1, None)
        img2 = cv2.drawKeypoints(raw_image_2, kp2, None)
        img = cv2.drawMatchesKnn(raw_image_1, kp1, raw_image_2,
                                 kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"match_{f1}_{f2}.jpg", img)
        cv2.imwrite(f"keypoints_{f1}.jpg", img1)
        cv2.imwrite(f"keypoints_{f2}.jpg", img2)


def read_rgb_file(fpath, shape):
    image = Image.open(fpath)
    sample = dict(image=image)
    sample = resize_sample(sample, image_shape=shape)
    image = cv2.cvtColor(np.array(sample['image']), cv2.COLOR_BGR2RGB)
    sample = to_tensor_sample(sample)

    return sample['image'].cuda(), image


if __name__ == "__main__":
    main()
