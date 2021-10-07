import os
import argparse

import cv2
from cv2.xfeatures2d import matchGMS
import numpy as np
import torch
from PIL import Image

from kp2d.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2d.datasets.augmentations import (resize_sample,
                                         to_tensor_sample)
DETECTION_THRESH = 0.5
TOP_N = 1000


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

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
    # CARLA SYNTHETIC EVAL
    f1 = "00000000_rgb_top_far.png"
    f2 = "00000000_rgb_top_far_far.png"
    # f1 = "00000000_rgb.png"
    # f2 = "00000000_rgb_L.png"
    # ATLAS IMAGES
    # f1 = "2cn03_0000000330_color.jpg"
    # f2 = "3cn04_0000000330_color.jpg"
    print(f"Detecting keypoints for: {f1} - {f2}")
    with torch.no_grad():
        # for params in eval_params:
        image_1, raw_image_1 = read_rgb_file(os.path.join(args.input_dir, f1), params['res'], warp=False)
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
        print(f"{Hc * Wc} keypoints generated")
        for i in range(Hc * Wc):
            x, y = coord1[:, i]
            kp = cv2.KeyPoint(x, y, 5, response=score1[i])
            kp1.append(kp)
            x, y = coord2[:, i]
            kp = cv2.KeyPoint(x, y, 5, response=score2[i])
            kp2.append(kp)

        # Initiate SIFT detector
        # sift = cv2.SIFT_create()
        # # # find the keypoints and descriptors with SIFT
        # C, W, H = raw_image_1.shape
        # raw_image_1_gray = cv2.cvtColor(raw_image_1, cv2.COLOR_RGB2GRAY)
        # raw_image_2_gray = cv2.cvtColor(raw_image_2, cv2.COLOR_RGB2GRAY)

        # kp1, des1 = sift.detectAndCompute(raw_image_1_gray, None)
        # kp2, des2 = sift.detectAndCompute(raw_image_2_gray, None)
        MATCH_GMS = False
        bf = cv2.BFMatcher()
        if MATCH_GMS:
            raw_matches = bf.match(desc1.T, desc2.T)
            matches = matchGMS([Wc, Hc], [Wc, Hc], kp1, kp2, raw_matches,
                               withScale=True, withRotation=True, thresholdFactor=0.01)
        else:
            raw_matches = bf.knnMatch(desc1.T, desc2.T, k=2)
            matches = []
            for m, n in raw_matches:
                if m.distance < 0.75 * n.distance:
                    matches.append(m)
            # ratio test for filtering outliers

        # Apply ratio test
        good = []
        for match in matches:
            dist = np.linalg.norm(desc1.T[match.queryIdx] - desc2.T[match.trainIdx])
            s1 = score1[match.queryIdx]
            s2 = score2[match.trainIdx]
            print(f"Match found {match.queryIdx} - {match.trainIdx} with {dist:.2f}, score1: {s1}, score2: {s2}")
            if s1 > DETECTION_THRESH and s2 > DETECTION_THRESH:
                good.append([match])

        print(len(good), " matches detected")

        # cv.drawMatchesKnn expects list of lists as matches.
        img1 = cv2.drawKeypoints(raw_image_1, np.array(kp1)[np.flip(np.argsort(score1))][:TOP_N], None)
        img2 = cv2.drawKeypoints(raw_image_2, np.array(kp2)[np.flip(np.argsort(score2))][:TOP_N], None)
        img = cv2.drawMatchesKnn(raw_image_1, kp1, raw_image_2,
                                 kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"match_{f1}_{f2}.jpg", img)
        cv2.imwrite(f"keypoints_{f1}.jpg", img1)
        cv2.imwrite(f"keypoints_{f2}.jpg", img2)


def read_rgb_file(fpath, shape, warp=False):
    image = Image.open(fpath)
    sample = dict(image=image)
    sample = resize_sample(sample, image_shape=shape)
    sample['image'] = np.array(sample['image'])
    if warp:
        sample['image'] = warp_birds_eye(sample['image'])
    image = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
    sample = to_tensor_sample(sample)
    return sample['image'].cuda(), image


def warp_birds_eye(image):
    IMAGE_H, IMAGE_W, _ = image.shape

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[IMAGE_W / 2 - 50, IMAGE_H], [IMAGE_W / 2 + 50, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    # Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

    img = image[200:(200 + IMAGE_H), 0:IMAGE_W, ]
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))

    return warped_img


if __name__ == "__main__":
    main()

