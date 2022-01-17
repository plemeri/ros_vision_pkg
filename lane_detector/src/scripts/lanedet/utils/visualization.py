import cv2
import os
import os.path as osp
import numpy as np
from sklearn.linear_model import RANSACRegressor

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),  (255, 255, 255)]

def imshow_lanes(img, lanes, show=False, out_file=None):
    for i, lane in enumerate(lanes):
        color = colors[i]
        # # print(lane.shape)
        # reg = RANSACRegressor().fit(lane[:, 0:1], lane[:, 1:])
        # for (x, y), inlier in zip(lane, reg.inlier_mask_):
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            # if inlier is False:
                color = (0, 0, 0)
                print('out')
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, color, 2)
    
    return img

    # if show:
    #     cv2.imshow('view', img)
    #     cv2.waitKey(0)

    # if out_file:
    #     if not osp.exists(osp.dirname(out_file)):
    #         os.makedirs(osp.dirname(out_file))
    #     cv2.imwrite(out_file, img)

