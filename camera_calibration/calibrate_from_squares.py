import numpy as np
import os
import cv2
from utils import compositeH, angles_between_planes, make_homogeneous, get_vanishing_point_from_lines, computeH


def calibrate_from_squares(imagepath='data/camera_calibration/squares.png', datapath='data/camera_calibration/squares.npy',
                           outpath='out/camera_calibration',
                           metric_polygon=None, whitebox=None):

    data = np.load(datapath, allow_pickle=True)
    img = cv2.imread(imagepath)
    homographies = []
    vanishing_points = []

    for i, P in enumerate(data):
        img2 = img.copy()
        p1 = P.T
        p2 = metric_polygon.T
        whitebox_copy = whitebox.copy() * 255
        H = computeH(p1, p2)
        homographies.append(H)
        whitebox_proj = compositeH(H, whitebox_copy, img2)
        output_file = os.path.join(outpath, f"whitebox_{i}vanishing.png")
        cv2.imwrite(output_file, whitebox_proj)

        points = make_homogeneous(data[i])
        p1, p2, p3, p4 = points

        vanishing_point1 = get_vanishing_point_from_lines(p1, p2, p3, p4)
        vanishing_point2 = get_vanishing_point_from_lines(p1, p4, p2, p3)
        vanishing_points.append([vanishing_point1, vanishing_point2])

    A = []
    for homography in homographies:
        h11, h12, h13 = homography[:, 0]
        h21, h22, h23 = homography[:, 1]
        A.append([
            h11 * h21,
            h11 * h22 + h12 * h21,
            h11 * h23 + h13 * h21,
            h12 * h22,
            h12 * h23 + h13 * h22,
            h13 * h23
        ])
        A.append([
            h11 ** 2 - h21 ** 2,
            2 * h11 * h12 - 2 * h21 * h22,
            2 * h11 * h13 - 2 * h21 * h23,
            h12 ** 2 - h22 ** 2,
            2 * h12 * h13 - 2 * h22 * h23,
            h13 ** 2 - h23 ** 2,
        ])
    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    w1, w2, w3, w4, w5, w6 = Vt[-1]
    W = np.array([
        [w1, w2, w3],
        [w2, w4, w5],
        [w3, w5, w6],
    ])

    L = np.linalg.cholesky(W)
    K = np.linalg.inv(L.T)
    K /= K[2, 2]
    K_inv = np.linalg.inv(K)

    angles_between_planes(vanishing_points, K_inv)
    return K
