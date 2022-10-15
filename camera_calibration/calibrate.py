import numpy as np
from annotations import plot_vanishing_points
from utils import get_line_from_points
import cv2


def get_principal_point(intrinsic):
    origin = np.array([0, 0, 1])
    principal_point = intrinsic.dot(origin)
    return principal_point


def calibrate(imgpath='data/camera_calibration/church.png', datapath='data/camera_calibration/church.npy', outpath='out/camera_calibration/church.png'):
    points = np.load(datapath)
    lines = []
    for i in range(len(points)):
        for j in range(2):
            points1 = points[i][j][:2]
            points2 = points[i][j][2:4]
            line = get_line_from_points(points1, points2)
            lines.append(line)
    lines = np.array(lines)

    points = []
    for i in range(0, len(lines), 2):
        point_intersection = np.cross(lines[i], lines[i + 1])
        point = point_intersection / point_intersection[-1]
        points.append(point)

    points = np.array(points)
    p1, p2, p3 = points
    vanishing_points = points[:, :2]

    A = []
    for u, v in [(p1, p2), (p2, p3), (p3, p1)]:
        u1, u2, u3 = u
        v1, v2, v3 = v
        A.append([u1 * v1 + u2 * v2, u1 * v3 + v1 * u3, u2 * v3 + u3 * v2, u3 * v3])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    w = Vt[-1]
    W = [[w[0], 0, w[1]], [0, w[0], w[2]], [w[1], w[2], w[3]]]
    L = np.linalg.cholesky(W)
    K = np.linalg.inv(L.T)
    K /= K[2, 2]

    img = cv2.imread(imgpath)
    principal_point = get_principal_point(K)
    plot_vanishing_points(
        img, vanishing_points, principal_point, outpath
    )
    return K
