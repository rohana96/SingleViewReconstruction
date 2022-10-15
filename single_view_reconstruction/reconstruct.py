import os
import numpy as np
import matplotlib.pyplot as plt
from camera_calibration import calibrate
import cv2
from utils import is_point_inside_boundary, get_plucker_repr, get_plane
from annotations import vis_annotated_planes
from tqdm import tqdm


def get_depth_from_corners(corners2d, corners3d):
    for corner in corners2d:
        if corner in corners3d:
            return corners3d[corner]
    return None


def get_arbitrary_depth(corners2d, origin, P):
    corner = corners2d[0]
    corner = np.array([corner[0], corner[1], 1.])
    ray_direction = P.dot(corner)
    step_size = 20. / ray_direction[-2]
    return origin + step_size * ray_direction


def get_boundary(square):
    min_x = min(square[:, 1])
    min_y = min(square[:, 0])
    max_x = max(square[:, 1])
    max_y = max(square[:, 0])
    return [min_x, min_y, max_x,max_y]


def get_3d_point(P_plus, point, origin, plane):
    ray_direction = P_plus.dot(point)
    ray_plucker = get_plucker_repr(origin, ray_direction)
    point_3d = ray_plucker.dot(plane)
    point_3d = point_3d[:-1] / point_3d[-1]
    return point_3d


def reconstruct(imagepath='data/single_view_reconstruction/building.png', datapath='data/single_view_reconstruction/building.npy',
                outpath='out/single_view_reconstruction/'):

    data = np.load(datapath).astype(np.int64)
    img = cv2.imread(imagepath)
    W, H = img.shape[:2]

    # vis_annotated_planes(datapath=datapath, imagepath=imagepath)

    points = []
    x1, y1, x2, y2 = data[0][:2].flatten()
    x3, y3, x4, y4 = data[0][2:4].flatten()
    lines = np.array([
        [x1, y1, x2, y2],
        [x3, y3, x4, y4]
    ])
    points.append(lines)

    x1, y1, x2, y2 = data[1][:2].flatten()
    x3, y3, x4, y4 = data[1][2:4].flatten()
    lines = np.array([
        [x1, y1, x2, y2],
        [x3, y3, x4, y4]
    ])
    points.append(lines)

    x1, y1, x3, y3 = data[1][:2].flatten()
    x4, y4, x2, y2 = data[1][2:4].flatten()
    lines = np.array([
        [x1, y1, x2, y2],
        [x3, y3, x4, y4]
    ])
    points.append(lines)
    points = np.array(points)

    points_datapath = os.path.join(outpath, 'building.npy')
    np.save(points_datapath, points)

    K = calibrate(imagepath, points_datapath, outpath)
    print(K)
    K_inv = np.linalg.inv(K)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    P = K.dot(np.eye(3, 4))
    P_plus = P.T.dot(np.linalg.inv(P.dot(P.T)))
    origin = np.array([0, 0, 0, 1.])
    point_cloud, colours = [], []
    corners3d = {}
    planes = []

    for idx, square in enumerate(data):
        corners2d = set([tuple(corner.tolist()) for corner in square])
        plane = get_plane(square, K_inv)

        if idx == 0:
            ref_point = get_arbitrary_depth(square, origin, P_plus)
        else:
            ref_point = get_depth_from_corners(corners2d, corners3d)

        ref_point = ref_point / ref_point[-1]
        residual = -plane.dot(ref_point[:3])
        plane = np.array([plane[0], plane[1], plane[2], residual])
        planes.append(plane)

        boundary = get_boundary(square)
        xrange = np.arange(boundary[0], boundary[2] + 1)
        yrange = np.arange(boundary[1], boundary[3] + 1)
        xys = np.meshgrid(xrange, yrange)
        xys = np.dstack(xys).reshape(-1, 2)

        for (i, j) in tqdm(xys):
            point = np.array([j, i, 1.])
            if is_point_inside_boundary(square, point[:2]):
                point_3d = get_3d_point(P_plus, point, origin, plane)
                point_cloud.append(point_3d)
                colours.append(img[i, j] / 255.)
                if (j, i) in corners2d:
                    corners3d[(j, i)] = np.array([point_3d[0], point_3d[1], point_3d[2], 1.])

    point_cloud = np.stack(point_cloud)
    xx, yy, zz = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    ax.scatter(xx, yy, zz, c=colours)
    plt.show()
