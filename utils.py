import numpy as np
import cv2


def compute_area_triangle(p1, p2, p3):
    area = 1 / 2 * abs(
        p1[0] * (p2[1] - p3[1]) + \
        p2[0] * (p3[1] - p1[1]) + \
        p3[0] * (p1[1] - p2[1]))
    return area


def is_point_inside_boundary(vertices, x):
    area1 = compute_area_triangle(vertices[0], vertices[1], vertices[2])
    area2 = compute_area_triangle(vertices[0], vertices[3], vertices[2])

    area3 = compute_area_triangle(vertices[0], vertices[1], x)
    area4 = compute_area_triangle(vertices[1], vertices[2], x)
    area5 = compute_area_triangle(vertices[2], vertices[3], x)
    area6 = compute_area_triangle(vertices[3], vertices[0], x)

    return area1 + area2 == area3 + area4 + area5 + area6


def get_plucker_repr(p1, p2):
    return p1[:, None].dot(p2[None]) - p2[:, None].dot(p1[None])


def angle_between_two_planes(plane1, plane2):
    dot_product = plane1.dot(plane2) / (np.linalg.norm(plane1) * np.linalg.norm(plane2))
    return np.degrees(np.arccos(dot_product))


def make_homogeneous(X):
    X = np.concatenate([X, np.ones(shape=(X.shape[0], 1))], -1)
    return X


def compositeH(H2to1, foreground, background):
    H, W, _ = background.shape
    warp_template = cv2.warpPerspective(foreground, H2to1, (W, H))
    template_ind = np.nonzero(warp_template)
    background[template_ind] = warp_template[template_ind]
    return background


def project(P, X):
    X = make_homogeneous(X)
    x = P @ X.T
    x = x / x[-1]
    x = x[:2]
    return x


def get_line_from_points(p1, p2):
    """
    :param p1: (x1, y1) coordinates in euclidean space
    :param p2: (x2, y2) coordinates in euclidean space
    :return: line in projective space l = (x1, y1, 1) x (x2, y2, 1)
    """
    p1_proj = np.array([p1[0], p1[1], 1])
    p2_proj = np.array([p2[0], p2[1], 1])
    return np.cross(p1_proj, p2_proj)


def get_vanishing_point_from_lines(p1, p2, p3, p4):
    line1 = get_line_from_points(p1, p2)
    line2 = get_line_from_points(p3, p4)
    vanishing_point = np.cross(line1, line2)
    return vanishing_point / vanishing_point[-1]


def angles_between_planes(vanishing_points, K_inv):
    planes = []
    for i in range(len(vanishing_points)):
        v1, v2 = vanishing_points[i]
        d1 = K_inv.dot(v1.T).T
        d2 = K_inv.dot(v2.T).T
        planes.append(np.cross(d1, d2))

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            print(f"Angle between plane {i + 1} and {j + 1}:", angle_between_two_planes(planes[i], planes[j]))


def get_plane(corners, K_inv):
    p1, p2, p3, p4 = corners
    v1 = get_vanishing_point_from_lines(p1, p2, p3, p4)
    v2 = get_vanishing_point_from_lines(p1, p4, p2, p3)
    d1 = K_inv.dot(v1.T).T
    d2 = K_inv.dot(v2.T).T
    plane = np.cross(d1, d2)
    return plane


def computeH(p1, p2):
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2

    A = np.zeros(shape=(2 * p1.shape[1], 9))
    for i in range(0, p1.shape[1]):
        A[2 * i] = np.array([-p2[0, i], -p2[1, i], -1, 0, 0, 0, p2[0, i] * p1[0, i], p1[0, i] * p2[1, i], p1[0, i]])
        A[2 * i + 1] = np.array([0, 0, 0, -p2[0, i], -p2[1, i], -1, p2[0, i] * p1[1, i], p1[1, i] * p2[1, i], p1[1, i]])
    U, S, Vt = np.linalg.svd(A)
    H2to1 = Vt[-1, :].reshape(3, 3)
    return H2to1


def get_parallel_lines(corners):
    x1, y1, x2, y2 = corners[:2].flatten()
    x3, y3, x4, y4 = corners[2:4].flatten()
    lines = np.array([
        [x1, y1, x2, y2],
        [x3, y3, x4, y4]
    ])
    return lines
