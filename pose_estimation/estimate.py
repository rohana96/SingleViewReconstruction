import os
import numpy as np

def parse_file(filepath):
    with open(filepath, 'r') as f:
        pts = f.readlines()
    pts_array = []
    for pt in pts:
        pts_array.append(pt.strip('\n').split(' '))
    pts_array = np.array(pts_array)
    pts_array = pts_array.astype(float)
    pts2D = pts_array[:, :2]
    pts3D = pts_array[:, 2:]
    # print(pts2D, pts3D)
    return pts2D, pts3D


def estimateP(datapath='data/pose_estimation', filename='bunny.txt'):
    filepath = os.path.join(datapath, filename)
    pts2d, pts3d = parse_file(filepath)
    A = []
    for i in range(len(pts3d)):
        x, y = pts2d[i]
        X, Y, Z = pts3d[i]
        curr_entry = [X, Y, Z, 1, 0, 0, 0, 0, -1*x*X, -1*x*Y, -1*x*Z, -1*x]
        A.append(curr_entry)
        curr_entry = [0, 0, 0, 0, X, Y, Z, 1, -1*y*X, -1*y*Y, -1*y*Z, -1*y]
        A.append(curr_entry)

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1]
    P = P.reshape(3, 4)
    # print(A, P)
    return P


