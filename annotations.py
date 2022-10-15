import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255)]


def annotate_points_manually(
        imagepath='data/pose_estimation/box.jpg',
        save_annotation=True,
        annopath='annotations/pose_estimation/box.npy',
        outpath='out/'
):
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(x, ' ', y)
            points.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(img, str(x) + ',' +
                        str(y), (x, y), font,
                        3, (255, 0, 0), 2)
            cv2.imshow('image', img)

    img = cv2.imread(imagepath)
    H, W, _ = img.shape
    print(H, W)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    if save_annotation is True:
        points = np.array(points)
        np.save(annopath, points)
        cv2.imwrite(outpath, img)
    cv2.destroyAllWindows()
    return points


def annotate_points(
        imagepath='data/pose_estimation/bunny.jpeg',
        pts=None,
        outpath='out/pose_estimation/bunny.jpeg',
        pt_size=2
):
    img = cv2.imread(imagepath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y in tqdm(pts):
        cv2.circle(img, (int(x), int(y)), pt_size, (0, 255, 0), -1)
        cv2.imshow('image', img)
    cv2.imwrite(outpath, img)


annotate_line = False
change_color = True
color = (0, 255, 0)


def draw_lines(annopath='data/camera_calibration/box.npy', imagepath='data/church.png', savepath='out/pose_estimation/bunny_bd.jpg'):
    """
	:annotations: (n, 4)
	n lines, each group has 2 lines, each line annotated as (x1, y1, x2, y2)
	"""
    annotations = np.load(annopath)
    img = cv2.imread(imagepath)
    for i in range(annotations.shape[0]):
        COLOR = COLORS[i % 5]
        line = annotations[i]
        x1, y1, x2, y2 = line
        cv2.circle(img, (x1, y1), 3, COLOR, -1)
        cv2.circle(img, (x2, y2), 3, COLOR, -1)
        cv2.line(img, (x1, y1), (x2, y2), COLOR, 10)

    # cv2.imshow('q2a', img)
    # cv2.waitKey(0)
    cv2.imwrite(savepath, img)


def vis_annotated_planes(datapath='data/single_view_reconstruction/building.npy', imagepath='data/single_view_reconstruction/building.png'):
    '''
	annotations: (5, 4, 2)
	5 planes in the scene, 4 points for each plane, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''

    annotations = np.load(datapath).astype(np.int64)
    img = cv2.imread(imagepath)

    for i in range(annotations.shape[0]):
        COLOR = COLORS[i]
        square = annotations[i]
        for j in range(square.shape[0]):
            x, y = square[j]
            cv2.circle(img, (x, y), 3, COLOR, -1)
            cv2.putText(img, str(j + i * 4), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
            cv2.line(img, square[j], square[(j + 1) % 4], COLOR, 2)

        cv2.imshow('single_view_reconstruction', img)
        cv2.waitKey(0)


def plot_vanishing_points(
        img,
        vanishing_points,
        principal_point,
        outpath
):
    vanishing_points = vanishing_points.astype(int)
    x1, y1 = vanishing_points[0]
    x2, y2 = vanishing_points[1]
    x3, y3 = vanishing_points[2]
    p1, p2, p3 = principal_point
    p1, p2 = p1 / p3, p2 / p3

    plt.imshow(img)
    plt.scatter(p1, p2, c='r', s=20)
    plt.plot([x1, x2], [y1, y2], linewidth=1)
    plt.plot([x1, x3], [y1, y3], linewidth=1)
    plt.plot([x2, x3], [y2, y3], linewidth=1)
    plt.scatter([x1, x2, x3], [y1, y2, y3], s=30)
    plt.savefig(outpath)
    plt.close()
