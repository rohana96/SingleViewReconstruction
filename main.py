import os
import numpy as np
from pose_estimation import estimateP
import argparse
from utils import project
from annotations import annotate_points, annotate_points_manually, draw_lines
from camera_calibration import calibrate, calibrate_from_squares
from single_view_reconstruction import reconstruct


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='run type', required=True)
    parser.add_argument('--subject', help='subject', required=False)
    parser.add_argument('--task', required=False)
    parser.add_argument('--force_annotate', action='store_true', default=False, help="prevent use of cached annotations")
    return parser


def run(args):
    if args.type == 'pose_estimation':
        if args.subject == 'bunny':
            P = estimateP('data/pose_estimation', 'bunny.txt')
            print(P)

            if args.task == 'draw_bbox':
                datapath = 'data/pose_estimation/bunny_bd.npy'
            elif args.task == 'project_points':
                datapath = 'data/pose_estimation/bunny_pts.npy'

            X = np.load(datapath)

            if args.task == 'draw_bbox':
                new_X = []
                for x in X:
                    new_X.append(x[:3])
                    new_X.append(x[3:6])
                new_X = np.array(new_X)
                X = new_X

            # import pdb; pdb.set_trace()

            x = project(P, X)
            x = x.T

            if args.task == 'draw_bbox':
                pts = []
                for i in range(0, len(x), 2):
                    pt1 = x[i]
                    pt2 = x[i + 1]
                    pts.append(np.concatenate([pt1, pt2], -1))
                annotations = np.array(pts)
                annotations = np.array(annotations).astype(int)
                np.save('annotations/pose_estimation/bunny_proj.npy', annotations)
                draw_lines(annopath='annotations/pose_estimation/bunny_proj.npy',
                           imagepath='data/pose_estimation/bunny.jpeg',
                           savepath='out/pose_estimation/bunny_bd.jpg')
            elif args.task == 'project_points':
                annotate_points(imagepath='data/pose_estimation/bunny.jpeg', pts=x, outpath='out/pose_estimation/bunny.jpg')

        elif args.subject == 'cuboid':

            # annotate point manually and create 2d-3d correspondences
            if args.force_annotate:
                points2d = annotate_points_manually(imagepath='data/pose_estimation/box.jpg', annopath='annotations/pose_estimation/box.npy',
                                                    outpath='out/pose_estimation/box_anno.jpg')
                points3d = [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1.5],
                    [0, 1, 1.5],
                    [1, 0, 0],
                    [1, 0, 1.5]
                ]
                points3d = np.array(points3d)
                np.save('data/pose_estimation/box_pts.npy', points3d)

                points = np.concatenate([points2d, points3d], -1)
                # points = str(points)
                with open('data/pose_estimation/box.txt', 'w') as f:
                    np.savetxt(f, points, delimiter=' ', newline='\n', header='', footer='', comments='# ')
                    # f.write(points)

            P = estimateP('data/pose_estimation', 'box.txt')
            print(P)
            datapath = 'data/pose_estimation/box_pts.npy'

            X = np.load(datapath)
            X = np.concatenate([X,
                                [[1, 1, 0],
                                 [1, 1, 1.5]]], axis=0)

            x = project(P, X)
            x = x.T
            annotate_points(imagepath='data/pose_estimation/box.jpg', pts=x, outpath='out/pose_estimation/box.jpg', pt_size=20)

            annotations2d = [
                [x[0][0], x[0][1], x[1][0], x[1][1]],
                [x[2][0], x[2][1], x[3][0], x[3][1]],
                [x[0][0], x[0][1], x[2][0], x[2][1]],
                [x[1][0], x[1][1], x[3][0], x[3][1]],
                [x[0][0], x[0][1], x[4][0], x[4][1]],
                [x[2][0], x[2][1], x[5][0], x[5][1]],

                [x[6][0], x[6][1], x[7][0], x[7][1]],
                [x[6][0], x[6][1], x[4][0], x[4][1]],
                [x[7][0], x[7][1], x[5][0], x[5][1]],
                [x[6][0], x[6][1], x[1][0], x[1][1]],
                [x[7][0], x[7][1], x[3][0], x[3][1]],
                [x[4][0], x[4][1], x[5][0], x[5][1]]
            ]

            annotations = np.array(annotations2d).astype(int)
            np.save('annotations/pose_estimation/box_proj.npy', annotations)
            draw_lines(annopath='annotations/pose_estimation/box_proj.npy', imagepath='data/pose_estimation/box.jpg',
                       savepath='out/pose_estimation/box_edges.jpg')

    if args.type == 'camera_calibration':
        K = np.zeros((3, 3))
        if args.task == 'from_van_points':
            K = calibrate()
        elif args.task == 'from_squares':
            metric_polygon = np.array([
                [0., 99.],
                [99., 99.],
                [99., 0.],
                [0., 0.],
            ])
            whitebox = np.ones((100, 100, 3), dtype=np.uint8)
            K = calibrate_from_squares(metric_polygon=metric_polygon, whitebox=whitebox)
        elif args.task == 'test_image':

            if args.force_annotate:
                points2d = annotate_points_manually(imagepath='data/camera_calibration/test2.jpg',
                                                    annopath='annotations/camera_calibration/test2.npy',
                                                    outpath='out/camera_calibration/test2.jpg')
                points2d = points2d.reshape(3, 4, 2)
                np.save('data/camera_calibration/test2.npy', points2d)
            metric_polygon = np.array([
                [0., 149.],
                [99, 149.],
                [99, 0.],
                [0., 0.]
            ])
            whitebox = np.ones((150, 100, 3), dtype=np.uint8)
            K = calibrate_from_squares(imagepath='data/camera_calibration/test2.jpg', datapath='data/camera_calibration/test2.npy',
                                       outpath='out/camera_calibration/',
                                       metric_polygon=metric_polygon, whitebox=whitebox)
        print(K)

    if args.type == 'reconstruction':
        print(reconstruct())


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    run(args)
