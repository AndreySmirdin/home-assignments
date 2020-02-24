#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import frameseq
import numpy as np
from _camtrack import (
    build_correspondences,
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    TriangulationParameters,

)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

MAX_REPRODUCTION_ERROR = 1.
MIN_DEPTH = 0.1
MIN_ANGLE_FIRST_ITERATION = 0.
MIN_ANGLE_NEXT_ITERATIONS = 1.


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    triangulation_parameters = TriangulationParameters(max_reprojection_error=MAX_REPRODUCTION_ERROR,
                                                       min_triangulation_angle_deg=MIN_ANGLE_FIRST_ITERATION,
                                                       min_depth=MIN_DEPTH)
    points3d, common_points, _ = triangulate_correspondences(correspondences,
                                                             pose_to_view_mat3x4(known_view_1[1]),
                                                             pose_to_view_mat3x4(known_view_2[1]),
                                                             intrinsic_mat,
                                                             triangulation_parameters)

    view_mats, point_cloud_builder = [], PointCloudBuilder()
    point_cloud_builder.add_points(common_points, points3d)

    triangulation_parameters = TriangulationParameters(max_reprojection_error=MAX_REPRODUCTION_ERROR,
                                                       min_triangulation_angle_deg=MIN_ANGLE_NEXT_ITERATIONS,
                                                       min_depth=MIN_DEPTH)

    for i in range(len(rgb_sequence)):
        intersection = np.zeros_like(common_points, dtype='bool')
        for j in range(len(intersection)):
            if common_points[j] in corner_storage[i].ids:
                intersection[j] = True

        points2d = []
        for ind, pt in zip(corner_storage[i].ids, corner_storage[i].points):
            if ind in common_points:
                points2d.append(pt)

        points2d = np.array(points2d)

        res, rvec, tvec, inliers = cv2.solvePnPRansac(points3d[intersection], points2d, intrinsic_mat, distCoeffs=None)
        assert res, f"Something bad has happened at frame {i}, please have a look."
        view_mats.append(rodrigues_and_translation_to_view_mat3x4(rvec, tvec))

        new_points = 0
        for j in range(i):
            correspondences = build_correspondences(corner_storage[j], corner_storage[i],
                                                    ids_to_remove=point_cloud_builder.ids)
            if not len(correspondences.ids):
                continue
            extra_points3d, extra_common_points, _ = triangulate_correspondences(correspondences,
                                                                                 view_mats[j],
                                                                                 view_mats[i],
                                                                                 intrinsic_mat,
                                                                                 triangulation_parameters)
            common_points = np.concatenate([common_points, extra_common_points])
            points3d = np.concatenate([points3d, extra_points3d])

            new_points += len(extra_common_points)
            point_cloud_builder.add_points(extra_common_points, extra_points3d)
            assert len(common_points) == len(np.unique(common_points))

        print(f'Added {new_points} new points during frame {i}, position found with {len(inliers)} inliers.')

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
