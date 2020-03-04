#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from collections import defaultdict
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

MAX_REPRODUCTION_ERROR = 1.0
MIN_DEPTH = 0.1
MIN_ANGLE = 1.


def try_frame_tracking(frame: int,
                       corner_storage: CornerStorage,
                       intrinsic_mat: np.ndarray,
                       point_cloud_builder) -> Optional[np.ndarray]:
    intersection = np.zeros(point_cloud_builder.ids.shape[0], dtype='bool')
    for j in range(len(intersection)):
        if point_cloud_builder.ids[j] in corner_storage[frame].ids:
            intersection[j] = True

    points2d = []
    for ind, pt in zip(corner_storage[frame].ids, corner_storage[frame].points):
        if ind in point_cloud_builder.ids:
            points2d.append(pt)

    points2d = np.array(points2d)

    if points2d.shape[0] < 6:
        return None

    # In case cv2 throws us any strange exceptions.
    try:
        res, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[intersection], points2d, intrinsic_mat,
                                                      distCoeffs=None)
        if res:
            print(f'Solved frame {frame}, found {len(inliers)} inliers.')
            return rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        return None
    except:
        return None


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
                                                       min_triangulation_angle_deg=MIN_ANGLE,
                                                       min_depth=MIN_DEPTH)

    points3d, common_points, _ = triangulate_correspondences(correspondences,
                                                             pose_to_view_mat3x4(known_view_1[1]),
                                                             pose_to_view_mat3x4(known_view_2[1]),
                                                             intrinsic_mat,
                                                             triangulation_parameters)

    view_mats, point_cloud_builder = [None] * len(rgb_sequence), PointCloudBuilder()
    point_cloud_builder.add_points(common_points, points3d)

    progress = True
    stats = defaultdict(int)
    epoch = 0

    while progress:
        epoch += 1
        progress = False
        for i in range(len(rgb_sequence)):
            if view_mats[i] is None:
                res = try_frame_tracking(i, corner_storage, intrinsic_mat, point_cloud_builder)
                if res is not None:
                    progress = True
                    view_mats[i] = res
                    stats[epoch] += 1

            new_points = 0
            for j in range(i):
                correspondences = build_correspondences(corner_storage[j], corner_storage[i],
                                                        ids_to_remove=point_cloud_builder.ids)
                if not len(correspondences.ids) or view_mats[i] is None or view_mats[j] is None:
                    continue
                extra_points3d, extra_common_points, _ = triangulate_correspondences(correspondences,
                                                                                     view_mats[j],
                                                                                     view_mats[i],
                                                                                     intrinsic_mat,
                                                                                     triangulation_parameters)

                new_points += len(extra_common_points)
                point_cloud_builder.add_points(extra_common_points, extra_points3d)

    unsuccessful_frames = list(map(lambda x: x[0], filter(lambda x: x[1] is None, enumerate(view_mats))))
    if unsuccessful_frames:
        raise RuntimeError(f"Could not find views for {len(unsuccessful_frames)}: {unsuccessful_frames}")

    for epoch, cnt in stats.items():
        print(f'Made {cnt} frames during epoch {epoch}')

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
