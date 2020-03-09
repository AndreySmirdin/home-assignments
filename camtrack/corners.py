#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import create_cli
from scipy.spatial.distance import cdist


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _float2uint8(img):
    return np.round(img * 255).astype('uint8')


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]

    MAX_CORNERS = 1000
    QUALITY_LEVEL = 0.005
    MIN_DIST = 10
    SHI_PARAMS = dict(blockSize=3,
                      gradientSize=3,
                      useHarrisDetector=False,
                      k=0.04)

    LK_PARAMS = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     minEigThreshold=0.001)

    last_id = 0
    prev_corners = None

    for frame, image_1 in enumerate(frame_sequence):
        detected = cv2.goodFeaturesToTrack(image_1, MAX_CORNERS, QUALITY_LEVEL, MIN_DIST, None, **SHI_PARAMS)
        detected = detected.reshape(-1, 2)
        if frame == 0:
            corners = FrameCorners(
                np.arange(len(detected)),
                detected,
                np.array([10] * len(detected))
            )
            last_id = len(detected)
        else:
            points_1, st, _ = cv2.calcOpticalFlowPyrLK(_float2uint8(image_0), _float2uint8(image_1),
                                                       prev_corners.points, None, **LK_PARAMS)
            st = st.reshape(-1)
            good_new = points_1[st == 1]
            ids = prev_corners.ids[st == 1]

            # Now, let's add new points
            dists = cdist(detected, good_new).min(axis=1)
            points2add = min(MAX_CORNERS - len(good_new), len(detected))
            extra_indices = np.argsort(dists)[-points2add:]

            corners = FrameCorners(
                np.concatenate([ids.reshape(-1), last_id + np.arange(points2add)]),
                np.concatenate([good_new, detected[extra_indices]]),
                np.array([10] * (len(good_new) + points2add))
            )

            last_id += points2add

        builder.set_corners_at_frame(frame, corners)

        prev_corners = corners
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
