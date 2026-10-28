from collections import deque

import numpy as np


class HeadPoseMonitor:
    """
    Head-pose distraction monitor following the documented baseline.

    1. Use the first valid pose as the reference pose.
    2. Calculate alignment relative to that reference.
    3. Maintain a 30-observation distraction buffer.
    4. Apply the documented distraction thresholds.
    """

    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.pose_zero = None
        self.distraction_history = deque(maxlen=buffer_size)

    def reset(self):
        self.pose_zero = None
        self.distraction_history.clear()

    def calibrate(self, head_pose):
        """Set the reference head pose from a valid observation."""
        try:
            pose = np.asarray(head_pose, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return False

        if pose.size != 3 or not np.all(np.isfinite(pose)):
            return False

        self.pose_zero = pose.copy()
        self.distraction_history.clear()

        print(
            "Head-pose calibration complete. "
            f"Reference pose: {self.pose_zero}"
        )

        return True

    def update(self, head_pose, tracking_valid=True):
        def unknown():
            return {
                "status": "UNKNOWN",
                "pose_score": None,
                "distraction_ratio": None,
                "calibrated": self.pose_zero is not None,
            }

        if not tracking_valid or head_pose is None:
            return unknown()

        try:
            pose = np.asarray(head_pose, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return unknown()

        if pose.size != 3 or not np.all(np.isfinite(pose)):
            return unknown()

        # --------------------------------------------------
        # 1. Initial reference-pose calibration
        # --------------------------------------------------
        if self.pose_zero is None:
            if not self.calibrate(pose):
                return unknown()

            return {
                "status": "CALIBRATION",
                "pose_score": None,
                "distraction_ratio": None,
                "calibrated": True,
            }

        # --------------------------------------------------
        # 2. Calculate deviation from the reference pose
        # --------------------------------------------------
        pitch, yaw, roll = pose - self.pose_zero

        pose_score = 1.0 - (
            abs(pitch / 90.0)
            + abs(yaw / 90.0)
            + abs(roll / 90.0)
        ) / 3.0

        distracted = pose_score < 0.9
        self.distraction_history.append(int(distracted))

        # Use only the observations collected so far until
        # the 30-sample buffer becomes full.
        distraction_ratio = (
            sum(self.distraction_history)
            / len(self.distraction_history)
        )

        # --------------------------------------------------
        # 3. Apply documented alert thresholds
        # --------------------------------------------------
        if distraction_ratio > 0.9:
            status = "DANGER"
        elif distraction_ratio > 0.7:
            status = "LOW_AWARENESS"
        else:
            status = "SAFE"

        return {
            "status": status,
            "pose_score": pose_score,
            "distraction_ratio": distraction_ratio,
            "calibrated": True,
        }