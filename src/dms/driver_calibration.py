import numpy as np


class DriverCalibration:
    """
    Establishes driver-specific baselines for head pose and eye opening.

    Calibration should be performed while the driver is alert,
    facing forward, and keeping both eyes normally open.
    """

    def __init__(self, required_samples=60):
        if required_samples < 1:
            raise ValueError("required_samples must be >= 1")

        self.required_samples = required_samples

        self.head_pose_samples = []
        self.eye_opening_samples = []

        self.baseline_head_pose = None
        self.baseline_eye_opening = None

        self.is_calibrated = False
        self.head_pose_calibrated = False
        self.eye_calibrated = False

    def update(self, euler, eyes=None, tracking_valid=True):
        """
        Collect valid head-pose and eye-opening samples.

        Eye output layout:
            eyes[0:2] -> eye visibility
            eyes[2:4] -> eye opening

        The eye-output mapping must be validated for the supplied model.
        """

        if not tracking_valid:
            return self._progress_result()

        # Collect head-pose samples.
        pose = np.asarray(euler, dtype=float)

        if pose.shape == (3,) and np.all(np.isfinite(pose)):
            if not self.head_pose_calibrated:
                self.head_pose_samples.append(pose.copy())

                if len(self.head_pose_samples) >= self.required_samples:
                    self.baseline_head_pose = np.median(
                        np.asarray(self.head_pose_samples),
                        axis=0,
                    )
                    self.head_pose_calibrated = True

        # Collect eye-opening samples only when both eyes are visible
        # and appear normally open.
        if not self.eye_calibrated and eyes is not None:
            try:
                values = np.asarray(eyes, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = np.array([])

            if values.size == 4 and np.all(np.isfinite(values)):
                visibility = values[:2]
                opening = values[2:]

                values_in_range = (
                    np.all((visibility >= 0) & (visibility <= 1))
                    and np.all((opening >= 0) & (opening <= 1))
                )

                eyes_visible = np.all(visibility >= 0.50)

                # Require both eyes to appear open before accepting
                # a sample as a normal-eye calibration sample.
                eyes_open = np.all(opening >= 0.50)

                if values_in_range and eyes_visible and eyes_open:
                    self.eye_opening_samples.append(opening.copy())

                    if (
                        len(self.eye_opening_samples)
                        >= self.required_samples
                    ):
                        self.baseline_eye_opening = np.median(
                            np.asarray(self.eye_opening_samples),
                            axis=0,
                        )
                        self.eye_calibrated = True

        self.is_calibrated = (
            self.head_pose_calibrated and self.eye_calibrated
        )

        return self._progress_result()

    def _progress_result(self):
        head_progress = min(
            len(self.head_pose_samples) / self.required_samples * 100.0,
            100.0,
        )

        eye_progress = min(
            len(self.eye_opening_samples) / self.required_samples * 100.0,
            100.0,
        )

        return {
            "status": (
                "CALIBRATED" if self.is_calibrated else "CALIBRATING"
            ),
            "head_pose_progress": head_progress,
            "eye_progress": eye_progress,
            "progress": min(head_progress, eye_progress),
            "baseline_head_pose": (
                self.baseline_head_pose.copy()
                if self.baseline_head_pose is not None
                else None
            ),
            "baseline_eye_opening": (
                self.baseline_eye_opening.copy()
                if self.baseline_eye_opening is not None
                else None
            ),
        }

    def get_head_pose_deviation(self, euler):
        """Return absolute yaw, pitch and roll deviations."""

        if not self.head_pose_calibrated:
            return None

        pose = np.asarray(euler, dtype=float)

        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return None

        return np.abs(pose - self.baseline_head_pose)