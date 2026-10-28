from collections import deque
import time

import numpy as np


class PerclosMonitor:
    """
    Eye-closure monitor following the documented baseline.

    1. Collect valid eye-opening observations for 5 seconds.
    2. Calculate a driver-specific baseline for each eye.
    3. Detect closure when both eyes are below 50% of baseline.
    4. Maintain a 100-observation buffer.
    5. Determine status using the documented thresholds.
    """

    def __init__(self, calibration_seconds=5.0, buffer_size=100):
        self.calibration_seconds = calibration_seconds
        self.buffer_size = buffer_size

        self.calibration_start = None
        self.calibration_samples = []

        self.baseline_eye_opening = None
        self.samples = deque(maxlen=buffer_size)

    def reset(self):
        self.calibration_start = None
        self.calibration_samples.clear()
        self.baseline_eye_opening = None
        self.samples.clear()

    def update(self, eyes, tracking_valid=True):
        now = time.monotonic()

        def unknown(reason):
            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "status": "UNKNOWN",
                "calibrated": self.baseline_eye_opening is not None,
                "reason": reason,
            }

        if not tracking_valid:
            return unknown("Tracking invalid")

        if eyes is None:
            return unknown("Eye output missing")

        try:
            values = np.asarray(eyes, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return unknown("Invalid eye output")

        if values.size != 4 or not np.all(np.isfinite(values)):
            return unknown("Invalid eye output")

        visibility = values[:2]
        opening = values[2:]

        if (
            np.any((visibility < 0) | (visibility > 1))
            or np.any((opening < 0) | (opening > 1))
        ):
            return unknown("Eye values outside expected range")

        # Require both eyes to be visible.
        if np.any(visibility < 0.50):
            return unknown("Eye visibility insufficient")

        # --------------------------------------------------
        # 1. Initial eye calibration
        # --------------------------------------------------
        if self.baseline_eye_opening is None:

            if self.calibration_start is None:
                self.calibration_start = now

            # Collect samples during the calibration period.
            self.calibration_samples.append(opening.copy())

            elapsed = now - self.calibration_start

            if elapsed < self.calibration_seconds:
                return {
                    "eye_state": "CALIBRATING",
                    "perclos": None,
                    "status": "CALIBRATION",
                    "calibrated": False,
                    "reason": "Collecting eye calibration samples",
                }

            # Calculate average opening for each eye.
            baseline = np.mean(
                self.calibration_samples,
                axis=0,
            )

            # Reject an unusable baseline instead of dividing by zero.
            if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
                self.calibration_start = None
                self.calibration_samples.clear()
                return unknown("Invalid eye calibration baseline")

            self.baseline_eye_opening = baseline

            print(
                "Eye calibration complete. "
                f"Left-eye baseline: {baseline[0]:.3f}, "
                f"Right-eye baseline: {baseline[1]:.3f}"
            )

        # --------------------------------------------------
        # 2. Compare current eye opening with the baseline
        # --------------------------------------------------
        relative_opening = opening / self.baseline_eye_opening

        # Both eyes must be below 50% of their calibrated values.
        closed = bool(np.all(relative_opening < 0.50))

        self.samples.append(1 if closed else 0)

        # --------------------------------------------------
        # 3. Calculate the documented open-eye proportion
        # --------------------------------------------------
        # The documentation calls this PERCLOS, but this
        # particular implementation calculates the proportion
        # of observations classified as OPEN.
        open_eye_ratio = 1.0 - (
            sum(self.samples) / self.buffer_size
        )

        # --------------------------------------------------
        # 4. Apply the documented alert thresholds
        # --------------------------------------------------
        if open_eye_ratio < 0.40:
            status = "DANGER"
        elif open_eye_ratio < 0.60:
            status = "LOW_AWARENESS"
        else:
            status = "SAFE"

        return {
            "eye_state": "CLOSED" if closed else "OPEN",
            "perclos": open_eye_ratio,
            "status": status,
            "calibrated": True,
            "reason": "Valid eye observation",
        }