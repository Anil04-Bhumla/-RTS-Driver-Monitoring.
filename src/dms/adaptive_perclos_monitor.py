from collections import deque
import time

import numpy as np


class AdaptivePerclosMonitor:
    """
    Experimental adaptive temporal-window eye monitor.

    The original PerclosMonitor remains unchanged for baseline comparison.

    Design:
    1. Calibrate eye-opening values for a driver.
    2. Classify each valid observation as OPEN or CLOSED.
    3. Maintain a rolling history of valid observations.
    4. Select a temporal window based on recent observation stability.
    5. Use the selected window to estimate the open-eye ratio.
    6. Apply the same status thresholds as the documented baseline.

    Important:
    This is an experimental implementation. Its performance must be
    evaluated against the original baseline using comparable test data.
    """

    def __init__(
        self,
        calibration_seconds=5.0,
        min_window_seconds=1.0,
        max_window_seconds=6.0,
        stability_window_size=10,
        closure_threshold=0.50,
        danger_threshold=0.40,
        low_awareness_threshold=0.60,
        min_observations=5,
    ):
        if calibration_seconds <= 0:
            raise ValueError("calibration_seconds must be positive.")

        if min_window_seconds <= 0:
            raise ValueError("min_window_seconds must be positive.")

        if max_window_seconds < min_window_seconds:
            raise ValueError(
                "max_window_seconds must be >= min_window_seconds."
            )

        if stability_window_size < 2:
            raise ValueError("stability_window_size must be at least 2.")

        if min_observations < 1:
            raise ValueError("min_observations must be positive.")

        self.calibration_seconds = calibration_seconds
        self.min_window_seconds = min_window_seconds
        self.max_window_seconds = max_window_seconds
        self.stability_window_size = stability_window_size
        self.closure_threshold = closure_threshold
        self.danger_threshold = danger_threshold
        self.low_awareness_threshold = low_awareness_threshold
        self.min_observations = min_observations

        # Driver-specific calibration.
        self.calibration_start = None
        self.calibration_samples = []
        self.baseline_eye_opening = None

        # Store valid observations as (timestamp, closed).
        self.history = deque()

        # Recent classifications are used to estimate stability.
        self.recent_states = deque(maxlen=stability_window_size)

        self.last_valid_observation_time = None
        self.current_window_seconds = max_window_seconds

    def reset(self):
        """Reset calibration and temporal history."""
        self.calibration_start = None
        self.calibration_samples.clear()
        self.baseline_eye_opening = None

        self.history.clear()
        self.recent_states.clear()

        self.last_valid_observation_time = None
        self.current_window_seconds = self.max_window_seconds

    def _unknown(self, reason):
        return {
            "eye_state": "UNKNOWN",
            "perclos": None,
            "status": "UNKNOWN",
            "calibrated": self.baseline_eye_opening is not None,
            "reason": reason,
            "window_seconds": self.current_window_seconds,
            "valid_observations": len(self.history),
            "stability": None,
        }

    def _select_window(self):
        """
        Select the window using recent classification stability.

        Stability = the fraction of recent classifications matching
        the most common classification.

        High stability -> shorter window.
        Low stability -> longer window.
        """
        if len(self.recent_states) < self.stability_window_size:
            return self.max_window_seconds, None

        states = list(self.recent_states)
        open_count = states.count(0)
        closed_count = states.count(1)

        stability = max(open_count, closed_count) / len(states)

        # Map stability from 0.5..1.0 to a window from max..min.
        normalized_stability = np.clip(
            (stability - 0.50) / 0.50,
            0.0,
            1.0,
        )

        window_seconds = (
            self.max_window_seconds
            - normalized_stability
            * (self.max_window_seconds - self.min_window_seconds)
        )

        return float(window_seconds), float(stability)

    def update(self, eyes, tracking_valid=True):
        """
        Update the eye monitor.

        Expected eye output:
            [left_visibility, right_visibility,
             left_opening, right_opening]

        Visibility and opening values are expected to be in [0, 1].
        """
        now = time.monotonic()

        if not tracking_valid:
            return self._unknown("Tracking invalid")

        if eyes is None:
            return self._unknown("Eye output missing")

        try:
            values = np.asarray(eyes, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return self._unknown("Invalid eye output")

        if values.size != 4 or not np.all(np.isfinite(values)):
            return self._unknown("Invalid eye output")

        visibility = values[:2]
        opening = values[2:]

        if (
            np.any((visibility < 0.0) | (visibility > 1.0))
            or np.any((opening < 0.0) | (opening > 1.0))
        ):
            return self._unknown("Eye values outside expected range")

        if np.any(visibility < 0.50):
            return self._unknown("Eye visibility insufficient")

        # ---------------------------------------------------------
        # 1. Driver-specific eye calibration
        # ---------------------------------------------------------
        if self.baseline_eye_opening is None:
            if self.calibration_start is None:
                self.calibration_start = now

            self.calibration_samples.append(opening.copy())

            elapsed = now - self.calibration_start

            if elapsed < self.calibration_seconds:
                return {
                    "eye_state": "CALIBRATING",
                    "perclos": None,
                    "status": "CALIBRATION",
                    "calibrated": False,
                    "reason": "Collecting eye calibration samples",
                    "window_seconds": self.current_window_seconds,
                    "valid_observations": 0,
                    "stability": None,
                }

            baseline = np.mean(self.calibration_samples, axis=0)

            if (
                not np.all(np.isfinite(baseline))
                or np.any(baseline <= 0)
            ):
                self.calibration_start = None
                self.calibration_samples.clear()
                return self._unknown("Invalid eye calibration baseline")

            self.baseline_eye_opening = baseline

            print(
                "Adaptive eye calibration complete. "
                f"Left baseline: {baseline[0]:.3f}, "
                f"Right baseline: {baseline[1]:.3f}"
            )

        # ---------------------------------------------------------
        # 2. Classify the current valid eye observation
        # ---------------------------------------------------------
        relative_opening = opening / self.baseline_eye_opening

        closed = bool(
            np.all(relative_opening < self.closure_threshold)
        )

        state = 1 if closed else 0

        self.recent_states.append(state)

        # ---------------------------------------------------------
        # 3. Select the adaptive window
        # ---------------------------------------------------------
        self.current_window_seconds, stability = self._select_window()

        # ---------------------------------------------------------
        # 4. Store this valid observation
        # ---------------------------------------------------------
        self.history.append((now, state))
        self.last_valid_observation_time = now

        # Remove observations outside the currently selected window.
        cutoff = now - self.current_window_seconds

        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        # ---------------------------------------------------------
        # 5. Calculate open-eye ratio from valid observations only
        # ---------------------------------------------------------
        if len(self.history) < self.min_observations:
            return {
                "eye_state": "CLOSED" if closed else "OPEN",
                "perclos": None,
                "status": "WARMING_UP",
                "calibrated": True,
                "reason": "Collecting observations for adaptive window",
                "window_seconds": self.current_window_seconds,
                "valid_observations": len(self.history),
                "stability": stability,
            }

        closed_count = sum(
            observation_state
            for _, observation_state in self.history
        )

        open_eye_ratio = 1.0 - (
            closed_count / len(self.history)
        )

        # ---------------------------------------------------------
        # 6. Apply the documented baseline status thresholds
        # ---------------------------------------------------------
        if open_eye_ratio < self.danger_threshold:
            status = "DANGER"
        elif open_eye_ratio < self.low_awareness_threshold:
            status = "LOW_AWARENESS"
        else:
            status = "SAFE"

        return {
            "eye_state": "CLOSED" if closed else "OPEN",
            "perclos": float(open_eye_ratio),
            "status": status,
            "calibrated": True,
            "reason": "Valid adaptive-window observation",
            "window_seconds": self.current_window_seconds,
            "valid_observations": len(self.history),
            "stability": stability,
        }