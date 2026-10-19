from collections import deque
import time

import numpy as np


class PerclosMonitor:
    """
    Rolling-window PERCLOS monitor with continuous eye-closure
    detection.

    Thresholds are initial experimental values, not validated
    automotive safety thresholds.
    """

    def __init__(
        self,
        window_seconds=60.0,
        opening_threshold=0.20,
        visibility_threshold=0.50,
        relative_closure_threshold=0.30,
        max_sample_gap_seconds=0.25,
        warning_closure_seconds=0.5,
        danger_closure_seconds=1.5,
        perclos_warning_threshold=15.0,
        perclos_danger_threshold=25.0,
    ):
        self.window_seconds = window_seconds
        self.opening_threshold = opening_threshold
        self.visibility_threshold = visibility_threshold
        self.relative_closure_threshold = relative_closure_threshold
        self.max_sample_gap_seconds = max_sample_gap_seconds

        self.warning_closure_seconds = warning_closure_seconds
        self.danger_closure_seconds = danger_closure_seconds

        self.perclos_warning_threshold = perclos_warning_threshold
        self.perclos_danger_threshold = perclos_danger_threshold

        self.samples = deque()

        self.closed_since = None
        self.last_valid_time = None

    def reset(self):
        self.samples.clear()
        self.closed_since = None
        self.last_valid_time = None

    def update(
        self,
        eyes,
        tracking_valid=True,
        baseline_eye_opening=None,
    ):
        now = time.monotonic()

        def unknown(reason):
            # Do not count an unobserved interval as continuous
            # eye closure.
            self.closed_since = None
            self.last_valid_time = None

            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "closed_duration": 0.0,
                "status": "UNKNOWN",
                "reason": reason,
            }

        if not tracking_valid:
            return unknown("Tracking invalid")

        if eyes is None:
            return unknown("Eye output missing")

        try:
            values = np.asarray(
                eyes,
                dtype=float,
            ).reshape(-1)
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

        if np.any(visibility < self.visibility_threshold):
            return unknown("Eye visibility insufficient")

        # Determine whether both eyes are closed.
        if baseline_eye_opening is not None:
            try:
                baseline = np.asarray(
                    baseline_eye_opening,
                    dtype=float,
                ).reshape(-1)
            except (TypeError, ValueError):
                return unknown("Invalid eye calibration baseline")

            if (
                baseline.size != 2
                or not np.all(np.isfinite(baseline))
                or np.any((baseline <= 0) | (baseline > 1))
            ):
                return unknown("Invalid eye calibration baseline")

            relative_opening = opening / baseline

            closed = bool(
                np.all(
                    relative_opening
                    <= self.relative_closure_threshold
                )
            )
        else:
            closed = bool(
                np.all(opening <= self.opening_threshold)
            )

        # Reset continuous-closure tracking if observations are
        # separated by an excessively large time gap.
        if (
            self.last_valid_time is not None
            and now - self.last_valid_time
            > self.max_sample_gap_seconds
        ):
            self.closed_since = None

        self.last_valid_time = now

        if closed:
            if self.closed_since is None:
                self.closed_since = now
        else:
            self.closed_since = None

        closed_duration = (
            now - self.closed_since
            if self.closed_since is not None
            else 0.0
        )

        # Update rolling PERCLOS history.
        self.samples.append((now, closed))

        cutoff = now - self.window_seconds

        while (
            len(self.samples) >= 2
            and self.samples[1][0] <= cutoff
        ):
            self.samples.popleft()

        closed_seconds = 0.0
        observed_seconds = 0.0

        records = list(self.samples)

        for i in range(len(records) - 1):
            t0, closed0 = records[i]
            t1, _ = records[i + 1]

            gap = t1 - t0

            if gap <= 0 or gap > self.max_sample_gap_seconds:
                continue

            duration = min(t1, now) - max(t0, cutoff)

            if duration <= 0:
                continue

            observed_seconds += duration

            if closed0:
                closed_seconds += duration

        perclos = None

        if observed_seconds > 0:
            perclos = (
                100.0 * closed_seconds / observed_seconds
            )

        # Combine continuous closure and rolling PERCLOS.
        status = "SAFE"

        if (
            closed_duration >= self.danger_closure_seconds
            or (
                perclos is not None
                and perclos >= self.perclos_danger_threshold
            )
        ):
            status = "DANGER"

        elif (
            closed_duration >= self.warning_closure_seconds
            or (
                perclos is not None
                and perclos >= self.perclos_warning_threshold
            )
        ):
            status = "WARNING"

        return {
            "eye_state": "CLOSED" if closed else "OPEN",
            "perclos": perclos,
            "closed_duration": closed_duration,
            "status": status,
            "reason": "Valid eye observation",
        }