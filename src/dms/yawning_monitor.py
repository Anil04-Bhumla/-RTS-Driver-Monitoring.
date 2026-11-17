# from collections import deque
# import numpy as np


# class YawningMonitor:
#     """
#     Yawning detection based on the author's documentation.

#     - Moving average over 15 observations.
#     - Yawn condition: 0.4 < mouth average < 0.8.
#     - Four consecutive positive evaluations register a yawn.
#     - Counts yawning events over 180 seconds.
#     """

#     def __init__(self, average_window=15, event_window_seconds=180.0):
#         self.average_window = average_window
#         self.event_window_seconds = event_window_seconds

#         self.mouth_history = deque(maxlen=average_window)
#         self.yawn_events = deque()

#         self.consecutive_positive = 0

#     def reset(self):
#         self.mouth_history.clear()
#         self.yawn_events.clear()
#         self.consecutive_positive = 0

#     def update(self, mouth_score, tracking_valid=True):
#         import time

#         now = time.monotonic()

#         if not tracking_valid or mouth_score is None:
#             self.consecutive_positive = 0

#             return {
#                 "status": "UNKNOWN",
#                 "mouth_average": None,
#                 "yawn_count": len(self.yawn_events),
#                 "new_yawn": False,
#             }

#         try:
#             score = float(mouth_score)
#         except (TypeError, ValueError):
#             self.consecutive_positive = 0

#             return {
#                 "status": "UNKNOWN",
#                 "mouth_average": None,
#                 "yawn_count": len(self.yawn_events),
#                 "new_yawn": False,
#             }
#         if not np.isfinite(score) or score not in (0.0, 1.0, 2.0):
#             self.consecutive_positive = 0

#             return {
#                 "status": "UNKNOWN",
#                 "mouth_average": None,
#                 "yawn_count": len(self.yawn_events),
#                 "new_yawn": False,
#             }


#         # The documentation defines the mouth-opening score as mouth / 2.
#         normalized_score = score / 2.0
#         self.mouth_history.append(normalized_score)

#         mouth_average = sum(self.mouth_history) / len(self.mouth_history)

#         # Author's documented yawn condition.
#         positive = 0.4 < mouth_average < 0.8

#         if positive:
#             self.consecutive_positive += 1
#         else:
#             self.consecutive_positive = 0

#         new_yawn = False

#         # Register a yawn after four consecutive positive evaluations.
#         if self.consecutive_positive >= 4:
#             self.yawn_events.append(now)
#             self.consecutive_positive = 0
#             new_yawn = True

#         # Retain events from the last 180 seconds.
#         cutoff = now - self.event_window_seconds

#         while self.yawn_events and self.yawn_events[0] < cutoff:
#             self.yawn_events.popleft()

#         yawn_count = len(self.yawn_events)

#         # Author's documented event-count thresholds.
#         if yawn_count > 3:
#             status = "DANGER"
#         elif yawn_count > 1:
#             status = "LOW_AWARENESS"
#         else:
#             status = "SAFE"

#         return {
#             "status": status,
#             "mouth_average": mouth_average,
#             "yawn_count": yawn_count,
#             "new_yawn": new_yawn,
#         }



from collections import deque
import time
import math


class YawningMonitor:
    """
    Yawning detector for the MtDMS project.

    Rules:
    - Fully open mouth must persist for 2 seconds.
    - One continuous opening registers at most one yawn.
    - More than 4 yawns within 180 seconds triggers DANGER.
    - DANGER remains active for 5 seconds.
    - After DANGER, clear the yawn history and start a fresh window.

    Assumption:
    - mouth_class == 2 represents a fully open mouth.
      Verify this against the model's actual class mapping.
    """

    def __init__(
        self,
        full_open_class=2,
        open_duration_seconds=1.0,
        event_window_seconds=180.0,
        danger_threshold=3,
        danger_duration_seconds=5.0,
    ):
        self.full_open_class = full_open_class
        self.open_duration_seconds = open_duration_seconds
        self.event_window_seconds = event_window_seconds
        self.danger_threshold = danger_threshold
        self.danger_duration_seconds = danger_duration_seconds

        self.yawn_events = deque()

        self.open_start_time = None
        self.event_registered = False

        self.danger_start_time = None
        self.window_start_time = time.monotonic()

    def reset(self):
        """Reset the monitor and begin a fresh counting window."""
        self.yawn_events.clear()

        self.open_start_time = None
        self.event_registered = False

        self.danger_start_time = None
        self.window_start_time = time.monotonic()

    def update(self, mouth_class, tracking_valid=True):
        """Process one mouth-class observation."""

        now = time.monotonic()
        new_yawn = False

        # --------------------------------------------------
        # 1. If DANGER is active, hold it for 5 seconds.
        # --------------------------------------------------
        if self.danger_start_time is not None:
            elapsed = now - self.danger_start_time

            if elapsed < self.danger_duration_seconds:
                return self._result(
                    status="DANGER",
                    new_yawn=False,
                    mouth_state="UNKNOWN" if not tracking_valid else "MONITORING",
                )

            # Danger hold has ended.
            # Clear old events and begin a completely fresh window.
            self.yawn_events.clear()

            self.danger_start_time = None
            self.window_start_time = now

            self.open_start_time = None
            self.event_registered = False

            return self._result(
                status="SAFE",
                new_yawn=False,
                mouth_state="MONITORING",
            )

        # --------------------------------------------------
        # 2. Remove expired events from the rolling window.
        # --------------------------------------------------
        cutoff = now - self.event_window_seconds

        while self.yawn_events and self.yawn_events[0] <= cutoff:
            self.yawn_events.popleft()

        # --------------------------------------------------
        # 3. Validate tracking and mouth-class observation.
        # --------------------------------------------------
        if not tracking_valid or mouth_class is None:
            self.open_start_time = None

            return self._result(
                status=self._get_status(),
                new_yawn=False,
                mouth_state="UNKNOWN",
            )

        try:
            value = float(mouth_class)
        except (TypeError, ValueError):
            self.open_start_time = None

            return self._result(
                status=self._get_status(),
                new_yawn=False,
                mouth_state="UNKNOWN",
            )

        if not math.isfinite(value) or value not in (0.0, 1.0, 2.0):
            self.open_start_time = None

            return self._result(
                status=self._get_status(),
                new_yawn=False,
                mouth_state="UNKNOWN",
            )

        mouth_class = int(value)

        # --------------------------------------------------
        # 4. Detect a sustained fully open mouth.
        # --------------------------------------------------
        if mouth_class == self.full_open_class:

            if self.open_start_time is None:
                self.open_start_time = now

            open_duration = now - self.open_start_time

            if (
                open_duration >= self.open_duration_seconds
                and not self.event_registered
            ):
                self.yawn_events.append(now)
                self.event_registered = True
                new_yawn = True

        else:
            # Mouth closed or not fully open.
            self.open_start_time = None
            self.event_registered = False

        # --------------------------------------------------
        # 5. Trigger DANGER when count exceeds 4.
        # --------------------------------------------------
        if len(self.yawn_events) > self.danger_threshold:
            self.danger_start_time = now

            return self._result(
                status="DANGER",
                new_yawn=new_yawn,
                mouth_state=(
                    "FULLY_OPEN"
                    if mouth_class == self.full_open_class
                    else "NOT_FULLY_OPEN"
                ),
            )

        # --------------------------------------------------
        # 6. Return normal status.
        # --------------------------------------------------
        return self._result(
            status="SAFE",
            new_yawn=new_yawn,
            mouth_state=(
                "FULLY_OPEN"
                if mouth_class == self.full_open_class
                else "NOT_FULLY_OPEN"
            ),
        )

    def _get_status(self):
        """Return the current yawning status."""
        if len(self.yawn_events) > self.danger_threshold:
            return "DANGER"

        return "SAFE"

    def _result(self, status, new_yawn, mouth_state):
        """Return consistent output fields for main.py."""
        return {
            "status": status,
            "yawn_count": len(self.yawn_events),
            "new_yawn": new_yawn,
            "mouth_state": mouth_state,
            "mouth_average": None,
        }