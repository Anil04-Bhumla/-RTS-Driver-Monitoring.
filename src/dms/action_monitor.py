import time


class ActionMonitor:
    def __init__(
        self,
        confirmation_duration=1.0,
        history_duration=2.0,
        minimum_detection_ratio=0.7,
        normal_clear_duration=1.0,
    ):
        self.confirmation_duration = confirmation_duration
        self.history_duration = history_duration
        self.minimum_detection_ratio = minimum_detection_ratio
        self.normal_clear_duration = normal_clear_duration

        self.history = []
        self.last_timestamp = None

        # Currently confirmed alert
        self.active_action = None

        # Candidate action being evaluated
        self.candidate_action = None
        self.candidate_start = None

        # Continuous Normal prediction timer
        self.normal_start = None

    def update(self, action, tracking_valid=True):
        current_time = time.monotonic()

        # Tracking is invalid, so predictions cannot be trusted.
        if not tracking_valid or action not in (0, 1, 2):
            self.history.clear()
            self.last_timestamp = None
            self.candidate_action = None
            self.candidate_start = None
            self.normal_start = None

            return {
                "status": "UNKNOWN",
                "action": self.active_action,
                "duration": 0.0,
            }

        # Handle an unexpected timestamp reset.
        if (
            self.last_timestamp is not None
            and current_time < self.last_timestamp
        ):
            self.history.clear()
            self.candidate_action = None
            self.candidate_start = None
            self.normal_start = None

        self.last_timestamp = current_time

        # Store the latest prediction.
        self.history.append((current_time, action))

        # Keep only recent predictions.
        cutoff = current_time - self.history_duration
        self.history = [
            item for item in self.history
            if item[0] >= cutoff
        ]

        # -------------------------------------------------
        # Case 1: Normal is predicted
        # -------------------------------------------------
        if action == 0:
            self.candidate_action = None
            self.candidate_start = None

            if self.normal_start is None:
                self.normal_start = current_time

            normal_duration = current_time - self.normal_start

            # Clear the alert only after sustained Normal.
            if normal_duration >= self.normal_clear_duration:
                self.active_action = None

                return {
                    "status": "SAFE",
                    "action": "Normal",
                    "duration": 0.0,
                }

            # Preserve an existing alert during brief Normal predictions.
            if self.active_action is not None:
                return {
                    "status": "DETECTED",
                    "action": self.active_action,
                    "duration": normal_duration,
                }

            return {
                "status": "SAFE",
                "action": "Normal",
                "duration": 0.0,
            }

        # A non-normal prediction interrupts the Normal timer.
        self.normal_start = None

        # -------------------------------------------------
        # Case 2: Phone Use or Smoking is predicted
        # -------------------------------------------------
        if action != self.candidate_action:
            self.candidate_action = action
            self.candidate_start = current_time
            self.history = [
                (current_time, action)
            ]

        # Count predictions matching the current candidate.
        matching = [
            item for item in self.history
            if item[1] == self.candidate_action
        ]

        total = len(self.history)
        ratio = len(matching) / total if total else 0.0
        duration = current_time - self.candidate_start

        # Require both sustained detection and consistency.
        confirmed = (
            duration >= self.confirmation_duration
            and ratio >= self.minimum_detection_ratio
        )

        if confirmed:
            self.active_action = {
                1: "Phone Use",
                2: "Smoking",
            }[self.candidate_action]

            return {
                "status": "DETECTED",
                "action": self.active_action,
                "duration": duration,
            }

        # Preserve an existing alert while evaluating a new candidate.
        if self.active_action is not None:
            return {
                "status": "DETECTED",
                "action": self.active_action,
                "duration": duration,
            }

        return {
            "status": "MONITORING",
            "action": {
                1: "Phone Use",
                2: "Smoking",
            }[self.candidate_action],
            "duration": duration,
        }