import time


class DecisionUnit:
    """
    Combines the five driver-monitoring indicators
    according to the author's documented baseline.
    """

    RISK_LEVELS = {
        "SAFE": 0,
        "LOW_AWARENESS": 1,
        "DANGER": 2,
    }

    def __init__(self, recovery_seconds=5.0):
        self.recovery_seconds = recovery_seconds
        self.current_status = "CALIBRATION"
        self.safe_since = None

    def reset(self):
        self.current_status = "CALIBRATION"
        self.safe_since = None

    def update(
        self,
        eye_result,
        head_pose_result,
        yawning_result,
        cellphone_result,
        smoking_result,
        tracking_valid=True,
        calibrated=True,
    ):
        # Calibration must finish before risk decisions.
        if not calibrated:
            self.current_status = "CALIBRATION"
            self.safe_since = None

            return {
                "status": "CALIBRATION",
                "message": "Calibrating driver baseline",
            }

        if not tracking_valid:
            self.safe_since = None

            return {
                "status": "UNKNOWN",
                "message": "Insufficient valid tracking",
            }

        # Read the five documented indicator states.
        signals = {
            "eye_drowsiness": eye_result.get("status", "UNKNOWN"),
            "head_pose_distraction": head_pose_result.get(
                "status", "UNKNOWN"
            ),
            "yawning": yawning_result.get("status", "UNKNOWN"),
            "cellphone": cellphone_result.get("status", "UNKNOWN"),
            "smoking": smoking_result.get("status", "UNKNOWN"),
        }

        # Do not interpret UNKNOWN as SAFE.
        known_signals = [
            status
            for status in signals.values()
            if status in self.RISK_LEVELS
        ]

        if not known_signals:
            self.safe_since = None
            return {
                "status": "UNKNOWN",
                "message": "No valid risk observations",
                "signals": signals,
            }

        # The overall risk is the highest risk among indicators.
        highest_level = max(
            self.RISK_LEVELS[status]
            for status in known_signals
        )

        target_status = next(
            status
            for status, level in self.RISK_LEVELS.items()
            if level == highest_level
        )

        now = time.monotonic()
        current_level = self.RISK_LEVELS.get(
            self.current_status, -1
        )
        target_level = self.RISK_LEVELS[target_status]

        # Escalate immediately.
        if target_level > current_level:
            self.current_status = target_status
            self.safe_since = None

        # When all indicators are SAFE, wait five seconds,
        # then downgrade only one level at a time.
        elif target_status == "SAFE":
            if self.safe_since is None:
                self.safe_since = now

            elif now - self.safe_since >= self.recovery_seconds:
                if self.current_status == "DANGER":
                    self.current_status = "LOW_AWARENESS"
                elif self.current_status == "LOW_AWARENESS":
                    self.current_status = "SAFE"

                self.safe_since = None

        # LOW_AWARENESS while in DANGER must not cause
        # an immediate downgrade.
        elif target_level < current_level:
            self.safe_since = None

        else:
            self.current_status = target_status
            self.safe_since = None

        return {
            "status": self.current_status,
            "message": f"Combined driver risk: {target_status}",
            "signals": signals,
        }