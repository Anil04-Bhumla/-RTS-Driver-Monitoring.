from collections import deque


class ActionMonitor:
    """
    Phone-use and smoking monitor based on the author's documentation.

    Action classes:
        0 = Normal
        1 = Phone Use
        2 = Smoking

    Each activity uses its own 30-observation buffer.
    """

    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size

        self.phone_history = deque(maxlen=buffer_size)
        self.smoking_history = deque(maxlen=buffer_size)

    def reset(self):
        self.phone_history.clear()
        self.smoking_history.clear()

    @staticmethod
    def _get_status(history, buffer_size):
        ratio = sum(history) / buffer_size

        if ratio > 0.9:
            status = "DANGER"
        elif ratio > 0.7:
            status = "LOW_AWARENESS"
        else:
            status = "SAFE"

        return status, ratio

    def update(self, action, tracking_valid=True):
        if not tracking_valid or action not in (0, 1, 2):
            return {
                "status": "UNKNOWN",
                "phone_status": "UNKNOWN",
                "smoking_status": "UNKNOWN",
                "phone_ratio": None,
                "smoking_ratio": None,
                "action": None,
            }

        # Record whether each activity was detected.
        self.phone_history.append(1 if action == 1 else 0)
        self.smoking_history.append(1 if action == 2 else 0)

        phone_status, phone_ratio = self._get_status(
            self.phone_history, self.buffer_size
        )

        smoking_status, smoking_ratio = self._get_status(
            self.smoking_history, self.buffer_size
        )

        # Overall action status is the highest risk.
        levels = {
            "SAFE": 0,
            "LOW_AWARENESS": 1,
            "DANGER": 2,
        }

        if levels[phone_status] >= levels[smoking_status]:
            overall_status = phone_status
        else:
            overall_status = smoking_status

        action_name = {
            0: "Normal",
            1: "Phone Use",
            2: "Smoking",
        }[action]

        return {
            "status": overall_status,
            "phone_status": phone_status,
            "smoking_status": smoking_status,
            "phone_ratio": phone_ratio,
            "smoking_ratio": smoking_ratio,
            "action": action_name,
        }