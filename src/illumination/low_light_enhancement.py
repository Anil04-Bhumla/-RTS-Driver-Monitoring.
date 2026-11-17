import cv2


def calculate_brightness(frame):
    """
    Calculate average brightness of an input frame.

    Returns:
        float: Average brightness between 0 and 255.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    return float(brightness)

def gamma_correction(frame, gamma=1.5):
    """
    Brighten a dark image using Gamma Correction.
    """

    import numpy as np

    inv_gamma = 1.0 / gamma

    lookup_table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    enhanced_frame = cv2.LUT(frame, lookup_table)

    return enhanced_frame


def enhance_low_light(frame, on_threshold=45, off_threshold=55, gamma=1.5):

    brightness = calculate_brightness(frame)

    # Remember the previous enhancement state
    if not hasattr(enhance_low_light, "enhancement_active"):
        enhance_low_light.enhancement_active = False

    # Turn enhancement ON
    if brightness < on_threshold:
        enhance_low_light.enhancement_active = True

    # Turn enhancement OFF
    elif brightness > off_threshold:
        enhance_low_light.enhancement_active = False

    # Apply enhancement based on the current state
    if enhance_low_light.enhancement_active:
        enhanced_frame = gamma_correction(frame, gamma)
        return enhanced_frame, brightness, True

    return frame, brightness, False