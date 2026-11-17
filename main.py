'''
Copyright (C) University of Modena and Reggio Emilia - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Carmelo Scribano <carmelo.scribano@unimore.it>
'''

import cv2
import numpy as np
import time
import csv

from datetime import datetime
from collections import deque

from src.driver_tracker import DriverTacker
from src.illumination.low_light_enhancement import enhance_low_light

from src.dms.adaptive_perclos_monitor import AdaptivePerclosMonitor
from src.dms.head_pose_monitor import HeadPoseMonitor
from src.dms.yawning_monitor import YawningMonitor
from src.dms.action_monitor import ActionMonitor
from src.dms.decision_unit import DecisionUnit


# ==================================================
# INITIALIZE WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    cap.release()

    raise SystemExit(1)

print("Webcam initialized successfully.")


# ==================================================
# INITIALIZE MONITORING COMPONENTS
# ==================================================

# 1. ADAPTIVE EYE MONITOR

perclos_monitor = AdaptivePerclosMonitor(

    calibration_seconds=5.0,

    min_window_seconds=1.0,

    max_window_seconds=6.0,

    stability_window_size=10,

    closure_threshold=0.50,

    danger_threshold=0.40,

    low_awareness_threshold=0.60,

    min_observations=5,
)


# 2. AUTHOR'S HEAD POSE MONITOR

head_pose_monitor = HeadPoseMonitor(

    buffer_size=30
)


# 3. EXISTING CUSTOM YAWNING MONITOR

yawning_monitor = YawningMonitor(

    full_open_class=2,

    open_duration_seconds=1.0,

    event_window_seconds=180.0,

    danger_threshold=3,

    danger_duration_seconds=5.0,
)


# 4. AUTHOR'S ACTION MONITOR

action_monitor = ActionMonitor()


# 5. AUTHOR'S DECISION UNIT

decision_unit = DecisionUnit(

    recovery_seconds=5.0
)


print("\nAll monitoring components initialized.")

print("Eye Monitoring       : Adaptive PERCLOS")
print("Head Pose Monitoring : Author Baseline")
print("Yawning Monitoring   : Existing Custom Algorithm")
print("Phone Detection      : Author Baseline")
print("Smoking Detection    : Author Baseline")
print("Final Decision       : Author Decision Unit")


# ==================================================
# RTS PERFORMANCE CONFIGURATION
# ==================================================

latency_history = deque(maxlen=10000)

deadline_ms = 100.0

deadline_misses = 0

total_frames = 0


# ==================================================
# CSV LOGGING
# ==================================================

log_filename = (
    f"integrated_dms_rts_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


try:

    with open(
        log_filename,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as log_file:

        csv_writer = csv.writer(log_file)

        csv_writer.writerow([

            "timestamp",
            "frame_number",

            "camera_read_latency_ms",
            "processing_latency_ms",
            "inference_latency_ms",
            "processing_fps",

            "average_latency_ms",
            "worst_case_latency_ms",

            "deadline_ms",
            "deadline_miss",
            "deadline_miss_percentage",

            "eye_state",
            "eye_status",
            "eye_metric",
            "eye_window_seconds",
            "eye_valid_observations",

            "head_pose_status",
            "head_pose_score",
            "head_pose_ratio",

            "yawning_status",
            "yawning_events",

            "phone_status",
            "phone_ratio",

            "smoking_status",
            "smoking_ratio",

            "final_driver_status",

            "tracking_valid",
            "calibrated",

        ])


        # ==========================================
        # INITIALIZE FACE DETECTOR AND DMS MODEL
        # ==========================================

        face_detector = "face_detect/version-RFB-320_sim.onnx"

        dms_model = "dms/reborn.onnx"

        driver_tracker = DriverTacker(

            face_detector,

            dms_model,

            5,

            20,

            0,

            device="cpu",
        )


        print("\nIntegrated Driver Monitoring System Started.")

        print("Eye calibration: 5 seconds")

        print("Adaptive eye window: 1-6 seconds")

        print("Head pose buffer: 30 observations")

        print("Yawning duration: 2 seconds")

        print("RTS deadline: 100 ms")

        print("\nPress 'q' to quit.\n")


        # ==========================================
        # MAIN WEBCAM PROCESSING LOOP
        # ==========================================

        while cap.isOpened():

            # --------------------------------------
            # CAMERA READ
            # --------------------------------------

            read_start_time = time.perf_counter()

            ret, frame = cap.read()

            read_end_time = time.perf_counter()

            if not ret:

                print("ERROR: Unable to read webcam frame.")

                break

            camera_read_latency_ms = (

                read_end_time - read_start_time

            ) * 1000.0


            # --------------------------------------
            # START PROCESSING MEASUREMENT
            # --------------------------------------

            frame_start_time = time.perf_counter()

            # --------------------------------------

            # LOW-LIGHT ENHANCEMENT

            # --------------------------------------

            processed_frame, brightness, enhancement_active = enhance_low_light(

                frame,

                on_threshold=45,

                off_threshold=55,

                gamma=1.5

            )


            # --------------------------------------
            # FACE TRACKING AND DMS INFERENCE
            # --------------------------------------

            (
                is_tracking,
                lmks,
                euler,
                eyes,
                mouth_s,
                action,

            ) = driver_tracker.update(frame)


            tracking_valid = (

                bool(is_tracking)

                and lmks is not None

                and len(lmks) > 0

            )


            # ======================================
            # 1. ADAPTIVE EYE MONITORING
            # ======================================

            eye_result = perclos_monitor.update(

                eyes,

                tracking_valid=tracking_valid,
            )

            eye_state = eye_result.get(
                "eye_state", "UNKNOWN"
            )

            eye_status = eye_result.get(
                "status", "UNKNOWN"
            )

            eye_metric = eye_result.get(
                "perclos", None
            )

            eye_window_seconds = eye_result.get(
                "window_seconds", ""
            )

            eye_valid_observations = eye_result.get(
                "valid_observations", ""
            )

            eye_calibrated = eye_result.get(
                "calibrated", False
            )


            # --------------------------------------
            # PREPARE EYE RESULT FOR DECISION UNIT
            # --------------------------------------

            eye_decision_result = dict(eye_result)

            # WARMING_UP is not a risk level.
            # Other valid indicators can still contribute.

            if eye_decision_result.get("status") == "WARMING_UP":

                eye_decision_result["status"] = "UNKNOWN"


            # ======================================
            # 2. AUTHOR'S HEAD POSE MONITORING
            # ======================================

            head_pose_result = head_pose_monitor.update(

                euler,

                tracking_valid=tracking_valid,
            )

            head_pose_status = head_pose_result.get(
                "status", "UNKNOWN"
            )

            head_pose_score = head_pose_result.get(
                "pose_score", ""
            )

            head_pose_ratio = head_pose_result.get(
                "distraction_ratio", ""
            )

            head_calibrated = head_pose_result.get(
                "calibrated", False
            )


            # ======================================
            # 3. EXISTING CUSTOM YAWNING MONITORING
            # ======================================

            yawning_result = yawning_monitor.update(

                mouth_s,

                tracking_valid=tracking_valid,
            )
            print(
                "Mouth Output:", mouth_s,
                "| Yawning Result:", yawning_result
            )
            

            yawning_status = yawning_result.get(
                "status", "UNKNOWN"
            )

            yawning_events = yawning_result.get(
                "event_count", ""
            )


            # ======================================
            # 4. AUTHOR'S ACTION MONITORING
            # ======================================

            action_result = action_monitor.update(

                action,

                tracking_valid=tracking_valid,
            )


            # ======================================
            # 5. SEPARATE PHONE AND SMOKING RESULTS
            # ======================================

            cellphone_result = {

                "status": action_result.get(
                    "phone_status", "UNKNOWN"
                )

            }

            smoking_result = {

                "status": action_result.get(
                    "smoking_status", "UNKNOWN"
                )

            }


            phone_status = cellphone_result["status"]

            smoking_status = smoking_result["status"]


            phone_ratio = action_result.get(
                "phone_ratio", ""
            )

            smoking_ratio = action_result.get(
                "smoking_ratio", ""
            )


            # ======================================
            # 6. CALIBRATION STATUS
            # ======================================

            all_calibrated = (

                eye_calibrated

                and head_calibrated

            )


            # ======================================
            # 7. AUTHOR'S FINAL DECISION UNIT
            # ======================================

            final_result = decision_unit.update(

                eye_decision_result,

                head_pose_result,

                yawning_result,

                cellphone_result,

                smoking_result,

                tracking_valid=tracking_valid,

                calibrated=all_calibrated,
            )

            final_status = final_result.get(
                "status", "UNKNOWN"
            )


            # ======================================
            # 8. RTS PERFORMANCE MEASUREMENT
            # ======================================

            inference_latency_ms = (
                driver_tracker.inference_latency_ms
            )

            frame_end_time = time.perf_counter()

            processing_latency_ms = (

                frame_end_time - frame_start_time

            ) * 1000.0


            processing_fps = (

                1000.0 / processing_latency_ms

                if processing_latency_ms > 0

                else 0.0

            )


            latency_history.append(processing_latency_ms)

            total_frames += 1


            # --------------------------------------
            # DEADLINE CHECK
            # --------------------------------------

            deadline_miss = (

                processing_latency_ms > deadline_ms

            )

            if deadline_miss:

                deadline_misses += 1


            average_latency_ms = float(
                np.mean(latency_history)
            )

            worst_case_latency_ms = float(
                np.max(latency_history)
            )

            deadline_miss_percentage = (

                deadline_misses / total_frames

            ) * 100.0


            # ======================================
            # 9. CSV LOGGING
            # ======================================

            csv_writer.writerow([

                datetime.now().isoformat(
                    timespec="milliseconds"
                ),

                total_frames,

                round(camera_read_latency_ms, 2),

                round(processing_latency_ms, 2),

                round(inference_latency_ms, 2),

                round(processing_fps, 2),

                round(average_latency_ms, 2),

                round(worst_case_latency_ms, 2),

                deadline_ms,

                int(deadline_miss),

                round(deadline_miss_percentage, 2),

                eye_state,

                eye_status,

                (
                    round(eye_metric * 100.0, 2)
                    if eye_metric is not None
                    else ""
                ),

                eye_window_seconds,

                eye_valid_observations,

                head_pose_status,

                head_pose_score,

                head_pose_ratio,

                yawning_status,

                yawning_events,

                phone_status,

                phone_ratio,

                smoking_status,

                smoking_ratio,

                final_status,

                int(tracking_valid),

                int(all_calibrated),

            ])

            log_file.flush()


            # ======================================
            # 10. DISPLAY STATUS COLOURS
            # ======================================

            if final_status == "DANGER":

                status_color = (0, 0, 255)

                alert_text = "DANGER: DRIVER ATTENTION REQUIRED"

            elif final_status == "LOW_AWARENESS":

                status_color = (0, 165, 255)

                alert_text = "WARNING: LOW DRIVER AWARENESS"

            elif final_status == "SAFE":

                status_color = (0, 255, 0)

                alert_text = "DRIVER STATUS: SAFE"

            elif final_status == "CALIBRATION":

                status_color = (0, 255, 255)

                alert_text = "CALIBRATING..."

            else:

                status_color = (200, 200, 200)

                alert_text = "MONITORING UNAVAILABLE"


            # ======================================
            # 11. DISPLAY RTS INFORMATION
            # ======================================

            display_info = [

                (f"FPS: {processing_fps:.2f}", 30),

                (f"Inference: {inference_latency_ms:.2f} ms", 60),

                (f"Processing: {processing_latency_ms:.2f} ms", 90),

                (f"Camera Read: {camera_read_latency_ms:.2f} ms", 120),

                (f"Average: {average_latency_ms:.2f} ms", 150),

                (f"Worst: {worst_case_latency_ms:.2f} ms", 180),

                (f"Deadline Miss: {deadline_miss_percentage:.2f}%", 210),

            ]


            for text, y in display_info:

                cv2.putText(

                    frame,

                    text,

                    (20, y),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    2,

                )


            # ======================================
            # 12. DISPLAY INDIVIDUAL INDICATORS
            # ======================================

            indicator_info = [

                (f"Eye: {eye_status}", 255),

                (f"Eye State: {eye_state}", 285),

                (f"Head Pose: {head_pose_status}", 315),

                (f"Yawning: {yawning_status}", 345),

                (f"Phone: {phone_status}", 375),

                (f"Smoking: {smoking_status}", 405),

            ]


            for text, y in indicator_info:

                cv2.putText(

                    frame,

                    text,

                    (20, y),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2,

                )


            # ======================================
            # 13. DISPLAY ADAPTIVE EYE INFORMATION
            # ======================================

            if eye_metric is not None:

                eye_text = (

                    f"Open-eye ratio: "
                    f"{eye_metric * 100.0:.1f}%"

                )

            else:

                eye_text = "Open-eye ratio: Unavailable"


            cv2.putText(

                frame,

                eye_text,

                (20, 440),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 255),

                2,

            )


            cv2.putText(

                frame,

                f"Adaptive Window: {eye_window_seconds} sec",

                (20, 470),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 0),

                2,

            )


            # ======================================
            # 14. DISPLAY FINAL DRIVER STATUS
            # ======================================

            cv2.putText(

                frame,

                f"FINAL STATUS: {final_status}",

                (20, 520),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                status_color,

                2,

            )


            cv2.putText(

                frame,

                alert_text,

                (20, 555),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                status_color,

                2,

            )


            # ======================================
            # 15. DRAW FACIAL LANDMARKS
            # ======================================

            if lmks is not None and len(lmks) > 0:

                points = np.asarray(
                    lmks, dtype=int
                ).reshape(-1, 2)

                for point in points:

                    cv2.circle(

                        frame,

                        tuple(point),

                        2,

                        (255, 229, 10),

                        -1,

                    )


            # ======================================
            # 16. DISPLAY FRAME
            # ======================================

            cv2.imshow(

                "Integrated Adaptive Driver Monitoring System",

                frame,

            )

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break


finally:

    cap.release()

    cv2.destroyAllWindows()


# ==================================================
# FINAL RTS PERFORMANCE SUMMARY
# ==================================================

print("\n========== INTEGRATED DMS TEST SUMMARY ==========")

print(f"Log file: {log_filename}")

print(f"Total frames: {total_frames}")

print(f"Deadline: {deadline_ms} ms")

print(f"Deadline misses: {deadline_misses}")


if total_frames:

    print(

        f"Deadline miss percentage: "

        f"{deadline_misses / total_frames * 100.0:.2f}%"

    )

    print(

        f"Average processing latency: "

        f"{np.mean(latency_history):.2f} ms"

    )

    print(

        f"Worst-case processing latency: "

        f"{np.max(latency_history):.2f} ms"

    )

    print(

        f"Average processing FPS: "

        f"{np.mean([1000.0 / x for x in latency_history if x > 0]):.2f}"

    )


print("=================================================\n")