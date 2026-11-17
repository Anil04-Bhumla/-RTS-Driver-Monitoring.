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
from src.dms.perclos_monitor import PerclosMonitor
from src.dms.adaptive_perclos_monitor import AdaptivePerclosMonitor
from src.dms.head_pose_monitor import HeadPoseMonitor
from src.dms.action_monitor import ActionMonitor
from src.dms.yawning_monitor import YawningMonitor
from src.dms.decision_unit import DecisionUnit


def main():

    # --------------------------------------------------
    # 1. Initialize camera
    # --------------------------------------------------
    # cap = cv2.VideoCapture(0)

    # if not cap.isOpened():
    #     print("Error: Could not open webcam.")
    #     return

    # ================= VIDEO INPUT CONFIGURATION =================

    VIDEO_PATH = "/Users/anilgurjar/Desktop/test_data.mov"

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_PATH}")
        return

    print(f"Video input: {VIDEO_PATH}")
    print(f"Video FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
    print(f"Total video frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    # --------------------------------------------------
    # 2. Initialize monitoring modules
    # --------------------------------------------------

    # Experiment configuration:
    # Select "baseline" or "adaptive".
    EYE_MONITOR_MODE = "baseline"

    if EYE_MONITOR_MODE == "baseline":
        perclos_monitor = PerclosMonitor(
            calibration_seconds=5.0,
            buffer_size=100,
        )

    elif EYE_MONITOR_MODE == "adaptive":
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

    else:
        raise ValueError(
            f"Unknown EYE_MONITOR_MODE: {EYE_MONITOR_MODE}"
        )

    head_pose_monitor = HeadPoseMonitor(
        buffer_size=30,
    )

    action_monitor = ActionMonitor(
        buffer_size=30,
    )

    yawning_monitor = YawningMonitor(
        full_open_class=2,
        open_duration_seconds=2.0,
        event_window_seconds=180.0,
        danger_threshold=4,
        danger_duration_seconds=5.0,
    )

    decision_unit = DecisionUnit(
        recovery_seconds=5.0,
    )

    # --------------------------------------------------
    # 3. Real-time performance configuration
    # --------------------------------------------------
    latency_history = deque(maxlen=10000)

    deadline_ms = 100.0
    deadline_misses = 0
    total_frames = 0
    prev_time = time.perf_counter()

    # --------------------------------------------------
    # 4. CSV performance logging
    # --------------------------------------------------
    log_filename = (
        f"rts_performance_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    face_detector = "face_detect/version-RFB-320_sim.onnx"
    dms_model = "dms/reborn.onnx"

    driver_tracker = None

    try:
        with open(
            log_filename,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as log_file:

            csv_writer = csv.writer(log_file)

            # CSV header: 17 columns.
            csv_writer.writerow([
                "timestamp",
                "frame_number",
                "frame_latency_ms",
                "inference_latency_ms",
                "fps",
                "deadline_ms",
                "deadline_miss",
                "eye_monitor_mode",
                "eye_state",
                "eye_status",
                "eye_metric",
                "eye_window_seconds",
                "eye_valid_observations",
                "head_status",
                "yawning_status",
                "phone_status",
                "smoking_status",
                "combined_status",
                "danger_source",
                "active_risk_sources",
            ])

            # --------------------------------------------------
            # 5. Initialize face tracker and DMS model
            # --------------------------------------------------
            driver_tracker = DriverTacker(
                face_detector,
                dms_model,
                5,
                20,
                0,
                device="cpu",
            )

            print("Driver monitoring system initialized.")
            print(f"Eye monitor mode: {EYE_MONITOR_MODE}")
            print("Press 'q' to quit.")

            # --------------------------------------------------
            # 6. Main real-time processing loop
            # --------------------------------------------------
            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    print("Unable to read webcam frame.")
                    break

                frame_start_time = time.perf_counter()

                # Calculate FPS.
                current_time = time.perf_counter()
                elapsed = current_time - prev_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_time = current_time

                # Run face tracking and DMS inference.
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

                # --------------------------------------------------
                # 7. Eye monitoring
                # --------------------------------------------------
                perclos_result = perclos_monitor.update(
                    eyes,
                    tracking_valid=tracking_valid,
                )
                eye_state = perclos_result["eye_state"]
                eye_status = perclos_result["status"]
                eye_metric = perclos_result["perclos"]
                eye_window_seconds = perclos_result.get("window_seconds", "")
                eye_valid_observations = perclos_result.get("valid_observations", "")
                # --------------------------------------------------
                # 8. Head-pose monitoring
                # --------------------------------------------------
                head_pose_result = head_pose_monitor.update(
                    euler,
                    tracking_valid=tracking_valid,
                )

                # --------------------------------------------------
                # 9. Phone and smoking monitoring
                # --------------------------------------------------
                action_result = action_monitor.update(
                    action,
                    tracking_valid=tracking_valid,
                )

                # Get calibration states.
                eye_calibrated = perclos_result.get(
                    "calibrated", False
                )
                head_calibrated = head_pose_result.get(
                    "calibrated", False
                )

                # Get calibrated values from the monitors.
                eye_baseline = perclos_monitor.baseline_eye_opening
                head_baseline = head_pose_monitor.pose_zero

                # Eye calibration display.
                eye_state_text = (
                    f"Eye Calibration: DONE | Baseline: {eye_baseline}"
                    if eye_calibrated
                    else "Eye Calibration: CALIBRATING..."
                )

                # Head calibration display.
                head_state_text = (
                    f"Head Calibration: DONE | Baseline: {head_baseline}"
                    if head_calibrated
                    else "Head Calibration: CALIBRATING..."
                )

                cv2.putText(
                    frame,
                    eye_state_text,
                    (20, 540),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if eye_calibrated else (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    head_state_text,
                    (20, 570),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if head_calibrated else (0, 255, 255),
                    2,
                )

                # --------------------------------------------------
                # 10. Yawning monitoring
                # --------------------------------------------------
                yawning_result = yawning_monitor.update(
                    float(mouth_s),
                    tracking_valid=tracking_valid,
                )

                # Temporary debugging output.
                print(
                    f"Mouth class: {mouth_s}, "
                    f"Mouth state: {yawning_result['mouth_state']}, "
                    f"Yawn count: {yawning_result['yawn_count']}, "
                    f"Yawning status: {yawning_result['status']}"
                )

                # --------------------------------------------------
                # 11. Combine all five indicator results
                # --------------------------------------------------
                calibrated = (
                    bool(perclos_result.get("calibrated", False))
                    and bool(head_pose_result.get("calibrated", False))
                )

                combined_result = decision_unit.update(
                    eye_result=perclos_result,
                    head_pose_result=head_pose_result,
                    yawning_result=yawning_result,
                    cellphone_result={
                        "status": action_result["phone_status"]
                    },
                    smoking_result={
                        "status": action_result["smoking_status"]
                    },
                    tracking_valid=tracking_valid,
                    calibrated=calibrated,
                )

                combined_status = combined_result["status"]
                combined_message = combined_result["message"]

                # Identify which monitoring indicators currently report DANGER.
                danger_sources = []

                if eye_status == "DANGER":
                    danger_sources.append("eye_drowsiness")

                if head_pose_result["status"] == "DANGER":
                    danger_sources.append("head_pose")

                if yawning_result["status"] == "DANGER":
                    danger_sources.append("yawning")

                if action_result["phone_status"] == "DANGER":
                    danger_sources.append("cellphone")

                if action_result["smoking_status"] == "DANGER":
                    danger_sources.append("smoking")

                danger_source = (
                    "|".join(danger_sources)
                    if danger_sources
                    else "NONE"
                )
                # Identify all indicators that are currently not SAFE.
                active_risk_sources = []

                if eye_status not in ("SAFE", "UNKNOWN", "CALIBRATION", "WARMING_UP"):
                    active_risk_sources.append("eye_drowsiness")

                if head_pose_result["status"] not in ("SAFE", "UNKNOWN", "CALIBRATION", "WARMING_UP"):
                    active_risk_sources.append("head_pose")

                if yawning_result["status"] not in ("SAFE", "UNKNOWN", "CALIBRATION", "WARMING_UP"):
                    active_risk_sources.append("yawning")

                if action_result["phone_status"] not in ("SAFE", "UNKNOWN", "CALIBRATION", "WARMING_UP"):
                    active_risk_sources.append("cellphone")

                if action_result["smoking_status"] not in ("SAFE", "UNKNOWN", "CALIBRATION", "WARMING_UP"):
                    active_risk_sources.append("smoking")

                active_risk_sources_text = (
                    "|".join(active_risk_sources)
                    if active_risk_sources
                    else "NONE"
                )

                # Measure frame processing latency.
                frame_end_time = time.perf_counter()

                frame_latency_ms = (
                    frame_end_time - frame_start_time
                ) * 1000.0

                # --------------------------------------------------
                # 12. Real-time performance metrics
                # --------------------------------------------------
                inference_latency_ms = (
                    driver_tracker.inference_latency_ms
                )

                latency_history.append(frame_latency_ms)
                total_frames += 1

                deadline_miss = frame_latency_ms > deadline_ms

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

                # --------------------------------------------------
                # 13. Write performance data to CSV
                # --------------------------------------------------
                csv_writer.writerow([
                    datetime.now().isoformat(timespec="milliseconds"),
                    total_frames,
                    round(frame_latency_ms, 2),
                    round(inference_latency_ms, 2),
                    round(fps, 2),
                    deadline_ms,
                    int(deadline_miss),
                    EYE_MONITOR_MODE,
                    eye_state,
                    eye_status,
                    round(eye_metric * 100.0, 2) if eye_metric is not None else "",
                    eye_window_seconds,
                    eye_valid_observations,
                    head_pose_result["status"],
                    yawning_result["status"],
                    action_result["phone_status"],
                    action_result["smoking_status"],
                    combined_status,
                    danger_source,
                    active_risk_sources_text,
                ])

                log_file.flush()

                # --------------------------------------------------
                # 14. Display model outputs
                # --------------------------------------------------
                if euler is not None and len(euler) >= 3:
                    cv2.putText(
                        frame,
                        f"Pose: {euler[0]:.1f}, "
                        f"{euler[1]:.1f}, {euler[2]:.1f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Inference: {inference_latency_ms:.2f} ms",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Frame latency: {frame_latency_ms:.2f} ms",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Average latency: {average_latency_ms:.2f} ms",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Worst latency: {worst_case_latency_ms:.2f} ms",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Eye state: {eye_state}",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                if eye_metric is not None:
                    eye_text = (
                        f"Eye metric: {eye_metric * 100.0:.1f}%"
                    )
                else:
                    eye_text = "Eye metric: Calibrating/Unknown"

                cv2.putText(
                    frame,
                    eye_text,
                    (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Head pose: {head_pose_result['status']}",
                    (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Yawning: {yawning_result['status']}",
                    (20, 340),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Phone: {action_result['phone_status']}",
                    (20, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Smoking: {action_result['smoking_status']}",
                    (20, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Combined driver status.
                status_colors = {
                    "CALIBRATION": (0, 255, 255),
                    "SAFE": (0, 255, 0),
                    "LOW_AWARENESS": (0, 165, 255),
                    "DANGER": (0, 0, 255),
                    "UNKNOWN": (200, 200, 200),
                }

                status_color = status_colors.get(
                    combined_status,
                    (200, 200, 200),
                )

                cv2.putText(
                    frame,
                    f"Driver status: {combined_status}",
                    (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    status_color,
                    2,
                )

                cv2.putText(
                    frame,
                    combined_message,
                    (20, 480),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_color,
                    1,
                )

                cv2.putText(
                    frame,
                    f"Deadline misses: {deadline_miss_percentage:.2f}%",
                    (20, 520),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (
                        (0, 0, 255)
                        if deadline_miss_percentage > 0
                        else (0, 255, 0)
                    ),
                    2,
                )

                # Draw facial landmarks.
                if lmks is not None and len(lmks) > 0:
                    points = np.asarray(
                        lmks,
                        dtype=int,
                    ).reshape(-1, 2)

                    for point in points:
                        cv2.circle(
                            frame,
                            tuple(point),
                            2,
                            (255, 229, 10),
                            -1,
                        )

                cv2.imshow("Driver Monitoring System", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Performance log saved to: {log_filename}")
    print(f"Total frames recorded: {total_frames}")
    print(f"Deadline misses: {deadline_misses}")

    if total_frames:
        print(
            f"Deadline miss percentage: "
            f"{deadline_misses / total_frames * 100.0:.2f}%"
        )


if __name__ == "__main__":
    main()