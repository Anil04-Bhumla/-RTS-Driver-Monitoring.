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
from src.dms.head_pose_monitor import HeadPoseMonitor
from src.dms.action_monitor import ActionMonitor
from src.dms.yawning_monitor import YawningMonitor
from src.dms.decision_unit import DecisionUnit


def main():

    # --------------------------------------------------
    # 1. Initialize camera
    # --------------------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --------------------------------------------------
    # 2. Initialize monitoring modules
    # --------------------------------------------------
    perclos_monitor = PerclosMonitor(
        calibration_seconds=5.0,
        buffer_size=100,
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

            csv_writer.writerow([
                "timestamp",
                "frame_number",
                "frame_latency_ms",
                "inference_latency_ms",
                "fps",
                "deadline_ms",
                "deadline_miss",
                "eye_state",
                "eye_metric",
                "head_status",
                "yawning_status",
                "phone_status",
                "smoking_status",
                "combined_status",
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

                # Calculate FPS
                current_time = time.perf_counter()
                elapsed = current_time - prev_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_time = current_time

                # Run face tracking and DMS inference
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
                eye_metric = perclos_result["perclos"]

                # --------------------------------------------------
                # 8. Head-pose monitoring
                # Pass raw pose: HeadPoseMonitor calibrates itself.
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

                # Get calibration states
                eye_calibrated = perclos_result.get("calibrated", False)
                head_calibrated = head_pose_result.get("calibrated", False)

                # Get actual calibrated values from the monitors
                eye_baseline = perclos_monitor.baseline_eye_opening
                head_baseline = head_pose_monitor.pose_zero

                # Eye calibration display
                eye_state_text = (
                    f"Eye Calibration: DONE | Baseline: {eye_baseline}"
                    if eye_calibrated
                    else "Eye Calibration: CALIBRATING..."
                )

                # Head calibration display
                head_state_text = (
                    f"Head Calibration: DONE | Baseline: {head_baseline}"
                    if head_calibrated
                    else "Head Calibration: CALIBRATING..."
                )

                # Draw eye calibration information
                cv2.putText(
                    frame,
                    eye_state_text,
                    (20, 540),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if eye_calibrated else (0, 255, 255),
                    2,
                )

                # Draw head calibration information
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
                # The current DMSInfer implementation returns a
                # mouth class index: 0, 1, or 2.
                # YawningMonitor must accept this representation.
                # --------------------------------------------------
                yawning_result = yawning_monitor.update(
                    float(mouth_s),
                    tracking_valid=tracking_valid,
                )

                # tempory debugging line 
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
                # End-to-end frame processing latency
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
                    datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    total_frames,
                    round(frame_latency_ms, 2),
                    round(inference_latency_ms, 2),
                    round(fps, 2),
                    deadline_ms,
                    int(deadline_miss),
                    eye_state,
                    (
                        round(eye_metric * 100.0, 2)
                        if eye_metric is not None
                        else ""
                    ),
                    head_pose_result["status"],
                    yawning_result["status"],
                    action_result["phone_status"],
                    action_result["smoking_status"],
                    combined_status,
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
                    eye_text = f"Eye metric: {eye_metric * 100.0:.1f}%"
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

                # Combined driver status
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

                # Draw facial landmarks
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