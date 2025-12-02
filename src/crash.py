# src/crash.py
import cv2
import numpy as np
import os
import threading  # Added threading for non-blocking alarm
from ultralytics import YOLO

# Import alarm utilities from the project's utility script
from src.utils import play_alarm_sound, ALARM_SOUND_PATH

# --- FCW CONSTANTS ---
FCW_MODEL_PATH = "models/yolov8n.pt"

# Mapping from YOLO class index to readable names
CLASS_NAMES = {
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# --- FCW FUNCTIONS ---
def setup_fcw_environment(frame_w, frame_h):
    """Loads the YOLO model and sets up the ROI."""
    if not os.path.exists(FCW_MODEL_PATH):
        print(f"FCW Warning: Model file '{FCW_MODEL_PATH}' not found. FCW will be disabled.")
        return None, None

    print("Loading FCW model...")
    fcw_model = YOLO(FCW_MODEL_PATH)

    # Define a more restrictive trapezoid ROI
    ROI_POLYGON = np.array([
        (int(frame_w * 0.2), frame_h),
        (int(frame_w * 0.8), frame_h),
        (int(frame_w * 0.6), int(frame_h * 0.6)),
        (int(frame_w * 0.4), int(frame_h * 0.6))
    ], dtype=np.int32)

    ROI_POLYGON = ROI_POLYGON.reshape((-1, 1, 2))

    return fcw_model, ROI_POLYGON


def get_center_bottom_point(box):
    """Calculates the center-bottom point of the detection box."""
    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
    center_x = (x_min + x_max) // 2
    bottom_y = y_max
    return center_x, bottom_y


def is_point_in_roi(point, roi_polygon):
    """Checks if a point is inside the ROI polygon."""
    point_float = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(roi_polygon, point_float, False) >= 0


def run_fcw_detection(fcw_model, frame, roi_polygon=None):
    """
    Robust FCW detection (vehicle-only) with defensive types for pointPolygonTest.
    Detects only: bicycle(1), car(2), motorcycle(3), bus(5), truck(7).
    """
    h, w, _ = frame.shape
    fcw_alert = False
    processed_frame = frame.copy()

    VEHICLE_IDS = {1, 2, 3, 5, 7}

    SAFETY_ZONE_DEPTH = 0.20
    SAFETY_ZONE_WIDTH = 0.60
    MIN_BOX_REL = 0.06
    MIN_BOX_PX = int(h * 0.03)

    cx = int(w // 2)
    bottom_y = int(h)
    far_y = int(h - h * SAFETY_ZONE_DEPTH)
    near_w = int(w * 0.90)
    far_w = int(w * SAFETY_ZONE_WIDTH)

    poly = [
        (cx - near_w // 2, bottom_y),
        (cx + near_w // 2, bottom_y),
        (cx + far_w // 2, far_y),
        (cx - far_w // 2, far_y)
    ]
    safety_poly = np.asarray(poly, dtype=np.int32).reshape(-1, 2)

    # Draw translucent safety zone
    overlay = processed_frame.copy()
    cv2.fillPoly(overlay, [safety_poly], (0, 165, 255))
    cv2.addWeighted(overlay, 0.15, processed_frame, 0.85, 0)
    cv2.polylines(processed_frame, [safety_poly], True, (0, 120, 255), 2)

    # Run YOLO safely
    try:
        results = fcw_model(processed_frame, verbose=False)
        result = results[0]
    except Exception as e:
        print("FCW model call failed:", e)
        return False, processed_frame

    if not hasattr(result, "boxes"):
        return False, processed_frame

    # Iterate detections
    for box in result.boxes:
        try:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cls_id = int(box.cls.cpu().numpy())
        except Exception:
            continue

        if cls_id not in VEHICLE_IDS:
            continue

        box_h = max(1, y2 - y1)
        rel_h = float(box_h) / float(h)

        if (rel_h < MIN_BOX_REL) and (box_h < MIN_BOX_PX):
            continue

        col_x = int((x1 + x2) // 2)
        col_y = int(y2)
        pt = (col_x, col_y)

        inside = False
        try:
            val = cv2.pointPolygonTest(safety_poly, (float(pt[0]), float(pt[1])), False)
            inside = val >= 0
        except Exception:
            inside = False

        color = (0, 0, 255) if inside else (0, 255, 0)
        label = "WARNING" if inside else "VEHICLE"

        if inside:
            fcw_alert = True

        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(processed_frame, label, (x1, max(12, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Trigger alarm safely
    if fcw_alert:
        try:
            alarm_thread = threading.Thread(target=play_alarm_sound, args=(ALARM_SOUND_PATH,))
            alarm_thread.daemon = True
            alarm_thread.start()
        except Exception as e:
            print(f"Failed to start crash alarm thread: {e}")

    # Bottom status banner
    banner_color = (0, 0, 255) if fcw_alert else (0, 165, 0)
    status = "COLLISION WARNING!" if fcw_alert else "Monitoring: Safe"

    cv2.rectangle(processed_frame, (0, h - 70), (w, h), banner_color, -1)
    cv2.putText(processed_frame, status, (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    return fcw_alert, processed_frame
