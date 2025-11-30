# src/crash_detector.py

import cv2

# Imports: Core logic from src.crash
from src.crash import setup_fcw_environment, run_fcw_detection 

def initialize_fcw(frame_w, frame_h):
    """Initializes the FCW model and ROI."""
    try:
        fcw_model, ROI_POLYGON = setup_fcw_environment(frame_w, frame_h)
        return fcw_model, ROI_POLYGON
    except Exception as e:
        print(f"FCW setup failed: {e}")
        return None, None

def detect_crash_hazard(fcw_model, frame, roi_polygon):
    """Runs the FCW detection for a single frame."""
    if fcw_model is None:
        return False, frame
    try:
        fcw_alert_triggered, processed_frame = run_fcw_detection(fcw_model, frame, roi_polygon)
        return fcw_alert_triggered, processed_frame
    except Exception as e:
        print(f"FCW detection error: {e}")
        return False, frame