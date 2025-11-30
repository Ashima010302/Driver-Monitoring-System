# src/drowsiness_monitor.py
from collections import deque
import numpy as np
import threading

from src.drowsiness import DrowsinessMonitor, FACE_MESH, full_calibration 
from src.utils import EYE_AR_CONSEC_FRAMES, LEFT_EYE, RIGHT_EYE, play_alarm_sound, ALARM_SOUND_PATH

def initialize_dms(skip_calib, cap):
    if skip_calib or cap is None:
        # Use defaults
        default_thresholds = {'ear': 0.22, 'mar': 0.60, 'pitch': 12.0, 'yaw': 25.0}
      
    else:
      
        print("Starting calibration... Look straight ahead, eyes OPEN!")
        calibrated_thresholds = full_calibration(
            cap=cap, 
            facemesh=FACE_MESH, 
            calibration_frames=20,
            progress_cb=None
        )
        default_thresholds = {
            'ear': calibrated_thresholds[0],
            'mar': calibrated_thresholds[1], 
            'pitch': calibrated_thresholds[2],
            'yaw': calibrated_thresholds[3]
        }
        print(f" Calibrated: EAR={default_thresholds['ear']:.3f}, MAR={default_thresholds['mar']:.3f}")

    dms_monitor = DrowsinessMonitor(
        initial_ear_thresh=default_thresholds['ear'],
        initial_mar_thresh=default_thresholds['mar'],
        initial_pitch_thresh=default_thresholds['pitch'],
        initial_yaw_thresh=default_thresholds['yaw']
    )
    return dms_monitor, default_thresholds


class DMSProcessorState:
    def __init__(self, thresholds):
        self.dms_monitor = DrowsinessMonitor(
            initial_ear_thresh=thresholds['ear'],
            initial_mar_thresh=thresholds['mar'],
            initial_pitch_thresh=thresholds['pitch'],
            initial_yaw_thresh=thresholds['yaw']
        )

        self.ear_thresh = thresholds['ear']
        self.mar_thresh = thresholds['mar']
        self.pitch_thresh = thresholds['pitch']
        self.yaw_thresh = thresholds['yaw']

    
        self.ear_history = deque(maxlen=6)

        self.local_ear_counter = 0
        
        self.local_ear_consec_frames = 8
        self.drowsy_eye_counter = 0
        self.drowsy_eye_frames_threshold = 90 

      
        self.yawn_counter = 0
        self.yawn_frames_threshold = 90  

        
        self.distraction_counter = 0
        self.distraction_frames_threshold = 90  

        self.latest_ear = 0.0
        self.latest_mar = 0.0
        self.latest_pitch = 0.0
        self.latest_yaw = 0.0

    def update_thresholds(self, new_thresholds):
        self.ear_thresh = new_thresholds['ear']
        self.mar_thresh = new_thresholds['mar']
        self.pitch_thresh = new_thresholds['pitch']
        self.yaw_thresh = new_thresholds['yaw']

        self.dms_monitor.EAR_THRESH = new_thresholds['ear']
        self.dms_monitor.MAR_THRESH = new_thresholds['mar']
        self.dms_monitor.HPE_PITCH_THRESH = new_thresholds['pitch']
        self.dms_monitor.HPE_YAW_THRESH = new_thresholds['yaw']

    def process_frame(self, frame):
        dms_status, landmarks = self.dms_monitor.run_detection(frame)

        ear_val = float(dms_status.get('ear', 0.0))
       

        if ear_val > 0:
            self.ear_history.append(ear_val)
            smoothed_ear = float(np.median(np.array(self.ear_history)))
           
        else:
            smoothed_ear = ear_val

        self.latest_ear = smoothed_ear
        self.latest_mar = float(dms_status.get('mar', 0.0))
        self.latest_pitch = float(dms_status.get('pitch', 0.0))
        self.latest_yaw = float(dms_status.get('yaw', 0.0))

        # Only count as "closed" if smoothed_ear is positive and strictly below threshold.
        if smoothed_ear and (smoothed_ear < self.ear_thresh):
            self.local_ear_counter += 1
            
        else:
            if self.local_ear_counter != 0:
                print("EYES OPEN (reset counter)")
            self.local_ear_counter = 0

        # Fire only when we've seen enough consecutive frames
        if self.local_ear_counter >= self.local_ear_consec_frames:
            dms_status['is_alert_needed'] = True
            dms_status['alert_text'] = "DROWSINESS ALERT!"
          
            try:
                alarm_thread = threading.Thread(target=play_alarm_sound, args=(ALARM_SOUND_PATH,))
                alarm_thread.daemon = True
                alarm_thread.start()
            except Exception as e:
                print(f"Failed to start alarm thread: {e}")

        dms_status['ear'] = smoothed_ear
        dms_status['ear_thresh'] = self.ear_thresh
        dms_status['mar_thresh'] = self.mar_thresh
        dms_status['pitch_thresh'] = self.pitch_thresh
        dms_status['yaw_thresh'] = self.yaw_thresh

        return dms_status, landmarks
