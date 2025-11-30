# src/drowsiness.py
import mediapipe as mp
import cv2
import numpy as np
import time

from src.utils import (
    eye_aspect_ratio, mouth_aspect_ratio, get_head_pose,
    FIXED_EAR_DEVIATION, LEFT_EYE, RIGHT_EYE, 
    MOUTH_VERTICAL, MOUTH_HORIZONTAL, HPE_LANDMARKS,
    EYE_AR_CONSEC_FRAMES, HPE_CONSEC_FRAMES
)

FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class DrowsinessMonitor:
    def __init__(self, initial_ear_thresh, initial_mar_thresh, initial_pitch_thresh, initial_yaw_thresh):
        self.EAR_THRESH = initial_ear_thresh
        self.MAR_THRESH = initial_mar_thresh
        self.HPE_PITCH_THRESH = initial_pitch_thresh
        self.HPE_YAW_THRESH = initial_yaw_thresh
        
        self.counter = 0
        self.distraction_counter = 0
        self.EYE_AR_CONSEC_FRAMES = EYE_AR_CONSEC_FRAMES
        self.HPE_CONSEC_FRAMES = HPE_CONSEC_FRAMES

    def run_detection(self, frame):
        """Processes a single frame and determines the driver's current state."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = FACE_MESH.process(frame_rgb)
        
        h, w = frame.shape[:2]
        landmarks = []
        
        status = {
            'ear': 0.0, 'mar': 0.0, 'yaw': 0.0, 'pitch': 0.0,
            'counter': self.counter,
            'distraction_counter': self.distraction_counter,
            'alert_text': 'NO FACE DETECTED',
            'is_alert_needed': False
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]  
          
            mp_landmarks = face_landmarks.landmark
            
           
            left_ear = eye_aspect_ratio(mp_landmarks, LEFT_EYE, frame_w=w, frame_h=h)
            right_ear = eye_aspect_ratio(mp_landmarks, RIGHT_EYE, frame_w=w, frame_h=h)
            current_ear = (left_ear + right_ear) / 2.0
            
            current_mar = mouth_aspect_ratio(mp_landmarks, MOUTH_VERTICAL, MOUTH_HORIZONTAL, frame_w=w, frame_h=h)
            
       
            current_yaw, current_pitch, _ = get_head_pose(landmarks, w, h)
            
            status.update({
                'ear': current_ear,
                'mar': current_mar,
                'yaw': current_yaw,
                'pitch': current_pitch
            })
            status['alert_text'] = 'Monitoring Active'
            
            # A. EAR Counter (drowsiness tracking)
            if current_ear < self.EAR_THRESH:
                self.counter += 1
            else:
                self.counter = 0
            
            # B. Yawning (MAR)
            if current_mar > self.MAR_THRESH and not status['is_alert_needed']:
                status['is_alert_needed'] = True
                status['alert_text'] = 'YAWN ALERT!'
            
            # C. Distraction (HPE - Yaw/Pitch)
            is_extreme_turn = abs(current_yaw) > self.HPE_YAW_THRESH
            is_downward_nod = current_pitch > self.HPE_PITCH_THRESH
            
            if is_extreme_turn or is_downward_nod:
                self.distraction_counter += 1
                if self.distraction_counter >= self.HPE_CONSEC_FRAMES and not status['is_alert_needed']:
                    status['is_alert_needed'] = True
                    status['alert_text'] = 'DISTRACTION ALERT!'
            else:
                self.distraction_counter = 0
            
            status['counter'] = self.counter
            status['distraction_counter'] = self.distraction_counter
        else:
            self.counter = 0
            self.distraction_counter = 0
            
        return status, landmarks

def full_calibration(cap, facemesh, calibration_frames=20, max_frame_tries=120, progresscb=None):
    """Robust calibration with relaxed criteria for real-world use"""
    print("Starting FULL calibration... Look straight ahead with eyes OPEN!")
    print("Criteria: Eyes open, mouth closed, head mostly straight")
    
    ear_values, mar_values, pitch_values, yaw_values = [], [], [], []
    frame_count = 0
    valid_frames = 0
    
    while frame_count < max_frame_tries and valid_frames < calibration_frames:
        ret, frame = cap.read()
        if not ret:
            frame_count += 1
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = facemesh.process(frame_rgb)
        h, w = frame.shape[:2]
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            mp_landmarks = face_landmarks.landmark
            landmarks_2d = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
            
            left_ear = eye_aspect_ratio(mp_landmarks, LEFT_EYE, frame_w=w, frame_h=h)
            right_ear = eye_aspect_ratio(mp_landmarks, RIGHT_EYE, frame_w=w, frame_h=h)
            current_ear = (left_ear + right_ear) / 2.0
            
            current_mar = mouth_aspect_ratio(mp_landmarks, MOUTH_VERTICAL, MOUTH_HORIZONTAL, frame_w=w, frame_h=h)
            current_yaw, current_pitch, _ = get_head_pose(landmarks_2d, w, h)
            
          
            is_eyes_open = current_ear > 0.05  
            is_mouth_closed = current_mar < 0.60    
            is_head_straight = (abs(current_yaw) < 25.0) and (abs(current_pitch) < 25.0)  
            
           
            if is_eyes_open and is_mouth_closed:   
                ear_values.append(current_ear)
                mar_values.append(current_mar)
                pitch_values.append(abs(current_pitch))
                yaw_values.append(abs(current_yaw))
                valid_frames += 1
                
                # Progress callback
                try:
                    provisional_ear = float(np.median(ear_values))
                    if progresscb:
                        progresscb(valid_frames, calibration_frames, provisional_ear)
                except:
                    provisional_ear = 0.0
                
                progress = (valid_frames / calibration_frames) * 100
                print(f"Progress: {progress:.1f}% ({valid_frames}/{calibration_frames}) EAR: {current_ear:.3f}")
                
            frame_count += 1
            time.sleep(0.03)  # 30fps pacing
            
        else:
            frame_count += 1
    
    print("Calibration complete!")
    
    # Fallback if insufficient frames
    if len(ear_values) < calibration_frames * 0.5:
        return 0.22, 0.60, 12.0, 25.0  
    
    # Compute personalized thresholds
    median_ear = np.median(ear_values)
    median_mar = np.median(mar_values)
    mean_pitch = np.mean(pitch_values)
    mean_yaw = np.mean(yaw_values)
    
    ear_thresh = max(0.20, median_ear * 0.92 - np.std(ear_values) * 0.35)
    mar_thresh = max(0.40, median_mar * 3.0)
    pitch_thresh = max(10.0, mean_pitch + 8.0)
    yaw_thresh = max(15.0, mean_yaw + 8.0)
    
  
    
    return ear_thresh, mar_thresh, pitch_thresh, yaw_thresh  # Tuple!
