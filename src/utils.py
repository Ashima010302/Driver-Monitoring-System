# src/utils.py
import numpy as np
import cv2
from threading import Lock


ALARM_START_LOCK = Lock()
play_alarm = type('obj', (object,), {'is_playing': False})()
ALARM_SOUND_PATH = "alert.wav"

def is_alarm_playing():
    return getattr(play_alarm, 'is_playing', False)

def play_alarm_sound(sound_path):
    """Play alarm WAV using simpleaudio if available. Thread-safe guard."""
    if is_alarm_playing():
        return
    with ALARM_START_LOCK:
        if is_alarm_playing():
            return
        try:
            import simpleaudio as sa
            wave_obj = sa.WaveObject.from_wave_file(sound_path)
            play_obj = wave_obj.play()
            play_alarm.is_playing = True
            play_obj.wait_done()
            play_alarm.is_playing = False
        except Exception as e:
            print(f"Sound error: {e}")
            play_alarm.is_playing = False


# 3D model points for HPE

MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],         # Nose tip
    [0.0, -330.0, -65.0],    # Chin
    [-225.0, 170.0, -135.0], # Left eye left corner
    [225.0, 170.0, -135.0],  # Right eye right corner
    [-150.0, -150.0, -125.0],# Left Mouth corner
    [150.0, -150.0, -125.0]  # Right mouth corner
], dtype=np.float64)

# HPE landmark indices 
HPE_LANDMARKS = [1, 152, 226, 446, 57, 287]  # nose, chin, L-eye, R-eye, L-mouth, R-mouth

# Default eye & mouth indices 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 382, 381, 380, 374, 373]

MOUTH_VERTICAL = [13, 14]
MOUTH_HORIZONTAL = [61, 291]

# tuning constants
EYE_AR_CONSEC_FRAMES = 12
HPE_CONSEC_FRAMES = 20
FIXED_EAR_DEVIATION = 1.0
UTILS_DEBUG = False


def _has_indices(landmarks, indices):

    if not landmarks:
        return False
    try:
        return max(indices) < len(landmarks)
    except Exception:
        return False

# EAR calculation 

def eye_aspect_ratio(landmarks, eye_indices, frame_w=None, frame_h=None):
 
    try:
        if not _has_indices(landmarks, eye_indices):
            if UTILS_DEBUG:
                print("EAR DEBUG: missing indices -> returning 0")
            return 0.0

        # normalized points (N x 2)
        pts_all = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)
        pts = pts_all[list(eye_indices)]  # shape (6,2) expected

        # get indices of leftmost and rightmost points (horizontal endpoints)
        xs = pts[:, 0]
        left_idx = int(np.argmin(xs))
        right_idx = int(np.argmax(xs))
        p_left = pts[left_idx]
        p_right = pts[right_idx]

        # horizontal span
        h = np.linalg.norm(p_right - p_left)

        # safety: require horizontal span not tiny relative to face (use absolute MIN_H and relative MIN_H_REL)
        MIN_H = 1e-3        # absolute normalized units
        MIN_H_REL = 0.04    # relative fraction of typical inter-eye width (tunable)
        if h < MIN_H:
            if UTILS_DEBUG:
                print(f"EAR DEBUG: absolute horizontal too small h={h:.6f}")
            return 0.0

        # remaining indices (the four eyelid points)
        rem_idxs = [i for i in range(len(pts)) if i not in (left_idx, right_idx)]
        if len(rem_idxs) < 4:
            if UTILS_DEBUG:
                print("EAR DEBUG: not enough remaining points:", rem_idxs)
            return 0.0

        rem_pts = pts[rem_idxs]  # shape (4,2)

        # assign rem_pts to left or right group by x-distance to endpoints
        left_group = []
        right_group = []
        for rp in rem_pts:
            d_left = abs(rp[0] - p_left[0])
            d_right = abs(rp[0] - p_right[0])
            if d_left <= d_right:
                left_group.append(rp)
            else:
                right_group.append(rp)

        # if grouping uneven, fallback: split first two / last two
        if len(left_group) != 2 or len(right_group) != 2:
            # fallback deterministic grouping
            left_group = rem_pts[:2].tolist()
            right_group = rem_pts[2:].tolist()

        # compute vertical gap for each side as max(y)-min(y)
        left_ys = [p[1] for p in left_group]
        right_ys = [p[1] for p in right_group]
        v_left = abs(max(left_ys) - min(left_ys))
        v_right = abs(max(right_ys) - min(right_ys))

        # if both verticals are zero (bad data) -> fail
        if (v_left <= 0.0) and (v_right <= 0.0):
            if UTILS_DEBUG:
                print("EAR DEBUG: both vertical gaps zero -> returning 0")
            return 0.0

        ear = (v_left + v_right) / (2.0 * h)

        
        if np.isnan(ear) or np.isinf(ear):
            if UTILS_DEBUG:
                print("EAR DEBUG: nan/inf -> returning 0")
            return 0.0

        
        if ear > 0.8:
            if UTILS_DEBUG:
                print(f"EAR DEBUG: very large EAR {ear:.4f} -> clipping to 0.8")
            ear = 0.8

        if ear < 0.02:
            ear = 0.02

        
        ear *= FIXED_EAR_DEVIATION

        if UTILS_DEBUG:
            print(f"EAR DEBUG -> left_idx:{left_idx} right_idx:{right_idx} h:{h:.6f} vL:{v_left:.6f} vR:{v_right:.6f} ear:{ear:.4f}")

        return float(ear)

    except Exception as e:
        print(f"EAR calc error (robust): {e}")
        return 0.0



# MAR calculation (pixel-based)

def mouth_aspect_ratio(landmarks, vertical_indices, horizontal_indices, frame_w=None, frame_h=None):
   
    try:
        all_indices = list(vertical_indices) + list(horizontal_indices)
        if not _has_indices(landmarks, all_indices):
            return 0.0

        if frame_w is not None and frame_h is not None:
            pts = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in landmarks], dtype=float)
        else:
            pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)

        P_top = pts[vertical_indices[0]]
        P_bottom = pts[vertical_indices[1]]
        P_left = pts[horizontal_indices[0]]
        P_right = pts[horizontal_indices[1]]

        vertical_dist = np.linalg.norm(P_top - P_bottom)
        horizontal_dist = np.linalg.norm(P_left - P_right)
        mar = vertical_dist / horizontal_dist if horizontal_dist != 0 else 0.0

        if np.isnan(mar) or np.isinf(mar):
            return 0.0
        return float(mar)

    except Exception as e:
        print(f"MAR calc error (pixel): {e}")
        return 0.0


# Head Pose Estimation (HPE)

def get_head_pose(landmarks_2d, w, h):
    
    try:
        if not landmarks_2d:
            return 0.0, 0.0, 0.0
        if max(HPE_LANDMARKS) >= len(landmarks_2d):
            return 0.0, 0.0, 0.0

        image_points = np.array([landmarks_2d[idx] for idx in HPE_LANDMARKS], dtype=np.float64)
        focal_length = float(w) if w != 0 else 1.0
        cam_center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, cam_center[0]],
            [0.0, focal_length, cam_center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
           
            yaw = float(angles[1])
            pitch = float(angles[0])
            roll = float(angles[2])
            if UTILS_DEBUG:
                print(f"HPE angles (deg): yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
            return yaw, pitch, roll

        return 0.0, 0.0, 0.0

    except Exception as e:
        print(f"HPE error: {e}")
        return 0.0, 0.0, 0.0
