# app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
import threading
import numpy as np
import time

from src.drowsiness_monitor import initialize_dms, DMSProcessorState
from src.crash_detector import initialize_fcw, detect_crash_hazard
from src.drowsiness import FACE_MESH, full_calibration
from src.utils import LEFT_EYE, RIGHT_EYE, MOUTH_VERTICAL, MOUTH_HORIZONTAL, HPE_LANDMARKS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

current_video_path = None
stop_processing = False
processing_lock = threading.Lock()

# Global capture holder and calibration progress
CAP_HOLDER = {
    'cap': None,
    'processor': None,
    'ear_thresh': 0.24,  
    'mar_thresh': 0.60,
    'pitch_thresh': 12.0,
    'yaw_thresh': 25.0,
    'calib_progress': 0.0,
    'calib_candidate': None,
    'is_calibrated': False
}
CALIBRATION_FLAG = threading.Event()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/drowsiness')
def drowsiness_page():
    return render_template('drowsiness.html')

@app.route('/crash')
def crash_page():
    return render_template('crash.html')

def generate_drowsiness_webcam():
    global CAP_HOLDER, CALIBRATION_FLAG


    try:
        old = CAP_HOLDER.get('cap')
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
            CAP_HOLDER['cap'] = None
            CAP_HOLDER['processor'] = None
            print("Released previous camera handle in CAP_HOLDER")
    except Exception as e:
        print("Error releasing previous cap:", e)

  
    cap = None
    for attempt in range(5):
        cap = cv2.VideoCapture(0)
        time.sleep(0.15)
        if cap.isOpened():
            print(f"Camera opened on attempt {attempt+1}")
            break
        else:
            try:
                cap.release()
            except:
                pass
            cap = None
            time.sleep(0.3)

    if cap is None or not cap.isOpened():
        print("ERROR: Could not open webcam.")
        h, w = 480, 640
        err_frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(err_frame, "ERROR: Webcam not available", (20, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', err_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print("Warning: failed to set capture properties:", e)

    initial_thresholds = {
        'ear': CAP_HOLDER.get('ear_thresh', 0.24),
        'mar': CAP_HOLDER.get('mar_thresh', 0.60),
        'pitch': CAP_HOLDER.get('pitch_thresh', 12.0),
        'yaw': CAP_HOLDER.get('yaw_thresh', 25.0)
    }
    CAP_HOLDER['processor'] = DMSProcessorState(initial_thresholds)
    CAP_HOLDER['cap'] = cap

    print("Webcam opened and processor initialized:", cap.isOpened())

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from webcam")
                break

            h, w = frame.shape[:2]

            # Always process frames so we can show live view during calibration
            try:
                processor = CAP_HOLDER.get('processor')
                dms_status, landmarks = processor.process_frame(frame)
            except Exception as e:
                print(f"Detection error in webcam loop: {e}")
                dms_status = {
                    'ear': 0.25, 'mar': 0.60, 'yaw': 0.0, 'pitch': 0.0,
                    'alert_text': 'Detection Ready', 'is_alert_needed': False,
                    'ear_thresh': CAP_HOLDER['ear_thresh'],
                    'mar_thresh': CAP_HOLDER['mar_thresh'],
                    'pitch_thresh': CAP_HOLDER['pitch_thresh'],
                    'yaw_thresh': CAP_HOLDER['yaw_thresh']
                }
                landmarks = []

         
            if CALIBRATION_FLAG.is_set():
                progress = CAP_HOLDER.get('calib_progress', 0.0)
                dms_status['is_alert_needed'] = False
                dms_status['alert_text'] = f"CALIBRATING... {progress:.0f}%"
                # reset local counters to avoid accidental alarm
                if CAP_HOLDER.get('processor') is not None:
                    CAP_HOLDER['processor'].local_ear_counter = 0


            is_calibrated = bool(CAP_HOLDER.get('is_calibrated', False))
            ear_thresh_display = CAP_HOLDER.get('ear_thresh', 0.0)
            calib_candidate = CAP_HOLDER.get('calib_candidate', None)

       
            label = f"EAR T: {ear_thresh_display:.3f}"
            if is_calibrated:
                label = "✅ Calibrated " + label
            else:
                label = "Current " + label
            cv2.putText(frame, label, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

          
            if CALIBRATION_FLAG.is_set() and calib_candidate is not None:
                cv2.putText(frame, f"Candidate EAR: {calib_candidate:.3f}", (10, 58),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
                cv2.putText(frame, f"Calib: {CAP_HOLDER.get('calib_progress',0.0):.0f}%", (10, 88),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
            else:
                cv2.putText(frame, f"EAR: {dms_status.get('ear', 0):.3f}", (10, 58),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {dms_status.get('mar', 0):.3f} (T: {CAP_HOLDER['mar_thresh']:.3f})", (10, 88),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {dms_status.get('yaw', 0):.1f}° (T: {CAP_HOLDER['yaw_thresh']:.1f}°)", (10, 118),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {dms_status.get('pitch', 0):.1f}° (T: {CAP_HOLDER['pitch_thresh']:.1f}°)", (10, 148),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

           
            alert_text = dms_status.get('alert_text', 'Monitoring')
            is_alert = dms_status.get('is_alert_needed', False)
            color = (0, 0, 255) if is_alert else (0, 165, 0)
            cv2.rectangle(frame, (0, h-60), (w, h), color, -1)
            cv2.putText(frame, alert_text, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        try:
            if CAP_HOLDER.get('cap') == cap:
                CAP_HOLDER['cap'] = None
                CAP_HOLDER['processor'] = None
            cap.release()
        except Exception:
            pass
        print("Webcam released (robust generator)")


def generate_drowsiness_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Processing video: {video_path}")

    ear_thresh, mar_thresh, pitch_thresh, yaw_thresh = 0.22, 0.60, 12.0, 25.0
    initial_thresholds = {'ear': ear_thresh, 'mar': mar_thresh, 'pitch': pitch_thresh, 'yaw': yaw_thresh}
    processor = DMSProcessorState(initial_thresholds)

    while True:
        success, frame = cap.read()
        if not success:
            break
        try:
            dms_status, landmarks = processor.process_frame(frame)
        except Exception as e:
            print("Video processing error:", e)
            dms_status = {'ear': 0.25, 'mar': 0.60, 'yaw': 0.0, 'pitch': 0.0,
                          'alert_text': 'Detection Ready', 'is_alert_needed': False}
            landmarks = []

        h, w = frame.shape[:2]
        alert_text = dms_status.get('alert_text', 'Monitoring')
        is_alert = dms_status.get('is_alert_needed', False)

        cv2.putText(frame, f"EAR: {dms_status.get('ear', 0):.3f} (T: {processor.ear_thresh:.3f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        color = (0, 0, 255) if is_alert else (0, 165, 0)
        cv2.rectangle(frame, (0, h-60), (w, h), color, -1)
        cv2.putText(frame, alert_text, (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    print("Finished processing video.")

def generate_crash_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        return
    h, w = test_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fcw_model, roi_polygon = initialize_fcw(w, h)

    while True:
        if stop_processing:
            break

        success, frame = cap.read()
        if not success:
            break

        fcw_alert_triggered, processed_frame = detect_crash_hazard(fcw_model, frame, roi_polygon)

        

        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# Routes 
@app.route('/videofeed/webcam')
def video_feed_webcam():
    return Response(generate_drowsiness_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videofeed/drowsiness')
def video_feed_drowsiness():
    global current_video_path, stop_processing
    stop_processing = False
    if current_video_path and os.path.exists(current_video_path):
        return Response(generate_drowsiness_video(current_video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video uploaded", 400

@app.route('/videofeed/crash')
def video_feed_crash():
    global current_video_path, stop_processing
    stop_processing = False
    if current_video_path and os.path.exists(current_video_path):
        return Response(generate_crash_video(current_video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video uploaded", 400

@app.route('/get_current_thresholds', methods=['GET'])
def get_current_thresholds_route():
    global CAP_HOLDER, CALIBRATION_FLAG
    processor = CAP_HOLDER.get('processor')

    live_metrics = {
        'live_ear': 0.0, 'live_mar': 0.0,
        'live_pitch': 0.0, 'live_yaw': 0.0
    }

    if processor:
        try:
            live_metrics = {
                'live_ear': getattr(processor, 'latest_ear', 0.0),
                'live_mar': getattr(processor, 'latest_mar', 0.0),
                'live_pitch': getattr(processor, 'latest_pitch', 0.0),
                'live_yaw': getattr(processor, 'latest_yaw', 0.0)
            }
        except:
            pass

    thresholds = {
        'ear_thresh': f"{CAP_HOLDER.get('ear_thresh', 0.22):.3f}",
        'mar_thresh': f"{CAP_HOLDER.get('mar_thresh', 0.60):.3f}",
        'pitch_thresh': f"{CAP_HOLDER.get('pitch_thresh', 12.0):.1f}",
        'yaw_thresh': f"{CAP_HOLDER.get('yaw_thresh', 25.0):.1f}",
        'is_calibrating': CALIBRATION_FLAG.is_set(),
        'calib_progress': float(CAP_HOLDER.get('calib_progress', 0.0)),
        'calib_candidate': f"{CAP_HOLDER.get('calib_candidate', '')}" if CAP_HOLDER.get('calib_candidate', None) is not None else "",
        'is_calibrated': bool(CAP_HOLDER.get('is_calibrated', False)),
        **live_metrics
    }
    return jsonify(thresholds), 200

@app.route('/start_calibration', methods=['POST'])
def start_calibration_route():
    global CAP_HOLDER, CALIBRATION_FLAG

    if CALIBRATION_FLAG.is_set():
        return jsonify({'error': 'Calibration already running'}), 409

    cap = CAP_HOLDER.get('cap')
    if cap is None:
        return jsonify({'error': 'Webcam not active. Please start webcam first.'}), 400

    CALIBRATION_FLAG.set()
    CAP_HOLDER['calib_progress'] = 0.0
    CAP_HOLDER['calib_candidate'] = None
    CAP_HOLDER['is_calibrated'] = False

    def progress_updater(valid, total, provisional_ear=None):
        try:
            pct = (valid / total) * 100.0 if total > 0 else 0.0
            CAP_HOLDER['calib_progress'] = pct
            if provisional_ear is not None:
                CAP_HOLDER['calib_candidate'] = float(provisional_ear)
        except Exception:
            CAP_HOLDER['calib_progress'] = 0.0
            CAP_HOLDER['calib_candidate'] = None

    def calibration_task():  
        try:
            print("CalibrationTask: starting")
         
            calibrated_tuple = full_calibration(cap, FACE_MESH, calibration_frames=20, max_frame_tries=120)
            print("CalibrationTask: full_calibration returned:", calibrated_tuple)
            
            new_thresholds = {
                'ear': calibrated_tuple[0],
                'mar': calibrated_tuple[1],
                'pitch': calibrated_tuple[2],
                'yaw': calibrated_tuple[3]
            }
            print("CalibrationTask: converted to dict:", new_thresholds)

            CAP_HOLDER.update({
                'ear_thresh': new_thresholds['ear'],
                'mar_thresh': new_thresholds['mar'],
                'pitch_thresh': new_thresholds['pitch'],
                'yaw_thresh': new_thresholds['yaw']
            })

          
            proc = CAP_HOLDER.get('processor')
            if proc is not None:
                proc.update_thresholds(new_thresholds)  
              
                proc.latest_ear = 0.0
                proc.latest_mar = 0.0
                proc.latest_pitch = 0.0
                proc.latest_yaw = 0.0
              

            CAP_HOLDER['is_calibrated'] = True
            CAP_HOLDER['calib_progress'] = 100.0
            CAP_HOLDER['calib_candidate'] = None

            print(f"✅ Calibration SUCCESS! New: EAR={new_thresholds['ear']:.3f}, MAR={new_thresholds['mar']:.3f}")

        except Exception as e:
            print(f"❌ Calibration failed: {e}")
        finally:
            CALIBRATION_FLAG.clear()
            print("Calibration task finished. Detection resumed.")

    threading.Thread(target=calibration_task, daemon=True).start()
    return jsonify({'success': True, 'message': 'Calibration started.'}), 200

@app.route('/uploadvideo', methods=['POST'])
def upload_video():
    global current_video_path, stop_processing

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        stop_processing = True
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            current_video_path = filepath
            return jsonify({'success': True, 'message': 'Video uploaded successfully'}), 200
        except Exception as e:
            print(f"--- SERVER CONSOLE ERROR: FILE SAVE FAILED ---\n{e}")
            return jsonify({'error': f'Server failed to save video file. Error: {e}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/stopprocessing', methods=['POST'])
def stop_processing_route():
    global stop_processing
    stop_processing = True
    return jsonify({'success': True}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
