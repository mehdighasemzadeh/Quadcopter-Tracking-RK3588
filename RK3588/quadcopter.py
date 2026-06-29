#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:24:12 2026

@author: mehdi
"""

import cv2
import threading
import queue
import time
import json
import socket
import psutil
import subprocess
import numpy as np
import asyncio
import math 
import struct
import sys

# AI Imports
from rknnlite.api import RKNNLite
import yolo
from yolo import CLASSES
from py_utils.coco_utils import COCO_test_helper

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed


# --- CONFIGURATION ---
STATION_IP = "192.168.1.3"  # IP of your PC running the Dashboard and Gazebo
CMD_PORT = 6000
TELEM_PORT = 6001
MAVLINK_PORT = "/dev/ttyUSB0"
MAVLINK_BAUD = 57600
Gazebo_sim = True
Gazebo_sim_port = 5005

DEMO_VIDEO_PATH = "car1_demo.mp4" 

INPUT_SIZE = (640, 640)

# --- GLOBAL AI CONFIGURATION (Synced from Station) ---
AI_CONFIDENCE = 0.25
ACTIVE_MODEL = 'yolov8n.rknn'
TARGET_CLASSES = ["person", "car", "motorbike ", "truck "]
AUTO_RECOVERY = False
AUTO_TRACK = False
AUTO_TRACK_STRATEGY = "largest"
MAX_LOST_FRAMES = 20 # How long Kalman filter will predict blind
ENABLE_KALMAN = True
ENABLE_SYNC = True

# --- QUEUES & THREAD-SAFE STATE ---
rgb_queue = queue.Queue(maxsize=1)
detection_queue = queue.Queue(maxsize=1)
command_queue = queue.Queue(maxsize=5)
velocity_queue = queue.Queue(maxsize=1)

coord_lock = threading.Lock()
current_coords = {
    "cx": 0, "cy": 0, "locked": False, "bbox": None, 
    "tracker_fps": 0, "detections": [], "last_class": None,
    "kalman_predicting": False, 
    "sync_count": 0,
    "v_cx": 0.0, "v_cy": 0.0 # Added Kalman velocity states
}
current_tracker_type = "CSRT"

# --- DYNAMIC CONTROL PARAMETERS ---
control_params_lock = threading.Lock()
control_params = {
    "kp_x": 5.5, "ki_x": 0.0, "kd_x": 4.2,
    "kp_y": 4.5, "ki_y": 0.0, "kd_y": 3.2,
    "max_forward_speed": 10.0, # Now used as Max X (Forward/Back) Speed
    "forward_acceleration": 2.0, # Now used as Max X (Forward/Back) Accel
    "max_accel_y": 3.0,
    "max_accel_z": 3.0, # Unused in 2D demo, kept for dashboard compatibility
    "sma_window": 20
}

# ==========================================
# MATH & GEOMETRY HELPERS
# ==========================================
def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def bb_containment(tracker_box, yolo_box):
    xA = max(tracker_box[0], yolo_box[0])
    yA = max(tracker_box[1], yolo_box[1])
    xB = min(tracker_box[0] + tracker_box[2], yolo_box[0] + yolo_box[2])
    yB = min(tracker_box[1] + tracker_box[3], yolo_box[1] + yolo_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    yolo_area = yolo_box[2] * yolo_box[3]
    
    if yolo_area == 0: 
        return 0.0
    return interArea / float(yolo_area)

# ==========================================
# KALMAN FILTER TRACKER
# ==========================================
class KalmanBBoxTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0

    def init_state(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32).reshape(8, 1)

    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = pred[0][0], pred[1][0], pred[2][0], pred[3][0]
        w, h = max(1.0, w), max(1.0, h)
        return [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]

    def correct(self, bbox, source="tracker"):
        if source == "yolo":
            self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.01 
        else:
            self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.2

        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        measurement = np.array([cx, cy, w, h], np.float32).reshape(4, 1)
        self.kf.correct(measurement)

# ==========================================
# ULTRA-OPTIMIZED AI NMS
# ==========================================
_GRID_CACHE = {}

def ultra_optimized_post_process(input_data):
    global AI_CONFIDENCE
    OBJ_THRESH = AI_CONFIDENCE
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    
    all_boxes, all_scores, all_classes = [], [], []
    
    for i in range(defualt_branch):
        box_tensor = input_data[pair_per_branch * i]
        cls_tensor = input_data[pair_per_branch * i + 1]
        h, w = cls_tensor.shape[2:4]
        
        cls_flat = cls_tensor.transpose(0, 2, 3, 1).reshape(-1, 80)
        classes = np.argmax(cls_flat, axis=-1)
        scores = np.max(cls_flat, axis=-1)
        
        keep_idx = np.where(scores >= OBJ_THRESH)[0]
        if len(keep_idx) == 0: continue
            
        branch_scores = scores[keep_idx]
        branch_classes = classes[keep_idx]
        
        box_flat = box_tensor.transpose(0, 2, 3, 1).reshape(-1, 64)
        filtered_boxes = box_flat[keep_idx]
        
        K = filtered_boxes.shape[0]
        y = filtered_boxes.reshape(K, 4, 16)
        y_max = np.max(y, axis=2, keepdims=True)
        exp_y = np.exp(y - y_max)
        y = exp_y / np.sum(exp_y, axis=2, keepdims=True)
        acc_matrix = np.arange(16, dtype=np.float32)
        dfl_output = np.sum(y * acc_matrix, axis=2)
        
        cache_key = (h, w)
        if cache_key not in _GRID_CACHE:
            col, row = np.meshgrid(np.arange(0, w), np.arange(0, h))
            _GRID_CACHE[cache_key] = (col.reshape(-1), row.reshape(-1), 640 // h, 640 // w)
        
        col_flat, row_flat, stride_y, stride_x = _GRID_CACHE[cache_key]
        grid_x = col_flat[keep_idx]
        grid_y = row_flat[keep_idx]
        
        x1 = (grid_x + 0.5 - dfl_output[:, 0]) * stride_y
        y1 = (grid_y + 0.5 - dfl_output[:, 1]) * stride_x
        x2 = (grid_x + 0.5 + dfl_output[:, 2]) * stride_y
        y2 = (grid_y + 0.5 + dfl_output[:, 3]) * stride_x
        
        branch_boxes = np.stack([x1, y1, x2, y2], axis=1)
        all_boxes.append(branch_boxes)
        all_scores.append(branch_scores)
        all_classes.append(branch_classes)
        
    if len(all_boxes) == 0: return None, None, None
        
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, c_ids, s = boxes[inds], classes[inds], scores[inds]
        keep = yolo.nms_boxes(b, s) 
        if len(keep) != 0:
            nboxes.append(b[keep]); nclasses.append(c_ids[keep]); nscores.append(s[keep])
            
    if not nclasses and not nscores: return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

yolo.post_process = ultra_optimized_post_process

def clean_and_sort_outputs(raw_outputs):
    stride_8 = [t for t in raw_outputs if t.shape[2] == 80]
    stride_16 = [t for t in raw_outputs if t.shape[2] == 40]
    stride_32 = [t for t in raw_outputs if t.shape[2] == 20]
    sorted_list = []
    for branch in [stride_8, stride_16, stride_32]:
        if len(branch) == 0: continue
        box_tensor = [t for t in branch if t.shape[1] == 64][0]
        cls_tensor = [t for t in branch if t.shape[1] != 64 and t.shape[1] != 1][0]
        sorted_list.extend([box_tensor, cls_tensor])
        sum_tensors = [t for t in branch if t.shape[1] == 1]
        if sum_tensors: sorted_list.append(sum_tensors[0])
    return sorted_list


# ==========================================
# CAMERA CLASS (OPENCV UDP CHUNKING)
# ==========================================
class RGBCamera(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.width = 1280
        self.height = 720
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def run(self):
        video_fps = 10.0
        frame_delay = 1.0 / video_fps

        print(f"[RGB] Streaming {self.width}x{self.height} @ {video_fps} fps (OpenCV UDP)...")
        
        cap = cv2.VideoCapture(DEMO_VIDEO_PATH)
        if not cap.isOpened():
            print(f"[RGB] ERROR: Could not open {DEMO_VIDEO_PATH}")
            return
            
        frame_id = 0
        # Send packets well below the typical 65535 UDP limit
        max_chunk_size = 60000 

        while True:
            start_time = time.time()
            
            # 1. Read Video Frame directly via OpenCV
            ret, frame = cap.read()
            if not ret:
                print("[RGB] Video file ended. Restarting to loop seamlessly...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame = cv2.resize(frame, (self.width, self.height))

            # Push to local AI and display queues
            try: rgb_queue.put_nowait(frame)
            except queue.Full:
                rgb_queue.get_nowait()
                rgb_queue.put_nowait(frame)
                
            try: detection_queue.put_nowait(frame)
            except queue.Full:
                detection_queue.get_nowait()
                detection_queue.put_nowait(frame)

            # 2. Encode to JPEG for network stream
            # Quality 60 provides excellent clarity while remaining lightweight
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
            if result:
                data = encoded_img.tobytes()
                total_chunks = math.ceil(len(data) / max_chunk_size)
                
                # Split large JPEGs into smaller UDP packets to avoid drops
                for i in range(total_chunks):
                    start = i * max_chunk_size
                    end = min(len(data), start + max_chunk_size)
                    chunk_data = data[start:end]
                    
                    # 3-byte simple Header: [frame_id (0-255), total_chunks, chunk_index]
                    header = bytes([frame_id % 256, total_chunks, i])
                    
                    try:
                        self.sock.sendto(header + chunk_data, (STATION_IP, 5000))
                    except Exception as e:
                        pass # Ignore standard UDP network interruptions
                        
                frame_id += 1

            # Maintain the 10fps framerate
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        cap.release()

# AI YOLO NPU THREAD
# ==========================================
def detection_thread():
    global ACTIVE_MODEL, TARGET_CLASSES, AUTO_RECOVERY, AUTO_TRACK, current_tracker_type
    print("[AI] Initializing NPU Subsystem...")
    rknn = RKNNLite()
    loaded_model = None
    co_helper = COCO_test_helper(enable_letter_box=True)

    while True:
        current_model = ACTIVE_MODEL
        if loaded_model != current_model:
            if loaded_model is not None:
                rknn.release()
                rknn = RKNNLite()
            if rknn.load_rknn(current_model) != 0:
                time.sleep(2); continue
            if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                time.sleep(2); continue
            loaded_model = current_model
            print(f"[AI] {loaded_model} Loaded and Active!")

        try: frame = detection_queue.get(timeout=1.0)
        except queue.Empty: continue

        img_padded = co_helper.letter_box(im=frame.copy(), new_shape=(INPUT_SIZE[1], INPUT_SIZE[0]), pad_color=(0,0,0))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_input = np.ascontiguousarray(np.expand_dims(img_rgb, axis=0))

        outputs = rknn.inference(inputs=[img_input])
        if outputs is None: continue
        try: outputs = clean_and_sort_outputs(outputs)
        except Exception: pass

        boxes, classes, scores = yolo.post_process(outputs)
        frame_detections = []
        
        if boxes is not None:
            real_boxes = co_helper.get_real_box(boxes)
            for box, class_id, score in zip(real_boxes, classes, scores):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                w, h = x2 - x1, y2 - y1
                class_name = CLASSES[int(class_id)] if int(class_id) < len(CLASSES) else "Unknown"
                frame_detections.append([x1, y1, w, h, float(score), class_name])
                
        with coord_lock:
            current_coords['detections'] = frame_detections
            locked = current_coords['locked']
            last_class = current_coords.get('last_class', None)
            tracker_bbox = current_coords.get('bbox', None)

        if ENABLE_SYNC and locked and tracker_bbox is not None:
            best_yolo_det = None
            sync_triggered = False
            sync_reason = ""
            
            for det in frame_detections:
                dx, dy, dw, dh, conf, cls_name = det
                
                if last_class is not None and cls_name != last_class:
                    continue
                
                yolo_box = [dx, dy, dw, dh]
                iou = bb_iou(tracker_bbox, yolo_box)
                containment = bb_containment(tracker_bbox, yolo_box)
                
                trk_area = tracker_bbox[2] * tracker_bbox[3]
                yolo_area = dw * dh
                
                if iou > 0.25 and (trk_area > yolo_area * 1.35 or trk_area < yolo_area * 0.65):
                    best_yolo_det = det
                    sync_triggered = True
                    sync_reason = f"Scale drift (IoU: {iou:.2f}, Area Ratio: {trk_area / yolo_area:.2f})"
                    break
                
                is_tracker_bloated = (trk_area > yolo_area * 2.8) or (tracker_bbox[2] > 550 or tracker_bbox[3] > 400)
                if containment > 0.85 and is_tracker_bloated:
                    best_yolo_det = det
                    sync_triggered = True
                    sync_reason = f"Tracker explosion detected! (YOLO enclosed {containment*100:.1f}%, Tracker area {trk_area} vs YOLO {yolo_area})"
                    break
            
            if sync_triggered and best_yolo_det:
                dx, dy, dw, dh, conf, cls_name = best_yolo_det
                if last_class is None:
                    with coord_lock: current_coords['last_class'] = cls_name
                    last_class = cls_name

                if not command_queue.full():
                    with coord_lock: 
                        current_coords['sync_count'] += 1
                    print(f"\n[SYNC TRIGGERED] {sync_reason}. Forcing template re-init to: {cls_name} @ [{dx}, {dy}, {dw}, {dh}]")
                    command_queue.put({
                        "cmd": "SYNC_TRACKER", 
                        "bbox": [dx, dy, dw, dh], 
                        "class_name": cls_name,
                        "source": "yolo"
                    })

        if not locked:
            target_to_lock = None
            def select_best_target(targets):
                if AUTO_TRACK_STRATEGY == "confidence": return max(targets, key=lambda d: d[4])
                elif AUTO_TRACK_STRATEGY == "center": return min(targets, key=lambda d: ((d[0] + d[2]/2) - 640)**2 + ((d[1] + d[3]/2) - 360)**2)
                else: return max(targets, key=lambda d: d[2] * d[3])
            
            if AUTO_RECOVERY and last_class is not None:
                valid_targets = [d for d in frame_detections if d[5] == last_class]
                if valid_targets:
                    target_to_lock = select_best_target(valid_targets)
            elif AUTO_TRACK:
                valid_targets = [d for d in frame_detections if d[5] in TARGET_CLASSES]
                if valid_targets:
                    target_to_lock = select_best_target(valid_targets)

            if target_to_lock:
                tx, ty, tw, th, _, cls_name = target_to_lock
                with coord_lock: current_coords['last_class'] = cls_name
                if not command_queue.full():
                    command_queue.put({"cmd": "ROI", "bbox": [tx, ty, tw, th], "tracker": current_tracker_type, "class_name": cls_name})

# ==========================================
# TRACKING THREAD 
# ==========================================
def tracking_thread():
    global current_tracker_type
    print("[TRACKER] Thread started with Kalman Filter.")
    tracker = None
    target_locked = False
    kalman = KalmanBBoxTracker()
    lost_frames_count = 0
    tracker_type = "CSRT"
    
    frame_count = 0
    start_time = time.time()
    tracker_fps = 0
    
    def init_tracker(frame, bbox, type_):
        nonlocal tracker
        x, y, w, h = bbox
        x = max(0, min(x, 1279))
        y = max(0, min(y, 719))
        w = max(10, min(w, 1280 - x))
        h = max(10, min(h, 720 - y))
        safe_bbox = (x, y, w, h)

        if type_ == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, safe_bbox)
        elif type_ == "ViT":
            try:
                params = cv2.TrackerVit_Params()
                params.net = "object_tracking_vittrack_2023sep.onnx"
                tracker = cv2.TrackerVit_create(params)
                tracker.init(frame, safe_bbox)
            except:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, safe_bbox)
                
    while True:
        try:
            cmd = command_queue.get_nowait()
            if cmd['cmd'] == 'ROI':
                bbox = tuple(cmd['bbox'])
                try:
                    init_frame = rgb_queue.get(timeout=0.1)
                    tracker_type = cmd['tracker']
                    current_tracker_type = tracker_type
                    init_tracker(init_frame, bbox, tracker_type)
                    
                    kalman.init_state(bbox)
                    lost_frames_count = 0
                    target_locked = True
                    
                    with coord_lock: 
                        current_coords['last_class'] = cmd.get('class_name', None)
                        current_coords['kalman_predicting'] = False
                except queue.Empty:
                    print("[TRACKER WARNING] Skipped ROI init: rgb_queue was empty!")
                    
            elif cmd['cmd'] == 'SYNC_TRACKER':
                bbox = tuple(cmd['bbox'])
                try:
                    init_frame = rgb_queue.get(timeout=0.1)
                    init_tracker(init_frame, bbox, current_tracker_type)
                    kalman.correct(bbox, source="yolo")
                    lost_frames_count = 0
                except queue.Empty:
                    pass
                
            elif cmd['cmd'] == 'RESET' or cmd['cmd'] == 'STOP':
                target_locked = False; tracker = None
                with coord_lock: 
                    current_coords['locked'] = False
                    current_coords['bbox'] = None
                    current_coords['last_class'] = None
                    current_coords['kalman_predicting'] = False
            elif cmd['cmd'] == 'SET_TRACKER': 
                current_tracker_type = cmd['tracker']
        except queue.Empty: pass

        try:
            frame = rgb_queue.get(timeout=0.05)
        except queue.Empty: continue

        if target_locked and tracker is not None:
            frame_count += 1
            if time.time() - start_time >= 1.0:
                tracker_fps = frame_count / (time.time() - start_time)
                frame_count = 0; start_time = time.time()
                
            success, bbox = tracker.update(frame)
            
            # Extract Kalman velocities directly from the physics model
            v_cx = kalman.kf.statePost[4][0] if ENABLE_KALMAN else 0.0
            v_cy = kalman.kf.statePost[5][0] if ENABLE_KALMAN else 0.0
            
            with coord_lock:
                if success:
                    if ENABLE_KALMAN:
                        kalman.correct(bbox, source="tracker")
                        smoothed_bbox = kalman.predict() 
                    else:
                        smoothed_bbox = bbox
                    lost_frames_count = 0
                    
                    current_coords['cx'] = int(smoothed_bbox[0] + smoothed_bbox[2] / 2)
                    current_coords['cy'] = int(smoothed_bbox[1] + smoothed_bbox[3] / 2)
                    current_coords['locked'] = True
                    current_coords['bbox'] = smoothed_bbox
                    current_coords['kalman_predicting'] = False
                    current_coords['v_cx'] = v_cx
                    current_coords['v_cy'] = v_cy
                else:
                    lost_frames_count += 1
                    
                    if ENABLE_KALMAN and lost_frames_count < MAX_LOST_FRAMES:
                        predicted_bbox = kalman.predict()
                        current_coords['cx'] = int(predicted_bbox[0] + predicted_bbox[2] / 2)
                        current_coords['cy'] = int(predicted_bbox[1] + predicted_bbox[3] / 2)
                        current_coords['locked'] = True
                        current_coords['bbox'] = predicted_bbox
                        current_coords['kalman_predicting'] = True
                        current_coords['v_cx'] = v_cx
                        current_coords['v_cy'] = v_cy
                    else:
                        current_coords['locked'] = False
                        current_coords['bbox'] = None
                        current_coords['kalman_predicting'] = False
                        current_coords['v_cx'] = 0.0
                        current_coords['v_cy'] = 0.0
                        target_locked = False
                        
                current_coords['tracker_fps'] = round(tracker_fps)

# ==========================================
# FILTERING & PID CLASSES
# ==========================================
class SimpleMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data_buffer = []
    def update(self, new_value):
        self.data_buffer.append(new_value)
        if len(self.data_buffer) > self.window_size: self.data_buffer.pop(0)
        return sum(self.data_buffer) / len(self.data_buffer)

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(-20.0, 20.0), integral_limits=(-5.0, 5.0)):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.reset()
    def reset(self):
        self._integral = 0.0; self._prev_error = None; self._last_output = 0.0
    def __call__(self, measurement, dt=0.5):
        error = measurement - self.setpoint
        p_term = self.kp * error
        
        self._integral += error * dt
        if self.integral_limits[0] is not None:
            self._integral = max(self._integral, self.integral_limits[0])
        if self.integral_limits[1] is not None:
            self._integral = min(self._integral, self.integral_limits[1])
            
        i_term = self.ki * self._integral
        d_term = 0.0
        if self._prev_error is not None: d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error
        
        output = p_term + i_term + d_term
        if self.output_limits[0] is not None: output = max(output, self.output_limits[0])
        if self.output_limits[1] is not None: output = min(output, self.output_limits[1])
        return output

# ==========================================
# DYNAMIC PID CONTROL THREAD
# ==========================================
def pid_control_thread():
    with control_params_lock: current_sma_window = int(control_params["sma_window"])
    sma_x = SimpleMovingAverage(window_size=current_sma_window)
    sma_y = SimpleMovingAverage(window_size=current_sma_window)
    sma_yaw = SimpleMovingAverage(window_size=current_sma_window)
    
    pid_x = PIDController(kp=0, ki=0, kd=0, setpoint=0.0, output_limits=(-20.0, 20.0))
    pid_y = PIDController(kp=0, ki=0, kd=0, setpoint=0.0, output_limits=(-20.0, 20.0))

    prev_vx, prev_vy = 0.0, 0.0
    last_time = time.time()

    while True:
        current_time = time.time()
        actual_dt = max(current_time - last_time, 0.001)
        last_time = current_time

        with control_params_lock: p = control_params.copy()

        pid_x.kp, pid_x.ki, pid_x.kd = p["kp_x"], p["ki_x"], p["kd_x"]
        pid_y.kp, pid_y.ki, pid_y.kd = p["kp_y"], p["ki_y"], p["kd_y"]
        
        if int(p["sma_window"]) != current_sma_window:
            current_sma_window = int(p["sma_window"])
            sma_x = SimpleMovingAverage(window_size=max(1, current_sma_window))
            sma_y = SimpleMovingAverage(window_size=max(1, current_sma_window))

        max_delta_x = p["forward_acceleration"] * actual_dt
        max_delta_y = p["max_accel_y"] * actual_dt

        with coord_lock:
            locked, cx, cy = current_coords['locked'], current_coords['cx'], current_coords['cy']
            v_cx, v_cy = current_coords.get('v_cx', 0.0), current_coords.get('v_cy', 0.0)

        if locked:
            norm_err_x = (cx - 1280/2) / (0.5 * 1280)
            norm_err_y = (720/2 - cy) / (0.5 * 720)
            
            target_vy = pid_x(sma_x.update(norm_err_x), actual_dt)
            target_vx = pid_y(sma_y.update(norm_err_y), actual_dt)

            vy = prev_vy + max(-max_delta_y, min(max_delta_y, target_vy - prev_vy))
            vx = prev_vx + max(-max_delta_x, min(max_delta_x, target_vx - prev_vx))
            
            max_spd = p["max_forward_speed"]
            vy = max(-max_spd, min(max_spd, vy))
            vx = max(-max_spd, min(max_spd, vx))

            prev_vy, prev_vx = vy, vx
            
            target_speed_px = math.sqrt(v_cx**2 + v_cy**2)
            if target_speed_px > 2.0: 
                yaw_error_deg = math.degrees(math.atan2(v_cx, -v_cy))
                target_yaw_rate = 0.5 * sma_yaw.update(yaw_error_deg)
                yaw_rate = max(-45.0, min(45.0, target_yaw_rate)) 
            else:
                yaw_rate = 0.0 
                
        else:
            vx = vy = prev_vx = prev_vy = yaw_rate = 0.0
            pid_x.reset(); pid_y.reset()
            sma_x = SimpleMovingAverage(window_size=max(1, current_sma_window))
            sma_y = SimpleMovingAverage(window_size=max(1, current_sma_window))
            sma_yaw = SimpleMovingAverage(window_size=max(1, current_sma_window))

        try: 
            velocity_queue.put_nowait({"vx": vx, "vy": vy, "vz": 0.0, "yaw": yaw_rate})
        except queue.Full:
            velocity_queue.get_nowait()
            velocity_queue.put_nowait({"vx": vx, "vy": vy, "vz": 0.0, "yaw": yaw_rate})

        time.sleep(max(0, (1.0 / 50.0) - (time.time() - current_time)))

# ==========================================
# COMMUNICATION & MAVSDK THREADS 
# ==========================================
def telemetry_sender():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        with coord_lock: coords = current_coords.copy()
        data = {
            "cpu": psutil.cpu_percent(), 
            "locked": coords.get("locked", False),
            "bbox": coords.get("bbox", None),
            "tracker_fps": coords.get("tracker_fps", 0),
            "detections": coords.get("detections", []),
            "kalman_predicting": coords.get("kalman_predicting", False),
            "sync_count": coords.get("sync_count", 0)
        }
        sock.sendto(json.dumps(data).encode('utf-8'), (STATION_IP, TELEM_PORT))
        time.sleep(0.1)

def station_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", CMD_PORT))
    while True:
        data, addr = sock.recvfrom(1024)
        try:
            cmd = json.loads(data.decode('utf-8'))
            if cmd.get("cmd") == "UPDATE_PARAMS":
                with control_params_lock:
                    params = cmd.get("params", {})
                    for k, v in params.items():
                        if k in control_params: control_params[k] = float(v)
            elif cmd.get("cmd") == "UPDATE_AI":
                global AI_CONFIDENCE, ACTIVE_MODEL, TARGET_CLASSES, AUTO_RECOVERY, AUTO_TRACK, AUTO_TRACK_STRATEGY, ENABLE_KALMAN, ENABLE_SYNC
                AI_CONFIDENCE = float(cmd.get("conf", 0.25))
                ACTIVE_MODEL = cmd.get("model", "yolov8n.rknn")
                TARGET_CLASSES = cmd.get("classes", [])
                AUTO_TRACK = cmd.get("auto_track", False)
                AUTO_RECOVERY = cmd.get("auto_recovery", False)
                AUTO_TRACK_STRATEGY = cmd.get("strategy", "largest")
                ENABLE_KALMAN = cmd.get("kalman", True)
                ENABLE_SYNC = cmd.get("sync", True)
            else:
                if not command_queue.full(): command_queue.put(cmd)
        except: pass


def mavsdk_offboard_thread():
    async def run():
        drone = System()
        await drone.connect(system_address=f"serial://{MAVLINK_PORT}:{MAVLINK_BAUD}")
        print("[MAVSDK] Waiting for drone to connect...")
        
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("[MAVSDK] Gazebo SITL discovered!")
                break
                
        print("[MAVSDK] Arming and taking off to demo altitude...")
        try:
            await drone.action.arm()
            await drone.action.takeoff()
            await asyncio.sleep(5) 
        except Exception as e:
            print(f"[MAVSDK] Warning during Arm/Takeoff: {e}")

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        print("[MAVSDK] Starting offboard mode")
        try:
            await drone.offboard.start()
        except OffboardError as e:
            print(f"[MAVSDK] Offboard start failed: {e._result.result}")
            return
            
        await asyncio.sleep(0.04)
        print("[MAVSDK] Sending 2D Translation commands to Gazebo – live status:")
        
        last_velocity_time = time.time()
        loop = asyncio.get_running_loop()
        vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0}
        
        while True:
            try:
                vel = await loop.run_in_executor(None, velocity_queue.get, True, 0.05)
                last_velocity_time = time.time()
            except queue.Empty:
                if time.time() - last_velocity_time > 2.0:
                    vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0}
                    print("\n[WARNING] Velocity timeout! Failsafe Hover Engaged.", end='')
                else:
                    pass 
            
            vel["yaw"] = 0 
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vel["vx"], vel["vy"], vel["vz"], vel["yaw"])
            )
            
            print(f"\rVX: {vel['vx']:+6.2f}  VY: {vel['vy']:+6.2f}  YAW: {vel['yaw']:+6.2f} deg/s",
                  end='', flush=True)
            
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        

# ---------- Persistent sender ----------
def velocity_sender_thread():
    sock = None
    while True:
        try:
            # Get latest velocity command (non-blocking)
            try:
                vel = velocity_queue.get_nowait()
            except queue.Empty:
                vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0}

            # Ensure we have a working socket
            if sock is None:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)  # timeout for connect
                sock.connect((STATION_IP, Gazebo_sim_port))
                sock.settimeout(None)  # disable timeout for send

            # Pack and send
            payload = struct.pack('!3f', vel['vx'], vel['vy'], vel['vz'])
            sock.sendall(payload)

            print(f"\rVX: {vel['vx']:+6.2f}  VY: {vel['vy']:+6.2f}  VZ: {vel['vz']:+6.2f} m/s",
                  end='', flush=True)

        except (ConnectionRefusedError, ConnectionResetError,
                ConnectionAbortedError, BrokenPipeError, OSError) as e:
            print(f"\nConnection lost: {e}. Reconnecting...")
            if sock:
                sock.close()
                sock = None
            time.sleep(1.0)      # wait before reconnecting
        except Exception as e:
            print(f"Unexpected error: {e}")
            if sock:
                sock.close()
                sock = None
            time.sleep(1.0)

        time.sleep(0.02)
        


if __name__ == "__main__":
    RGBCamera().start()
    threading.Thread(target=detection_thread, daemon=True).start()
    threading.Thread(target=tracking_thread, daemon=True).start()
    threading.Thread(target=pid_control_thread, daemon=True).start()
    
    if Gazebo_sim:
        threading.Thread(target=velocity_sender_thread, daemon=True).start()
    else:
        threading.Thread(target=mavsdk_offboard_thread, daemon=True).start()
    
    threading.Thread(target=station_listener, daemon=True).start()
    threading.Thread(target=telemetry_sender, daemon=True).start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        
        
        
