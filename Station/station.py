#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import socket
import json
import threading
import subprocess
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import tkinter as tk
from tkinter import ttk

# --- CONFIGURATION ---
DRONE_IP = "192.168.1.2"
CMD_PORT = 6000
TELEM_PORT = 6001

# Global State
active_tracker = "CSRT"
telemetry_data = {}
drawing = False
roi_pts = []

# ==========================================
# MULTIPROCESS VIDEO STREAM RECEIVER (OPENCV UDP)
# ==========================================
def video_stream_process(port, width, height, name, ready_event, shm_name_out, stop_event, fps_value, bitrate_value, shm_lock):
    frame_size = width * height * 3
    
    # Initialize Shared Memory
    shm = shared_memory.SharedMemory(create=True, size=frame_size)
    shm_name_out.put(shm.name)
    frame_buffer = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
    frame_buffer.fill(0)
    
    # Create UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.1) # Short timeout allows checking the stop_event frequently
    
    ready_event.set()
    
    frame_count, byte_count = 0, 0
    metrics_start = time.time()
    
    # Buffer to reconstruct split JPEG UDP chunks
    buffer = {}
    
    while not stop_event.is_set():
        try:
            packet, _ = sock.recvfrom(65536)
            
            # Make sure it has our 3-byte header
            if len(packet) < 4: continue
            
            byte_count += len(packet)
            
            frame_id = packet[0]
            total_chunks = packet[1]
            chunk_idx = packet[2]
            chunk_data = packet[3:]
            
            if frame_id not in buffer:
                buffer[frame_id] = {}
                
            buffer[frame_id][chunk_idx] = chunk_data
            
            # If all chunks are successfully received
            if len(buffer[frame_id]) == total_chunks:
                # Reassemble the pieces
                frame_bytes = b''.join([buffer[frame_id][i] for i in range(total_chunks)])
                
                # Decode the raw JPEG back to a frame matrix
                img_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Sanity check frame size
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                        
                    # Safely push it to UI Shared Memory
                    with shm_lock:
                        np.copyto(frame_buffer, frame)
                        
                    frame_count += 1
                
                # We finished this frame, remove it to save memory
                del buffer[frame_id]
                
            # Periodically wipe extremely old, incomplete frames (dropped packets)
            if len(buffer) > 5:
                oldest_keys = list(buffer.keys())[:-3]
                for k in oldest_keys:
                    del buffer[k]
                    
        except socket.timeout:
            pass
        except Exception:
            pass
            
        # FPS and Bitrate tracking
        now = time.time()
        if now - metrics_start >= 1.0:
            fps_value.value = frame_count / (now - metrics_start)
            bitrate_value.value = (byte_count * 8) / (now - metrics_start) / 1e6
            frame_count = 0
            byte_count = 0
            metrics_start = now
            
    shm.close()

# ==========================================
# COMMUNICATION & CALLBACKS
# ==========================================
def telemetry_listener():
    global telemetry_data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", TELEM_PORT))
    while True:
        try:
            data, _ = sock.recvfrom(8192) 
            telemetry_data = json.loads(data.decode('utf-8'))
        except Exception:
            pass

def send_command(cmd_dict):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(cmd_dict).encode('utf-8'), (DRONE_IP, CMD_PORT))
    except Exception:
        pass

def mouse_callback(event, x, y, flags, param):
    global drawing, roi_pts, active_tracker
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if telemetry_data and telemetry_data.get("detections") and not telemetry_data.get("locked"):
            for det in telemetry_data["detections"]:
                dx, dy, dw, dh, conf, cls_name = det
                if dx <= x <= dx + dw and dy <= y <= dy + dh:
                    send_command({"cmd": "ROI", "bbox": [dx, dy, dw, dh], "tracker": active_tracker, "class_name": cls_name})
                    return

        drawing = True
        roi_pts = [(x, y)]
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_pts.append((x, y))
        
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        roi_pts.append((x, y))
        x_min, y_min = min(roi_pts[0][0], roi_pts[-1][0]), min(roi_pts[0][1], roi_pts[-1][1])
        w, h = abs(roi_pts[-1][0] - roi_pts[0][0]), abs(roi_pts[-1][1] - roi_pts[0][1])
        if w > 10 and h > 10:
            send_command({"cmd": "ROI", "bbox": [x_min, y_min, w, h], "tracker": active_tracker})
        roi_pts = [] 

# ==========================================
# TKINTER DASHBOARD
# ==========================================
def create_dashboard(stop_event, rgb_fps, rgb_bitrate, shm_lock, rgb_frame):
    root = tk.Tk()
    root.title("Quadcopter Advanced Station Dashboard")
    root.geometry("600x750")

    cv2.namedWindow("Drone Station")
    cv2.setMouseCallback("Drone Station", mouse_callback)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    tab_dash = ttk.Frame(notebook)
    tab_ai = ttk.Frame(notebook)
    tab_pid = ttk.Frame(notebook)

    notebook.add(tab_dash, text="Dashboard & Telemetry")
    notebook.add(tab_ai, text="AI & Auto-Tracking")
    notebook.add(tab_pid, text="Flight Tuning")

    # ==========================================
    # TAB 1: DASHBOARD
    # ==========================================
    telemetry_frame = ttk.LabelFrame(tab_dash, text="Real-Time Telemetry", padding=10)
    telemetry_frame.pack(fill=tk.X, pady=10, padx=10)
    
    lbl_trk = ttk.Label(telemetry_frame, text="Tracker: -")
    lbl_lock = ttk.Label(telemetry_frame, text="Locked: NO")
    lbl_fps = ttk.Label(telemetry_frame, text="FPS: -")
    lbl_bit = ttk.Label(telemetry_frame, text="Bitrate: -")
    lbl_cpu = ttk.Label(telemetry_frame, text="Drone CPU: -")
    
    lbl_trk.grid(row=0, column=0, padx=10, sticky=tk.W); lbl_lock.grid(row=0, column=1, padx=10, sticky=tk.W)
    lbl_fps.grid(row=1, column=0, padx=10, sticky=tk.W); lbl_bit.grid(row=1, column=1, padx=10, sticky=tk.W)
    lbl_cpu.grid(row=2, column=0, padx=10, sticky=tk.W)

    def set_trk(trk):
        global active_tracker; active_tracker = trk
        send_command({"cmd": "SET_TRACKER", "tracker": trk})

    btn_frame = ttk.Frame(tab_dash)
    btn_frame.pack(fill=tk.X, pady=10, padx=10)
    ttk.Button(btn_frame, text="Use CSRT", command=lambda: set_trk("CSRT")).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Use ViT", command=lambda: set_trk("ViT")).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Reset Target", command=lambda: send_command({"cmd": "RESET"})).pack(side=tk.RIGHT, padx=5)

    stop_btn = tk.Button(tab_dash, text="EMERGENCY STOP (KILL TRACKING)", bg="red", fg="white", font=("Arial", 12, "bold"), command=lambda: send_command({"cmd": "STOP"}))
    stop_btn.pack(fill=tk.X, pady=20, padx=10)

    # ==========================================
    # TAB 2: AI & VISION SETTINGS
    # ==========================================
    ai_frame = ttk.LabelFrame(tab_ai, text="YOLOv8 Edge NPU Configuration", padding=10)
    ai_frame.pack(fill=tk.X, pady=10, padx=10)

    ttk.Label(ai_frame, text="Model Version:").grid(row=0, column=0, sticky=tk.W, pady=5)
    model_var = tk.StringVar(value="yolov8n.rknn")
    model_combo = ttk.Combobox(ai_frame, textvariable=model_var, values=["yolov8n.rknn", "yolov8s.rknn"], state="readonly")
    model_combo.grid(row=0, column=1, padx=10, pady=5)

    ttk.Label(ai_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, pady=5)
    conf_var = tk.DoubleVar(value=0.25)
    conf_slider = ttk.Scale(ai_frame, from_=0.1, to=0.9, variable=conf_var, orient=tk.HORIZONTAL)
    conf_slider.grid(row=1, column=1, padx=10, pady=5)
    conf_lbl = ttk.Label(ai_frame, text="0.25")
    conf_lbl.grid(row=1, column=2, sticky=tk.W)
    conf_var.trace_add("write", lambda *args: conf_lbl.config(text=f"{conf_var.get():.2f}"))

    tgt_frame = ttk.LabelFrame(tab_ai, text="Allowed Target Types", padding=10)
    tgt_frame.pack(fill=tk.X, pady=10, padx=10)
    
    var_person = tk.BooleanVar(value=True); ttk.Checkbutton(tgt_frame, text="Person", variable=var_person, command=lambda: push_ai_config()).grid(row=0, column=0, padx=10)
    var_car = tk.BooleanVar(value=True); ttk.Checkbutton(tgt_frame, text="Car", variable=var_car, command=lambda: push_ai_config()).grid(row=0, column=1, padx=10)
    var_moto = tk.BooleanVar(value=True); ttk.Checkbutton(tgt_frame, text="Motorbike", variable=var_moto, command=lambda: push_ai_config()).grid(row=0, column=2, padx=10)
    var_truck = tk.BooleanVar(value=True); ttk.Checkbutton(tgt_frame, text="Truck", variable=var_truck, command=lambda: push_ai_config()).grid(row=0, column=3, padx=10)

    auto_frame = ttk.LabelFrame(tab_ai, text="Autonomous Tracking Modes", padding=10)
    auto_frame.pack(fill=tk.X, pady=10, padx=10)

    var_autotrack = tk.BooleanVar(value=False)
    ttk.Checkbutton(auto_frame, text="Enable Auto-Track (Lock onto first seen allowed target)", variable=var_autotrack, command=lambda: push_ai_config()).pack(anchor=tk.W, pady=5)
    
    var_autorecover = tk.BooleanVar(value=False)
    ttk.Checkbutton(auto_frame, text="Enable Auto-Recovery (If tracker lost, re-lock onto same type)", variable=var_autorecover, command=lambda: push_ai_config()).pack(anchor=tk.W, pady=5)

    var_kalman = tk.BooleanVar(value=True)
    ttk.Checkbutton(auto_frame, text="Enable Kalman Filter Tracking & Path Coasting", variable=var_kalman, command=lambda: push_ai_config()).pack(anchor=tk.W, pady=5)
    
    var_sync = tk.BooleanVar(value=True)
    ttk.Checkbutton(auto_frame, text="Enable YOLO Scale-Drift Sync (Fix tracking explosions)", variable=var_sync, command=lambda: push_ai_config()).pack(anchor=tk.W, pady=5)

    ttk.Label(auto_frame, text="Target Selection Mechanism (If multiple targets found):").pack(anchor=tk.W, pady=(10, 2))
    strategy_var = tk.StringVar(value="Largest Size")
    strategy_combo = ttk.Combobox(auto_frame, textvariable=strategy_var, values=["Largest Size", "Highest Confidence", "Nearest to Center"], state="readonly")
    strategy_combo.pack(anchor=tk.W, padx=5, pady=(0, 5))

    def push_ai_config(*args):
        classes = []
        if var_person.get(): classes.append("person")
        if var_car.get(): classes.append("car")
        if var_moto.get(): classes.append("motorbike ")
        if var_truck.get(): classes.append("truck ")
        
        strategy_map = {
            "Largest Size": "largest",
            "Highest Confidence": "confidence",
            "Nearest to Center": "center"
        }
        
        cmd = {
            "cmd": "UPDATE_AI",
            "model": model_var.get(),
            "conf": conf_var.get(),
            "classes": classes,
            "auto_track": var_autotrack.get(),
            "auto_recovery": var_autorecover.get(),
            "strategy": strategy_map.get(strategy_var.get(), "largest"),
            "kalman": var_kalman.get(),
            "sync": var_sync.get()
        }
        send_command(cmd)

    model_combo.bind("<<ComboboxSelected>>", push_ai_config)
    strategy_combo.bind("<<ComboboxSelected>>", push_ai_config)
    conf_slider.bind("<ButtonRelease-1>", push_ai_config)

    # ==========================================
    # TAB 3: PID TUNING
    # ==========================================
    DEFAULT_PARAMS = {
        "kp_x": 5.5, "ki_x": 0.0, "kd_x": 4.2,
        "kp_y": 4.5, "ki_y": 0.0, "kd_y": 3.2,
        "max_forward_speed": 10.0, "forward_acceleration": 2.0,
        "max_accel_y": 3.0, "max_accel_z": 3.0, "sma_window": 20
    }

    kp_x_var, ki_x_var, kd_x_var = tk.DoubleVar(value=DEFAULT_PARAMS["kp_x"]), tk.DoubleVar(value=DEFAULT_PARAMS["ki_x"]), tk.DoubleVar(value=DEFAULT_PARAMS["kd_x"])
    kp_y_var, ki_y_var, kd_y_var = tk.DoubleVar(value=DEFAULT_PARAMS["kp_y"]), tk.DoubleVar(value=DEFAULT_PARAMS["ki_y"]), tk.DoubleVar(value=DEFAULT_PARAMS["kd_y"])
    max_fwd_var = tk.DoubleVar(value=DEFAULT_PARAMS["max_forward_speed"]); fwd_acc_var = tk.DoubleVar(value=DEFAULT_PARAMS["forward_acceleration"])
    max_acc_y_var = tk.DoubleVar(value=DEFAULT_PARAMS["max_accel_y"]); max_acc_z_var = tk.DoubleVar(value=DEFAULT_PARAMS["max_accel_z"]); sma_var = tk.DoubleVar(value=DEFAULT_PARAMS["sma_window"])

    def push_parameters(*args):
        params = {
            "kp_x": kp_x_var.get(), "ki_x": ki_x_var.get(), "kd_x": kd_x_var.get(),
            "kp_y": kp_y_var.get(), "ki_y": ki_y_var.get(), "kd_y": kd_y_var.get(),
            "max_forward_speed": max_fwd_var.get(), "forward_acceleration": fwd_acc_var.get(),
            "max_accel_y": max_acc_y_var.get(), "max_accel_z": max_acc_z_var.get(),
            "sma_window": int(sma_var.get())
        }
        send_command({"cmd": "UPDATE_PARAMS", "params": params})

    def create_slider(parent, row, label, var, from_, to_):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        slider = ttk.Scale(parent, from_=from_, to=to_, variable=var, orient=tk.HORIZONTAL, length=150, command=push_parameters)
        slider.grid(row=row, column=1, padx=10, pady=2)
        val_label = ttk.Label(parent, text="0.0")
        val_label.grid(row=row, column=2, sticky=tk.W)
        var.trace_add("write", lambda *args, v=var, l=val_label: l.config(text=f"{v.get():.2f}"))
        var.set(var.get())

    lf_x = ttk.LabelFrame(tab_pid, text="X-Axis (Lateral) PID", padding=10)
    lf_x.pack(fill=tk.X, pady=5, padx=10)
    create_slider(lf_x, 0, "Kp:", kp_x_var, 0, 15); create_slider(lf_x, 1, "Ki:", ki_x_var, 0, 5); create_slider(lf_x, 2, "Kd:", kd_x_var, 0, 15)

    lf_y = ttk.LabelFrame(tab_pid, text="Y-Axis (Vertical) PID", padding=10)
    lf_y.pack(fill=tk.X, pady=5, padx=10)
    create_slider(lf_y, 0, "Kp:", kp_y_var, 0, 15); create_slider(lf_y, 1, "Ki:", ki_y_var, 0, 5); create_slider(lf_y, 2, "Kd:", kd_y_var, 0, 15)

    lf_v = ttk.LabelFrame(tab_pid, text="Kinematic Limits", padding=10)
    lf_v.pack(fill=tk.X, pady=5, padx=10)
    create_slider(lf_v, 0, "Max Fwd Speed (m/s):", max_fwd_var, 1, 20); create_slider(lf_v, 1, "Fwd Accel (m/s²):", fwd_acc_var, 0.1, 10)
    create_slider(lf_v, 2, "Max Accel Y (m/s²):", max_acc_y_var, 0.1, 10); create_slider(lf_v, 3, "Max Accel Z (m/s²):", max_acc_z_var, 0.1, 10)
    create_slider(lf_v, 4, "SMA Window (frames):", sma_var, 1, 60)

    root.after(1000, push_ai_config)

    def update_gui():
        if stop_event.is_set():
            cv2.destroyAllWindows()
            root.destroy()
            return
            
        if not hasattr(update_gui, 'last_sync_count'):
            update_gui.last_sync_count = 0
            update_gui.sync_display_timer = 0

        stream_fps, bitrate = rgb_fps.value, rgb_bitrate.value
        locked = telemetry_data.get('locked', False) if telemetry_data else False
        trk_fps = telemetry_data.get('tracker_fps', 0) if telemetry_data else 0
        sync_count = telemetry_data.get('sync_count', 0) if telemetry_data else 0
        is_kalman = telemetry_data.get('kalman_predicting', False) if telemetry_data else False
        
        if sync_count > update_gui.last_sync_count:
            update_gui.sync_display_timer = 15 
            update_gui.last_sync_count = sync_count

        lbl_trk.config(text=f"Tracker: {active_tracker}")
        lbl_fps.config(text=f"Video FPS: {stream_fps:.1f} | Track FPS: {trk_fps}")
        lbl_bit.config(text=f"Bitrate: {bitrate:.2f} Mbps")
        lbl_cpu.config(text=f"Drone CPU: {telemetry_data.get('cpu', 0) if telemetry_data else 0}%")
        
        if locked and is_kalman:
            lbl_lock.config(text="Locked: KALMAN COASTING", foreground="orange")
        elif locked:
            lbl_lock.config(text="Locked: YES", foreground="red")
        else:
            lbl_lock.config(text="Locked: NO", foreground="black")

        with shm_lock:
            display_frame = rgb_frame.copy()

        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
        cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        
        # 1. Draw YOLO AI Detections
        if telemetry_data and telemetry_data.get("detections"):
            for det in telemetry_data["detections"]:
                dx, dy, dw, dh, conf, cls_name = det
                cv2.rectangle(display_frame, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 1)
                cv2.putText(display_frame, f"{cls_name} {conf:.2f}", (dx, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Draw Tracker Box & Kalman Physics Status
        if locked and telemetry_data.get("bbox"):
            bbox = telemetry_data.get("bbox")
            x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            if is_kalman:
                color = (0, 165, 255) # Orange
                status_text = "KALMAN COASTING"
            else:
                color = (0, 0, 255) # Red
                status_text = "TRACKING"
                
            cv2.rectangle(display_frame, (x, y), (x+bw, y+bh), color, 3)
            cv2.drawMarker(display_frame, (x + bw//2, y + bh//2), color, cv2.MARKER_CROSS, 15, 2)
            cv2.putText(display_frame, status_text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. Draw SYNC_TRACKER Event
        if update_gui.sync_display_timer > 0:
            cv2.putText(display_frame, "YOLO SYNC TRIGGERED (Scale Corrected)", (w//2 - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
            update_gui.sync_display_timer -= 1

        # 4. Draw Manual Selection
        if drawing and len(roi_pts) > 0:
            cv2.rectangle(display_frame, roi_pts[0], roi_pts[-1], (255, 255, 0), 2)

        cv2.imshow("Drone Station", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): stop_event.set()

        root.after(33, update_gui)

    root.protocol("WM_DELETE_WINDOW", lambda: stop_event.set())
    update_gui()
    root.mainloop()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn')
    stop_event = mp.Event()

    rgb_fps = mp.Value('d', 0.0)
    rgb_bitrate = mp.Value('d', 0.0)

    rgb_ready = mp.Event()
    rgb_shm_name = mp.Queue(maxsize=1)
    
    shm_lock = mp.Lock()

    rgb_proc = mp.Process(target=video_stream_process, args=(5000, 1280, 720, "RGB", rgb_ready, rgb_shm_name, stop_event, rgb_fps, rgb_bitrate, shm_lock))
    rgb_proc.start()

    rgb_ready.wait()
    rgb_shm = shared_memory.SharedMemory(name=rgb_shm_name.get())
    rgb_frame = np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=rgb_shm.buf)

    threading.Thread(target=telemetry_listener, daemon=True).start()

    create_dashboard(stop_event, rgb_fps, rgb_bitrate, shm_lock, rgb_frame)

    stop_event.set()
    rgb_proc.join(timeout=2)
    rgb_shm.close(); rgb_shm.unlink()