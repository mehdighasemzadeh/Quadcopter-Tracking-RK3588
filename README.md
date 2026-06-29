🚁 Quadcopter Target Tracking with RK3588

An ultra-low latency, high-performance edge AI tracking system for Quadcopters. Designed to run entirely on the Rockchip RK3588 NPU (companion computer), this project provides autonomous object tracking (vehicles, people) using YOLOv8, classical computer vision trackers, a Kalman Filter for path coasting, and a dynamic PID controller linked directly to PX4 via MAVSDK.

Demo

System Architecture

This project splits the workload between an edge companion computer (RK3588) handling the heavy AI and flight dynamics, and a ground control station providing a real-time operator interface.

The hardware architecture consists of two main nodes. The Edge Computer (RK3588) is mounted on the drone, capturing frames directly from the camera, running the NPU AI inference and classical tracking, and calculating flight velocities. It sends telemetry and compressed video to the Ground Station, which reassembles the real-time feed and provides an operator interface.

Hybrid Tracking Logic

To achieve robustness against occlusions and AI scaling issues, the software uses a multi-threaded hybrid approach.

Features

High-Speed AI Inference: Runs YOLOv8 Nano/Small at 45 FPS / 25 FPS utilizing the RK3588's NPU.

Hybrid Tracking Pipeline: Combines AI detections with optical tracking (CSRT @ 30 FPS / ViT @ 45 FPS).

Kalman Filter Coasting: Predicts target trajectory and coasts the drone if the target is temporarily lost.

Auto-Recovery: AI thread constantly evaluates the IoU (Intersection over Union) to force resyncs if optical drift occurs.

Installation

Clone Repository

git clone https://github.com/mehdighasemzadeh/Quadcopter-Tracking-RK3588.git
cd Quadcopter-Tracking-RK3588


Setup Ground Station (PC)

# Install requirements
pip3 install opencv-python numpy tk mavsdk asyncio


Setup Drone (RK3588 Companion Computer)

You need a Debian/Ubuntu OS with Rockchip NPU drivers installed and Python 3.8.

# Install core Python requirements
pip3 install opencv-python opencv-contrib-python numpy psutil asyncio mavsdk

# Install Rockchip NPU Toolkit 1.6
pip3 install rknn_toolkit_lite2-1.6.0-cp38-cp38-linux_aarch64.whl


Requirements

Core Dependencies

Companion Computer: RK3588 based board (e.g., Orange Pi 5, Radxa Rock 5B, NanoPC-T6).

Flight Controller: Any PX4 or ArduPilot flight controller connected via UART/USB to the RK3588.

Ground Station: Any PC (Windows/Linux/Mac) to run the Tkinter dashboard and receive video.

Python 3.8 (Strictly required for RKNN Toolkit compatibility)

Running the System

Step 1: Maximize RK3588 Performance

Before running the drone script, overdrive the RK3588 to ensure maximum FPS:

cd RK3588
sudo chmod +x boost.sh
sudo ./boost.sh


Step 2: Configure IP Addresses

Edit RK3588/quadcopter.py and Station/station.py. Ensure STATION_IP and DRONE_IP match your network configuration.

Step 3: Run SITL (Simulation)

Open Gazebo with PX4 SITL on your Ground Station.

Start the Gazebo Bridge on PC: cd Station && python3 send_data_to_gazebo.py

Start the Dashboard on PC: python3 station.py

Start Drone script on RK3588 (Ensure Gazebo_sim = True): cd RK3588 && python3 quadcopter.py

Step 4: Run HITL / Real Life

Connect the RK3588 to your Flight Controller via UART/USB.

Update RK3588/quadcopter.py: set Gazebo_sim = False and update MAVLINK_PORT.

Start Dashboard on PC: python3 station.py

Start Drone script on RK3588: python3 quadcopter.py

Project Structure

Quadcopter-Tracking-RK3588/
│
├── images/                             
│   ├── architecture.png
│   ├── demo.gif
│   ├── demo.mp4
│   └── tracking_flow.png
│
├── RK3588/                             
│   ├── py_utils/                       
│   ├── boost.sh                        
│   ├── car1_demo.mp4                   
│   ├── object_tracking_vittrack...     
│   ├── quadcopter.py                   
│   ├── yolo.py                         
│   ├── yolov8n.rknn                    
│   └── yolov8s.rknn                    
│
├── Station/                            
│   ├── send_data_to_gazebo.py          
│   └── station.py                      
│
└── README.md


License

This project is licensed under the MIT License - see the LICENSE file for details.
