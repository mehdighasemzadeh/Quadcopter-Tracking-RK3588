<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quadcopter Target Tracking with RK3588</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Justify text for paragraphs and list items */
        p, .justify-text {
            text-align: justify;
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        li {
            text-align: justify;
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }
        /* Ensure code blocks and specific elements remain left-aligned */
        pre, code, .no-justify {
            text-align: left;
        }
        pre {
            background-color: #1e293b;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin-bottom: 1.5rem;
            font-family: monospace;
            font-size: 0.875rem;
        }
        code:not(pre code) {
            background-color: #f1f5f9;
            color: #ef4444;
            padding: 0.125rem 0.375rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
        }
        h1, h2, h3 {
            color: #0f172a;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        h1 { font-size: 2.25rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
        h2 { font-size: 1.5rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; }
        h3 { font-size: 1.25rem; }
        ul, ol {
            padding-left: 1.5rem;
            margin-bottom: 1.5rem;
        }
        ul { list-style-type: disc; }
        ol { list-style-type: decimal; }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .badge-container img {
            display: inline-block;
            margin-right: 0.25rem;
            margin-bottom: 0;
            border-radius: 0;
        }
    </style>
</head>
<body class="bg-slate-100 text-slate-700 font-sans antialiased py-8 px-4 sm:px-8">

    <main class="max-w-4xl mx-auto bg-white p-6 sm:p-10 rounded-2xl shadow-xl">
        
        <!-- Header Section -->
        <header class="mb-8">
            <h1 class="flex items-center gap-3">
                <span>🚁</span> Quadcopter Target Tracking with RK3588 (YOLOv8 + CSRT/ViT + Kalman)
            </h1>
            
            <div class="badge-container my-4">
                <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
                <img src="https://img.shields.io/badge/NPU-RK3588_6_TOPS-green" alt="RK3588">
                <img src="https://img.shields.io/badge/AI-YOLOv8-yellow" alt="YOLOv8">
                <img src="https://img.shields.io/badge/Control-MAVSDK-orange" alt="MAVSDK">
            </div>

            <p class="text-lg">
                An ultra-low latency, high-performance edge AI tracking system for Quadcopters. Designed to run entirely on the <strong>Rockchip RK3588</strong> NPU (companion computer), this project provides autonomous object tracking (vehicles, people) using YOLOv8, classical computer vision trackers (CSRT/ViT), a Kalman Filter for path coasting, and a dynamic PID controller linked directly to PX4 via MAVSDK.
            </p>
            <p class="text-lg">
                Includes full support for <strong>Gazebo SITL</strong> (Software In The Loop) simulation and <strong>HITL/Real-world</strong> flight. Developed by <a href="https://github.com/mehdighasemzadeh/" class="text-blue-600 hover:underline font-semibold" target="_blank">Mehdi Ghasemzadeh</a>.
            </p>
        </header>

        <!-- System Demo -->
        <section>
            <h2>🎥 System Demo</h2>
            <div class="my-4 border border-slate-200 rounded-lg p-1 bg-slate-50">
                <img src="images/demo.gif" alt="Gazebo SITL Demo" onerror="this.src='https://via.placeholder.com/800x450?text=demo.gif+not+found';">
            </div>
            <p class="text-sm italic text-slate-500 text-center no-justify">
                The system successfully identifying, locking onto, and dynamically tracking a vehicle using YOLOv8 bounding box initialization, Kalman filtering, and MAVSDK offboard velocity control.
            </p>
        </section>

        <!-- Key Features -->
        <section>
            <h2>✨ Key Features</h2>
            <ul>
                <li><strong>High-Speed AI Inference</strong>: Runs YOLOv8 Nano/Small at <strong>45 FPS / 25 FPS</strong> utilizing the RK3588's built-in NPU.</li>
                <li><strong>Hybrid Tracking Pipeline</strong>: Combines AI detections with optical tracking (CSRT @ 30 FPS / ViT @ 45 FPS).</li>
                <li><strong>Kalman Filter Coasting</strong>: If a target is temporarily lost (occlusion), the system predicts the trajectory and coasts the drone until the target is re-acquired.</li>
                <li><strong>Custom Ground Station</strong>: A multiprocessed Python Tkinter dashboard showing real-time UDP video (custom chunking for ultra-low latency), telemetry, and live PID tuning.</li>
            </ul>
        </section>

        <!-- System Architecture -->
        <section>
            <h2>🏗️ System Architecture</h2>
            <p>
                This project splits the workload between an edge companion computer (RK3588) handling the heavy AI and flight dynamics, and a ground control station providing a real-time operator interface.
            </p>

            <h3>Hardware & Network Architecture</h3>
            <div class="my-4 border border-slate-200 rounded-lg p-1 bg-slate-50">
                <img src="images/architecture.png" alt="System Architecture" onerror="this.src='https://via.placeholder.com/800x400?text=architecture.png+not+found';">
            </div>
            <p>The architecture consists of two main nodes:</p>
            <ol>
                <li><strong>The Edge Computer (RK3588)</strong>: Captures frames, runs NPU AI inference, classical tracking, and calculates the necessary flight velocity using PID controllers.</li>
                <li><strong>The Ground Station</strong>: Receives highly-optimized JPEG UDP chunks, reassembles them into a real-time feed, and displays system diagnostics while sending back operator commands.</li>
            </ol>

            <h3>Hybrid Tracking Logic</h3>
            <div class="my-4 border border-slate-200 rounded-lg p-1 bg-slate-50">
                <img src="images/tracking_flow.png" alt="Tracking Flow" onerror="this.src='https://via.placeholder.com/800x600?text=tracking_flow.png+not+found';">
            </div>
            <p>To achieve robustness, the software uses a multi-threaded approach:</p>
            <ul>
                <li><strong>AI Thread</strong>: Searches for targets and constantly evaluates the "IoU" (Intersection over Union) against the classical tracker. If scale drift is detected, it forces a resync.</li>
                <li><strong>Tracking Thread</strong>: Updates the CSRT/ViT optical tracker. If it fails due to an obstacle, it falls back to a <strong>Kalman Filter</strong> to predict the target's location for a set number of frames before dropping the lock.</li>
            </ul>
        </section>

        <!-- Hardware Requirements -->
        <section>
            <h2>🛠️ Hardware Requirements</h2>
            <ul>
                <li><strong>Companion Computer</strong>: RK3588 based board (e.g., Orange Pi 5, Radxa Rock 5B, NanoPC-T6).</li>
                <li><strong>Flight Controller</strong>: Any PX4 or ArduPilot flight controller connected via UART/USB to the RK3588.</li>
                <li><strong>Ground Station</strong>: Any PC (Windows/Linux/Mac) to run the Tkinter dashboard and receive video.</li>
            </ul>
        </section>

        <!-- Software Setup -->
        <section>
            <h2>📦 Software Setup</h2>
            
            <h3>1. On the Ground Station (PC)</h3>
<pre><code># Clone the repository
git clone https://github.com/mehdighasemzadeh/Quadcopter-Tracking-RK3588.git
cd Quadcopter-Tracking-RK3588

# Install requirements
pip3 install opencv-python numpy tk mavsdk asyncio</code></pre>

            <h3>2. On the Drone (RK3588 Companion Computer)</h3>
            <p>You need a Debian/Ubuntu OS with Rockchip NPU drivers installed and <strong>Python 3.8</strong>.</p>
<pre><code>cd Quadcopter-Tracking-RK3588

# Install Python requirements (ensure Python 3.8 is being used)
pip3 install opencv-python opencv-contrib-python numpy psutil asyncio mavsdk

# Install Rockchip NPU Toolkit 1.6 (RKNNtoolkit2-1.6)
pip3 install rknn_toolkit_lite2-1.6.0-cp38-cp38-linux_aarch64.whl</code></pre>
        </section>

        <!-- Usage Guide -->
        <section>
            <h2>🚀 Usage Guide</h2>
            
            <h3>Step 1: Maximize RK3588 Performance</h3>
            <p>Before running the drone script, overdrive the RK3588 to ensure maximum FPS:</p>
<pre><code>cd RK3588
sudo chmod +x boost.sh
sudo ./boost.sh</code></pre>

            <h3>Step 2: Configure IP Addresses</h3>
            <p>Edit <code>RK3588/quadcopter.py</code> and <code>Station/station.py</code>. Ensure <code>STATION_IP</code> and <code>DRONE_IP</code> match your network configuration.</p>

            <h3>Step 3: Running in SITL (Simulation)</h3>
            <ol>
                <li>Open Gazebo with PX4 SITL on your Ground Station.</li>
                <li>Start the Gazebo Bridge on your Ground Station:
<pre><code>cd Station
python3 send_data_to_gazebo.py</code></pre>
                </li>
                <li>Start the Dashboard on your Ground Station:
<pre><code>python3 station.py</code></pre>
                </li>
                <li>Start the Drone script on the RK3588 (Ensure <code>Gazebo_sim = True</code> in <code>quadcopter.py</code>):
<pre><code>cd RK3588
python3 quadcopter.py</code></pre>
                </li>
            </ol>

            <h3>Step 4: Running in Real Life (HITL/Real)</h3>
            <ol>
                <li>Connect the RK3588 to your Flight Controller via UART/USB.</li>
                <li>Update <code>RK3588/quadcopter.py</code>: set <code>Gazebo_sim = False</code> and update <code>MAVLINK_PORT</code>.</li>
                <li>Start the Dashboard on your Ground Station: <code>python3 station.py</code>.</li>
                <li>Start the Drone script on the RK3588: <code>python3 quadcopter.py</code>.</li>
            </ol>
        </section>

        <!-- Control & Dashboard -->
        <section>
            <h2>🎛️ Control & Dashboard</h2>
            <ul>
                <li><strong>Click and Drag</strong> on the video feed to manually initialize the tracker on a specific region.</li>
                <li><strong>Click an AI Bounding Box</strong> to lock onto an AI-recognized target.</li>
                <li><strong>Auto-Track / Auto-Recovery</strong>: Toggle these in the "AI & Auto-Tracking" tab to let the drone fly entirely autonomously.</li>
                <li><strong>PID Tuning</strong>: Use the "Flight Tuning" tab to adjust the X/Y Axis responsiveness of the drone in real time.</li>
            </ul>
        </section>

        <!-- Repository Structure -->
        <section>
            <h2>📂 Repository Structure</h2>
<pre><code>.
├── images/                             # Documentation assets
│   ├── architecture.png
│   ├── demo.gif
│   ├── demo.mp4
│   └── tracking_flow.png
├── RK3588/                             # Code to run on the Edge Companion Computer
│   ├── py_utils/                       # Helper scripts for YOLO processing
│   ├── boost.sh                        # CPU/NPU/GPU performance overdrive script
│   ├── car1_demo.mp4                   # Sample top-down video for SITL testing
│   ├── object_tracking_vittrack...     # ViT ONNX model weights
│   ├── quadcopter.py                   # MAIN tracking, AI, and flight dynamics loop
│   ├── yolo.py                         # YOLO processing and ultra-fast NMS
│   ├── yolov8n.rknn                    # YOLOv8 Nano NPU weights
│   └── yolov8s.rknn                    # YOLOv8 Small NPU weights
├── Station/                            # Code to run on the Ground Control PC
│   ├── send_data_to_gazebo.py          # TCP/MAVSDK bridge for Gazebo simulation
│   └── station.py                      # Tkinter multiprocessing telemetry dashboard
└── README.md</code></pre>
        </section>

        <!-- License -->
        <section class="mt-12 pt-6 border-t border-slate-200">
            <h2>📝 License</h2>
            <p>
                This project is licensed under the MIT License - see the LICENSE file for details.
            </p>
        </section>

    </main>
</body>
</html>
