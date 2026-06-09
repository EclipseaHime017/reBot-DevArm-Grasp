# 🦾 reBot Arm B601-DM Visual Grasping Demo (Jetson Edition)

<p align="center">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601" width="600">
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/Platform-Jetson%20(ARM64)-orange.svg" alt="Platform: Jetson">
    <img src="https://img.shields.io/badge/Camera-Orbbec%20Gemini%202-green.svg" alt="Camera: Orbbec Gemini 2">
    <img src="https://img.shields.io/badge/Detection-GraspNet%2BYOLO-yellow.svg" alt="Detection: GraspNet + YOLO">
</p>

<p align="center">
  <strong>RGB-D Perception · Object Detection · Hand-Eye Calibration · GraspNet 6-DoF Pose · Robot Control</strong>
</p>

<p align="center">
  <a href="#readme-en"><strong>English Guide</strong></a>
  &nbsp;|&nbsp;
  <a href="./README_zh.md"><strong>中文文档</strong></a>
</p>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- ENGLISH SECTION                                             -->
<!-- ═══════════════════════════════════════════════════════════ -->
<div id="readme-en"></div>

## Table of Contents

> **Jump to:** [1. Overview](#1-overview--en) · [2. Hardware Requirements](#2-hardware-requirements--en) · [3. System Architecture](#3-system-architecture--en) · [4. Environment Setup](#4-environment-setup--en) · [5. Four-Step Environment Verification](#5-four-step-environment-verification--en) · [6. Hand-Eye Calibration](#6-hand-eye-calibration--en) · [7. YOLO / TensorRT Model Export](#7-yolo--tensorrt-model-export--en) · [8. Running the Web Demo](#8-running-the-web-demo--en) · [9. Web UI Guide](#9-web-ui-guide--en) · [10. Script Reference](#10-script-reference--en) · [11. Configuration Reference](#11-configuration-reference--en) · [12. CLI Grasping](#12-cli-grasping--en) · [13. Troubleshooting](#13-troubleshooting--en) · [14. FAQ](#14-faq--en) · [15. References](#15-references--en)

---

<div id="1-overview--en"></div>

## 1. Overview <span style="font-size:0.6em">[<a href="./README_zh.md#1-overview--zh">中文</a>]</span>

This project implements a complete visual grasping pipeline for the **reBot Arm B601-DM** on **NVIDIA Jetson**, combining multi-modal perception, deep-learning-based grasp pose estimation, and real robot control.

### Pipeline Flow

```
Orbbec Gemini 2 RGB-D Camera
        ↓
  YOLO Instance Segmentation (target filtering)
        ↓
  GraspNet 6-DoF Grasp Pose Estimation
        ↓
  Hand-Eye Calibration (Eye-in-Hand, TSAI algorithm)
        ↓
  Coordinate Transform: Camera Frame → Robot Base Frame
        ↓
  Arm IK Trajectory + Gripper Force Control
        ↓
  Base Rotation + Object Placement
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Dual grasp estimation** | GraspNet (6-DoF, pretrained) + ordinary grasp (depth-based, high-frequency) |
| **Eye-in-Hand calibration** | TSAI algorithm, fully automatic collection |
| **YOLO target filtering** | Segment anything in the scene, or pick by class name |
| **Web UI** | Real-time MJPEG stream, target selection, grasp preview, extrinsic tuning |
| **Base motor jog** | Direct joint1 control via web panel |
| **Multi-camera support** | Orbbec Gemini 2, RealSense D435i, RealSense D405 |
| **Extrinsic compensation** | Per-frame XYZ + RPY offsets for gripper / camera / base |

### Default Configuration

| Item | Default |
|------|---------|
| Conda environment | `graspnet` |
| Camera | Orbbec Gemini 2 (`pyorbbecsdk`) |
| Object detection | YOLO `yolo11n-seg.engine` |
| Grasp estimation | GraspNet `checkpoint-rs.tar` |
| Hand-eye calibration | `config/calibration/orbbec_gemini2/hand_eye.npz` |
| Main entry point | `scripts/grasp_web.py` |

---

<div id="2-hardware-requirements--en"></div>

## 2. Hardware Requirements <span style="font-size:0.6em">[<a href="./README_zh.md#2-hardware-requirements--zh">中文</a>]</span>

| Component | Model / Notes |
|-----------|---------------|
| Robot arm | reBot Arm B601-DM (DAMIAO motor version) |
| Depth camera | Orbbec Gemini 2 |
| Connectivity | USB2CAN adapter (robot CAN bus); USB 3.0 (camera) |
| Host | Jetson (Ubuntu 22.04, Python 3.10, ARM64) |
| ArUco calibration board | 4x4, ID=0, 0.1 m edge length (for hand-eye calibration) |

### Wiring

```bash
# 1. Connect Gemini 2 to Jetson via USB 3.0
# 2. Connect USB2CAN adapter to robot CAN bus and insert into Jetson USB port
# 3. Set device permissions (required on first use)
sudo chmod a+rw /dev/bus/usb/*/*
sudo chmod 666 /dev/ttyUSB0   # Adjust port number as needed
```

---

<div id="3-system-architecture--en"></div>

## 3. System Architecture <span style="font-size:0.6em">[<a href="./README_zh.md#3-system-architecture--zh">中文</a>]</span>

```
┌──────────────────────────────────────────────────────────────┐
│                 grasp_web.py (Web UI)                        │
│  Live MJPEG Stream · Target Class Selector · Grasp Button  │
│              Base Motor Jog Panel (joint1)                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐        ┌─────────────────────────┐
│  YOLO Seg.       │        │  GraspNet 6-DoF        │
│  Target Filter   │        │  Grasp Pose Estimation │
│  (TensorRT)      │        │  + Ordinary Grasp      │
└────────┬─────────┘        └────────────┬────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
              ┌──────────────────┐
              │  Hand-Eye Transform │ ← hand_eye.npz (calibration result)
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Arm IK Trajectory │
              │  + Gripper FSM     │
              └──────────────────┘
```

---

<div id="4-environment-setup--en"></div>

## 4. Environment Setup <span style="font-size:0.6em">[<a href="./README_zh.md#4-environment-setup--zh">中文</a>]</span>

> **Recommended order**: Follow each step in sequence. Do not skip the verification steps in Section 5.

### 4.1 Clone the Repository

Prefer the official Seeed-Projects repository:

```bash
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

You can also use the current development repository:

```bash
git clone https://github.com/EclipseaHime017/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

### 4.2 Create a Conda Environment

```bash
conda create -n graspnet python=3.10 -y
conda activate graspnet
```

### 4.3 Install NVIDIA PyTorch for Jetson

> **Critical:** On Jetson, **do not** install the generic PyPI `torch`. You must install a JetPack-matched wheel from NVIDIA.

```bash
# Query JetPack version
dpkg -l | grep jetpack
# or
cat /etc/nv_tegra_release
```

Reference: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

```bash
# Example (JetPack 6.0 + CUDA 12.6, Python 3.10 aarch64)
pip install torch-2.6.0 torchvision-0.21.0 -f https://developer.download.nvidia.com/compute/pytorchicrob/links.html
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.4 Install Non-PyTorch Dependencies

```bash
pip install -r requirements-graspnet-jetson.txt
```

Do not install pip `pin>=3.9.0`: the pip `pin` package may require `numpy>=2.2,<2.3`, which conflicts with this project and several vision / point-cloud dependencies that still use `numpy<2.0`.

### 4.5 Set CUDA Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

> Add to `~/.bashrc` to avoid repeating this every session:
> ```bash
> echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
> echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
> echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```

### 4.6 Install the Robotic Arm SDK

#### 4.6.1 reBotArm SDK

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

### 4.7 Install the Depth Camera SDK

**This project uses the Orbbec Gemini2 depth camera. If you use a different depth camera, install the matching SDK for your camera and skip this step.**

The Orbbec Gemini2 depth camera depends on **pyorbbecsdk** — the Python wrapper for Orbbec SDK v2. Prefer installing the prebuilt Python package first:

**Option 1: Install from pip (recommended)**

```bash
pip install pyorbbecsdk2
```

**Option 2: Get it from GitHub**

```bash
# Install build dependencies
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

# Get pyorbbecsdk
cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .

# Build from source if pre-built version is incompatible:
# bash ../../scripts/install_pyorbbecsdk.sh --from-source
```

When installing from source, make sure the native extension has been built with CMake first so `install/lib` contains `pyorbbecsdk*.so` and the Orbbec shared libraries before running `pip install -e .`.

Mainland China users can use:

```bash
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git
```

If all installation methods above fail, please refer to the official Orbbec documentation below.

**SDK Resources**

| Resource | Link |
|----------|------|
| Gemini 2 product page | https://www.orbbec.com/products/stereo-vision-camera/gemini-2/ |
| All developer resources | https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1 |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| SDK v2 API guide | https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/ |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| pyorbbecsdk docs | https://orbbec.github.io/pyorbbecsdk/index.html |
| ROS2 Wrapper | https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main |

**Verify installation**

```bash
python scripts/verify_pyorbbec_stream.py
```

### 4.8 Configure Orbbec udev Rules (Required)

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/pyorbbecsdk/scripts/env_setup
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4.9 Download YOLO Model Weights

```bash
# Create models directory
mkdir -p models

# Download YOLO11n-seg model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt -O models/yolo11n-seg.pt
```

### 4.10 Export YOLO TensorRT Engine (Must Run on Target Jetson)

> TensorRT `.engine` files are tightly coupled to the Jetson device, GPU, CUDA, and TensorRT versions. **Do not** copy engines from other machines — export on the target Jetson:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
```

Output: `models/yolo11n-seg.engine`

If FP16 export fails, fall back to FP32:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 device=0 workspace=4
```

### 4.11 Configure GraspNet (Optional)

You do not need GraspNet for `scripts/main.py` or `scripts/ordinary_grasp_pipeline.py`. Configure it only when you want to run `scripts/graspnet_camera_demo.py` or `scripts/grasp.py`, which require GraspNet baseline, CUDA-enabled PyTorch, the PointNet2/knn CUDA operators, and a pretrained checkpoint.

The GraspNet `pointnet2` / `knn` extensions require a CUDA compiler. Before starting, make sure the active environment can find `nvcc`, and check that the CUDA version reported by `nvcc` matches the CUDA version used to build PyTorch:

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

If `nvcc` is missing, or if the CUDA version reported by `nvcc` does not match `torch.version.cuda`, install a CUDA compiler that matches your current PyTorch CUDA version. For example, if PyTorch reports `13.0`:

```bash
conda install -c nvidia cuda-nvcc=13.0
```

You can also install a PyTorch build that matches your current `nvcc` version instead. The two versions must match, otherwise building `pointnet2` / `knn` will fail with `The detected CUDA version (...) mismatches the version that was used to compile PyTorch (...)`.

```bash
cd sdk
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline

# Install PyTorch for your CUDA version first, then install GraspNet runtime dependencies
pip install open3d tensorboard Pillow tqdm

# Build CUDA operators
cd pointnet2
pip install . --no-build-isolation
cd ../knn
pip install . --no-build-isolation
cd ..

# Install GraspNet API
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
sed -i "s/'sklearn'/'scikit-learn'/" setup.py
sed -i "s/'numpy==1.23.4'/'numpy>=1.24.0'/" setup.py
pip install .
cd ../..
```

***Note: If you follow the official graspnet-baseline repository documentation and use `python setup.py install`, CUDA / PyTorch related errors may occur. We recommend using `pip install . --no-build-isolation` so the extension is built against the PyTorch and CUDA configuration already installed in the active conda environment.***

***In addition, older GraspNet API dependencies may still use the deprecated `sklearn` package name. The `sed` commands replace it with the currently recommended `scikit-learn` package name to avoid `The 'sklearn' PyPI package is deprecated` during installation. They also adjust `numpy==1.23.4` to `numpy>=1.24.0`, preventing GraspNet API installation from downgrading NumPy and conflicting with the robotic arm control dependencies.***

Refer to the official graspnet-baseline repository to download the official GraspNet pretrained weight, then place `checkpoint-rs.tar` at:

```bash
sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar
```

Then verify `config/default.yaml`:

```yaml
graspnet:
  checkpoint: "checkpoint-rs.tar"
```

The `checkpoint` field supports three forms: a file name is resolved under `sdk/graspnet-baseline/checkpoints/`; a relative path is resolved from the project root; an absolute path is used directly.

---

## 📁 Directory Structure

```
rebot_grasp/
├── config/
│   ├── default.yaml              # Main configuration
│   └── calibration/
│       └── orbbec_gemini2/
│           ├── intrinsics.npz    # Camera intrinsics
│           └── hand_eye.npz      # Hand-eye calibration result
├── drivers/
│   ├── camera/
│   │   ├── base.py               # Abstract camera base class
│   │   ├── orbbec_gemini2.py     # Gemini 2 driver
│   │   └── realsense.py          # RealSense driver (alternative)
│   └── robot/
│       └── rebot_arm.py          # reBotArm wrapper + gripper FSM
├── calibration/
│   ├── aruco_pose.py             # ArUco pose estimation
│   └── hand_eye.py               # Hand-eye calibration solver
├── utils/
│   ├── ordinary_grasp.py         # OBB grasp estimation and visualization
│   └── transforms.py             # Coordinate transform utilities
├── scripts/
│   ├── main.py                   # Main grasping program
│   ├── ordinary_grasp_pipeline.py
│   ├── object_detection.py
│   ├── collect_handeye_eih.py
│   ├── grasp_web.py              # Web UI entry point
│   ├── grasp.py                  # GraspNet CLI entry point
│   ├── graspnet_camera_demo.py   # GraspNet camera demo
│   ├── verify_pyorbbec_stream.py
│   ├── verify_rebot_arm_motion.py
│   ├── verify_handeye_calibration.py
│   └── verify_graspnet_stack.py
├── sdk/
│   ├── pyorbbecsdk/              # Orbbec SDK Python wrapper
│   └── reBotArm_control_py/      # reBot Arm SDK
├── models/                       # YOLO model files (.pt, .engine)
├── environment.yml               # Recommended conda environment
└── requirements-graspnet-jetson.txt  # pip dependencies for Jetson
```

---

<div id="5-four-step-environment-verification--en"></div>

## 5. Four-Step Environment Verification <span style="font-size:0.6em">[<a href="./README_zh.md#5-four-step-environment-verification--zh">中文</a>]</span>

> **Run after every deployment.** Complete all four steps in order before proceeding to calibration or demo.

### Step 1: Orbbec RGB-D Stream

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp

# Text-only check
python scripts/verify_pyorbbec_stream.py

# With preview window
python scripts/verify_pyorbbec_stream.py --preview --seconds 10
```

**Expected:** RGB and Depth resolution + frame rate reported, no errors.

### Step 2: Robot Arm Connection

```bash
# Read-only check (safe — does not move the arm)
python scripts/verify_rebot_arm_motion.py --read-only

# Small-angle jog test (make sure the arm path is clear!)
python scripts/verify_rebot_arm_motion.py --deg 5
```

**Expected:** `[OK]` or normal pose data returned.

### Step 3: Hand-Eye Calibration File

```bash
python scripts/verify_handeye_calibration.py
```

**Expected:**

```
[OK] hand-eye calibration looks usable
```

> If you get `No such file`, hand-eye calibration has not been done yet. Proceed to [Section 6](#6-hand-eye-calibration--en).

### Step 4: GraspNet Stack

```bash
python scripts/verify_graspnet_stack.py
```

**Expected:**

```
[OK] GraspNet stack is ready
```

> If you see `No module named pointnet2._ext`, re-run [Section 4.6.4](#464-build-graspnet-cuda-operators-critical).

---

<div id="6-hand-eye-calibration--en"></div>

## 6. Hand-Eye Calibration <span style="font-size:0.6em">[<a href="./README_zh.md#6-hand-eye-calibration--zh">中文</a>]</span>

This system uses **Eye-in-Hand** calibration: the camera is mounted on the robot end-effector and moves with it. The ArUco marker is fixed on the table.

### 6.1 When to Recalibrate

Recalibrate when any of the following occurs:

- Camera mount position changes (bracket moved or re-attached)
- Gripper geometry changes (different gripper or position moved)
- ArUco marker size changed
- Table layout significantly changed
- Grasp accuracy noticeably degraded

### 6.2 Calibration Board Setup

Use an **ArUco DICT_4X4_50, ID=0** board with edge length **0.1 m** (must match `aruco.marker_length_m` in `config/default.yaml`).

Print the board and affix it flat on the table, or use a physical calibration target. The marker must be clearly visible from the arm's viewing angles.

### 6.3 Automatic Collection (Recommended)

```bash
python scripts/collect_handeye_eih.py
```

The arm automatically traverses 50 predefined poses. When the ArUco marker is stably detected, it automatically samples. A minimum of **5 samples** is required; **15+** is recommended for stable results.

| Key | Action |
|-----|--------|
| `c` or `q` | End collection and compute calibration |
| `Ctrl+C` | Also triggers computation on collected data mid-collection |

### 6.4 Manual Collection Mode

```bash
python scripts/collect_handeye_eih.py --manual
```

The arm enters gravity-compensation mode. Manually push it to various viewing angles:

| Key | Action |
|-----|--------|
| `Enter` | Sample current pose |
| `pos` | Print current end-effector pose |
| `c` / `q` | Finish and compute |

### 6.5 Calibration Output

Results are auto-saved to:

```
config/calibration/orbbec_gemini2/hand_eye.npz
config/calibration/orbbec_gemini2/intrinsics.npz
```

### 6.6 Verify Calibration

```bash
python scripts/verify_handeye_calibration.py
```

---

<div id="7-yolo--tensorrt-model-export--en"></div>

## 7. YOLO / TensorRT Model Export <span style="font-size:0.6em">[<a href="./README_zh.md#7-yolo--tensorrt-model-export--zh">中文</a>]</span>

### 7.1 Export ONNX from .pt (Intermediate Format)

```bash
yolo export model=models/yolo11n-seg.pt format=onnx imgsz=640 opset=12 simplify=True
```

### 7.2 Export TensorRT Engine Directly from .pt (Recommended)

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
```

### 7.3 Convert ONNX to TensorRT Engine

```bash
trtexec \
  --onnx=models/yolo11n-seg.onnx \
  --saveEngine=models/yolo11n-seg.engine \
  --fp16 \
  --workspace=4096
```

### 7.4 Verify the Exported Engine

```bash
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
python scripts/object_detection.py
```

### 7.5 Switching Models

Edit `config/default.yaml`:

```yaml
yolo:
  model_name: "yolo11n-seg.engine"  # Replace with your engine filename
  device: "auto"
```

Or override from the command line:

```bash
python scripts/grasp_web.py --yolo-model yolo11n-seg.engine
```

> **Note:** `.engine` files are device-specific and cannot be shared across machines. If JetPack / TensorRT / CUDA is updated, re-export the engine on the target Jetson.

---

<div id="8-running-the-web-demo--en"></div>

## 8. Running the Web Demo <span style="font-size:0.6em">[<a href="./README_zh.md#8-running-the-web-demo--zh">中文</a>]</span>

### 8.1 Preview Mode (No Robot Connection)

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

Open in browser:

```
http://<jetson_ip>:8000
```

In this mode you can:
- View live RGB-D video stream
- See YOLO detection results overlaid
- Preview GraspNet grasp points
- Tune extrinsic compensation parameters (gripper, camera, base)
- Jog the base motor

### 8.2 Enable Real Robot Execution

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

Click **Real Grasp** in the web interface to trigger the full grasp sequence.

### 8.3 Disable Post-Grasp Actions for Debugging

Base rotation and placement after each grasp interfere with continuous grasp debugging. Disable them:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --no-place-after-grasp
```

### 8.4 Full-Scene GraspNet (No YOLO Filtering)

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --no-yolo
```

### 8.5 Common Command-Line Options

| Flag | Description |
|------|-------------|
| `--host 0.0.0.0 --port 8000` | Bind to all interfaces |
| `--enable-robot` | Allow real arm movement |
| `--no-yolo` | Skip YOLO detection, show full-scene GraspNet |
| `--camera-type orbbec_gemini2` | Force camera type |
| `--target-class cup` | Auto-select this class |
| `--no-place-after-grasp` | Skip base rotation and placement |
| `--no-auto-graspnet` | Disable automatic GraspNet updates |
| `--graspnet-interval 2.0` | GraspNet update interval (seconds) |

---

<div id="9-web-ui-guide--en"></div>

## 9. Web UI Guide <span style="font-size:0.6em">[<a href="./README_zh.md#9-web-ui-guide--zh">中文</a>]</span>

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [reBot Grasp Web] [EN/中文] Target:[▼] [Set] [Refresh] [Real Grasp] │
│ mode hint...                                                │
├─────────────────────────────────┬───────────────────────────┤
│                                 │  ┌─ Compensation ───────┐ │
│                                 │  │ Gripper F/L/U (m)    │ │
│     Live MJPEG Camera Stream    │  │ [___] [___] [___]    │ │
│     + YOLO boxes                │  │ Gripper RPY (deg)    │ │
│     + GraspNet grasp point      │  │ [___] [___] [___]    │ │
│                                 │  │ Camera XYZ (m)       │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ Camera RPY (deg)     │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ Base XYZ (m)         │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ Base RPY (deg)       │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ [Set Compensation]   │ │
│                                 │  └──────────────────────┘ │
│                                 │  ┌─ Base Motor ────────┐ │
│                                 │  │ joint1 jog: [___]   │ │
│                                 │  │ Duration: [___]     │ │
│                                 │  │ Margin:  [___]       │ │
│                                 │  │ [-30°] [Apply] [+30°]│ │
│                                 │  └──────────────────────┘ │
│                                 │  status line...            │
│                                 │  {state JSON}              │
└─────────────────────────────────┴───────────────────────────┘
```

### Extrinsic Compensation Fields

The page has three categories of compensation fields:

| Category | Fields | Effect |
|----------|--------|--------|
| **Gripper Fwd/Lat/Up** | forward, lateral, vertical (meters) | Corrects final grasp TCP pose |
| **Gripper RPY** | roll, pitch, yaw (degrees) | Local TCP rotation offset |
| **Camera XYZ/RPY** | 6-DOF (meters / degrees) | Corrects camera extrinsics |
| **Base XYZ/RPY** | 6-DOF (meters / degrees) | Corrects base-frame extrinsics (**does not rotate the base motor**) |

### Base Motor Jog Panel

The base motor debug panel **directly jogs `joint1`** and returns JSON with:

```json
{
  "before_deg": 12.5,
  "after_deg": -17.5,
  "limit_deg": -30.0,
  "safe_limit_deg": -25.0
}
```

Use this panel to:
- Test base motor response
- Find safe joint limits before running grasp sequences
- Tune `base_delta_deg` and `base_safety_margin_deg` in `config/default.yaml`

---

<div id="10-script-reference--en"></div>

## 10. Script Reference <span style="font-size:0.6em">[<a href="./README_zh.md#10-script-reference--zh">中文</a>]</span>

| Script | Purpose |
|--------|---------|
| `verify_pyorbbec_stream.py` | Orbbec RGB-D stream check (text / preview) |
| `verify_rebot_arm_motion.py` | Robot connection and jog check |
| `verify_handeye_calibration.py` | Hand-eye calibration file integrity check |
| `verify_graspnet_stack.py` | GraspNet / CUDA / YOLO engine sanity check |
| `collect_handeye_eih.py` | Eye-in-Hand hand-eye calibration (auto / manual modes) |
| `graspnet_camera_demo.py` | Camera-side GraspNet preview with Open3D window |
| `object_detection.py` | Pure YOLO detection demo |
| `grasp_web.py` | **Main entry:** Web UI, target selection, real grasp, base jog |
| `grasp.py` | CLI real grasp pipeline (non-Web, OpenCV window) |
| `install_pyorbbecsdk.sh` | pyorbbecsdk installation helper script |

---

<div id="11-configuration-reference--en"></div>

## 11. Configuration Reference <span style="font-size:0.6em">[<a href="./README_zh.md#11-configuration-reference--zh">中文</a>]</span>

Main config file: `config/default.yaml`

### 11.1 Camera Configuration

```yaml
camera:
  type: realsense_d435i
  serial: null
  color_width: 1280
  color_height: 720
  depth_width: 1280
  depth_height: 720
  fps: 30
```

### 11.2 Calibration Configuration

```yaml
calibration:
  aruco:
    marker_length_m: 0.1
    dict_id: 0
    target_marker_id: 0
  hand_eye_method: TSAI
```

### 11.3 Detection Configuration

```yaml
detection:
  conf_threshold: 0.5
  iou_threshold: 0.45
```

### 11.4 Robot Configuration

```yaml
robot:
  repo_root: null   # auto-detects sdk/reBotArm_control_py
  config_path: null
  urdf_path: null
  pose_convention: xyz_euler_rad
  ready_pose:
    x: 0.3
    y: 0.0
    z: 0.3
    roll: 0.0
    pitch: 0.7
    duration: 3.0
```

### 11.5 YOLO Configuration

```yaml
yolo:
  model_name: "yoloe-26l-seg.pt"
  device: "cpu"          # use "cuda:0" for GPU
  use_world: true
  custom_classes:
    - "yellow banana"
    - "water bottle"
    - "light blue coffee cup"
    - "cup"
    - "green object"
    - "red object"
    - "tool"
```

### 11.6 Grasp Pipeline Configuration

```yaml
grasp_pipeline:
  infer_every_live: 3
  grasp:
    depth_quantile: 0.6
    pregrasp_offset_m: 0.080
```

### YAML parameter notes

- `camera.type`: camera type. Available values: `realsense_d435i`, `realsense_d405`, `orbbec_gemini2`.
- `camera.serial`: specific device serial number; `null` means use the first available device.
- `calibration.aruco.marker_length_m`: ArUco marker side length used for hand-eye calibration, in meters.
- `detection.conf_threshold`: YOLO confidence threshold.
- `detection.iou_threshold`: YOLO NMS IoU threshold.
- `robot.repo_root`: root directory of `reBotArm_control_py`; when `null`, the code auto-detects `cameraws/sdk/reBotArm_control_py`.
- `robot.config_path` / `robot.urdf_path`: robot control config and URDF; `null` means use the SDK defaults.
- `robot.ready_pose`: the ready pose reached on startup and after each completed grasp.
- `grasp_pipeline.infer_every_live`: run detection once every N frames during live preview to reduce CPU/GPU load.
- `grasp_pipeline.grasp.depth_quantile`: depth quantile used by the ordinary grasp pipeline; larger values usually place the grasp point deeper.
- `grasp_pipeline.grasp.pregrasp_offset_m`: distance, in meters, to retreat along the tool approach direction when generating the pre-grasp pose.

### Model selection

YOLO models are loaded from `cameraws/models/`. If the file is missing, Ultralytics will usually try to download it automatically.

Common choices:

| Model | Description |
| --- | --- |
| `yoloe-26l-seg.pt` | Open-vocabulary + segmentation, current default |
| `yoloe-26s-seg.pt` | Lighter and faster |
| `yolov8n-seg.pt` | Closed-set segmentation, small model |
| `yolov8s-seg.pt` | Closed-set segmentation, higher accuracy |

If the model name contains `world` or `yoloe`, and `yolo.use_world=true`, the program calls `model.set_classes(custom_classes)` and injects `yolo.custom_classes` as open-vocabulary categories. Standard `yolov8*-seg.pt` models ignore these open-vocabulary class entries.

### Grasp Offset Configuration (from camera to gripper)

```yaml
grasp_pipeline:
  grasp:
    pregrasp_offset_m: 0.080         # Pre-grasp height offset (meters)
    grasp_forward_offset_m: 0.000    # Move grasp point forward along approach axis
    camera_x_offset_m: 0.000         # Camera extrinsic compensation X
    camera_y_offset_m: 0.000         # Camera extrinsic compensation Y
    camera_z_offset_m: 0.000         # Camera extrinsic compensation Z
    camera_roll_offset_deg: 0.0      # Camera extrinsic compensation roll
    camera_pitch_offset_deg: 0.0     # Camera extrinsic compensation pitch
    camera_yaw_offset_deg: 0.0       # Camera extrinsic compensation yaw
    base_x_offset_m: 0.000           # Base extrinsic compensation X
    base_yaw_offset_deg: 0.0         # Base extrinsic compensation yaw
```

### Placement Configuration

```yaml
grasp_pipeline:
  place:
    enabled: true
    base_joint: joint1
    base_delta_deg: 90.0             # Base rotation angle (negative = negative direction)
    base_direction: auto              # auto / positive / negative
    base_rotate_duration: 2.5        # Rotation duration (seconds)
    base_safety_margin_deg: 5.0      # Safety margin from joint limits
    return_home: true                # Whether to return to home after placing
```

---

<div id="12-cli-grasping--en"></div>

## 12. CLI Grasping <span style="font-size:0.6em">[<a href="./README_zh.md#12-cli-grasping--zh">中文</a>]</span>

Suitable for headless or automated invocation.

### 12.1 Dry-Run (No Arm Movement)

```bash
# Full-scene grasping
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --no-yolo

# With target class
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --target-class cup
```

### 12.2 Real Execution

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup
```

### 12.3 Disable Post-Grasp Placement

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup --no-place-after-grasp
```

### 12.4 Custom Base Rotation

```bash
python scripts/grasp.py \
  --camera-type orbbec_gemini2 \
  --target-class cup \
  --place-base-delta-deg -30 \
  --place-base-rotate-duration 2.5
```

---

<div id="13-troubleshooting--en"></div>

## 13. Troubleshooting <span style="font-size:0.6em">[<a href="./README_zh.md#13-troubleshooting--zh">中文</a>]</span>

### Q1: `pyorbbecsdk import failed`

```bash
bash scripts/install_pyorbbecsdk.sh
# If still failing, build from source:
bash scripts/install_pyorbbecsdk.sh --from-source
```

### Q2: Camera frame is all black

```bash
# Check udev rules
ls -la /dev/bus/usb/*/*
# Reload udev
sudo udevadm control --reload-rules && sudo udevadm trigger
# Check USB 3.0 connection
```

### Q3: `nvbufsurftransform: Could not get EGL display connection`

This is normal on Jetson without a desktop display. Text-only checks (`--preview` omitted) are unaffected.

### Q4: `torch.cuda.is_available()` returns False

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Reinstall the wheel matching your JetPack version
```

### Q5: `No module named pointnet2._ext`

Rebuild the CUDA operators:

```bash
cd sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install
```

### Q6: YOLO engine fails to load

Common causes:

- Engine was copied from another machine
- JetPack / TensorRT / CUDA versions changed
- Ultralytics version changed

Solution: Re-export on the target Jetson:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
```

### Q7: ArUco marker not detected

- Verify the marker is `DICT_4X4_50` with ID `0`
- Verify `marker_length_m` matches the actual black square edge length
- Marker must be flat, complete, and under stable lighting
- Verify `config/calibration/orbbec_gemini2/intrinsics.npz` exists

### Q8: Base jog does not change the angle

First check: joint1 motor enable state, encoder feedback, limit settings, and low-level control mode.

### Q9: Grasp offset tuning

If the gripper consistently misses the object:

| Symptom | Tune |
|---------|------|
| Gripper approaches from wrong depth | `grasp_forward_offset_m` |
| Gripper misses left/right | `grasp_lateral_offset_m` or `camera_x_offset_m` |
| Gripper misses height | `grasp_vertical_offset_m` or `camera_z_offset_m` |
| Rotation is off | `grasp_roll_offset_deg` / `grasp_pitch_offset_deg` / `grasp_yaw_offset_deg` |
| Camera mount is slightly off | `camera_x/y/z_offset_m` and `camera_r/p/y_offset_deg` |

Use the web UI sliders for live tuning, then copy values to `config/default.yaml` for persistence.

---

<div id="14-faq--en"></div>

## ❓ FAQ

### 1. `ModuleNotFoundError: No module named 'motorbridge'`

This usually means the robotic arm SDK dependencies are not installed in the current Python environment. Make sure the project environment is active, then update the environment and install the robotic arm SDK:

```bash
conda activate rebotarm
conda env update -n rebotarm -f environment.yml
cd sdk/reBotArm_control_py && pip install -e .
```

### 2. Pressing `G` does not execute grasping

Common causes include:

- `hand_eye.npz` does not exist
- The hand-eye calibration mode is not `eye_in_hand`
- The target pose is not reachable by IK

It is recommended to validate the perception result and target pose in dry-run mode first:

```bash
python scripts/main.py --dry-run
```

### 3. The grasp depth is unstable

Check and adjust these items first:

- `grasp_pipeline.grasp.depth_quantile`
- The installation height of the camera relative to the workspace
- Reflective properties of the target surface

### 4. GraspNet reports that `pointnet2_utils` cannot be imported from `pointnet2`

This usually means the local CUDA extension under `sdk/graspnet-baseline/pointnet2` was not built in the active conda environment, or Python is resolving a different `pointnet2` package. Make sure the project environment is active, then rebuild both `pointnet2` and `knn` in that same environment:

```bash
conda activate rebotarm
cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

Verify:

```bash
python -c "from pointnet2 import pointnet2_utils; print('Submodule import works')"
```

### 5. CUDA architecture compatibility issues on newer GPUs

If you see `no kernel image is available for execution on the device`, or PyTorch reports that the current GPU CUDA capability is unsupported, the installed PyTorch wheel likely does not include CUDA kernels for that GPU architecture. Install a PyTorch build that supports your current CUDA/GPU architecture, then rebuild the GraspNet local CUDA extensions.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

If you need to specify the build architecture manually, set `TORCH_CUDA_ARCH_LIST` before rebuilding. Choose the value according to your GPU architecture and PyTorch/CUDA version.

### 6. GraspNet inference reports `RuntimeError: CPU not supported`

The sampling operators in `pointnet2` only support CUDA tensors. Confirm that CUDA is available, the GraspNet network and input point cloud are on GPU, and `pointnet2` / `knn` were built against the PyTorch version in the active environment.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `False`, fix the CUDA / PyTorch installation first. If it is `True` but the error remains, rebuild `pointnet2` and `knn`.

### `scripts/graspnet_camera_demo.py` — GraspNet camera estimation demo

Runs GraspNet 6D grasp pose estimation with only the RGB-D camera, without connecting to the robotic arm. The script keeps a live camera preview, uses YOLO bounding boxes to select the target area, and filters feasible GraspNet full-scene candidates by the target bbox. Press `G` or `Space` to infer the current frame, `R` to resume live preview, and `Q` or `Esc` to quit. After inference, Open3D can visualize the point cloud and grasp candidates.

```bash
python scripts/graspnet_camera_demo.py
```

### `scripts/grasp.py` — GraspNet robotic grasping program

Connects the GraspNet estimate to the robotic arm execution flow. YOLO selects the target, GraspNet outputs a 6D grasp pose, hand-eye calibration transforms it into the robot base frame, and the script checks IK reachability before running the pre-grasp, grasp, and retreat motion sequence. For debugging, start with `--dry-run` to print the target poses and candidate filtering result without moving the arm.

```bash
python scripts/grasp.py --dry-run
python scripts/grasp.py --target-class "light blue coffee cup"
```

### `scripts/object_detection.py` — Basic detection demo

---

<div id="15-references--en"></div>

## 15. References <span style="font-size:0.6em">[<a href="./README_zh.md#15-references--zh">中文</a>]</span>

| Resource | Link |
|----------|------|
| Seeed wiki | https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/ |
| reBotArm_control_py | https://github.com/vectorBH6/reBotArm_control_py |
| reBot-DevArm | https://github.com/Seeed-Projects/reBot-DevArm |
| Orbbec Gemini 2 | https://www.orbbec.com/products/stereo-vision-camera/gemini-2/ |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| GraspNet baseline | https://github.com/graspnet/graspnet-baseline |
| GraspNet API | https://github.com/graspnet/graspnetAPI |
| Ultralytics YOLOv11 | https://github.com/ultralytics/ultralytics |
| NVIDIA PyTorch for Jetson | https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html |

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- END OF ENGLISH SECTION                                     -->
<!-- ═══════════════════════════════════════════════════════════ -->

<p align="center">
  <strong>If this project is helpful to you, feel free to star it!</strong>
</p>
