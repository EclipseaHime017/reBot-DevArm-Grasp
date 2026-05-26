# 🦾 reBot Arm B601-DM Visual Grasping Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601">
</p>

<p align="center">
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04+-orange.svg" alt="Platform">
    <img src="https://img.shields.io/badge/Camera-Orbbec%20Gemini%202-green.svg" alt="Camera">
    <img src="https://img.shields.io/badge/Detection-YOLO-yellow.svg" alt="YOLO">
</p>

<p align="center">
  <strong>Depth Perception · Object Detection · Hand-Eye Calibration · Autonomous Grasping · Fully Open Source</strong>
</p>

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a>
  </strong>
</p>

---

## 📖 Introduction

**reBot Arm B601-DM Visual Grasping Demo** is a vision-based grasping demo that integrates the [reBot Arm B601](https://github.com/vectorBH6/reBotArm_control_py) robotic arm control library with the **Orbbec Gemini 2** depth camera. The system uses a YOLO model to detect tabletop objects in real time, estimates grasp poses via OBB minimum bounding rectangles, transforms grasp points from camera space to robot base space through hand-eye calibration, and drives the arm to perform autonomous grasping.

### ✨ Core Features

- 📷 **Depth Perception** — Orbbec Gemini 2 provides aligned RGB + depth frames (1280×720 @ 30fps)
- 🔍 **Object Detection** — YOLO model-based recognition with open-vocabulary custom classes
- 📐 **Pose Estimation** — OBB short-axis direction for gripper orientation; depth quantile for grasp height
- 🔄 **Coordinate Transform** — TSAI hand-eye calibration (Eye-in-Hand) to map camera-frame grasp points to robot base frame
- 🦾 **Motion Execution** — reBotArm_control_py IK + trajectory controller with built-in gripper force-control state machine

---

## ⚙️ Hardware Setup

| Component | Model / Requirement |
|-----------|-------------------|
| Robotic Arm | reBot Arm B601-DM (DAMIAO motor variant) |
| Depth Camera | Orbbec Gemini 2 |
| Communication | USB2CAN serial bridge (arm); USB 3.0 (camera) |
| Host PC | Ubuntu 22.04+, Python 3.10, x86_64 |

**Wiring**

1. Connect the Gemini 2 to the host via USB 3.0
2. Connect the USB2CAN adapter to the arm's CAN bus and plug it into the host
3. Set device permissions:

```bash
sudo chmod a+rw /dev/bus/usb/*/*   # Orbbec camera
sudo chmod 666 /dev/ttyUSB0        # USB2CAN (adjust port as needed)
```

---

## 🚀 Quick Start

### Step 1. Clone the repository

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

### Step 2. Create and install the conda environment

```bash
conda env create -f environment.yml
conda activate rebotarm
```

Do not install pip `pin>=3.9.0`: the pip `pin` package may require `numpy>=2.2,<2.3`, which conflicts with this project and several vision / point-cloud dependencies that still use `numpy<2.0`.

### Step 3. Install the robotic arm control library

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

### Step 4. Install the depth camera SDK

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

cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
```

When installing from source, make sure the native extension has been built with CMake first so `install/lib` contains `pyorbbecsdk*.so` and the Orbbec shared libraries before running `pip install -e .`.

Mainland China users can use:

```bash
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git
```

If all installation methods above fail, please refer to the official Orbbec documentation below.

**Verify installation**

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
```

**Configure udev rules (required on first use)**

```bash
cd sdk/pyorbbecsdk
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**OrbbecViewer (optional — verify camera)**

Download the prebuilt package and run `OrbbecViewer` to confirm the camera connection and depth stream are working before running the demo.

- GitHub: https://github.com/orbbec/OrbbecSDK_v2/releases
- Gitee: https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases

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

### Step 5. Configure GraspNet (optional)

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
cd ../../..
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
│   └── collect_handeye_eih.py
├── sdk/
│   ├── pyorbbecsdk/              # Orbbec SDK Python wrapper
│   └── reBotArm_control_py/      # reBot Arm SDK
└── environment.yml               # Recommended conda environment
```

---

## 🛠️ Configuration

### Config file

Edit `config/default.yaml` and verify the key parameters:

```yaml
camera:
  type: realsense_d435i
  serial: null
  color_width: 1280
  color_height: 720
  depth_width: 1280
  depth_height: 720
  fps: 30

calibration:
  aruco:
    marker_length_m: 0.1
    dict_id: 0
    target_marker_id: 0
  hand_eye_method: TSAI

detection:
  conf_threshold: 0.5
  iou_threshold: 0.45

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

### Hand-eye calibration (first-time setup)

```bash
python scripts/collect_handeye_eih.py
```

In automatic mode, the arm traverses 50 preset poses and records a sample whenever the ArUco marker is detected stably. If the run finishes normally or is interrupted midway, the script still attempts to compute and save the calibration result; at least 5 samples are required, and 15 or more are recommended.

If you want to move the arm by hand during calibration, use:

```bash
python scripts/collect_handeye_eih.py --manual
```

In manual mode, the arm enters gravity-compensation mode. Push the end effector to a suitable viewpoint, press `Enter` to capture, and use `c` or `q` to finish and compute the result.

---

## 🎬 Demo Description

### `scripts/main.py` — Main grasping program

The full vision-grasping pipeline:

1. Enable the arm and move to the ready pose
2. Live camera preview with YOLO object detection and instance segmentation
3. OBB short-axis estimation for gripper orientation; depth quantile for grasp height
4. Press `G` to freeze the frame; hand-eye transform computes the target arm pose
5. Arm moves to pre-grasp point → descends → gripper closes → lifts → returns to ready pose

### `scripts/ordinary_grasp_pipeline.py` — Simplified grasp test

Runs OBB grasp pose estimation and visualization without connecting to the arm. Useful for debugging the perception module in isolation.

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

Pure YOLO detection with real-time bounding boxes and confidence scores. No grasping logic.

### `scripts/collect_handeye_eih.py` — Hand-eye calibration data collection

Eye-in-Hand hand-eye calibration using ArUco markers, with both automatic pose traversal and manual gravity-compensation sampling. Supports TSAI, PARK, and HORAUD solvers.

---

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

---

## 📄 References

- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) — Robotic arm control library
- [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — reBot arm open-source project
- [Orbbec Gemini 2](https://www.orbbec.com/products/stereo-vision-camera/gemini-2/)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)

---

## ☎ Contact Us

- **Technical Support**: [Submit an Issue](https://github.com/Seeed-Projects/reBot-DevArm-Grasp/issues)

---

<p align="center">
  <strong>🌟 If this project helps you, please give us a Star!</strong>
</p>
