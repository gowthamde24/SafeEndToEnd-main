# SafeEndToEnd: SA-TCP Architecture & Control Barrier Functions in CARLA

##  Overview
This repository contains the implementation and Phase 1 baseline replication of the Safe Autonomous Trajectory Control Prediction (SA-TCP) architecture. The system executes closed-loop autonomous driving in the CARLA simulator (Town 04) by combining an End-to-End (E2E) neural network with a mathematical Control Barrier Function (CBF) safety gate

##  Architecture Breakdown
The pipeline processes live RGB camera feeds via a PyTorch ResNet backbone, split into two primary branches:

* **Branch A (Primary Control):** Extracts visual features and concatenates them with current velocity vectors to predict raw steering commands.
* **Branch B (Safety Predictors):** Three separate ResNet-18 models predict spatial telemetry:
  * Cross-Track Error (X)
  * Heading Error (Theta)
  * Road Curvature
* **Control Barrier Function (CBF):** Acts as the system's "Safety Gate." It evaluates the raw steering command from Branch A against the spatial state predicted by Branch B. By projecting the vehicle's trajectory using a Kinematic or Dynamic bicycle model, the CBF actively overrides the neural network with a mathematically guaranteed safe steering angle if lane boundaries are threatened.

---

##  Phase 1 Milestones Achieved
The baseline environment, training loop, and inference pipeline have been successfully stabilized. The following critical work has been completed:

1. **Phantom Error Resolution (X):** Investigated and eliminated a scaling artifact that caused the network to hallucinate a 300-meter lateral offset. The model now predicts cross-track error with centimeter-level precision.
2. **Angular Discontinuity Handling (Theta):** Implemented a wrapped circular difference metric in the evaluation pipeline to mathematically handle +/- Pi coordinate boundary jumps, eliminating artificial Mean Squared Error (MSE) spikes during heading alignment evaluation.
3. **Legacy CARLA 0.9.x Bridge:** Deployed adapter classes (`Carla09Bridge` and `MockImageConverter`) to cleanly interface modern 0.9.x sensor queues and transforms with the legacy 0.8.x physics and controller logic.
4. **Closed-Loop Evaluation:** Successfully executed a full inference loop (`RUN_NO = 1`) without CARLA Autopilot. The Lincoln MKZ navigated Town 04 driven entirely by the trained ResNet weights and CBF override.
5. **Automated Telemetry Processing:** Upgraded `plots_curves.py` to automatically generate validation curves and "Predicted vs. Observed" spatial tracking graphs, outputting directly to the `results/` directory.
6. **Video Compilation Pipeline:** Optimized `create_video.py` to stitch real-time third-person inference frames into a final `.mp4` format, with built-in safety checks to handle dropped simulation frames.

---

##  Dataset Naming Convention
To prevent desynchronization between image frames and telemetry labels, ground-truth physics are embedded directly into the dataset file strings during expert Autopilot collection (`-r 0`). 

The extraction format is:
`frame_<ID>_<Steering>_<Curvature>_<X>_<Theta>_<Speed>_<Perp_Speed>.png`

| Variable | Physical Meaning | Integer Scaling Math |
| :--- | :--- | :--- |
| **`ID`** | Sequential frame index | `frame//5` |
| **`Steering`** | Expert steering command | Degrees * 100 |
| **`Curvature`** | True road map curvature | Curvature * 10000 |
| **`X`** | Lateral cross-track error | Meters * 100 |
| **`Theta`** | Vehicle heading error | Degrees * 100 |
| **`Speed`** | Longitudinal velocity | m/s * 100 |
| **`Perp_Speed`** | Lateral slip velocity | m/s * 100 |

---

##  Execution Commands

**1. Run Closed-Loop Inference:**
Executes the real-time driving loop using the trained PyTorch models and the active CBF constraint.
```bash
python run_iter.py -r 1
```

**2. Generate Evaluation Plots:**
Parses the training and inference CSV logs to generate MSE convergence and performance graphs, saving them directly into the `results/` folder.
```bash
python plots_curves.py
```

**3. Compile Demo Video:**
Assembles the saved third-person chase camera frames into an MP4 video playback.
```bash
python create_video.py -n 1
```