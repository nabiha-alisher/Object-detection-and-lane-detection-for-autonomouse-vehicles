# Object-detection-and-lane-detection-for-autonomouse-vehicles

## Overview

This project implements a real-time AI Driving assistance system for autonomous vehicles using deep learning and NVIDIA inference technologies.

The system performs:

- Lane Detection
- Vehicle Detection
- Traffic Light Detection
- Depth Estimation
- Real-time Inference using NVIDIA Triton Inference Server
- TensorRT optimized inference for high FPS

The project combines multiple deep learning models into a single inference pipeline for efficient autonomous driving assistance.

---

## Project Structure

```
app/
pipelines/
scripts/
experiments/
verification/
triton_model_repo/
triton_configs/
model_sources/
docs/
assets/
```

---

## Technologies Used

- Python
- PyTorch
- TensorRT
- NVIDIA Triton Inference Server
- ONNX
- OpenCV
- CUDA
- YOLOPX
- Depth Anything V2

---

## Features

- Real-time lane detection
- Object detection
- Traffic light detection
- Depth estimation
- Triton HTTP & gRPC inference
- TensorRT optimized pipelines
- FPS benchmarking
- Lane optimization experiments
- Raw TensorRT inference support

---

## Repository Contents

This repository contains:

- Source code
- Triton configuration files
- Benchmark scripts
- Verification scripts
- Pipeline implementations
- Documentation

Large model files are **not included** in this repository.

---

# Model Files

The following files must be downloaded separately.

## Weights

Place inside:

```
weights/
```

Required files:

- yolopx.pth
- depth_anything_v2.pth
- traffic_light_detector.pth

Download:

> Google Drive Link Here

---

## ONNX Models

Place inside:

```
onnx/
```

Required files:

- yolopx.onnx
- depth_anything_v2.onnx
- traffic_light.onnx

Download:

> Google Drive Link Here

https://drive.google.com/file/d/1K16TcPjNHW-XLdnmy4NMz-pN0h_o7ZC6/view?usp=sharing

## TensorRT Engines

Place inside:

```
engines/
```

Required files:

- yolopx.engine
- depth.engine
- traffic.engine

Download:

> [Google Drive Link Here](https://drive.google.com/file/d/1DY13nr8brxRBJUdySYhS2zZEFS6Xg9P1/view?usp=sharing)



## Triton Model Repository

Copy the downloaded models into

```
triton_model_repo/
```

and update the configuration if necessary.

---

## Installation

Clone the repository

```bash
git clone https://github.com/nabiha-alisher/Object-detection-and-lane-detection-for-autonomouse-vehicles.git

cd Object-detection-and-lane-detection-for-autonomouse-vehicles
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

Run the application

```bash
python app/main.py
```

---

## Experiments

The repository includes:

- Original FPS benchmark scripts
- Optimized FPS benchmark scripts
- Triton HTTP inference
- Triton gRPC inference
- Raw TensorRT inference
- Lane optimization verification

---

## Documentation

Additional documentation is available in

```
docs/
```

including

- RunPod setup
- FPS optimization summary
- Assistance logic
- Model setup instructions

---

## Notes

The following files are intentionally excluded from GitHub:

- *.engine
- *.onnx
- *.pth
- *.pt
- videos
- logs
- output folders
- datasets
- checkpoints

These files can be downloaded from the provided Google Drive links.

---

## Authors

Nabiha Khalid

Final Year Project

Department of Computer Engineering

University of Engineering and Technology, Taxila
