# GaussianSemantic_app

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%2B-brightgreen.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15.4-blue.svg)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

**GaussianSemantic_app** is a powerful PyQt5-based graphical user interface (GUI) application designed for loading, rendering, and training on point cloud data using Gaussian models. It provides intuitive controls for camera positioning and orientation, allowing users to visualize and manipulate 3D point clouds seamlessly. Whether you're a researcher, developer, or enthusiast working with 3D data, this application offers a user-friendly platform to explore and enhance your projects.

## Features

- **Load Point Cloud**: Import point cloud data from PLY files.
- **Real-Time Rendering**: Visualize point clouds with customizable camera positions and orientations.
- **Camera Controls**: Adjust camera position using Euler angles (Yaw, Pitch, Roll) for intuitive navigation.
- **Training Module**: Simulate training processes with progress tracking.
- **Multi-threaded Rendering**: Efficient rendering operations handled in the background.
- **User-Friendly UI**: Clean and responsive interface built with PyQt5.
- **Future Enhancements**:
  - **Semantic Segmentation Queries**: Upcoming feature to perform and visualize semantic segmentation on point clouds.


## Installation

### Prerequisites

- **Python 3.12 or higher**
- **CUDA-enabled GPU** (optional, for accelerated rendering)
- **diff_gaussian_rasterization** library

### Clone the Repository

```bash
git clone https://github.com/yourusername/GaussianSemantic_app.git
cd GaussianSemantic_app
```

### Install Dependencies

1. **Install `diff_gaussian_rasterization`**

   Before installing other dependencies, ensure that ![diff_gaussian_rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) is installed. 

2. **Install the required Python libraries**

```bash
pip install -r requirements.txt
```
The requirements.txt includes the following key libraries:

- **PyQt5**: For building the graphical user interface.
- **NumPy**: For numerical operations and data manipulation.
- **PyTorch**: For machine learning and deep learning model training.
- **Open3D**: For point cloud processing and visualization.
- **Matplotlib**: For plotting and visualization.

## Usage

### Running the Application

After installing all dependencies, you can start the application by running the following command:

```bash
python main.py
```

### Camera Control
Use the sliders and input fields to control the camera's position and orientation. The camera's position is represented by three values: X, Y, and Z, while the orientation is given by Yaw, Pitch, and Roll (Euler angles).

- **Load Point Cloud**: Load a point cloud file (e.g., .ply format) for rendering.
- **Render**: Display the loaded point cloud in the 3D viewer.
- **Start Training**: Begin the training process with the current point cloud data.
### Adjusting Camera Position and Orientation
- **Position**: You can control the camera's position in 3D space by adjusting the X, Y, and Z coordinates.
- **Orientation**: You can adjust the camera's orientation using Euler angles, represented as Yaw, Pitch, and Roll. These angles define the rotation of the camera in 3D space.

## License
This project is licensed under the MIT License
