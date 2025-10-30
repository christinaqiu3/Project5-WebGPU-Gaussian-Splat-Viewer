# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Christina Qiu
  * [LinkedIn](https://www.linkedin.com/in/christina-qiu-6094301b6/), [personal website](https://christinaqiu3.github.io/), [twitter](), etc.
* Tested on: Windows 11, Intel Core i7-13700H @ 2.40GHz, 16GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (Personal laptop)

## Overview

This project implements a GPU-driven 3D Gaussian splatting renderer using WebGPU. The goal is to take a set of 3D Gaussians and render high-quality, order-independent, depth-sorted 2D splats (quads) on the screen. The implementation follows the assignment stages: loading scene and camera data, preprocessing Gaussians on the GPU (culling, covariance → 2D conic projection, color evaluation), sorting by depth, and rendering the resulting splats with an indirect draw call.

### Live Demo

[![](<Screenshot 2025-10-29 235258.png>)](https://christinaqiu3.com/Project5-WebGPU-Gaussian-Splat-Viewer/)

### Demo Video/GIF

[![](img/hw_5_1.gif)](TODO)

## Performance Analysis

### Comparing point-cloud and gaussian renderer

The point-cloud renderer displays each point as a simple dot or quad with uniform size and color, producing a sparse and noisy appearance without smooth transitions. It lacks depth-dependent opacity, shading, or blending between nearby points.

The Gaussian renderer, produces smooth, continuous surfaces because each point is rendered as an anisotropic 2D Gaussian ellipse instead of a fixed-size point. The splats blend together, giving a more realistic and soft appearance, especially around regions with dense point samples. 

### Workgroup size affecting performance

insert graphs

### View-frustum culling give performance improvement

insert graphs

### Number of guassians affects performance

insert graphs

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
