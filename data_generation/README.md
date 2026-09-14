# Data Preparation Interfaces

This directory contains code only. It does not distribute raw channel data.

- `generate_extrapolation_v3.m`: QuaDRiGa pretraining and axis-scale tests.
- `generate_controlled_quadriga_diagnostics.m`: controlled speed, delay, and
  angle diagnostics.
- `deepmimo/`: DeepMIMO v4 CSI-to-beam preparation entry points.
- `mamimo_uav/`: MaMIMO-UAV recording split, window, row-crop, and mask
  preparation entry point.
Every public script accepts a user-supplied input and output location. Dataset
licenses and download conditions remain with their respective providers.
