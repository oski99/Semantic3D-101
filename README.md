# Semantic3D Point Cloud Segmentation with RandLA-Net

This is a minimal project for Semantic3D segmentation using RandLA-Net, based on the Open3D-ML implementation.

## Features

- Complete data preparation pipeline (download, preprocessing)
- Open3D-ML RandLA-Net implementation optimized for Semantic3D
- Training and evaluation scripts
- Visualization tools (PLY conversion, label histograms)

## Requirements

- Python 3.10

## Usage

1. **Install dependencies** (using Poetry):
   ```bash
   poetry install
   ```

2. **Download the Semantic3D dataset**:
   ```bash
   ./download_semantic3d.sh
   ```

3. **Preprocess data (splitting and downsampling)**:
   ```bash
   poetry run python src/preprocess_semantic3d.py
   ```

4. **Training**:
   ```bash
   poetry run python src/train.py
   ```

## References

- [Open3D-ML](https://github.com/isl-org/Open3D-ML)
- [Semantic3D Dataset](http://www.semantic3d.net/)