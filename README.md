# Semantic3D Point Cloud Segmentation with RandLA-Net

This is a minimal project for Semantic3D segmentation using RandLA-Net, based on the Open3D-ML implementation.

## Features

- Complete data preparation pipeline (download, preprocessing)
- Open3D-ML RandLA-Net implementation optimized for Semantic3D
- Training script
- Visualization tools (PLY conversion, label histograms)

## Requirements

- Python 3.10

## Notes on Evaluation

   The official Semantic3D test set is not publicly annotated.
   I have applied for the Point Cloud Classification Challenge access, but approval is still pending.

   Therefore, for evaluation purposes, I use a subset of the training set (the clouds bildstein_station1_xyz_intensity_rgb and domfountain_station1_xyz_intensity_rgb) as a validation set.

   The Open3D-ML pipeline.run_test implementation currently fails during evaluation (metric computation step) on Semantic3D.

   It correctly generates predictions, but crashes while calculating metrics.

   To overcome this, I implemented:

   A custom wrapper Semantic3DForEval that sets up a validation split as test for evaluation.

   A standalone evaluation script src/eval.py that calculates per-class IoU, per-class Accuracy, and global metrics from saved prediction files.

## Training artifacts

   Trained model checkpoint, logs, tensorboard event file, raw test predictions and class-colored test .ply can be accessed at following link: [gdrive](https://drive.google.com/drive/folders/180G0s2eyBpIvrE1DbcIdCp--wEAOOw0A?usp=sharing).


## Evaluation on *bildstein_station1_xyz_intensity_rgb* and *domfountain_station1_xyz_intensity_rgb*

| Class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| IoU | 0.9313 | 0.4435 | 0.3807 | 0.1689 | 0.8730 | 0.5703 | 0.5676 | 0.8861 |
| Accuracy | 0.9675 | 0.4787 | 0.6422 | 0.5121 | 0.8796 | 0.7899 | 0.7741 | 0.9000 |

| Metric | Value |
|---|---|
| Global IoU | 0.8080 |
| Global Accuracy | 0.8938 |

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
   poetry run python src/processing/preprocess_semantic3d.py
   ```

4. **Training**:
   ```bash
   poetry run python src/train.py
   ```

5. **Evaluatoin**:
   ```bash
   poetry run python src/test_inference.py
   poetry run python src/eval.py
   ```

## References

- [Open3D-ML](https://github.com/isl-org/Open3D-ML)
- [Semantic3D Dataset](http://www.semantic3d.net/)