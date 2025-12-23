# Source Code Directory

This directory contains modular Python code for the AI Virtual Try-On system.

## 📂 Directory Structure

```
src/
├── preprocessing/   # Data preprocessing modules
├── models/          # Model architectures
├── training/        # Training utilities
├── inference/       # Inference pipeline
└── utils/           # Helper functions
```

## 🔧 Module Descriptions

### preprocessing/
- `human_parser.py` - Human body segmentation
- `pose_estimator.py` - Pose detection and keypoints
- `garment_processor.py` - Garment extraction and processing
- `data_augmentation.py` - Data augmentation utilities

### models/
- `tps_warping.py` - Thin-Plate Spline warping module
- `garment_encoder.py` - Garment feature extraction
- `diffusion_vton.py` - Diffusion-based VTON model
- `refinement_net.py` - Post-processing refinement
- `controlnet_wrapper.py` - ControlNet integration

### training/
- `trainer.py` - Main training loop
- `dataset.py` - Dataset classes and loaders
- `losses.py` - Loss functions
- `callbacks.py` - Training callbacks

### inference/
- `pipeline.py` - End-to-end inference pipeline
- `optimization.py` - Model optimization (TensorRT, ONNX)
- `batch_processing.py` - Batch inference utilities

### utils/
- `visualization.py` - Visualization utilities
- `metrics.py` - Evaluation metrics
- `config.py` - Configuration management
- `image_utils.py` - Image processing helpers
- `logger.py` - Logging utilities

## 🚀 Usage

Import modules in your notebooks:

```python
from src.preprocessing.human_parser import HumanParser
from src.models.diffusion_vton import VTONDiffusionModel
from src.inference.pipeline import VTONPipeline
```

## 📝 Notes

- All modules are designed to be imported in Jupyter notebooks
- Each module is self-contained and well-documented
- Use `__init__.py` files for package imports
