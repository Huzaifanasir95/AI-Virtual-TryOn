# Models Directory

This directory contains model checkpoints, pretrained weights, and configuration files.

## 📂 Directory Structure

```
models/
├── checkpoints/     # Training checkpoints and saved models
├── pretrained/      # Pretrained model weights
└── configs/         # Model configuration files
```

## 🤖 Model Components

### Core Models

1. **Stable Diffusion XL**
   - Base generative model
   - Location: `pretrained/stable-diffusion-xl/`
   - Size: ~6.9GB

2. **ControlNet**
   - Pose conditioning model
   - Location: `pretrained/controlnet/`
   - Size: ~1.4GB

3. **Human Parser (Graphonomy)**
   - Semantic segmentation
   - Location: `pretrained/graphonomy/`
   - Size: ~200MB

4. **DensePose**
   - Pose estimation
   - Location: `pretrained/densepose/`
   - Size: ~250MB

### Custom Models

5. **TPS Warping Module**
   - Garment alignment
   - Location: `checkpoints/tps_warping/`

6. **Refinement Network**
   - Post-processing
   - Location: `checkpoints/refinement/`

7. **Garment Encoder**
   - Feature extraction
   - Location: `checkpoints/garment_encoder/`

## 📥 Downloading Pretrained Models

Use the download script:
```bash
python scripts/download_models.py
```

Or download manually from:
- Stable Diffusion: [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- ControlNet: [Hugging Face](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose)
- Graphonomy: [GitHub](https://github.com/Gaoyiminggithub/Graphonomy)
- DensePose: [Detectron2](https://github.com/facebookresearch/detectron2)

## 💾 Storage Requirements

- **Pretrained Models**: ~10GB
- **Checkpoints**: ~5GB (during training)
- **Total**: ~15GB

## 🔧 Configuration Files

Configuration files in `configs/` define:
- Model architectures
- Hyperparameters
- Training settings
- Inference parameters

Example: `configs/train_config.yaml`

## 📝 Notes

- Model files are excluded from git (see `.gitignore`)
- Checkpoints are saved every N epochs during training
- Best model is saved based on validation metrics
