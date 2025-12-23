# Data Directory

This directory contains all datasets used for training and evaluation.

## 📂 Directory Structure

```
data/
├── raw/              # Original, unprocessed datasets
├── processed/        # Preprocessed and augmented data
├── train/            # Training dataset
├── test/             # Test dataset
├── validation/       # Validation dataset
└── models/           # Model-specific data (pose, segmentation, etc.)
```

## 📥 Required Datasets

### 1. VITON-HD Dataset
- **Size**: ~13,679 image pairs
- **Resolution**: 1024x768
- **Download**: [VITON-HD GitHub](https://github.com/shadow2496/VITON-HD)
- **Location**: `data/raw/viton-hd/`

### 2. DeepFashion Dataset
- **Size**: 800K+ images
- **Categories**: Multiple clothing types
- **Download**: [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- **Location**: `data/raw/deepfashion/`

### 3. Custom E-Commerce Dataset (Optional)
- **Purpose**: Domain-specific fine-tuning
- **Location**: `data/raw/custom/`

## 📋 Data Format

### Training Data Structure
```
train/
├── person/          # Person images (1024x768)
├── garment/         # Garment images
├── parse/           # Segmentation masks
├── pose/            # Pose keypoints (JSON)
└── pairs.txt        # Image pair mappings
```

## 🔧 Preprocessing Steps

1. **Image Resizing**: Standardize to 512x384 or 1024x768
2. **Human Parsing**: Generate segmentation masks
3. **Pose Estimation**: Extract keypoints
4. **Data Augmentation**: Rotation, flipping, color jittering

## 💾 Storage Requirements

- **Raw Data**: ~50GB
- **Processed Data**: ~30GB
- **Total**: ~80GB minimum

## 📝 Notes

- Large files are excluded from git (see `.gitignore`)
- Download datasets using `scripts/download_datasets.py`
- Preprocess data using `notebooks/03_data_preprocessing.ipynb`
