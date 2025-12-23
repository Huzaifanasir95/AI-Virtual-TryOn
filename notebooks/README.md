# Notebooks Directory

This directory contains Jupyter notebooks for the AI Virtual Try-On project, organized by development phase.

## 📓 Notebook Structure

### Phase 1: Setup & Data Preparation
- `01_environment_setup.ipynb` - Environment configuration and dependency installation
- `02_data_exploration.ipynb` - Dataset exploration and visualization
- `03_data_preprocessing.ipynb` - Data preprocessing and augmentation

### Phase 2: Model Development
- `04_human_parsing.ipynb` - Human body segmentation and parsing
- `05_pose_estimation.ipynb` - Pose detection and keypoint extraction
- `06_garment_processing.ipynb` - Garment feature extraction and warping

### Phase 3: Core VTON Models
- `07_tps_warping.ipynb` - Thin-Plate Spline warping implementation
- `08_diffusion_model.ipynb` - Stable Diffusion integration
- `09_controlnet_training.ipynb` - ControlNet fine-tuning

### Phase 4: Training & Optimization
- `10_model_training.ipynb` - Full model training pipeline
- `11_refinement_network.ipynb` - Post-processing and refinement
- `12_optimization.ipynb` - Model optimization and acceleration

### Phase 5: Evaluation & Inference
- `13_model_evaluation.ipynb` - Quantitative evaluation metrics
- `14_inference_demo.ipynb` - Single image inference demo
- `15_model_comparison.ipynb` - Multi-model comparison system

### Phase 6: Advanced Features
- `16_multi_garment_tryon.ipynb` - Multiple garment try-on
- `17_ar_integration.ipynb` - AR mobile integration
- `18_web_app_demo.ipynb` - Web application demo

## 🚀 Getting Started

1. Start with `01_environment_setup.ipynb` to set up your environment
2. Follow the notebooks in order for a complete walkthrough
3. Each notebook is self-contained with explanations and code

## 💡 Usage Tips

- Run notebooks in order for first-time setup
- Each notebook saves outputs to the `outputs/` directory
- Model checkpoints are saved to `models/checkpoints/`
- Use GPU for faster training and inference

## 📊 Expected Outputs

- Preprocessed datasets in `data/processed/`
- Trained models in `models/checkpoints/`
- Visualizations in `outputs/visualizations/`
- Evaluation metrics in `outputs/metrics/`
