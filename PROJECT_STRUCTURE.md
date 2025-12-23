# 📁 AI Virtual Try-On - Complete Folder Structure

## Project Directory Tree

```
AI-Virtual-TryOn/
│
├── 📓 notebooks/                          # Jupyter notebooks (main development area)
│   ├── README.md                          # Notebook organization guide
│   │
│   ├── 01_environment_setup.ipynb         # Environment setup & dependencies
│   ├── 02_data_exploration.ipynb          # Dataset exploration
│   ├── 03_data_preprocessing.ipynb        # Data preprocessing pipeline
│   │
│   ├── 04_human_parsing.ipynb             # Human body segmentation
│   ├── 05_pose_estimation.ipynb           # Pose detection
│   ├── 06_garment_processing.ipynb        # Garment feature extraction
│   │
│   ├── 07_tps_warping.ipynb               # TPS warping implementation
│   ├── 08_diffusion_model.ipynb           # Stable Diffusion integration
│   ├── 09_controlnet_training.ipynb       # ControlNet fine-tuning
│   │
│   ├── 10_model_training.ipynb            # Full training pipeline
│   ├── 11_refinement_network.ipynb        # Post-processing refinement
│   ├── 12_optimization.ipynb              # Model optimization
│   │
│   ├── 13_model_evaluation.ipynb          # Evaluation metrics
│   ├── 14_inference_demo.ipynb            # Single image inference
│   ├── 15_model_comparison.ipynb          # Multi-model comparison
│   │
│   ├── 16_multi_garment_tryon.ipynb       # Multiple garment try-on
│   ├── 17_ar_integration.ipynb            # AR mobile integration
│   └── 18_web_app_demo.ipynb              # Web application demo
│
├── 💾 data/                               # Dataset storage
│   ├── README.md                          # Dataset documentation
│   │
│   ├── raw/                               # Original datasets
│   │   ├── .gitkeep
│   │   ├── viton-hd/                      # VITON-HD dataset
│   │   ├── deepfashion/                   # DeepFashion dataset
│   │   └── custom/                        # Custom e-commerce data
│   │
│   ├── processed/                         # Preprocessed data
│   │   ├── .gitkeep
│   │   ├── images/                        # Processed images
│   │   ├── masks/                         # Segmentation masks
│   │   └── poses/                         # Pose data
│   │
│   ├── train/                             # Training split
│   │   ├── .gitkeep
│   │   ├── person/                        # Person images
│   │   ├── garment/                       # Garment images
│   │   ├── parse/                         # Parsing masks
│   │   ├── pose/                          # Pose keypoints
│   │   └── pairs.txt                      # Image pairs
│   │
│   ├── test/                              # Test split
│   │   ├── .gitkeep
│   │   └── [same structure as train/]
│   │
│   ├── validation/                        # Validation split
│   │   ├── .gitkeep
│   │   └── [same structure as train/]
│   │
│   └── models/                            # Model-specific data
│       ├── .gitkeep
│       ├── slim_female/                   # Model database
│       ├── athletic_female/
│       ├── plus_size_female/
│       ├── slim_male/
│       ├── athletic_male/
│       └── plus_size_male/
│
├── 🤖 models/                             # Model weights & configs
│   ├── README.md                          # Model documentation
│   │
│   ├── checkpoints/                       # Training checkpoints
│   │   ├── .gitkeep
│   │   ├── tps_warping/
│   │   ├── garment_encoder/
│   │   ├── refinement_net/
│   │   └── best_model.pth
│   │
│   ├── pretrained/                        # Pretrained weights
│   │   ├── .gitkeep
│   │   ├── stable-diffusion-xl/          # SD-XL weights
│   │   ├── controlnet/                    # ControlNet weights
│   │   ├── graphonomy/                    # Human parser
│   │   └── densepose/                     # Pose estimator
│   │
│   └── configs/                           # Model configurations
│       ├── .gitkeep
│       ├── train_config.yaml              # Training config
│       ├── inference_config.yaml          # Inference config
│       └── model_architectures.yaml       # Architecture specs
│
├── 🔧 src/                                # Source code modules
│   ├── README.md                          # Code documentation
│   ├── __init__.py
│   │
│   ├── preprocessing/                     # Preprocessing modules
│   │   ├── __init__.py
│   │   ├── human_parser.py                # Body segmentation
│   │   ├── pose_estimator.py              # Pose detection
│   │   ├── garment_processor.py           # Garment processing
│   │   └── data_augmentation.py           # Data augmentation
│   │
│   ├── models/                            # Model architectures
│   │   ├── __init__.py
│   │   ├── tps_warping.py                 # TPS warping module
│   │   ├── garment_encoder.py             # Feature extraction
│   │   ├── diffusion_vton.py              # Diffusion VTON
│   │   ├── refinement_net.py              # Refinement network
│   │   └── controlnet_wrapper.py          # ControlNet wrapper
│   │
│   ├── training/                          # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                     # Training loop
│   │   ├── dataset.py                     # Dataset classes
│   │   ├── losses.py                      # Loss functions
│   │   └── callbacks.py                   # Training callbacks
│   │
│   ├── inference/                         # Inference pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                    # Main pipeline
│   │   ├── optimization.py                # Model optimization
│   │   └── batch_processing.py            # Batch inference
│   │
│   └── utils/                             # Utility functions
│       ├── __init__.py
│       ├── visualization.py               # Visualization tools
│       ├── metrics.py                     # Evaluation metrics
│       ├── config.py                      # Config management
│       ├── image_utils.py                 # Image helpers
│       └── logger.py                      # Logging utilities
│
├── 📊 outputs/                            # Generated outputs
│   ├── results/                           # Try-on results
│   │   ├── .gitkeep
│   │   └── [generated images]
│   │
│   ├── visualizations/                    # Visualizations
│   │   ├── .gitkeep
│   │   ├── training_curves/
│   │   ├── attention_maps/
│   │   └── comparison_grids/
│   │
│   ├── comparisons/                       # Model comparisons
│   │   ├── .gitkeep
│   │   └── [comparison results]
│   │
│   └── metrics/                           # Evaluation metrics
│       ├── .gitkeep
│       ├── ssim_scores.json
│       ├── lpips_scores.json
│       └── fid_scores.json
│
├── 🌐 web/                                # Web application
│   ├── static/                            # Static assets
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   ├── templates/                         # HTML templates
│   │   ├── index.html
│   │   └── results.html
│   │
│   └── gradio_app.py                      # Gradio interface
│
├── 🔌 api/                                # REST API
│   ├── main.py                            # FastAPI application
│   ├── routes/                            # API routes
│   │   ├── tryon.py
│   │   └── comparison.py
│   │
│   └── schemas/                           # Pydantic schemas
│       └── models.py
│
├── 🧪 tests/                              # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_inference.py
│
├── 📜 scripts/                            # Utility scripts
│   ├── download_datasets.py               # Download datasets
│   ├── download_models.py                 # Download pretrained models
│   ├── prepare_dataset.py                 # Dataset preparation
│   ├── train.py                           # Training script
│   ├── evaluate.py                        # Evaluation script
│   └── inference.py                       # Inference script
│
├── 🖼️ examples/                           # Example images
│   ├── person1.jpg
│   ├── person2.jpg
│   ├── shirt1.jpg
│   ├── dress1.jpg
│   └── README.md
│
├── 📄 Configuration Files
│   ├── .gitignore                         # Git ignore rules
│   ├── requirements.txt                   # Python dependencies
│   ├── setup.py                           # Package setup
│   ├── README.md                          # Project documentation
│   └── LICENSE                            # License file
│
└── 🐳 Docker & Deployment (Optional)
    ├── Dockerfile                         # Docker image
    ├── docker-compose.yml                 # Docker compose
    └── deployment/                        # Deployment configs
        ├── aws/
        └── kubernetes/
```

## 📋 Directory Purposes

### 🎯 Core Development (Jupyter Notebooks)
- **notebooks/** - Main development area with 18 organized notebooks
- All experiments, training, and testing done here
- Self-contained with markdown explanations

### 💾 Data Management
- **data/raw/** - Original datasets (VITON-HD, DeepFashion)
- **data/processed/** - Preprocessed images, masks, poses
- **data/train/test/validation/** - Split datasets for training
- **data/models/** - Model database for comparison

### 🤖 Model Storage
- **models/checkpoints/** - Your trained model weights
- **models/pretrained/** - Downloaded pretrained models (~10GB)
- **models/configs/** - YAML configuration files

### 🔧 Reusable Code
- **src/** - Modular Python code imported in notebooks
- Organized by functionality (preprocessing, models, training, etc.)
- Each module has `__init__.py` for easy imports

### 📊 Outputs
- **outputs/results/** - Generated try-on images
- **outputs/visualizations/** - Training curves, attention maps
- **outputs/comparisons/** - Multi-model comparison results
- **outputs/metrics/** - Evaluation scores (SSIM, LPIPS, FID)

### 🌐 Web & API
- **web/** - Gradio/Streamlit web interface
- **api/** - FastAPI REST API for production

### 🧪 Testing & Scripts
- **tests/** - Unit tests for code validation
- **scripts/** - Standalone utility scripts
- **examples/** - Sample images for quick testing

## 🚀 Workflow

1. **Setup**: Run `notebooks/01_environment_setup.ipynb`
2. **Data**: Download and preprocess using notebooks 02-03
3. **Development**: Build models using notebooks 04-12
4. **Training**: Train full pipeline in notebook 10
5. **Evaluation**: Evaluate using notebooks 13-15
6. **Deployment**: Create web app using notebook 18

## 💡 Key Features

✅ **Notebook-Centric**: All development in Jupyter notebooks  
✅ **Modular Code**: Reusable modules in `src/`  
✅ **Organized Data**: Clear separation of raw/processed/split data  
✅ **Version Control**: `.gitignore` excludes large files  
✅ **Documentation**: README in every major directory  
✅ **Scalable**: Easy to extend with new notebooks/modules  

## 📝 Notes

- Large files (models, datasets) are gitignored
- `.gitkeep` files preserve empty directories in git
- All paths are relative to project root
- Notebooks are numbered for sequential workflow
- Each directory has its own README for details
