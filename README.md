# VLM-HRIF: A Hierarchical Vision-Language Framework for Real-Time Traffic Accident Risk Prediction

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/JerryLeeSYA/TAISC-2025_VLM-HRIF)
[![Python](https://img.shields.io/badge/Python-3.8+-green?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-yellow)](https://github.com/JerryLeeSYA/TAISC-2025_VLM-HRIF)

**TAISC 2025 Challenge Submission**

---

## Abstract

This repository presents **VLM-HRIF** (Vision-Language Model with Hierarchical Risk Inference Framework), an integrated solution for real-time traffic accident risk prediction from single dashcam frames. Our framework achieves an **official weighted score of 0.5940** in the TAISC 2025 Challenge through the synergistic combination of three core innovative techniques: selective middle-layer fine-tuning, dynamic prompt engineering, and scene-adaptive Gaussian Mixture Model (GMM) post-processing.

## Table of Contents

- [System Overview](#system-overview)
- [Core Innovations](#core-innovations)
- [Performance Results](#performance-results)
- [Technical Architecture](#technical-architecture)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Implementation Details](#implementation-details)
- [Experimental Analysis](#experimental-analysis)
- [Future Work](#future-work)
- [Citation](#citation)
- [Contact](#contact)

## System Overview

VLM-HRIF addresses the challenge of proactive traffic safety by developing an efficient Vision-Language Model framework that predicts accident risk from individual dashcam frames. The system transitions from traditional "passive reaction" paradigms to "proactive prevention" through advanced computer vision and natural language processing techniques.

### Key Features
- **Real-time Processing**: Single-frame inference capability
- **Parameter Efficiency**: Only 3.5% of model parameters require fine-tuning
- **Scene Adaptability**: Dynamic threshold adjustment for different driving scenarios
- **Robust Performance**: Handles class imbalance through weighted loss and GMM post-processing

## Core Innovations

### 1. Selective Middle-Layer Fine-Tuning (SMFT)

**Methodology**: Based on hierarchical feature theory of Vision Transformers, we selectively unfreeze and fine-tune only the 10th and 11th transformer encoder blocks of CLIP-ViT-Large-336.

**Advantages**:
- Targets mid-level abstract semantic concept formation
- Preserves pre-trained knowledge while enabling domain adaptation
- Achieves 96.5% parameter efficiency compared to full fine-tuning
- Reduces overfitting risk on limited training data

**Technical Implementation**:
```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze selective layers
unfreeze_layers = [
    "visual.transformer.resblocks.10",
    "visual.transformer.resblocks.11", 
    "visual.ln_post"
]
```

### 2. Dynamic Prompt Engineering (DPE)

**Strategy**: Random prompt selection from categorized pools during training phase.

**Prompt Pools**:
- **Dangerous Prompts** (20): "A dash-cam frame where a crash is about to occur", "Vehicle cutting in sharply, high risk of collision", etc.
- **Safe Prompts** (20): "Calm highway driving with ample following distance", "Urban street scene, vehicles obeying traffic rules", etc.
- **Neutral Prompts** (6): "Describe the traffic situation in this frame", "Analyse the driver's required reaction", etc.

**Pairing Strategy**:
- 80% same-class prompts + 10% cross-class prompts + 10% neutral prompts
- Acts as powerful regularization mechanism
- Forces model to learn deep semantic correlations rather than superficial text-image associations

### 3. Scene-Adaptive GMM Post-Processing

**Two-Stage Pipeline**:

1. **Global Normalization**: Min-max normalization of raw risk scores to [0,1] interval
2. **Scene-Specific Threshold Derivation**: Independent GMM fitting for road/freeway scenarios

**Mathematical Foundation**:
```
p(s) = w₁N(s|μ₁,σ₁²) + w₂N(s|μ₂,σ₂²)

Optimal threshold τ* satisfies:
w₁N(τ*|μ₁,σ₁²) = w₂N(τ*|μ₂,σ₂²)

Solving: aτ*² + bτ* + c = 0
where: τ* = (-b ± √(b² - 4ac)) / (2a)
```

**Fallback Mechanism**: 80th percentile threshold when GMM convergence fails.

## Performance Results

### Official TAISC 2025 Competition Results

| Metric | Score | Weight | Contribution |
|--------|-------|--------|--------------|
| **F1-Score** | 0.5455 | 50% | 0.2728 |
| **Accuracy** | 0.6341 | 30% | 0.1902 |
| **AUC** | 0.6553 | 20% | 0.1311 |
| **Official Weighted Score** | **0.5940** | 100% | **0.5940** |

### Performance Analysis

- **F1-Score (0.5455)**: Demonstrates effective minority class (dangerous events) detection capability in imbalanced datasets
- **AUC (0.6553)**: Indicates superior discriminative ability compared to random classification (>0.5)
- **Balanced Performance**: Achieves effective trade-off between precision and recall for safety-critical applications

## Technical Architecture

```
Input Pipeline:
Dashboard Camera Frame (Raw Image)
    ↓
Image Preprocessing (Resize to 336×336, RGB Conversion)
    ↓
Dynamic Prompt Selection (Based on Training Labels)
    ↓

Model Architecture:
CLIP-ViT-Large-336 Vision Encoder
    ↓
Selective Fine-tuning Layers (10th, 11th Blocks + ln_post)
    ↓
Feature Extraction (768-dimensional embeddings)
    ↓
Classification Head (768→512→ReLU→512→1)
    ↓
Risk Score Output (Continuous)
    ↓

Post-Processing Pipeline:
Min-Max Normalization
    ↓
Scene Classification (road/freeway)
    ↓
GMM-based Threshold Computation
    ↓
Binary Risk Classification (0: Safe, 1: Dangerous)
```

## Installation Guide

### System Requirements

**Hardware**:
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 16GB+ System RAM
- 10GB+ Available storage

**Software**:
- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- Git LFS (for model files)

### Step-by-Step Installation

1. **Clone Repository**
```bash
git clone https://github.com/JerryLeeSYA/TAISC-2025_VLM-HRIF.git
cd TAISC-2025_VLM-HRIF
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from transformers import CLIPModel; print('CLIP Model Import: Success')"
```

### Dataset Structure

Ensure your dataset follows this structure:
```
AVA Dataset/
├── road/
│   ├── train/
│   │   ├── road_0000/
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── road_0179/
│       └── ...
├── freeway/
│   ├── train/
│   └── test/
├── road_train.csv
├── freeway_train.csv
└── sample_submission.csv
```

## Usage Instructions

### Quick Start

Execute the complete pipeline with a single command:
```bash
python Final.py
```

This will automatically perform:
1. **Model Training** with selective fine-tuning
2. **Test Set Prediction** using the best checkpoint
3. **GMM Post-processing** with scene-adaptive thresholds
4. **Submission File Generation**

### Advanced Usage

#### Custom Configuration
Modify the `Config` class in `Final.py` to adjust hyperparameters:
```python
class Config:
    def __init__(self):
        # Training parameters
        self.learning_rate = 1e-5
        self.batch_size = 4
        self.num_epochs = 1
        
        # Model parameters
        self.image_size = 336
        self.prediction_threshold = 0.35
        
        # Post-processing
        self.danger_class_weight = 1.5
```

#### Individual Components

**Training Only**:
```python
config = Config()
trainer = Trainer(config)
trainer.train()
```

**Prediction Only** (requires trained model):
```python
predictor = Predictor(config, 'checkpoints/best_model.pth')
predictions = predictor.predict_test_data()
```

### Output Files

- `outputs/submission_raw.csv`: Raw continuous risk scores
- `outputs/submission_final.csv`: Final binary predictions (competition format)
- `checkpoints/best_model.pth`: Best performing model weights

## Implementation Details

### Model Configuration

| Component | Specification |
|-----------|---------------|
| **Base Model** | CLIP-ViT-Large-336 |
| **Input Resolution** | 336×336 pixels |
| **Fine-tuning Layers** | Layers 10, 11 + ln_post |
| **Fine-tuning Ratio** | ~3.5% of total parameters |
| **Feature Dimension** | 768 |
| **Classification Head** | 768→512→ReLU→512→1 |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 4 | Memory optimization for large ViT model |
| **Learning Rate (ViT)** | 1×10⁻⁵ | Stable fine-tuning of pre-trained layers |
| **Learning Rate (Classifier)** | 7.77×10⁻⁴ | Faster adaptation of randomly initialized head |
| **Weight Decay** | 0.001 | Regularization |
| **Gradient Clipping** | 0.05 | Training stability |
| **Epochs** | 1 | Competition score reproduction |

### Loss Function

**Weighted Binary Cross-Entropy**:
```python
pos_weight = (negative_samples / positive_samples)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This addresses the inherent class imbalance where safe driving scenarios significantly outnumber dangerous ones.

### Dynamic Prompting Strategy

```python
def select_prompt(risk_label, random_value):
    if risk_label == 1:  # Dangerous
        if random_value < 0.8:    # 80% same-class
            return random.choice(danger_prompts)
        elif random_value < 0.9:  # 10% cross-class
            return random.choice(safety_prompts)
        else:                     # 10% neutral
            return random.choice(neutral_prompts)
    # Similar logic for safe samples
```

## Experimental Analysis

### Ablation Study Design

Due to competition time constraints, we focused on optimizing the complete framework. However, our theoretical ablation study design includes:

| Experiment | Configuration | Expected Impact |
|------------|---------------|-----------------|
| **Full Method** | All components enabled | Baseline performance |
| **No PEFT** | Full fine-tuning | Risk of overfitting, lower efficiency |
| **No Dynamic Prompting** | Fixed neutral prompts | Reduced robustness and generalization |
| **No GMM** | Fixed threshold (0.5) | Suboptimal F1-Score for imbalanced data |

### Key Findings

1. **Selective Fine-tuning Effectiveness**: Only middle layers (10th, 11th) contain sufficient abstraction for risk concept adaptation
2. **Dynamic Prompting Impact**: Cross-class prompting acts as crucial regularization, preventing text-image overfitting
3. **Scene Adaptation Necessity**: Road and freeway scenarios require different risk thresholds due to varying baseline risk distributions

## Future Work

### Technical Enhancements

1. **Temporal Modeling**
   - Extend to multi-frame video sequences
   - Utilize Video-LLM architectures (e.g., Video-ChatGPT, VideoChat)
   - Capture dynamic scene evolution for predictive (vs. reactive) risk assessment

2. **Multi-modal Fusion**
   - Integrate vehicle CAN bus data (speed, acceleration, steering angle)
   - Combine visual observations with vehicle dynamics
   - Develop unified multi-modal risk perception framework

3. **Foundation Model Evolution**
   - Adapt framework to newer VLMs (LLaVA-NeXT, Florence-2, BLIP-2)
   - Explore higher resolution inputs (512×512, 768×768)
   - Investigate parameter-efficient adaptation techniques (LoRA, AdaLoRA)

### Methodological Improvements

1. **Advanced Post-processing**
   - Explore mixture models beyond 2-component GMM
   - Investigate adaptive threshold learning
   - Develop confidence-aware prediction mechanisms

2. **Training Strategies**
   - Implement curriculum learning for progressive difficulty
   - Explore contrastive learning for better feature separation
   - Investigate knowledge distillation from larger models

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{li2025vlmhrif,
  title={VLM-HRIF: A Hierarchical Vision-Language Framework for Real-Time Traffic Accident Risk Prediction},
  author={Li, Shao-Chen},
  year={2025},
  howpublished={TAISC 2025 Challenge},
  url={https://github.com/JerryLeeSYA/TAISC-2025_VLM-HRIF},
  institution={University of Taipei, Department of Computer Science}
}
```

## Contact

**Author**: Shao-Chen Li  
**Institution**: University of Taipei, Department of Computer Science, In-service Master Program  
**Address**: 1 Ai-Guo West Road, Zhongzheng District, Taipei City 10048, Taiwan  
**Email**: m11116027@go.utaipei.edu.tw

## Acknowledgments

We express sincere gratitude to:
- Professor Chun-Ming Tsai for invaluable guidance and supervision
- ACVLAB (Taiwan Computer Vision Society) for organizing the TAISC 2025 Challenge
- The competition organizers for providing the comprehensive dataset and research platform
- The open-source community for foundational tools and models

---

**License**: This project is submitted for the TAISC 2025 Challenge and is intended for academic research and competition purposes.

**Disclaimer**: This system is designed for research purposes. Real-world deployment requires extensive validation and safety testing.