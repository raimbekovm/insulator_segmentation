# Computer Vision Course Projects

A comprehensive collection of computer vision projects covering semantic segmentation, object detection, image classification, and deep learning techniques.

## 👨‍💻 About

This repository contains practical implementations and projects completed during an intensive computer vision course at AUCA. Each project demonstrates state-of-the-art computer vision techniques using PyTorch, with a focus on real-world applications.

## 📋 CV / Resume

**[📄 Download CV (PDF)](./Murat_Raimbekov_CV.pdf)**

---

## 🚀 Projects

### 1. Insulator Segmentation
**Semantic segmentation of power line insulators using U-Net**

- **Task**: Segment insulators on aerial power line photos captured by UAVs
- **Architecture**: U-Net with ResNet34 encoder
- **Performance**: **0.9895 Dice coefficient**
- **Techniques**: Transfer Learning, Data Augmentation, Test-Time Augmentation
- **Framework**: PyTorch, Albumentations

[View Project →](./01_insulator_segmentation)

---

### 2. Grain Classification
**Multi-class image classification for grain type identification**

- **Task**: Classify grain images into 4 categories (barley, flax, oats, wheat)
- **Architecture**: EfficientNetV2, ConvNeXt (ensemble)
- **Techniques**: 5-fold Cross-Validation, Test-Time Augmentation, Mixed Precision Training
- **Framework**: PyTorch, timm library
- **Key Features**: Ensemble methods, learning rate scheduling, data augmentation

[View Project →](./02_grain_classification)

---

## 📚 Seminar Assignments

Practical notebooks implementing various computer vision and deep learning concepts:

### Neural Networks & Deep Learning
- **Seminar 1**: Introduction to Computer Vision and PyTorch basics
- **Seminar 2 Part 1**: Gradient descent and optimization algorithms
- **Seminar 2 Part 2**: Wine classification with neural networks
- **Seminar 3**: Advanced classification techniques

### Convolutional Neural Networks
- **Seminar 5 Part 1**: CNN fundamentals and forward pass implementation
- **Seminar 5 Part 2**: Dense and convolutional neural networks
- **Seminar 5 Part 3**: AlexNet implementation and activation map visualization

### Advanced Topics
- **Seminar 6 Part 1**: Advanced CNN architectures
- **Seminar 6 Part 2**: Object detection techniques
- **Seminar 8**: Deep learning optimization and regularization
- **Seminar 10 Part 1**: Semantic segmentation (U-Net, SegNet, loss functions)

---

## 🗂️ Repository Structure

```
computer_vision_course/
├── README.md                        # This file
├── Murat_Raimbekov_CV.pdf          # Resume/CV
├── requirements.txt                 # Common dependencies
│
├── Seminar_1.ipynb                  # Seminar notebooks
├── Seminar_2_part1.ipynb
├── Seminar_2_part2.ipynb
├── Seminar_3.ipynb
├── Seminar_5_part1.ipynb           # CNN implementation
├── Seminar_5_part2.ipynb
├── Seminar_5_part3.ipynb
├── Seminar_6_part1.ipynb
├── Seminar_6_part2.ipynb
├── Seminar_8.ipynb
├── Seminar_10_part1.ipynb          # Segmentation
│
├── 01_insulator_segmentation/       # Project 1
│   ├── insulator_segmentation.ipynb
│   ├── requirements.txt
│   ├── README.md
│   └── samples/
│
└── 02_grain_classification/         # Project 2
    └── grain_classification.ipynb
```

---

## 🛠️ Technologies & Tools

### Deep Learning Frameworks
- **PyTorch** - Primary framework for all models
- **TorchVision** - Pre-trained models and transforms
- **timm** - PyTorch Image Models library
- **scikit-learn** - ML utilities and metrics

### Computer Vision
- **OpenCV** - Image processing and manipulation
- **Albumentations** - Advanced data augmentation
- **scikit-image** - Image processing algorithms

### Data Science & Visualization
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical visualizations

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git/GitHub** - Version control
- **CUDA** - GPU acceleration

---

## 🎯 Key Skills Demonstrated

### Computer Vision Techniques
- ✅ Semantic Segmentation (U-Net, SegNet)
- ✅ Image Classification (CNN, EfficientNet, ConvNeXt)
- ✅ Object Detection (YOLO - see hackathon project)
- ✅ Transfer Learning
- ✅ Data Augmentation & Preprocessing

### Deep Learning Best Practices
- ✅ Model Training & Optimization
- ✅ Cross-Validation & Ensemble Methods
- ✅ Test-Time Augmentation
- ✅ Mixed Precision Training
- ✅ Learning Rate Scheduling
- ✅ Loss Function Design (BCE, Dice, Focal)
- ✅ Performance Evaluation & Metrics

### Implementation Skills
- ✅ PyTorch Model Architecture Design
- ✅ Custom Dataset & DataLoader Implementation
- ✅ Training Pipeline Development
- ✅ Ablation Studies & Experimentation
- ✅ Result Visualization & Analysis

---

## 🏆 Related Projects

### Computer Vision Road Defects Detection
**Yandex Hackathon Project - December 2024**

Automated road defect detection system using YOLOv8 for real-time object detection on highway images.

- **Achievement**: 85%+ mAP on test set
- **Technologies**: YOLOv8, PyTorch, OpenCV, Albumentations
- **Repository**: [hackathon-urban-tech](https://github.com/raimbekovm/hackathon-urban-tech)

---

## 📦 Installation

### General Requirements

```bash
pip install -r requirements.txt
```

### Project-Specific Installation

Each project has its own dependencies. Navigate to the specific project directory:

```bash
cd 01_insulator_segmentation
pip install -r requirements.txt
```

### GPU Support

For CUDA/GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Running Jupyter Notebooks

```bash
jupyter notebook Seminar_1.ipynb
```

### Running Project Notebooks

```bash
cd 01_insulator_segmentation
jupyter notebook insulator_segmentation.ipynb
```

---

## 📊 Highlights & Results

| Project | Task | Best Model | Metric | Score |
|---------|------|------------|--------|-------|
| Insulator Segmentation | Semantic Segmentation | U-Net + ResNet34 | Dice Coefficient | **0.9895** |
| Medical Image Segmentation | Skin Lesion Detection | SegNet + BCE | IoU | **0.654** |
| Grain Classification | Multi-class Classification | Ensemble (EfficientNetV2 + ConvNeXt) | F1-Score | High |
| Road Defects Detection | Object Detection | YOLOv8 | mAP | **85%+** |

---

## 📧 Contact

**Murat Raimbekov**
Data Science & Computer Vision Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/murat-raimbekov)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/raimbekovm)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:raimbekov_m@auca.kg)

---

## 📜 License

This repository is for educational and portfolio purposes.

## 🙏 Acknowledgments

- American University of Central Asia (AUCA) - Neural Networks and Deep Learning Course
- Course instructor
- Open-source community (PyTorch, OpenCV, Albumentations)

---
