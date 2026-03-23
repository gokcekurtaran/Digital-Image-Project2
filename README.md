# Digital-Image-Project2

# Digital Image Feature Extraction & Fusion Project
A Python-based digital image processing project that performs **multi-level feature extraction and fusion** using spatial and wavelet domain techniques.


## Overview
This project processes a labeled image dataset and extracts a wide range of features including **texture, shape, color, and gradient-based descriptors**.
It combines multiple feature extraction methods and generates different feature sets for machine learning experiments, exported as `.ARFF` files compatible with **WEKA**.

## Features
- Automatic dataset scanning (folder-based labeling)
- Image preprocessing:
  - Resize (256×256)
  - Grayscale conversion
  - Gaussian noise addition
- Wavelet transform (DWT - db2)
- Multi-feature extraction:
  - Texture, shape, color, and gradient features
- Feature fusion (combining multiple descriptors)
- Multiple ARFF dataset generation for experimentation


##  Extracted Features

### Texture Features
- **GLCM (Gray-Level Co-occurrence Matrix)** (from wavelet LL band)

### Color Features
- HSV mean values (H, S, V)

### Shape Features
- Hu Moments (log-transformed)
- Area-based features:
  - Area
  - Perimeter
  - Aspect Ratio
  - Compactness
  - Extent
  - Solidity

### Local Texture Descriptors
- LBP (Local Binary Pattern histogram)
- LCP (Local Contrast Pattern histogram)

### Gradient Features
- PHOG (Pyramid Histogram of Oriented Gradients)

### Histogram Features
- Soft histogram (Gaussian-smoothed intensity distribution)


## Feature Fusion

Different feature combinations are created for comparison:

- GLCM  
- GLCM + LBP  
- GLCM + LBP + PHOG  
- Wavelet + GLCM + HSV + HU  
- Wavelet + GLCM + HSV + HU + LBP  
- Wavelet + GLCM + HSV + HU + LBP + PHOG  
- All features combined  



## Dataset Structure

images/
├── Class1/
│   ├── img1.jpg
│   ├── img2.png
├── Class2/
│   ├── img3.jpg
│   ├── img4.png

Folder names are automatically used as **class labels**.


## Getting Started

### 1. Requirements
pip install opencv-python numpy scikit-image pywavelets


### 2. Update Dataset Path

Modify the dataset path in the code:
DATASET_PATH = "your_dataset_path"


### 3. Run the Script
python Kod.py

### Output

The project generates multiple `.ARFF` files:
- `glcm.arff`
- `glcm_lbp.arff`
- `glcm_lbp_phog.arff`
- `wavelet_glcm_hsv_hu.arff`
- `wavelet_glcm_hsv_hu_lbp.arff`
- `wavelet_glcm_hsv_hu_lbp_phog.arff`
- `wavelet_all_features.arff`

These datasets can be directly used in **WEKA or other ML tools**.


## Workflow
1. Load dataset from folders  
2. Preprocess images  
3. Apply noise + wavelet transform  
4. Extract multiple features  
5. Perform feature fusion  
6. Export datasets for ML  


## Learning Outcomes
- Image preprocessing techniques  
- Wavelet-based feature extraction  
- Texture analysis (GLCM, LBP, LCP)  
- Shape analysis (Hu Moments, contour features)  
- Gradient-based descriptors (PHOG)  
- Feature fusion strategies  
- Dataset preparation for machine learning  

## Notes
- Supports `.jpg`, `.jpeg`, `.png` formats :contentReference[oaicite:0]{index=0}  
- Uses Gaussian noise for robustness testing  
- Generates multiple feature sets for comparison  


## Purpose
This project was developed for a **Digital Image Processing course** to explore advanced feature extraction and fusion techniques in real-world datasets.
