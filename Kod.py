import os
import cv2
import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.feature import hog


def extract_glcm_features(image_64):
  
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        image_64,
        distances=distances,
        angles=angles,
        levels=64,         
        symmetric=True,
        normed=True
    )

    props = ['contrast', 'dissimilarity', 'homogeneity',
             'energy', 'correlation', 'ASM']

    return [graycoprops(glcm, p).mean() for p in props]


def extract_hsv_mean(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return [np.mean(h), np.mean(s), np.mean(v)]


def extract_hu_moments(gray_image):
    moments = cv2.moments(gray_image)
    hu = cv2.HuMoments(moments).flatten()

    hu = [-np.sign(h) * np.log10(abs(h)) if h != 0 else 0 for h in hu]
    return hu

             #Area & Shape Features

def extract_shape_features(gray_image):
   
    # Otsu threshold -> binary
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    if np.sum(th == 255) < np.sum(th == 0):
        th = cv2.bitwise_not(th)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0, 0, 0, 0, 0, 0]

    c = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / float(h) if h != 0 else 0.0

    # compactness: 4*pi*A / P^2
    compactness = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter != 0 else 0.0

    # extent: A / (w*h)
    extent = (area / float(w * h)) if (w * h) != 0 else 0.0

    # solidity: A / convex_area
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area != 0 else 0.0

    return [area, perimeter, aspect_ratio, compactness, extent, solidity]


def extract_lbp_hist(gray_image, P=8, R=1, bins=16):
    
    lbp = local_binary_pattern(gray_image, P, R, method="uniform")
    
    # normalize LBP values to histogram bins
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, lbp.max() + 1))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()


def extract_lcp_hist(gray_image, ksize=3, bins=16):
  
    img = gray_image.astype(np.float32)

    kernel = np.ones((ksize, ksize), np.uint8)
    local_min = cv2.erode(img, kernel)
    local_max = cv2.dilate(img, kernel)
    contrast = local_max - local_min  # 0..255 approx

    hist, _ = np.histogram(contrast.ravel(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()

def extract_phog_features(gray_image,
                          levels=2,
                          orientations=9,
                          pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2)):
    
    phog_features = []
    h, w = gray_image.shape

    for level in range(levels + 1):
        num_cells = 2 ** level
        cell_h = h // num_cells
        cell_w = w // num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                cell = gray_image[
                    i * cell_h:(i + 1) * cell_h,
                    j * cell_w:(j + 1) * cell_w
                ]

                hog_feat = hog(
                    cell,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    block_norm='L2-Hys',
                    visualize=False,
                    feature_vector=True
                )

                phog_features.extend(hog_feat)

    return phog_features

def extract_soft_histogram(gray_image, bins=16, sigma=5):
   
    # Normal histogram
    hist, bin_edges = np.histogram(
        gray_image.ravel(),
        bins=bins,
        range=(0, 256),
        density=False
    )

    # Gaussian smoothing
    hist = hist.astype(np.float32)
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 5), sigma).flatten()

    # Normalize
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()


# Dataset path
DATASET_PATH = r"C:\Cmp447Final\images"

# Gaussian noise parameter 
SIGMA = 15   # fixed sigma same for all dataset

# Class names
class_names = sorted(os.listdir(DATASET_PATH))
print("Classes:", class_names)

# Data containers
images = []

# Dataset read loop
for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):

        # Only image files
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, file)

        # Image read
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path)
            continue

        # Resize 
        img = cv2.resize(img, (256, 256))

        # Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian noise addition for μ=0
        noise = np.random.normal(0, SIGMA, gray.shape)
        noisy_gray = gray.astype(np.float32) + noise
        noisy_gray = np.clip(noisy_gray, 0, 255).astype(np.uint8)

        # Wavelet Transform
        LL, (LH, HL, HH) = pywt.dwt2(noisy_gray, 'db2')
        
      
        # GLCM Features from LL band
        
        LL_uint8 = np.clip(LL, 0, 255).astype(np.uint8)
        # Quantization: 256 -> 64 level
        LL_64 = (LL_uint8 // 4).astype(np.uint8)
        
        glcm_features = extract_glcm_features(LL_64)
        
       
        # HSV mean features
        hsv_features = extract_hsv_mean(img)

        
        # Hu Moments (from grayscale)
        hu_features = extract_hu_moments(gray)
        
      
        # Area & Shape Features
        shape_features = extract_shape_features(gray)


        # LBP histogram features
        lbp_features = extract_lbp_hist(gray, P=8, R=1, bins=16)


        # LCP histogram features
        lcp_features = extract_lcp_hist(gray, ksize=3, bins=16)
        
      
        # PHOG features
        phog_features = extract_phog_features(gray, levels=2)
        
       
        # Soft histogram features
        soft_hist_features = extract_soft_histogram(gray, bins=16)
        
        
        # ! FEATURE FUSION !
        fused_features = (
           glcm_features +
           hsv_features +
           hu_features +
           shape_features +
           lbp_features +
           lcp_features +
           phog_features +
           soft_hist_features
       )



        # Store extracted data
        images.append({
          "GLCM": glcm_features,
          "HSV": hsv_features,
          "HU": hu_features,
          "SHAPE": shape_features,
          "LBP": lbp_features,
          "LCP": lcp_features,
          "PHOG": phog_features,
          "SOFT": soft_hist_features,
          
          # Feature fusion
          "features": fused_features,
          
          "label": class_name
        })


ARFF_DIR = r"C:\Cmp447Final\arff"
os.makedirs(ARFF_DIR, exist_ok=True)

def write_arff(file_path, feature_matrix, labels, relation_name):
    with open(file_path, "w") as f:
        # Relation
        f.write(f"@RELATION {relation_name}\n\n")

        # Feature attributes 
        num_features = len(feature_matrix[0])
        for i in range(num_features):
            f.write(f"@ATTRIBUTE f{i+1} NUMERIC\n")

        # Class attributes
        class_labels = sorted(list(set(labels)))
        class_str = ",".join(class_labels)
        f.write(f"@ATTRIBUTE class {{{class_str}}}\n\n")

        # Data
        f.write("@DATA\n")
        for feats, label in zip(feature_matrix, labels):
            row = ",".join(str(float(x)) for x in feats)
            f.write(f"{row},{label}\n")
     
            
    
labels_arff = [img["label"] for img in images]
    
 
    # Spatial domain
    
glcm_feats = [img["GLCM"] for img in images]

glcm_lbp_feats = [
    img["GLCM"] + img["LBP"]
    for img in images
]

glcm_lbp_phog_feats = [
    img["GLCM"] + img["LBP"] + img["PHOG"]
    for img in images
]

    # Wavelet domain
    
wavelet_glcm_hsv_hu_feats = [
    img["GLCM"] + img["HSV"] + img["HU"]
    for img in images
]

wavelet_glcm_hsv_hu_lbp_feats = [
    img["GLCM"] + img["HSV"] + img["HU"] + img["LBP"]
    for img in images
]

wavelet_glcm_hsv_hu_lbp_phog_feats = [
    img["GLCM"] + img["HSV"] + img["HU"] + img["LBP"] + img["PHOG"]
    for img in images
]

wavelet_all_features_feats = [
    img["features"]
    for img in images
]
    
write_arff(
    os.path.join(ARFF_DIR, "glcm.arff"),
    glcm_feats,
    labels_arff,
    "glcm"
)

write_arff(
    os.path.join(ARFF_DIR, "glcm_lbp.arff"),
    glcm_lbp_feats,
    labels_arff,
    "glcm_lbp"
)

write_arff(
    os.path.join(ARFF_DIR, "glcm_lbp_phog.arff"),
    glcm_lbp_phog_feats,
    labels_arff,
    "glcm_lbp_phog"
)

write_arff(
    os.path.join(ARFF_DIR, "wavelet_glcm_hsv_hu.arff"),
    wavelet_glcm_hsv_hu_feats,
    labels_arff,
    "wavelet_glcm_hsv_hu"
)

write_arff(
    os.path.join(ARFF_DIR, "wavelet_glcm_hsv_hu_lbp.arff"),
    wavelet_glcm_hsv_hu_lbp_feats,
    labels_arff,
    "wavelet_glcm_hsv_hu_lbp"
)

write_arff(
    os.path.join(ARFF_DIR, "wavelet_glcm_hsv_hu_lbp_phog.arff"),
    wavelet_glcm_hsv_hu_lbp_phog_feats,
    labels_arff,
    "wavelet_glcm_hsv_hu_lbp_phog"
)

write_arff(
    os.path.join(ARFF_DIR, "wavelet_all_features.arff"),
    wavelet_all_features_feats,
    labels_arff,
    "wavelet_all_features"
)

print("All ARFF files generated successfully.")

           


     

    



    
