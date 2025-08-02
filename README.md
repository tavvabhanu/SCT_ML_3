🐾 SkillCraft Task 03– Cat vs Dog Image Classification with SVM

This repository contains my complete submission for Task 03 of the SkillCraft Machine Learning Internship. I implemented an SVM classifier to distinguish images of cats from dogs using image feature descriptors.

 📌 Problem Statement

  Input: RGB (or optionally grayscale) images of size 150 × 150 pixels  
  Output: Binary label — cat or dog  
  Dataset: Kaggle Dogs vs. Cats dataset (~25,000 labeled images: ~12,500 per class)

---
 📊 Workflow Summary

1. Data Preprocessing

  Resized all images to 150 × 150 for consistency  
  Converted to grayscale to reduce dimensionality (RGB optional mode supported)  
  Applied min–max scaling (pixel values normalized to [0,1])  
  Used `train_test_split()` to split into 80% train / 20% test

2. Feature Extraction

Computed Histogram of Oriented Gradients (HOG) descriptors via skimage.feature.hog 
Parameters: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2) 
Explored optional LBP (Local Binary Patterns) and color histograms  
Evaluated combined HOG+LBP feature fusion for improved performance

 3. Model Building

Utilized scikit-learn’s LinearSVC and SVC  
Conducted hyperparameter tuning with `GridSearchCV` over:
C = [0.1, 1, 10]
kernel = {'linear', 'rbf'} 
Observed that HOG + Linear SVM often outperformed non-linear kernels for limited-data scenarios

 4. Model Evaluation

Reported metrics on test set:
Accuracy
Precision
Recall
F1-score
Confusion Matrix 
HOG-only baseline yielded ~60%–75% accuracy  
Feature fusion (HOG + LBP + RBF SVM) boosted accuracy to ~85%+



🧠 Key Learnings

Gained practical experience pre‑processing and feature‑extracting from image data  
Learned that HOG remains effective for edge/texture detection in classic ML pipelines  
Understood SVM hyperparameter sensitivity and model selection via cross-validation  
Discovered performance gains through feature fusion (HOG + LBP)
Gained insight into why simpler ML pipelines sometimes match CNN performance in low-resource settings

🛠️ Tech Stack

Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Notebook: Jupyter (Google Colab)
📁 Files Included

SCT_ML_1.ipynb: Complete notebook with data analysis, model training, and evaluation
📬 Connect

Feel free to explore the project and reach out for collaboration or feedback!

https://www.linkedin.com/in/bhanu-tavva-880030367/
