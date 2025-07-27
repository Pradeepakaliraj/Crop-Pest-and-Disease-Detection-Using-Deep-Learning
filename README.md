# Crop-Pest-and-Disease-Detection-Using-Deep-Learning
This project automates the classification of crop diseases and pests from leaf images using a convolutional neural network based on a pretrained VGG16 model. It focuses on four common crops: Cashew, Cassava, Maize, and Tomato.

**Objective**
Detect and classify crop diseases and pests using image data.

Reduce manual effort in agricultural diagnostics.

Build a scalable, AI-driven crop health assessment tool.

**Dataset**
Dataset from Kaggle : https://data.mendeley.com/datasets/bwh3zbpkpv/1

Sourced from a structured local directory containing labeled folders for each crop and disease.

Images are preprocessed:

Resized to 128x128

Normalized and cleaned (corrupt files removed)

Augmented using random flips and rotations
Sourced from a structured local directory containing labeled folders for each crop and disease.

**Images are preprocessed:**

Resized to 128x128

Normalized and cleaned (corrupt files removed)

Augmented using random flips and rotations


**Frameworks & Tools**
Python 3.x

TensorFlow / Keras

NumPy, Pandas, Matplotlib

Scikit-learn

**Improvement**
Run with Epoch = 50 with early stopping enabled
