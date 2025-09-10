## Blood Cancer Detection using Deep Learning

A deep learning project for accurate detection of blood cancers using Convolutional Neural Networks (CNNs) on medical image data.

Overview
--------
This repository contains a comprehensive solution for automated blood cancer detection using state-of-the-art CNN architectures. The model achieves high classification accuracy (>99%) in recognizing normal and cancerous blood cells, facilitating early diagnosis and supporting clinicians in decision making.

Features
--------
- End-to-end pipeline: data preprocessing, augmentation, training, evaluation, and inference
- Custom and transfer learning CNN architectures (VGG, ResNet, Inception, DenseNet, etc.)
- Supports multi-class classification and visualization (Grad-CAM for interpretability)
- Comprehensive experiment logs and result analytics

Dataset
-------
- Source: Kaggle - Blood Cancer Dataset https://www.kaggle.com/datasets/mahdinavaei/blood-cancer
- Description: 17,092 labeled images of normal and cancerous blood cells from healthy individuals and patients.
- Classes: Includes subtypes of acute lymphoblastic leukemia (ALL) and other blood cancers.

Model Architecture
------------------
- Modular CNN design with multiple convolutional blocks for feature extraction
- Max pooling layers for spatial reduction
- Fully connected layers for final classification
- Softmax output for multi-class probabilities
- Optionally supports transfer learning and ensemble methods for improved performance

Training
--------
- Training-validation split from the provided dataset
- Optimized with Adam/SGD, categorical cross-entropy loss
- Real-time augmentation and callbacks for best results
- Model checkpoints and logs saved for reproducibility

Results
-------
- Validation accuracy: >99%, competitive with or superior to leading research
- Robust to variations in image quality/environment
- Grad-CAM visualizations demonstrate reliable focus on critical regions for predictions
- Benchmarked against SVM, Random Forest, and deep transfer learning methods

Usage
-----
1. Clone the repository and install dependencies
2. Download the dataset from Kaggle and place in the `data` folder
3. Train the model:
   python train.py

4. Test inference on new blood cell images:
   python predict.py --img_path sample/test_image.png

5. Explore visualizations and reports in the `results` directory

Dependencies
------------
- Python 3.x
- TensorFlow >= 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn (optional for additional analysis)
- OpenCV (optional for image IO/augmentation)

Impacts & Benefits
------------------
- Enables accurate, rapid, and reproducible blood cancer diagnosis
- Reduces manual workload and variability in traditional microscopy
- Assists medical professionals by visualizing model’s decision process
- Ready for integration into web-based and clinical diagnostic tools
- Supports research extensions and real-time deployment

Contributing
------------
- Submit issues, feature requests, or pull requests via GitHub.
- Please ensure unit tests pass (`python -m unittest discover tests`) before PRs.

