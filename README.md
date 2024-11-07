# ImageNet Final Project

## Overview
This project explores various techniques in machine learning on the ImageNet and Tiny ImageNet datasets. It is part of the GEL521 - Machine Learning course, presented by Anthony El Chemaly and Catherina El Khoury.

## Project Structure
- **ImageNet Introduction**: Understanding what ImageNet is and its significance in visual object recognition research.
- **Tiny ImageNet Overview**: A subset of ImageNet, focusing on handling a smaller set of classes with 64x64 colored images.
- **Model Training**: Training CNNs using Keras on the Tiny ImageNet dataset from scratch, including techniques like Batch Normalization, L2 Regularization, and He Normalization.
- **Pre-Trained Models**: Using Keras to employ models like VGG16, ResNet50, and InceptionV3 that were pre-trained on ImageNet ILSVRC.
- **Fine-Tuning VGG16**: Adaptation of the VGG16 model to new classes involving notable figures.
- **Testing and Results**: Leveraging OpenCV for image processing and achieving high accuracy in both training and testing phases.

## Technologies Used
- **Keras**: For building and training neural network models.
- **OpenCV**: For image processing tasks.
- **Google Colab**: Used as the development and training environment.

## Results
- **Best Training Accuracy**: 99.87% (achieved on our specific dataset).
- **Best Testing Accuracy**: 98.50%.
- Confidence levels reached as high as 99.95% in individual tests.

## Setup and Installation
-  Ensure Python 3.x is installed.
-  Install necessary Python packages:
  ```bash
   pip install tensorflow keras opencv-python
