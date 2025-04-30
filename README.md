# Sign Language Recognition
Sign language recognition (SLR) is an important task in the field of human-computer interaction and accessibility, aiming to bridge the communication gap between hearing-impaired individuals and the rest of the world. In this project, we explore two different approaches to recognizing hand signs from images or videos:

MediaPipe + Machine Learning:
In the first approach, we utilize MediaPipe, a powerful framework by Google for real-time hand tracking and landmark detection. MediaPipe provides 21 keypoints for each detected hand, which we use as features to train a lightweight machine learning classifier (e.g., SVM, Random Forest, or MLP). This method is efficient, works well with low computational resources, and is suitable for real-time applications.

YOLO + Convolutional Neural Network (CNN):
In the second approach, we employ a YOLOv8 object detection model to locate and crop hand regions in the input images. The cropped hand images are then passed through a custom CNN-based classifier to predict the corresponding sign. This deep learning approach allows us to learn features directly from the raw image data and can potentially achieve higher accuracy, especially in complex or cluttered backgrounds.

By comparing both methods, we aim to evaluate their effectiveness in different scenarios in terms of accuracy, speed, and robustness. The project provides insights into the trade-offs between classical machine learning with structured features and deep learning with end-to-end representation learning.


### Project Structure

This project consists of three main folders: `yolo`, `ML`, and `Deep_learn`, each corresponding to different parts of the sign language recognition system.

- **`yolo/`**  
  Contains files related to the first stage of the second approach: hand detection using YOLO.  
  - Includes scripts to **train a YOLO model** on custom hand images.  
  - Also provides a **testing script** to run YOLO on individual images and visualize the detection results.

- **`Deep_learn/`**  
  Contains files for the second part of the deep learning pipeline, where cropped hand images are classified using a CNN model.  
  - `model_resnet.py`: Main training script for the CNN model (e.g., based on ResNet architecture).  
  - `predict.py`: Allows prediction on **individual hand images**.  
  - `ontime.py`: Runs the model in **real-time** using a webcam feed, displaying predictions live.

- **`ML/`**  
  (This folder contains the implementation for the first approach: MediaPipe + classical machine learning.)  
  - `collect_imgs.py`: This file is used to capture images for the dataset.
  - `create_data.py`: This file is used to extract hand landmark features from images in the ASL alphabet dataset and save the data into a .pickle file for easy use in model training later
  - `inference_classifier.py`: this file is used to detect and classify hand signs from live webcam feed using a pre-trained machine learning model
  - `train_classifier.py`: This script trains a Random Forest classifier to recognize hand signs.
  - `notebook.ipynb`: Contains data preprocessing steps and exploratory analysis.
11
  - Typically includes scripts to **extract hand landmarks** using MediaPipe.  
  - Uses lightweight classifiers (e.g., SVM, Random Forest) trained on landmark features for sign recognition.

Each folder is self-contained and serves a specific step in the overall system. You can choose to run either the full deep learning pipeline (`yolo` + `Deep_learn`) or the lightweight classical machine learning version in `ML`.



### Dataset

We use a combination of public and custom datasets for training and evaluation:

- **ASL Alphabet Dataset (Kaggle)**  
  The main dataset used is the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle, which contains over **60,000 labeled images** of American Sign Language hand gestures, covering all 26 letters of the alphabet as well as additional labels like "space", "nothing", and "delete".

- **Custom Captured Dataset**  
  To improve robustness and make the model more adaptive to real-world usage, we also created our **own custom dataset** by capturing images of our own hands. These images were taken under various lighting conditions, angles, and backgrounds to simulate more realistic scenarios.

The combination of both datasets allows us to train models that generalize better and perform well not only on clean datasets but also on more diverse, real-life hand gesture inputs.


### Import Model

Below are the important links to access pretrained models and datasets used in this project:

- **🅰️ YOLO Model:** [Download YOLO Hand Detection Model](<https://drive.google.com/file/d/1ZhW7f9w01vyZoqUMd3RwX5uFJBoMpH5K/view?usp=drive_link>)  
  Trained YOLOv8 model for hand detection.

- **🅱️ ResNet Model:** [Download ResNet Sign Classification Model](<https://drive.google.com/file/d/1NjCBJxkQQlsSIt8jYe279jQ5LSULz9JF/view?usp=drive_link>)  
  CNN model (based on ResNet) trained to classify hand signs from cropped images.

- **🇨 ML Model:** [Download MediaPipe + ML Classifier](<C>)  
  Classical machine learning model trained on hand landmarks extracted using MediaPipe.

- **🇩 Main Dataset:** [Download ASL + Custom Dataset](<https://drive.google.com/file/d/11QvvFUdx9NZTRbl2I-5uSmIJa1aiI_w6/view?usp=drive_link>)  
  Combined dataset including the ASL Alphabet dataset from Kaggle and our self-collected hand sign images.

- **🇪 YOLO Finetune Dataset:** [Download YOLO Fine-tuning Images](<https://drive.google.com/file/d/1njKLJ8u5ZdN03NEi0pYdm2O4v5xkURdY/view?usp=sharing>)  
  Additional hand images used to finetune the YOLO detection model for better performance on real hands 