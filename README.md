# Sign Language Recognition
Sign language recognition (SLR) is an important task in the field of human-computer interaction and accessibility, aiming to bridge the communication gap between hearing-impaired individuals and the rest of the world. In this project, we explore two different approaches to recognizing hand signs from images or videos:

MediaPipe + Machine Learning:
In the first approach, we utilize MediaPipe, a powerful framework by Google for real-time hand tracking and landmark detection. MediaPipe provides 21 keypoints for each detected hand, which we use as features to train a lightweight machine learning classifier (e.g., SVM, Random Forest, or MLP). This method is efficient, works well with low computational resources, and is suitable for real-time applications.

YOLO + Convolutional Neural Network (CNN):
In the second approach, we employ a YOLOv8 object detection model to locate and crop hand regions in the input images. The cropped hand images are then passed through a custom CNN-based classifier to predict the corresponding sign. This deep learning approach allows us to learn features directly from the raw image data and can potentially achieve higher accuracy, especially in complex or cluttered backgrounds.

By comparing both methods, we aim to evaluate their effectiveness in different scenarios in terms of accuracy, speed, and robustness. The project provides insights into the trade-offs between classical machine learning with structured features and deep learning with end-to-end representation learning.


