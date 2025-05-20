# Image-Segmentation-using-CNN
## Introduction
Image segmentation takes image analysis a step further than just detecting objects — it breaks down an image into specific regions that are meaningful. This is especially useful in fields like medical imaging, where it’s important to tell the difference between things like tissues, blood, or abnormal cells in order to highlight a specific organ.
This project focuses on image segmentation, specifically semantic segmentation, where the goal is to classify each pixel in an image into a specific category. I’m applying this to cardiac MRI scans to identify which pixels belong to the left ventricle (LV).
It’s essentially a pixel-level classification task, which makes it more detailed and precise than standard image classification. I’m using TensorFlow to build and train a convolutional neural network (CNN) that can learn to segment these images accurately.
This kind of pixel-by-pixel labeling can be really valuable in medical imaging, where isolating specific structures—like an organ or a region of interest—can support diagnostics and further analysis.

## Learning Objectives:
 - Understand how neural networks can be used to solve image-related problems
 - Work with transpose convolution layers for upsampling
 - Use Keras and TensorFlow 2 to train models on image data

## Walkthrough the dataset
For this project, I’m working with a set of cardiac MRI images—specifically short-axis (SAX) scans—that come with expert-labeled annotations. The original images are in DICOM format (256 x 256 grayscale), which is a standard in the medical imaging field.
I used a set of preprocessing steps to convert these raw images into a format that’s compatible with TensorFlow. While the full data preparation pipeline isn’t included in this repo, it involved extracting the images, converting them to TFRecords, and storing them for training. TFRecords are useful because they let me take advantage of TensorFlow’s built-in functions for efficient data loading, preprocessing, and augmentation.
Each label is a tensor of size 256 x 256 x 2, where the last dimension represents two classes—whether a pixel is part of the left ventricle or not. The training set includes 234 images, and I reserved 26 images for validation to evaluate the model’s performance.

## Deep Learning with TensorFlow (additional)
This project is also a great opportunity for me to work with TensorFlow—Google’s open-source deep learning framework that’s widely used in both research and industry. TensorFlow lets you define computations as data flow graphs operating on tensors (which is where the name comes from), and it’s designed to run efficiently across CPUs, GPUs, and even mobile devices.
I’m using TensorFlow 2 in this project, which introduced several big changes compared to the older 1.x versions. One of the most noticeable is the shift to eager execution by default. This means the code runs more like regular Python, which makes debugging and experimentation a lot easier.
For building and training the model, I’m using tf.keras, TensorFlow’s implementation of the high-level Keras API. It simplifies the process of defining, training, and evaluating neural networks. I used the Sequential API for straightforward layer stacking, but Keras also supports a Functional API for more complex architectures when needed.
The flexibility of TensorFlow, along with the ease of use provided by Keras, made it a good fit for experimenting with CNNs for image segmentation. While this project doesn't cover every detail of TensorFlow, it gave me a solid hands-on feel for how to use it effectively in a practical setting.
### TensorBoard
To track and visualize the training process, I used TensorBoard—TensorFlow’s built-in tool for monitoring models. It gives a visual view of things like loss, accuracy, and learning rate as training progresses, which really helps in understanding how the model is learning over time.
TensorBoard integrates nicely with Keras through a callback, so it was easy to plug in during training. Once it’s set up, I could view detailed logs, see how the metrics evolved, and even look at the model graph structure. It’s a really handy way to debug and fine-tune models while keeping everything organized.
