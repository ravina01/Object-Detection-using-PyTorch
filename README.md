# Object Detection using PyTorch
---

Object Detection for Computer Vision using Deep Learning with Python. Train and Deploy (Detectron2, Faster RCNN, YOLOv8)

#### Whats Object Detection ?
- The goal of Object detection is to train a Deep learning model which can look at the image of multiple objects to detect and localizing individual pbjects present in the image.
- Classify the object label and precise bounding box around them.
- Train a deep learning model which can look at the image of multiple objects to detectv an dlocalize individual objects presents in the image.
- It localizes and classifies one or more objects in an image.
- Detection = Classification + Localization.  

**GOAL**
- FrameWork - PyTorch
- Deep Learning Architectures - CNN, RCNN, Fast RCNN, Faster RCNN, Mask RCNN and YOLO8, Detectron2
- Outcome -
  - 1. Learn how to leverage pre-trained models, fine-tune them for Object Detection
    2. Single-Stage Object Detection vs Two-Stage Objection Detection
    3. Single-Stage Object Detection vs Two-Stage Objection Detection
    4. Perform Object Instance Segmentation at Pixel Level using Mask RCNN

![image](https://github.com/user-attachments/assets/44cf948b-e746-40a6-ba38-c0e3cd97e438)

Applications -
![image](https://github.com/user-attachments/assets/9fa67948-a8c6-4ced-b86b-da3b1407b085)


### Object Detection Pipeline -
- 1. Dataset Preparation
  2. Deep Learning Network Architecture Selection
  3. Model Training
  4. Inference (Testing and Deploy Model)
  5. Evaluaion (mAP, IOU)
  6. Results Visualization

#### How to identify objects ? Regions/Location/ Boudning Boxes:
- 1. Region Proposal Methods
  - 1. Selective search
    2. Edges boxes
    3. Regions Proposal Networks (RPNs)
    4. Superpixels
       

#### Single-stage vs two-stage Object Detection:

![image](https://github.com/user-attachments/assets/cb932c97-3aaf-4b42-96eb-555e4eca747b)


#### Single-shot/ Single-stage Object Detection -
- 1. Single-shot object detection uses a **single pass of the input image** to make predictions about the presence and location of objects in the image. It processes an entire image in a single pass, making them computationally efficient.
  2. single-shot object detection is generally **less accurate** than other methods.
  3. it’s less effective in detecting small objects.
  4. Such algorithms can be used to detect objects in real time in resource-constrained environments.
  5. YOLO is a single-shot detector that uses a fully convolutional neural network (CNN) to process an image.


#### Two-Shot Object Detection OR Two-Stage Object Detection -
- 1. Two-shot object detection uses two passes of the input image to make predictions about the presence and location of objects.
  2. The first pass is used to generate a set of proposals or potential object locations, and the second pass is used to refine these proposals and make final predictions.
  3. more accurate than single-shot object detection but is also more computationally expensive.
  4. RCNNs are  two stage object detectors.


**Note**: single-shot object detection is better suited for real-time applications, while two-shot object detection is better for applications where accuracy is more important.

![image](https://github.com/user-attachments/assets/d15e9b6c-1de3-44b6-9265-2c718328eaf7)

#### 1.Object Detection Approach:
1. YOLO: YOLO is a one-stage object detection algorithm. It divides the image into a grid and directly predicts bounding boxes and class probabilities for each grid cell.

2. R-CNN: R-CNN is a two-stage object detection approach. It first generates region proposals and then classifies each proposed region.

#### 2.Speed:

1. YOLO: YOLO is known for its real-time processing capabilities. It processes the entire image at once, making it faster compared to R-CNN.

2. R-CNN: R-CNN is relatively slower due to its two-stage nature, involving region proposal generation and classification.

#### 3. Accuracy:

1. YOLO: YOLO sacrifices some accuracy for speed. It may struggle with small objects and dense scenes but provides a good balance between speed and accuracy.

2. R-CNN: R-CNN typically achieves higher accuracy as it carefully selects region proposals for detailed processing. It is more suitable for applications where accuracy is critical.

#### 4. Training:

1. YOLO: YOLO involves end-to-end training, making it simpler to train and implement.

2. R-CNN: R-CNN has a more complex training pipeline, including region proposal generation and classification, making it computationally more intensive.

#### 5. Use Cases:

1. YOLO: YOLO is suitable for real-time applications such as object detection in videos and surveillance.

2. R-CNN: R-CNN is preferred in scenarios where high accuracy is crucial, such as medical image analysis.

#### 6. Model Variants:

1. YOLO: YOLO has seen multiple versions, with YOLOv4 and YOLOv5 being some of the latest versions like YOLOV8.

2. R-CNN: R-CNN has evolved into Faster R-CNN and Mask R-CNN, improving speed and capabilities

### Deep Learning Architectures for Object Detection -
---

#### 1. CNN Deep Learning -

1. VGGNet (VGG16, VGG19):

- Pros: Simple and uniform architecture with only 3x3 convolutions and 2x2 pooling layers.
- Cons: Very large number of parameters, leading to high computational cost and memory usage.
- Use Case: Image classification and feature extraction.

2. GoogLeNet (Inception V1):

- Pros: Efficient in terms of computational cost and memory usage due to inception modules.
- Cons: More complex architecture compared to VGGNet.
- Use Case: Image classification, object detection.

3. ResNet (Residual Networks):

- Pros: Introduced residual connections to mitigate the vanishing gradient problem, allowing for very deep networks.
- Cons: Can be computationally intensive.
- Use Case: Image classification, feature extraction, transfer learning.

4. DenseNet:

- Pros: Uses dense connections between layers, leading to better parameter efficiency and gradient flow.
- Cons: Computationally expensive due to concatenation of feature maps.
- Use Case: Image classification, segmentation.

5. EfficientNet:

- Pros: Achieves state-of-the-art accuracy with fewer parameters and FLOPS by scaling up width, depth, and resolution systematically.
- Cons: More complex design, requires compound scaling.
- Use Case: Image classification, transfer learning.

Choosing the Right CNN
- Accuracy: If achieving the highest possible accuracy is crucial, consider architectures like ResNet, DenseNet, or EfficientNet.
- Efficiency: For resource-constrained environments (e.g., mobile devices), MobileNet or EfficientNet is more suitable.
- Ease of Use: If you need a straightforward architecture for feature extraction or transfer learning, VGGNet or ResNet is a good choice.

