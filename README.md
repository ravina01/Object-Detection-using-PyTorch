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

1.1 VGG16
- Architecture: VGG16 has 16 weight layers, consisting of 13 convolutional layers and 3 fully connected layers. The convolutional layers use 3x3 filters with stride 1 and padding 1. The max-pooling layers use 2x2 filters with stride 2.
- Layer Configuration:
- Convolutional Layers: 13 layers
- Fully Connected Layers: 3 layers
- Max Pooling Layers: 5 layers
- Total Number of Parameters: Approximately 138 million

1.2 VGG19
- Architecture: VGG19 is an extension of VGG16, with 19 weight layers, consisting of 16 convolutional layers and 3 fully connected layers. The convolutional layers use the same 3x3 filters with stride 1 and padding 1. The max-pooling layers use 2x2 filters with stride 2.
- Layer Configuration:
- Convolutional Layers: 16 layers
- Fully Connected Layers: 3 layers
- Max Pooling Layers: 5 layers
- Total Number of Parameters: Approximately 143 million
  
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

6. Choosing the Right CNN
- Accuracy: If achieving the highest possible accuracy is crucial, consider architectures like **ResNet, DenseNet, or EfficientNet.**
- Efficiency: For resource-constrained environments (e.g., mobile devices), **MobileNet or EfficientNet**is more suitable.
- Ease of Use: If you need a straightforward architecture for feature extraction or transfer learning, VGGNet or ResNet is a good choice.

#### CNN vs R-CNN:

![image](https://github.com/user-attachments/assets/16766ef5-674f-4bdd-b38d-cba279dd3a50)

2. Training Process:

CNN: CNNs are trained using labeled images with their corresponding ground truth class labels. The training process focuses on optimizing the network's weights to accurately classify images into predefined categories.

R-CNN: R-CNN involves additional steps in the training process. It requires an initial pre-training step where a CNN is trained on a large-scale dataset (e.g., ImageNet) for image classification. Then, a region proposal algorithm (e.g., Selective Search) is used to generate potential object proposals, and these proposals are labeled with their corresponding object classes and refined bounding box coordinates. The CNN is fine-tuned on these labeled proposals to learn to classify and refine the proposed regions.

3. Inference Speed:

CNN: CNNs process images independently and do not consider objects' spatial information. They can be applied to images of any size, but they lack efficiency in localizing and detecting multiple objects within an image.

R-CNN: R-CNN, especially its later variants like Fast R-CNN and Faster R-CNN, have improved the efficiency of object detection by introducing shared feature extraction and region proposal networks. By sharing the computation for multiple region proposals, R-CNN significantly reduces the inference time compared to CNN-based approaches.

#### 2. R-CNN (Region Based Convolution Neural Network) Deep Learning - Selective search

![image](https://github.com/user-attachments/assets/fbb84b26-6d68-4b99-a7da-536b55edadde)

![image](https://github.com/user-attachments/assets/a507fa46-8338-4a57-8ace-e4e752c8112e)

![image](https://github.com/user-attachments/assets/554cd1cd-4ebd-4724-b313-c34cd59beaf0)

![image](https://github.com/user-attachments/assets/c74c3ff0-c839-406b-b358-530080f93540)


- Use Case : This family is mainly used for object Detection + Segmentation.

#### Three Stages of R-CNN

The original R-CNN is indeed a three-stage process involving region proposal, feature extraction, and classification with bounding box regression. While effective, it is computationally intensive and slow due to the separate processing of each region proposal.

1. Region Proposal:

- Purpose: To generate a set of candidate regions in the image that are likely to contain objects.
- Method: A region proposal algorithm, such as Selective Search, is used to generate around 2000 region proposals from the input image.
- Output: These proposals are potential bounding boxes around objects in the image.

2. Feature Extraction:

- Purpose: To extract features from each region proposal.
- Method: Each region proposal is resized to a fixed size and passed through a pre-trained CNN (such as AlexNet or VGG16) to extract a fixed-length feature vector.
- Output: A feature vector representing each region proposal.

3. Classification and Bounding Box Regression:
   
- Purpose: To classify the object within each region proposal and refine the bounding box.
Method:
- Classification: The feature vectors are fed into a set of class-specific linear SVM classifiers to predict the object class for each region proposal.
- Bounding Box Regression: A regression model is used to refine the coordinates of the bounding box to better fit the object.
- Output: The final object class and refined bounding box coordinates for each region proposal.

#### 3. Fast R-CNN (Region Based Convolution Neural Network) Deep Learning -

Introduced RoI (Region of Interest) pooling, allowing the entire image to be processed by the CNN in one forward pass, and then extracting features for region proposals. This significantly reduces computation time and improves efficiency.

Key Improvements in Fast R-CNN 

1. Single-Stage Training:

- Unlike R-CNN, which requires a multi-stage training process (training the CNN, SVMs, and bounding box regressors separately), Fast R-CNN trains the entire network in a single stage using a multi-task loss function.
- This approach simplifies the training process and makes it more efficient.

2. Region of Interest (RoI) Pooling:

**It allows the network to handle inputs of varying sizes by converting all region proposals into fixed-size feature maps.**

- Fast R-CNN introduces the Region of Interest (RoI) pooling layer, which allows the network to extract fixed-size feature maps from each region proposal.
- The entire image is processed by the CNN once, and feature maps are generated. Then, the RoI pooling layer extracts features for each region proposal directly from these feature maps, reducing the computational cost significantly.
This method **avoids the need to resize each region proposal to a fixed size before passing it through the CNN**, as done in R-CNN.

3. End-to-End Training:

- The RoI pooling layer allows for end-to-end training of both the region proposal and the object detection network, resulting in better performance and more accurate predictions.
- The network simultaneously optimizes classification and bounding box regression.

#### Architecture of Fast R-CNN
- Input: An input image and a set of region proposals.
- Convolutional Layers: The input image is passed through several convolutional and max-pooling layers to generate a feature map.
- RoI Pooling Layer: The region proposals are projected onto the feature map, and the RoI pooling layer extracts fixed-size feature maps for each proposal.
- Fully Connected Layers: The pooled feature maps are then flattened and passed through a series of fully connected layers.
Output Layers:
- Classification Layer: A softmax layer predicts the object class for each region proposal.
- Bounding Box Regression Layer: A regression layer refines the bounding box coordinates for each region proposal.


#### Detailed Explanation of RoI Pooling

1. Input to RoI Pooling:

- Feature Map: The feature map is generated by passing the entire input image through several convolutional and max-pooling layers of the CNN.
- Region Proposals: These are the candidate bounding boxes that might contain objects. They are generated using algorithms like Selective Search.

2. Purpose of RoI Pooling:

- To extract fixed-size feature maps from the feature map for each region proposal.
- To ensure that the fully connected layers that follow the RoI pooling layer can handle a consistent input size.

3. Operation of RoI Pooling:

- Projection: Each region proposal is projected onto the feature map. This projection is scaled according to the downsampling done by the previous convolutional layers.
- Division into Grids: The projected region on the feature map is divided into a grid of a fixed size, typically  H×W (e.g., 7x7).
- Max Pooling: Each grid cell is subjected to max pooling. This involves selecting the maximum value within each grid cell to form the final output of that cell. This operation converts the variable-sized regions into a fixed-size grid of features.


#### Step-by-Step Process

1. Projection:

- Suppose the input image is of size 800×600, and the feature map after several convolutional layers is of size 50×38.(Divide by 16 scale)
- For instance, if the region proposal is (200,150,400,350), the corresponding coordinates on the feature map might be (12.5,9.5,25,22) after scaling.

2. Grid Division
- The projected region on the feature map is divided into a fixed-size grid of H×W cells. For example, if the fixed size is 7x7, the projected region is divided into 49 cells.
- Each cell in the grid corresponds to a sub-region of the feature map.

3. Max Pooling:

- For each cell in the H×W grid, max pooling is performed over the corresponding sub-region of the feature map.
- The maximum value from each sub-region is taken as the representative value for that cell.
- This results in a fixed-size output of H×W values (e.g., a 7x7 grid).


#### 4. Faster RCNN (Region Based Convolution Neural Network) Deep Learning - 

![image](https://github.com/user-attachments/assets/7ecddddb-98f6-458c-b58a-5854637d98af)

- Faster R-CNN is an extension of the Fast R-CNN model designed to improve the speed and efficiency of the region proposal process.
- Doesn't use Selective search.
  
#### Key Components of Faster R-CNN

1. Region Proposal Network (RPN):

- Purpose: To generate high-quality region proposals directly from the convolutional feature map.
- Function: The RPN slides a small network over the convolutional feature map and outputs region proposals along with objectness scores indicating the likelihood of the proposals containing objects.

2. Shared Convolutional Layers:

- Both the RPN and the object detection network share the convolutional layers, allowing the model to perform both tasks simultaneously and efficiently.

3. RoI Pooling:

- As in Fast R-CNN, the RoI pooling layer is used to convert the region proposals into fixed-size feature maps.

4. Classification and Bounding Box Regression:

The fixed-size feature maps are used for final object classification and bounding box regression.

#### Architecture of Faster R-CNN

- 1. Input: An input image.
- 2. Convolutional Layers: The image is passed through several convolutional layers to produce a feature map.
- 3. Region Proposal Network (RPN):
  - Anchor Boxes: The RPN generates a set of anchor boxes of different sizes and aspect ratios for each position in the feature map.
  - Sliding Window: A sliding window mechanism is used to evaluate the anchor boxes.
  - Objectness Score: For each anchor, the RPN outputs an objectness score, indicating whether the anchor is likely to contain an object.
  - Bounding Box Regression: The RPN also adjusts the anchor boxes to better fit the objects.
- 4. RoI Pooling:
The top N region proposals (based on objectness scores) are selected and projected onto the feature map.
RoI pooling is used to convert the variable-sized proposals into fixed-size feature maps.
- 5. Fully Connected Layers:
The fixed-size feature maps are flattened and passed through fully connected layers.
- 6. Output Layers:
  - Classification: A softmax layer predicts the object class for each region proposal.
  - Bounding Box Regression: A regression layer refines the bounding box coordinates for each region proposal.
 
#### Training Faster R-CNN
Training Faster R-CNN involves two main stages:

1. Train RPN:

- The RPN is trained to generate region proposals with high recall and good localization.
- The loss function used is a combination of classification loss (object vs. non-object) and bounding box regression loss.

2. Train Detection Network:

- The detection network (Fast R-CNN) is trained using the region proposals generated by the RPN.
- The entire network is fine-tuned end-to-end, optimizing both the RPN and the detection network simultaneously.

Q What are Anchor Boxes?

- Anchor boxes are predefined bounding boxes of different scales and aspect ratios that are placed uniformly across the image. They act as reference points to predict region proposals.

Example:
- If your feature map is of size H×W (height and width), you will generate H×W×9 anchor boxes.
- Suppose you have a feature map of size 50×50.
- You have 9 anchor boxes per location.
- This results in 50×50×9=22,500 anchor boxes for the entire image.
- Sliding Window Mechanism:
  - The RPN uses a small sliding window (typically 3x3) that moves across the feature map.
  - For each position of the sliding window, it predicts:
  - Objectness Score: Whether the anchor box contains an object or not.
  - Bounding Box Regression: Adjustments to the anchor box coordinates to better fit the object.
- Assigning Labels to Anchor Boxes:
  - Positive Anchors:
    - Anchors that have an Intersection over Union (IoU) overlap greater than 0.7 with any ground-truth box.
    - Anchors that have the highest IoU overlap with a ground-truth box.
  - Negative Anchors:
    - Anchors that have an IoU overlap less than 0.3 with all ground-truth boxes.
    - Anchors that do not fall into either category are ignored during training.
   
#### Example Workflow
- 1. Input Image: An image is input into the network.
- 2. Convolutional Feature Map: The image is passed through several convolutional layers to produce a feature map.
- 3. Region Proposal Network:
  - Anchor boxes are generated for each location in the feature map.
  - The RPN predicts objectness scores and adjusts the anchor boxes.
  - Top N region proposals are selected based on objectness scores.
- 4. RoI Pooling: The selected region proposals are projected onto the feature map and converted into fixed-size feature maps using RoI pooling.
- 5. Object Detection:
  - The fixed-size feature maps are passed through fully connected layers.
  - The network outputs object class predictions and refined bounding box coordinates.

#### Process of Generating and Using Region Proposals in Faster RCNN -

1. Generating Region Proposals in Feature Map Coordinates:

- The Region Proposal Network (RPN) operates on the feature map produced by the convolutional layers of the CNN.
- Anchor boxes are placed on this feature map, and their coordinates are relative to the feature map grid.

2. Bounding Box Regression:

- The RPN outputs regression offsets that adjust these anchor boxes to better fit the objects.
- These offsets are applied to the anchor boxes to generate refined region proposals, still in the context of the feature map.

3. Transforming to Image Coordinates:

- Once the region proposals are generated, they are transformed back to the original image coordinates.
- This transformation is necessary because the final bounding box predictions and classifications need to correspond to the actual image space.

#### 5. Mask R-CNN (Region Based Convolution Neural Network) Deep Learning -

![image](https://github.com/user-attachments/assets/3bddec9b-89ae-4ede-983e-104114e7533a)

![image](https://github.com/user-attachments/assets/95455cdf-5eab-4df8-9d8a-6a3ea8414722)

- Use Case - Instance Segmentation + Object Detection

#### Overview of RCNN Family-
![image](https://github.com/user-attachments/assets/47d3f975-d2a3-4d1e-87b9-67199c2de84d)

