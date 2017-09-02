---
layout: post
title: Object detection, an overview in the age of Deep Learning
---

There is a nice article summarizing the advances of object detection via deep learning: [Object detection: an overview in the age of Deep Learning](https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/). For 

  - an overview of the object detection problem, 
  - the comparison to other related computer vision problems, 
  - the challenges, 
  - the applications,
  - and commonly used dataset, 
  
please read the original post. Here I only list the most important algorithms here.

# Overfeat

One of the first advances in using deep learning for object detection was [OverFeat](https://arxiv.org/abs/1312.6229) from NYU published in 2013. They proposed a multi-scale sliding window algorithm using Convolutional Neural Networks (CNNs).

# R-CNN

Quickly after OverFeat, [Regions with CNN features or R-CNN](https://arxiv.org/abs/1311.2524) from [Ross Girshick](http://www.rossgirshick.info/), et al. at the UC Berkeley was published which boasted an almost 50% improvement on the object detection challenge. What they proposed was a three stage approach:

  - Extract possible objects using a region proposal method (the most popular one being [Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)).
  - Extract features from each region using a CNN.
  - Classify each region with [SVMs](https://en.wikipedia.org/wiki/Support_vector_machine)</a>.

![R-CNN Architecture](https://tryolabs.com/images/blog/post-images/2017-08-30-object-detection/rcnn.jpg)

While it achieved great results, the training had lots of problems. To train it you first had to generate proposals for the training dataset, apply the CNN feature extraction to every single one (which usually takes over 200GB for the [Pascal 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) train dataset) and then finally train the SVM classifiers.

# Fast R-CNN

This approach quickly evolved into a purer deep learning one, when a year later Ross Girshick (now at Microsoft Research) published [Fast R-CNN](https://arxiv.org/abs/1504.08083). Similar to R-CNN, it used Selective Search to generate object proposals, but instead of extracting all of them independently and using SVM classifiers, it applied the CNN on the complete image and then used both Region of Interest (RoI) Pooling on the feature map with a final feed forward network for classification and regression. Not only was this approach faster, but having the RoI Pooling layer and the fully connected layers allowed the model to be end-to-end differentiable and easier to train. The biggest downside was that the model still relied on Selective Search (or any other region proposal algorithm), which became the bottleneck when using it for inference.

![Fast R-CNN Architecture](https://tryolabs.com/images/blog/post-images/2017-08-30-object-detection/fastrcnn.jpg)

# YOLO

Shortly after that, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (YOLO) paper published by Joseph Redmon (with Girshick appearing as one of the co-authors). YOLO proposed a simple convolutional neural network approach which has both great results and high speed, allowing for the first time real time object detection.

![YOLO Architecture](https://tryolabs.com/images/blog/post-images/2017-08-30-object-detection/yolo.jpg)

# Faster R-CNN

Subsequently, [Faster R-CNN](https://arxiv.org/abs/1506.01497) authored by Shaoqing Ren (also co-authored by Girshick, now at Facebook Research), the third iteration of the R-CNN series. Faster R-CNN added what they called a Region Proposal Network (RPN), in an attempt to get rid of the Selective Search algorithm and make the model completely trainable end-to-end. We won’t go into details on what the RPNs does, but in abstract it has the task to output objects based on an “objectness” score. These objects are used by the RoI Pooling and fully connected layers for classification. We will go into much more detail in a subsequent post where we will discuss the architecture in detail.

![Faster R-CNN Architecture](https://tryolabs.com/images/blog/post-images/2017-08-30-object-detection/fasterrcnn.jpg)

# SSD and R-FCN

Finally, there are two notable papers, [Single Shot Detector](https://arxiv.org/abs/1512.02325) (SSD) which takes on YOLO by using multiple sized convolutional feature maps achieving better results and speed, and [Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409) (R-FCN) which takes the architecture of Faster R-CNN but with only convolutional networks.
