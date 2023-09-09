---
layout: post
title: 3D Object Detection
tags:  deep-learning 3d-object-detection
---

![](https://paperswithcode.com/media/thumbnails/task/task-0000000785-04017634.jpg)

[3D Object Detection](https://paperswithcode.com/task/3d-object-detection) classifies the object category and estimates oriented 3D bounding boxes of physical objects from 3D sensor data.

Based on the data, 3D object detection methods can be classified into following categories:
- image based:
  - monocular
  - binocular
  - multi-view
- point-cloud based:
  - Lidar
  - depth camera
  
In this note, we will focus on image based 3D object detection methods.

# 3D Object Detection from Monocular Image

## [3D Bounding Box Estimation Using Deep Learning and Geometry](http://arxiv.org/abs/1612.00496)

![](https://camo.githubusercontent.com/6c6438830d080a1924e404a4ff48da6b2661b51a/687474703a2f2f736f726f7573686b686164656d2e636f6d2f696d672f32642d746f702d33642d626f74746f6d312e706e67)

> We present a method for 3D object detection and pose estimation from a single image. In contrast to current techniques that only regress the 3D orientation of an object, our method first regresses relatively stable 3D object properties using a deep convolutional neural network and then combines these estimates with geometric constraints provided by a 2D object bounding box to produce a complete 3D bounding box. The first network output estimates the 3D object orientation using a novel hybrid discrete-continuous loss, which significantly outperforms the L2 loss. The second output regresses the 3D object dimensions, which have relatively little variance compared to alternatives and can often be predicted for many object types. These estimates, combined with the geometric constraints on translation imposed by the 2D bounding box, enable us to recover a stable and accurate 3D object pose. We evaluate our method on the challenging KITTI object detection benchmark both on the official metric of 3D orientation estimation and also on the accuracy of the obtained 3D bounding boxes. Although conceptually simple, our method outperforms more complex and computationally expensive approaches that leverage semantic segmentation, instance level segmentation and flat ground priors and sub-category detection. Our discrete-continuous loss also produces state of the art results for 3D viewpoint estimation on the Pascal 3D+ dataset.

This paper models the location of 3D object as 3D bounding box which includes 4 parameters: size (3) and orientation (1). For size estimation, the paper classifies the car into four categories, the size within each categories are known.

## [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](http://arxiv.org/abs/1903.10955)

![](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThAClOxeE6whdeqsicRKL6eoia9Bpvt2KALQLUoJJpvIyEOn8w1LzJjxQ7QQr49v6alQptk9HeRFeGxeVA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> We present an efficient 3D object detection framework based on a single RGB image in the scenario of autonomous driving. Our efforts are put on extracting the underlying 3D information in a 2D image and determining the accurate 3D bounding box of the object without point cloud or stereo data. Leveraging the off-the-shelf 2D object detector, we propose an artful approach to efficiently obtain a coarse cuboid for each predicted 2D box. The coarse cuboid has enough accuracy to guide us to determine the 3D box of the object by refinement. In contrast to previous state-of-the-art methods that only use the features extracted from the 2D bounding box for box refinement, we explore the 3D structure information of the object by employing the visual features of visible surfaces. The new features from surfaces are utilized to eliminate the problem of representation ambiguity brought by only using a 2D bounding box. Moreover, we investigate different methods of 3D box refinement and discover that a classification formulation with quality aware loss has much better performance than regression. Evaluated on the KITTI benchmark, our approach outperforms current state-of-the-art methods for single RGB image based 3D object detection.

# 3D Object Detection from Stereo Image

## [3D Object Proposals for Accurate Object Class Detection]()

![](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThAClOxeE6whdeqsicRKL6eoia9BKeTIKIgz8w2d8bZPaT0iaV3kssTcOibS5RCy4D1lqdJlKyozSV447F4g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> The goal of this paper is to generate high-quality 3D object proposals in the context of autonomous driving. Our method exploits stereo imagery to place proposals in the form of 3D bounding boxes. We formulate the problem as minimizing an energy function encoding object size priors, ground plane as well as several depth informed features that reason about free space, point cloud densities and distance to the ground. Our experiments show significant performance gains over existing RGB and RGB-D object proposal methods on the challenging KITTI benchmark. Combined with convolutional neural net (CNN) scoring, our approach outper-forms all existing results on all three KITTI object classes.

Similar to Faster-RCNN, the proposed method takes RGBD (depth computed from stereo) as input and generates 3d object proposal with 6 parameters: size (3) and orientation (3).

## [Stereo R-CNN based 3D Object Detection for Autonomous Driving](http://arxiv.org/abs/1902.09738)

![](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThAClOxeE6whdeqsicRKL6eoia9B9893NaqFCVoxAcLrbicXDZcJpyFVicON71IOyr4433uxd9qpITRk23sQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> We propose a 3D object detection method for autonomous driving by fully exploiting the sparse and dense, semantic and geometry information in stereo imagery. Our method, called Stereo R-CNN, extends Faster R-CNN for stereo inputs to simultaneously detect and associate object in left and right images. We add extra branches after stereo Region Proposal Network (RPN) to predict sparse keypoints, viewpoints, and object dimensions, which are combined with 2D left-right boxes to calculate a coarse 3D object bounding box. We then recover the accurate 3D bounding box by a region-based photometric alignment using left and right RoIs. Our method does not require depth input and 3D position supervision, however, outperforms all existing fully supervised image-based methods. Experiments on the challenging KITTI dataset show that our method outperforms the state-of-the-art stereo-based method by around 30% AP on both 3D detection and 3D localization tasks. Code has been released at https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN.

Stereo-RCNN is also based on Faster-RCNN. However, it combines the stereo match and 3d object detection in a unified framework.
