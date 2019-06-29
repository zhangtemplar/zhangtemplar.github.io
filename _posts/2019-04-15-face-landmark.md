---
layout: post
title: Face Landmark Detection
tags: deep-learning face face-landmark face-landmark-detection
---

![](https://cdn-images-1.medium.com/max/1200/1*Db5cCH72jLsV5lrgdAs78Q.jpeg)

Facial landmark detection algorithms aim to auto- matically identify the locations of the facial key land- mark points on facial images or videos. Those key points are either the dominant points describing the unique lo- cation of a facial component (e.g., eye corner) or an interpolated point connecting those dominant points around the facial components and facial contour.

The facial landmark detection algorithms can be divided into three major categories, according to their ways to utilize the facial appearance and shape information:
- holistic methods: explicitly build models to represent the global facial appearance and shape information
- Constrained Local Model (CLM) methods: explicitly leverage the global shape model but build the local appearance models
- the regression-based methods: implicitly capture facial shape and appearance information. 

This note will focus on the regression-based methods, especially those based on deep learning.

# [Deep convolutional network cascade for facial point detection](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sun_Deep_Convolutional_Network_2013_CVPR_paper.pdf)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/57ebeff9273dea933e2a75c306849baf43081a8c/3-Figure2-1.png)

This paper (Cascaded CNN) proposes a three-stage neural network to predict the facial landmark locations, where each stage of neural network takes the output of all neural network (each network covers one or several landmarks) of previous stage, thus each landmark location detection will receive the context information of the whole face.

The first stage (takes large portion as input) is able to generate high precision prediction and the next two stages (take patches centered at the predicition of first stage) further refines the prediction.

Note the convolution layer used in this Cascaded CNN is slightly different from typical ones: each map in the convolutional layer is evenly divided into p by q regions, and weights are locally shared in each region.


# [Facial landmark detection by deep multi-task learning](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/8a3c5507237957d013a0fe0f082cab7f757af6ee/8-Figure3-1.png)

This work (TCDCN) proposes to we to optimize facial landmark detection together with heterogeneous but subtly correlated tasks, e.g.head pose estimation and facial attribute inference. This is motivated by one task (e.g., head pose estimation) could help others.

To address different tasks may have different learning difficulties and convergence rate, task-wise early stop criterion is proposed, which is based on training error and validation error.

Compare with Cascaded CNN mentioned above, the TCDCN has only one network and predict all landmarks together.

# [HyperFace](https://arxiv.org/pdf/1603.01249)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/b2cd92d930ed9b8d3f9dfcfff733f8384aa93de8/2-Figure2-1.png)

HyperFace also tries to solve face detection, landmarks localization, pose estimation and gender recognition jointly, but by fusing the intermediate layers of the network. This is based on observation that while the lower layer features are effective for landmarks localization and pose estimation, the higher layer features are suitable for more complex tasks such as detection or classification

The network is based on R-CNN.

# [Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.702.1120&rep=rep1&type=pdf)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/22e2066acfb795ac4db3f97d2ac176d6ca41836c/2-Figure1-1.png)

CFAN also proposes a cascade neural network but instead of using CNN it uses Auto-Encoder. The first stage predicts the landmark location quickly from low resolution images; the next stages refines the prediction of previous stage with higher resolution images.

The first stage uses the whole face image as input and generates the locations of all landmarks. The next stage uses the patches extract around each landmark and concates them together, to explore the face model constraint.

# [Mnemonic descent method: A recurrent process applied for end-to-end face alignment](https://ibug.doc.ic.ac.uk/media/uploads/documents/trigeorgis2016mnemonic.pdf)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/193debca0be1c38dabc42dc772513e6653fd91d8/4-Figure2-1.png)

This paper proposes a combine and jointly trained recurrent convolution neural network (MDM) for face landmark detection, where convolution modules extracts feature for each landmark, and recurrent module utilizes the information cross cascades.

Especially, at the first timestamp, the inputs are patches extractes from mean face model, convolution modules are applied to extract features from those inputs; the recurrent module computes the landmark location offsets given the feature. At next timestamp, the landmark locations are updated according to the offset.

# [Face alignment across large poses: A 3d solution](http://cvlab.cse.msu.edu/pdfs/Liu_StanLi_CVPR2016.pdf)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/2a4153655ad1169d482e22c468d67f3bc2c49f12/3-Figure2-1.png)

The proposed method 3DDFA addresses the face landmark detection for large pose variations. The major challenge is under large pose variations, not all face landmarks will be visible, which is addressed by fitting a 3D face model to a 2D face image.

The 3D face model is based on 3D morphable model (3DM-M):
$$S=\bar{S}+A\alpha+B\beta$$
where $\bar{S}$ is mean face model, $A$ is the principal compoennts computed from faces with neural expression and $B$ counts for expression change.


