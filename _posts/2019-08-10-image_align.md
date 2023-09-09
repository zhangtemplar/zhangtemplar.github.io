---
layout: post
title: Image Alignment
tags:  deep-learning alignment image
---

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85v9gh2Dv2npxyiagtYaQiabicjc5m97mHnpUGAmnMQGr5Q8tRiasZtABBJHcQQticIlnde1muq6gssHA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Image alignment aims to find a transform which transform a source image into the target image. Many different types of transforms are available:
- rigid transform (rotate, translate)
- affine transform (scale, shear)
- homography transform
- other

# Classical Methods

Classical method usually relies on feature matching between two images: the feature could be the image itself, or feature from some points from the image. The feature point based methods are more robust and thus more widely used, which typically contains the following steps:
- detect feature points from each of the image
- compute the feature vector for each feature points
- find the correspondance between feature points of one image and feature points of the other image
- find the transfrom from the correspondance, via, e.g., [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85v9gh2Dv2npxyiagtYaQiabU3PZJtqblEM0jGgfYXy2jRd75tRIATyicQo7F2EQKCibT6hrJgUUNGicA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

There are many choices for the feature points detector and descriptors:
- Scale-invariant feature transform (SIFT): by using histogram of gradient, it is generally robust over scale, rotation and translation, certain illumination change. Not free for commercial usage.
- Speeded Up Robust Features (SURF): a fast version of SIFT. Not free for commercial usage.
- Oriented FAST and rotated BRIEF (ORB): based on the FAST keypoint detector and the visual descriptor BRIEF (Binary Robust Independent Elementary Features). Its aim is to provide a fast and efficient alternative to SIFT. It is robust over scale, rotation and translation, certain illumination change and noise. Free to use.
- Accelerated-KAZE [AKAZE]: It is robust over scale, rotation and translation. Free to use.

# Deep Learning Base Methods

## Deep Learning Based Keypoints Descriptor

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85v9gh2Dv2npxyiagtYaQiab2X5VFjeEq7NoT4qlAp2zFZEhUJKbYf6X4cP4QL0CgHbEfiasojTbGrg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Several works have been proposed to learn better keypoints descriptor, e.g.,
- [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)
- [Multi-temporal Remote Sensing Image Registration Using Deep Convolutional Features](https://ieeexplore.ieee.org/document/8404075)

## End to End Methods for Learning Transform

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85v9gh2Dv2npxyiagtYaQiabo67ibNyj3ZNLe6Kicgcqzec987dethvbXJnpicINAbSqMviaBkKhm594VQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

The transform can be also learned end to end in deep learning models, e.g., [Deep Image Homography Estimation](https://arxiv.org/pdf/1606.03798.pdf) concatenate two images to align in channels and then utilize VGG network to learn the homography transform (8 parameters).

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85v9gh2Dv2npxyiagtYaQiabENJYOica14hiagaWw41jy3hMnEkibpDbmY4KbGv4omWuTJvl6SU93NMmQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Instead of requiring the ground truth transform label as supervision signal, [Unsupervised Deep Homography: A Fast and Robust Homography Estimation Model](https://arxiv.org/pdf/1709.03966.pdf) removed this requirement by comparing the similarity of an image aligned to the other.

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85v9gh2Dv2npxyiagtYaQiabqpJaMuc54fzoQjhguH2XRoMNE55Edrtj4RK2VaRHicTJlkXxbicn1UiaQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

[An Artificial Agent for Robust Image Registration](https://arxiv.org/pdf/1611.10336.pdf) proposed a reinforcement learning based approach for learning the transform between two images, where the state is the two images and the action is the transform. Similar idea was also studied in [Robust non-rigid registration through agent-based action learning](https://hal.inria.fr/hal-01569447/document）
