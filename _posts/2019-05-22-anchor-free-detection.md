---
layout: post
title: Anchor Free Object Detection
tags: deep-learning object-detection anchor-free centernet cornernet extremenet unitbox ssd yolo densebox psaf fcos foveabox ga-rpn
---

The most sucessfull single stage object detection algorithms, e.g., [YOLO](http://arxiv.org/abs/1506.02640), [SSD](https://arxiv.org/pdf/1512.02325.pdf), all relies all some anchor to refine to the final detection location. For those algorithms, the anchor are typically defined as the grid on the image coordinates at all possible locations, with different scale and aspect ratio.

![](https://cdn-images-1.medium.com/max/1600/1*7heX-no7cdqllky-GwGBfQ.png)

Though much faster than their two-stage counterparts, single stage algorithms' speed and performance is still limited by the choice of the anchor boxes: fewer than anchor leads better speed but deteroiates the accuracy. As a result, many new works are trying to design anchor free object detection algorithms.

The table summarizes the performance of some of the best anchor free methods:

| Methods | mAP | FPS | Code |
| --- | --- | --- | --- |
| FSAF | 42.9 | 5.3 | N.A. |
| FCOS | 43.2 | Nona | [tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS) |
| CenterNet：Objects as Points | 42.1 | 7.8 | [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet) |
| **CenterNet: Keypoint Triplets for Object Detection** | 44.9 | 3 | [Duankaiwen/CenterNet](https://github.com/Duankaiwen/CenterNet) |
| **AlignDet** | 44.1 | 5.6 | N.A | 

# UnitBox: An Advanced Object Detection Network

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LzOcSkG8Bnd39q5UYWylv8sEayaWWZqIQOpQ7xuVibPCPYt2G9l3JPIg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LK9iazsL16aRpIqzCrIpKhliarDDGHEDxEVhX2XibXe1QMtkmkgkGDkeGg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

UnitBox uses Intersection over Union (IoU) loss function for bounding box prediction.

# DenseBox: Unifying Landmark Localization and Object Detection

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LYc8cvGVeZfACEf8JUwicZohbb8YXfhbJfIwkNZSib0J8shyAuU03uFDA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LpwWjKNia0I5zgS6p96MmEskkicDhGf7AqUmH4dx8ibdMf9qAEELgzLjkQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LaHjeTJhxYbJFkEdEP0nWicNqXJxtBmia8kNt5zwpEZbw4BsicPzt9AmGA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

DenseBox directly compute the bounding box and its label from the feature map.

# CornerNet: Detecting Objects as Paired Keypoints

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0L3ctVVSmzxIVCkExZNkzg30wGzxHkf06TiaVslj44EF58ycicaRXicDjgA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

In CornerNet, the bounding box is uniquely defined by its top-left corner and bottom-right corner, which is detected by each of the two branches. Corner-pooling is applied to detect the corners, which utilizes the ideas of integral image (see below)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LD86GAmEbLxZxVIbaFJMOhCbppdicrGpWqxftBtLIu7qQBMGSS5kic4Sg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LhYtzpjiagDy1mx727pv1lGZLT9P4MN4J3FJKxdOKv5gY78niccdcXLwQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

# [ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247486939&idx=2&sn=a0c328e245f3f103175efa752604ad58&scene=21#wechat_redirect)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LT64vAia9vmoTmpt0jcBjibFcp3C0odVAiajwoQZQVaicSvsjiaKkz5HDafA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LIgjyDI3Jkf6hgOiaOiaHPr8ZAbKHoTnxFz2oTh03FEUibEuAZXEjAicFHQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Similar as CornerNet, it formulates the problem of finding bounding box as finding some corner points. But instead of two corners as in CornerNet, it requires four corner points and one center point, which is computed via peaks of heatmaps of each corner points.

# [FSAF: Feature Selective Anchor-Free](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247487645&idx=2&sn=74ea4e6e524468493fbf305ae86f9f52&scene=21#wechat_redirect)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LJCRy91ibAlAsEB7ZNfEMEvxJkMef7893P2x5R3b5UBxlqUIA3wpMhhg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0Lsc7XibLzicf0r7PHdRBjz3JZqvYeN9hxicn6O4gibGpZHSG4KrQLwc5mIw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0Lcuibb06My5dSicDRxicrlM6rbGjYa8d4cW8xaL0FKAiaia0jS5V3glnLKuQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

It is based on feature pyramid network, where the final result is dynamically selected from the optimal resolution.

# [FCOS: Fully Convolutional One-Stage](https://mp.weixin.qq.com/s?__biz=MzU4OTg3Nzc3MA==&mid=2247483704&idx=1&sn=01c6d16be8e3990e9f5ccae13599f7de&scene=21#wechat_redirect)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LvBBpibWrfwdSXa6bbKM1HLzfhMZQIwJhCxoCm5rYSqaghIs4oicLo5MA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

FCOS is anchor-box free, as well as proposal free. FCOS works by predicting a 4D vector (l, t, r, b) encoding the location of a bounding box at each foreground pixel (supervised by ground-truth bounding box information during training). 

This done in a per-pixel prediction way, i.e., for each pixel, the network try to predict a bounding box from it, together with the label of class. To counter for the pixel which are far from the ground truth object (center), a centerness score is also predicted which downweights the prediction for those pixels.

If a location falls into multiple bounding boxes, it is considered as an ambiguous sample. For now, we simply choose the bounding box with minimal area as its regression target.

Feature Pyramid Network is used as the backbone.

# FoveaBox: Beyond Anchor-based Object Detector

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LW4b4okqEw31tt7o4IZ0WNEFsaiciaXRYhfTJg9Fn3FpfIMuU4icFoB2qg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0Ljs1AqgtEc9jQpeibLT5RpCyibViccLj4yDK5CDLfOaUeN4fsibQjOaZHdg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

It is very similar to FCOS.


# [Region Proposal by Guided Anchoring(GA-RPN)](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247487049&idx=1&sn=820a5adcb6beb444326ac1d90630f481&scene=21#wechat_redirect)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0Lg9aBScmc5vbx7eibpICPfndpGpiaUbkzmcicnKZEiabsbq5UOAmgM4NtJQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

In GA-RPN, the anchor (defined as a tuple of its location and shape) is learned instead of manually defined. Then feature extraction is then adapted to this computed anchor.
CenterNet: Objects as Points

# CenterNet: Objects as Points

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0Lnia89KDm0CMyzL67zXt7qv43vA0mFKpIWLS3ibOvwYZ83CPKATmASNXQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0L2TgNoVbvYQCoUE984rPsuiccDfKmta3XVajT5834m9Ikm69k1NnhlMA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

CenterNet defines the bounding box by its center. After the center is computed, its shape and pose can be further computed.

# [CenterNet: Object Detection with Keypoint Triplets](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247488634&idx=1&sn=877122d09512321bc6a1cc94a3d75fc2&scene=21#wechat_redirect)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LH5iczkq229yLv31ciaH9AhlWmoGicYBtiacZia1PbAnsf0udVvfbMjGv0Yw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0L1nriabcTVdz6X2rz3oX1DQXICtZTuJiaVDiaG6icEyTsrH23uNg9rOQRlA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

It is based on CenterNet but very similar to ExtremeNet or CornerNet, where the bounding box is now defined by a pair of corner points and the label is defined by the response of the center point.

# [CornerNet-Lite: Efficient Keypoint Based Object Detection](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247488676&idx=1&sn=d2cb1f991687379756ca26dd4bf072db&scene=21#wechat_redirect)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LejkaY57fbNONtcBCUl6fuYkFI1jZQiaJt9rOuTgJ8YtBz3qp2AaWdcw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

CornerNet-Lite：CornerNet-Saccade（attention mechanism）+ CornerNet-Squeeze

# [Center and Scale Prediction: A Box-free Approach for Object Detection]

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0LTHBjImc9t9y9mnnEnH1MiaKJEevS5heHA47OTAoicIpFFiawKs1CLQYmg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVtmWMCMAYFgyQHf8Yn8V0Lia8OfXXecHXMfwfzeEogOCNncoLMmliabHYjRcN5OkiakOYAYSylibuHxA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

As GA-RPN, the bounding box is defined by its center and shape, which is computed from two branches of the neural network.

# [Matrix Nets](https://arxiv.org/abs/1908.04646)

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9eTcrcwFsib63LiacvgWwib9lZCVgVNXvTYfic0cnRLciap91gibXTiaQYs57M6ibCtCzyhTOuzk4IiabP7nw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Maxtrix Nets addresses different aspect ratio of the objects. Compared with image pyramid, it generates feature across different scales and aspect ratio, like a matrix. Especially, for feature x_{ij} at layer i and j, feature x_{i+1, j+1} is generated via a 3x3 kernel with stride 2x2, feature x_{i+1, j} is generated via a 3x3 kernel with stride 2x1 and feature x_{i, j+1} is generated via a 3x3 kernel with stride 1x2. Note those three convolutions share the same parameter.

To detect objects, it utilizes the similar idea of center net: the location of top left corner and bottom right corner are detected, the centerness is also computed. Then the result for feature x_{i+k, i+[0:k]} and x_{i+[0:k], i+k} is aggreated via nonmax suppression.
