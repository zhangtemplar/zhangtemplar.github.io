---
layout: post
title: Object Detections
tags:  mask-rcnn yolo r-cnn deep-learning faster-rcnn ssd object-detection cornernet hourglass retina-net fcos r-fcn focal-loss fast-rcnn centernet
---

Here is the comparison of the most popular object detection frameworks.

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.1.2.png)

# [R-CNN](https://arxiv.org/abs/1311.2524)
Use [selective search](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html#selective-search) to generate region proposal, extract patches from those proposal and apply image classification algorithm.
![](https://lilianweng.github.io/lil-log/assets/images/RCNN.png)

# [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
Apply CNN on image then use ROI pooling layer to convert the feature map of ROI to fix length for future classification. Note it still requires an external region proposal generator.
![](https://lilianweng.github.io/lil-log/assets/images/fast-RCNN.png)

# [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
Combine region proposal network and classification CNN, which share the same feature extraction layer. RPN generates region proposals and confidence for each anchor point (e.g., regular grid) and classification CNN apply ROI pooling layer for those proposal as well.
![](https://lilianweng.github.io/lil-log/assets/images/faster-RCNN.png)

# [Mask-RCNN](https://arxiv.org/pdf/1703.06870.pdf)
Similar to Faster RCNN, but add a branch to generate a binary mask for segmentation for each ROI.
![](https://lilianweng.github.io/lil-log/assets/images/mask-rcnn.png)

# [YOLO](http://arxiv.org/abs/1506.02640)

Similar to Faster RCNN, the network generates a lot of region proposal (from regular grid) and its confidence, then apply non-maximal suppression.
![](https://camo.githubusercontent.com/c54ee9c13e406046c35553e5da32175801a25b93/687474703a2f2f706a7265646469652e636f6d2f6d656469612f696d6167652f6d6f64656c5f322e706e67)

# [R-FCN](http://arxiv.org/abs/1605.06409v2)

Similar to Faster RCNN, but it apply ROI pooling layer at the very end of network and the output of that layer is directly the classification probability.
![](https://www.groundai.com/media/arxiv_projects/35313/eps/overall.svg)

# [SSD](https://arxiv.org/pdf/1512.02325.pdf)

Like YOLO, but it use fuse the response from not only the last convolution layer but also layers before them.

![](https://cdn-images-1.medium.com/max/2400/1*up-gIJ9rPkHXUGRoqWuULQ.jpeg)

# [RetinaNet](https://arxiv.org/abs/1708.02002)

> Focal Loss for Dense Object Detection

![](https://cdn-images-1.medium.com/max/1600/1*IIuPgetzAtJM0OAW35QiVA.png)

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the **extreme foreground-background class imbalance encountered during training of dense detectors is the central cause.**

We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. The focal loss function can be written as:

$$FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t)$$
where $p_t=p$ for $y=1$ otherwise $p_t=1-p$. For an easy example, $p_t\approx 1$, thus doesn't affect $FL(p_t)$. $\gamma=2$ and $\alpha=0.5$ is the typical choice.

Feature Pyramid Network is used as the backbone.

# [FCOS](https://arxiv.org/abs/1904.01355)

> Fully Convolutional One-Stage Object Detection

![](http://www.ishenping.com/Images/artImg/20190411090121699_UPUXVE.jpg)

FCOS is anchor-box free, as well as proposal free. FCOS works by predicting a 4D vector (l, t, r, b) encoding the location of a bounding box at each foreground pixel (supervised by ground-truth bounding box information during training). 

This done in a per-pixel prediction way, i.e., for each pixel, the network try to predict a bounding box from it, together with the label of class. To counter for the pixel which are far from the ground truth object (center), a centerness score is also predicted which downweights the prediction for those pixels.

If a location falls into multiple bounding boxes, it is considered as an ambiguous sample. For now, we simply choose the bounding box with minimal area as its regression target.

Feature Pyramid Network is used as the backbone.

# [Objects as Points](http://arxiv.org/abs/1904.07850v1)

![Screen Shot 2019-04-17 at 9.55.24 PM.png](quiver-image-url/EA406B3C60A69B9AFBBD58FDDA2105BB.png =799x233)

This paper proposes a CenterNet, which formulates the object detection problem into the problem of detection the center of object and their size of the bounding box is then inferred from the neighbor around the center.

# [CornerNet-Lite](https://arxiv.org/pdf/1904.08900)

![](https://media.arxiv-vanity.com/render-output/572847/x2.png)

CornerNet is yet another a single-stage object detection algorithm. CornerNet detects and groups the top-left and bottom-right corners of bounding boxes; it uses a stacked hourglass network to predict the heatmaps of the cor- ners and then uses associate embeddings to group them.

To improve the efficiency of CornerNet, CornerNet-Lite combines two of its variants, CornerNet-Saccade, which uses an attention mechanism to eliminate the need for exhaustively processing all pixels of the image, and CornerNet-Squeeze, which introduces a new compact backbone architecture. 

CornerNet-Saccade compute three attention maps for predicting locations of objects at three scales: small (<32), medium (32~96) and lare (>96). The attention maps are generated from multi-scale of a hourglass network.
