---
layout: post
title: Object Detection 
tags: deep-learning object-detection r-cnn fast-rcnn faster-rcnn mask-rcnn yolo r-fcn ssd
---

Here is the comparison of the most popular object detection frameworks.

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/img/ch8/8.1.2.png)

- **[R-CNN](https://arxiv.org/abs/1311.2524)** use [selective search](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html#selective-search) to generate region proposal, extract patches from those proposal and apply image classification algorithm.
![](https://lilianweng.github.io/lil-log/assets/images/RCNN.png)

- **[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)** apply CNN on image then use ROI pooling layer to convert the feature map of ROI to fix length for future classification. Note it still requires an external region proposal generator.
![](https://lilianweng.github.io/lil-log/assets/images/fast-RCNN.png)

- **[Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)** combine region proposal network and classification CNN, which share the same feature extraction layer. RPN generates region proposals and confidence for each anchor point (e.g., regular grid) and classification CNN apply ROI pooling layer for those proposal as well.
![](https://lilianweng.github.io/lil-log/assets/images/faster-RCNN.png)

- **[Mask-RCNN](https://arxiv.org/pdf/1703.06870.pdf)** similar to Faster RCNN, but add a branch to generate a binary mask for segmentation for each ROI.
![](https://lilianweng.github.io/lil-log/assets/images/mask-rcnn.png)

- **[YOLO](http://arxiv.org/abs/1506.02640)** similar to Faster RCNN, the network generates a lot of region proposal (from regular grid) and its confidence, then apply non-maximal suppression.
![](https://camo.githubusercontent.com/c54ee9c13e406046c35553e5da32175801a25b93/687474703a2f2f706a7265646469652e636f6d2f6d656469612f696d6167652f6d6f64656c5f322e706e67)

- **[R-FCN](http://arxiv.org/abs/1605.06409v2)** similar to Faster RCNN, but it apply ROI pooling layer at the very end of network and the output of that layer is directly the classification probability.
![](https://www.groundai.com/media/arxiv_projects/35313/eps/overall.svg)

- **[SSD](https://arxiv.org/pdf/1512.02325.pdf)** like YOLO, but it use fuse the response from not only the last convolution layer but also layers before them.
![](https://cdn-images-1.medium.com/max/2400/1*up-gIJ9rPkHXUGRoqWuULQ.jpeg)
