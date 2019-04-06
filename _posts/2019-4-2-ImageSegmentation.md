---
layout: post
title: Image Segmentation
tags: deep-learning image-segmentation fcn unet segnet dilated-convolutions refinenet pspnet deeplab mask-rcnn skip scribble
---

There are two types of image segmentation:
- semantic segmentation: assign labels to each pixel
- instance segmentation: extract the bounary/mask for each instance in the image

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/Instance-01.png)

# Examples
- Fully Convolution Neural Network (FCN): FCN convert the fully layers in traditional network used for image classifcation, e.g., Alexnet, VGG16, to convolution layer. Thus is could generate a probability map of each pixel.
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.1.1_2.jpg)
- UNet: it combines two parts: left part uses convolution and max-pooling to extract feature; right part uses upsampling and skip (input from lower layer of left part) to generate the label map.
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.2_1.png)
- SegNet: similar to UNet, but it doesn't use skip to combine the input from lower layer of left part (refered encoder network).
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.3_1.jpg)
- Dilated Convolutions: the problem in FCN is that, using pooling and then up-sampling will cause data loss. Dilate convolution resolves this problem by adding `dilate` to convolution, which increases the field size while doesn't reduce the output size as pooling dose.
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.3_4.jpg)
- RefineNet: similar to UNet, but utilizes the ResNet as the base.
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.4_2.png)
- PspNet: is applies the idea of spatial pyramid pooling to image segmentation,
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.6_2.png)
- DeepLab: combines Atrous Convolution (similar to Dilated Convolutions) with PspNet.
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.6_6.png)
- Mask-R-CNN: uses the idea of object detection for semantic segmentation, where the probability of each boundary box is used as a response map, where softmax is then applied to generate the mask.
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.8_1.png)

# Common Techniques
- transposed convolution: a conjugate pair of convolution operator, whose forward propgation is the backward propagation of convolution operation and vice versa
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.1.8_1.png)
- skip: combine the output of intermidiate layers to have multiple level features
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.1.9_1.png)

# Labels for Training Data

- [Scribble](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lin_ScribbleSup_Scribble-Supervised_Convolutional_CVPR_2016_paper.pdf) uses a few simple scribbles as the label of the training image. Cost function is $$ \sum_{i}\psi _i^{scr}\left(y_i|X,S\right)+\sum i-logP\left(y_i| X,\theta\right)+\sum{i,j}\psi _{ij}\left(y_i,y_j|X\right) $$
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.9_1.png)
- [Image-level label](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Pathak_Constrained_Convolutional_Neural_ICCV_2015_paper.pdf) the label if provides to image level and there is no pixel level label, like image classificaition case. Cost function is $$ \underset{\theta ,P}{minimize}\qquad D(P(X)||Q(X|\theta ))\ subject\to\qquad A\overrightarrow{P} \geqslant \overrightarrow{b},\sum_{X}^{ }P(X)=1 $$
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.9_4.png)
- [Bounding box and label](https://arxiv.org/pdf/1502.02734.pdf): the label is some bounding boxes and their labels, as object detection case. Cost function is $$ P\left ( x,y,z;\theta \right ) = P\left ( x \right )\left (\prod_{m=1}^{M} P\left ( y_m|x;\theta \right )\right )P\left ( z|y \right ) $$
![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch09_%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/img/ch9/figure_9.9_6.png)
