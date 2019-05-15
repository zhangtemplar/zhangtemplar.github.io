---
layout: post
title: Siamese Network Based Single Object Tracking
tags: deep-learing single-object-tracking siamese siamfc siamrpn siammask dasiamrpn
---

Siamese network is an artificial neural network that use the same weights while working in tandem on two different input vectors to compute comparable output vectors.

![](https://cdn-images-1.medium.com/max/1600/1*hBJRs10uBc9a2Ol10N-jlg.png)

# Algorithms

## [SiamFC](https://arxiv.org/abs/1606.09549)
> Fully-Convolutional Siamese Networks for Object Tracking, Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr., The European Conference on Computer Vision (ECCV) Workshops, 2016.

![](https://www.robots.ox.ac.uk/~luca/stuff/siamesefc_conv-explicit.jpg)

SiamFC formulates the tracking problem into a similarity learning problem, for which a fully convolutional Siamese network is trained to locate an exemplar image within a larger search image. The network compares an exemplar image z (the initial appearance of the object) to a candidate image x of the same size and returns a high score if the two images depict the same object and a low score otherwise. The backbone is Alexnet.

For training, pairs are obtained from a dataset of annotated videos by extracting exemplar and search images that are centred on the target. The images are extracted from two frames of a video that both contain the object and are at most T frames apart. The elements of the score map are considered to belong to a positive example if they are within radius R of the centre. Note the network is learned offline and **does not** update via tracking.

## [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
> High Performance Visual Tracking with Siamese Region Proposal Network, Bo Li, Wei Wu, Zheng Zhu, Junjie Yan, Xiaolin Hu., IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

![](http://bo-li.info/SiamRPN/img/SiamRPN.jpg)

SiamRPN focuses on the speed of the deep neural network based tracking algorithm. It consists of Siamese subnetwork for feature extraction and region proposal subnetwork including the template branch and detection branch. In the inference phase, the proposed framework is formulated as a local one-shot detection task, where the bounding box in the first frame is the only exemplar.

Similar as Faster-RCNN, the template branch predicts foreground/background and the regression branch compute the bounding box offset for each of the k anchor points. The template branch of the Siamese subnetwork is pre-computed and the correlation layers (denoted as star) as trivial convolution layers is used to perform online tracking.

Alexnet is used as the backbone.

## [DaSiamRPN](https://arxiv.org/abs/1808.06048)
> Distractor-aware Siamese Networks for Visual Object Tracking, Zheng Zhu, Qiang Wang, Bo Li, Wu Wei, Junjie Yan, Weiming Hu., The European Conference on Computer Vision (ECCV), 2018.

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/776bc8955e801f6965e85b35d8e2dd6f2f1498ad/6-Figure2-1.png)

DaSiamRPN considers that features used in most Siamese tracking approaches can only discriminate foreground from the non-semantic backgrounds, whereas the semantic backgrounds are always considered as distractors. To solve this problem, an effective sampling strategy is introduced in the training stage to address the imbalanced distribution of training data. During inference, a novel distractor-aware module is designed to perform incremental learning.

Sampling strategies:
- Diverse categories of positive pairs can promote the generalization ability, which can be achieved via data augmentation on dataset for object detection tasks.
- Semantic negative pairs can improve the discriminative ability. For existing algorithms, most negative samples are non-semantic (not real object, just background), and they can be easily classified. To address this, the constructed negative pairs consist of labelled targets both in the same cate- gories and different categories.
- Customizing effective data augmentation for visual tracking, e.g., motion blur.

During inference, the detection with the highest score is selected as the tracking result, whereas others detections with sufficient high score is used as hard negative samples, which are then used to incrementally update the tracker. For long term tracking, the faliure case (object out of scene) need to handled, where an iterative local-to-global search strategy is designed to re-detect the target.

## [SiamRPN++](https://arxiv.org/abs/1812.11703)
> SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks, Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan., IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

![](http://bo-li.info/SiamRPN++/img/SiamRPN_plus_plus.png)

This paper studies how to apply modern neural network backbone (e.g., ResNet) for siamese network based tracker. It is achieved via a simple yet effective spatial aware sampling strategy. Moreover, we propose a new model architecture to perform layer-wise and depth-wise aggregations, which not only further improves the accuracy but also reduces the model size.

When using deep networks for siamese network based tracker, the decrease in accuracy comes from the destroy of the strict translation invariance because of padding for convolution. To address this, spatial aware sampling strategy is introduced.

## [SiamMask](https://arxiv.org/abs/1812.05050)
> Fast Online Object Tracking and Segmentation: A Unifying Approach, Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr., IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

![](http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask.jpg)

SiamMask formulates the problems of visual tracking and visual object segmentation as a joint learning of three tasks:
- to learn a measure of similarity between the target object and multiple candidates in a sliding window fashion
- bounding box regression using a Region Proposal Network
- class-agnostic binary segmentation: binary labels are only required during offline training to compute the segmentation loss and not online during segmentation/tracking

Once trained, SiamMask solely relies on a single bounding box initialisation, operates online without updates and produces object segmentation masks and rotated bounding boxes at 55 frames per second.

# [PySOT](https://github.com/STVIR/pysot)

![](https://github.com/STVIR/pysot/raw/master/demo/output/bag_demo.gif)

[**PySOT**](https://github.com/STVIR/pysot) is a software system designed by SenseTime Video Intelligence Research team. It implements state-of-the-art single object tracking algorithms, including [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) and [SiamMask](https://arxiv.org/abs/1812.05050). It is written in Python and powered by the [PyTorch](https://pytorch.org) deep learning framework. This project also contains a Python port of toolkit for evaluating trackers.

PySOT includes implementations of the following visaul tracking algorithms:

- [SiamMask](https://arxiv.org/abs/1812.05050)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [SiamFC](https://arxiv.org/abs/1606.09549)

using the following backbone network architectures:

- [ResNet{18, 34, 50}](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

Evaluation toolkit can support the following datasets:

- [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 
- [VOT16/18/19](http://votchallenge.net) 
- [VOT18-LT](http://votchallenge.net/vot2018/index.html) 
- [LaSOT](https://arxiv.org/pdf/1809.07845.pdf) 
- [UAV123](https://arxiv.org/pdf/1804.00518.pdf)
