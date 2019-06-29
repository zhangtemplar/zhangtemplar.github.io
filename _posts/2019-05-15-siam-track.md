---
layout: post
title: Siamese Network Based Single Object Tracking
tags: deep-learning single-object-tracking siamese siamfc siamrpn siammask dasiamrpn sint cfnet dsiam sint++ sa-siam rasnet siamfc-tri StructSiam DenseSiam MBST Siam-BM C-RPN CIR
---

Siamese network is an artificial neural network that use the same weights while working in tandem on two different input vectors to compute comparable output vectors.

![](https://cdn-images-1.medium.com/max/1600/1*hBJRs10uBc9a2Ol10N-jlg.png)

> Partially based on [基于孪生网络的目标跟踪算法汇总](https://blog.csdn.net/WZZ18191171661/article/details/88369667)

# Algorithms

The figure belows summaries the history of Siamese network based trackrs.

![](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzbcic1aOuEqZa4XxqPxDp79duqS4kngYIWB90dOK86mfGdFCadWxpSZw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nz5NHiclDkB1IGhibvUgCBKex6OMOnuKvUZcKj48icnmnO2cIElmp8t3bAQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## [SINT](https://arxiv.org/abs/1605.05863)

> [project page](https://taotaoorange.github.io/projects/SINT/SINT_proj.html) and [code](https://github.com/taotaoorange/SINT)

![](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzBv2ZKCXNILxvcTXYKOLLSribJVeEttibyibYnJVu0ibCaH13bjT4GiaJb8A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

This is the first work proposing to use Siamese network for visual tracking.

![](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzuTib8olRaJ04YrcKatoBkagPV3NChGOD8OiaowOuJjqeBaPSadvNsmvQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## [SiamFC](https://arxiv.org/abs/1606.09549)
> Fully-Convolutional Siamese Networks for Object Tracking, Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr., The European Conference on Computer Vision (ECCV) Workshops, 2016.
> [project](http://www.robots.ox.ac.uk/~luca/siamese-fc.html) [code](https://github.com/bertinetto/siamese-fc)

![](https://www.robots.ox.ac.uk/~luca/stuff/siamesefc_conv-explicit.jpg)

SiamFC formulates the tracking problem into a similarity learning problem, for which a fully convolutional Siamese network is trained to locate an exemplar image within a larger search image. The network compares an exemplar image z (the initial appearance of the object) to a candidate image x of the same size and returns a high score if the two images depict the same object and a low score otherwise. The backbone is Alexnet.

For training, pairs are obtained from a dataset of annotated videos by extracting exemplar and search images that are centred on the target. The images are extracted from two frames of a video that both contain the object and are at most T frames apart. The elements of the score map are considered to belong to a positive example if they are within radius R of the centre. Note the network is learned offline and **does not** update via tracking.

## [CFNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Valmadre_End-To-End_Representation_Learning_CVPR_2017_paper.pdf)

> [project](http://www.robots.ox.ac.uk/~luca/cfnet.html) and [code](https://github.com/bertinetto/cfnet)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzDeuEwVXApNibCX7MXrzic0r6icRzsaPxPaoGreakbPd7oAoCa8EzB6ib8w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

The improvement over SiamFC of CFNet is to have a dedicated layer for correlation operation.

## [DSiam](http://openaccess.thecvf.com/content_ICCV_2017/papers/Guo_Learning_Dynamic_Siamese_ICCV_2017_paper.pdf)

> [code](https://github.com/tsingqguo/DSiam)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzhjSB6hvbv8WQ36YY2wiaFx4gWnArdwoicedl4wZrUzZG8uy8jwwJ5D3g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Compared with SiamFC, DSiam introduces two online fast transformations, i.e. target variation transformation and background suppression transformation, which makes SiamFC adapt target changes while excluding background interferences.

## [SINT++](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SINT_Robust_Visual_CVPR_2018_paper.pdf)

> [project](https://sites.google.com/view/cvpr2018sintplusplus/)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzAfkg72dQ7Wjk3G8yYEK8vz3w6LE83icCZjsP5ecWhu9cZwrccHWIHibQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

SINT++ improves SINT via introducing AutoEncoder and GAN to generate variations of positive sample to enhance the robustness of the tracker.

## [SA-Siam](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_A_Twofold_Siamese_CVPR_2018_paper.pdf)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzvun6rB34bGWpZoZ6FGRM3G4DjKiaFAJnNJGe1LWV4XjjeyibrUZIzNRA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Two networks, namely A-Net and S-NET, are used to extract the appearance feature and semantic feature accordingly. In addition, an attention mechanism is introduced.

## [RASNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Attentions_Residual_CVPR_2018_paper.pdf)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzpffHmrKPK2MiaDicCVpL4t7G8BnLlTBLhDrT6pGHEhTzl7zZ92x0DHXQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

RASNet further explores the idea of the attention mechanism. It utilizes three attention mechenisms to remove the requirement of model online update during tracking:
- residual attentions: distinctiveness of current tracking target over the common samples
- general attentions: learned from all training samples
- channel attentions: selecting semantic attributes for different contexts

The backbone of the attention module in the RASNet is an Hourglass-like Convolutional Neural Network (CNN) model to learn contextualized and multi-scaled feature representation.

## [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)

> High Performance Visual Tracking with Siamese Region Proposal Network, Bo Li, Wei Wu, Zheng Zhu, Junjie Yan, Xiaolin Hu., IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
> [project](http://bo-li.info/SiamRPN/) [code in pytorch](https://github.com/songdejia/Siamese-RPN-pytorch) [code in tensorflow](https://github.com/makalo/Siamese-RPN-tensorflow)

![](http://bo-li.info/SiamRPN/img/SiamRPN.jpg)

SiamRPN focuses on the speed of the deep neural network based tracking algorithm. It consists of Siamese subnetwork for feature extraction and region proposal subnetwork including the template branch and detection branch. In the inference phase, the proposed framework is formulated as a local one-shot detection task, where the bounding box in the first frame is the only exemplar.

Similar as Faster-RCNN, the template branch predicts foreground/background and the regression branch compute the bounding box offset for each of the k anchor points. The template branch of the Siamese subnetwork is pre-computed and the correlation layers (denoted as star) as trivial convolution layers is used to perform online tracking.

Alexnet is used as the backbone.

## [SiamFC-tri](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xingping_Dong_Triplet_Loss_with_ECCV_2018_paper.pdf)

> [code](https://github.com/shenjianbing/TripletTracking)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzITsdsSL6G1lA7lxlrE18vAFvYk5OvoqJoBYZ2kJTHia3qkyus7H9jDg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Similar as SiamFC, but introduces triplet loss.

## [StructSiam](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yunhua_Zhang_Structured_Siamese_Network_ECCV_2018_paper.pdf)

> [code](https://github.com/xiaobai1217/StructSiam)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nz21Dy4k2v2tIOiaibLBn1lXlMxiagXjyrZ9T0ibShSZ30VhqL1UiazSC4AbA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

StructSiam proposes a local structure learning method, which simultaneously considers the local patterns of the target and their structural relationships for more accurate target tracking. To this end, a local pattern detection module is designed to automatically identify discriminative regions of the target objects.

## [DaSiamRPN](https://arxiv.org/abs/1808.06048)
> Distractor-aware Siamese Networks for Visual Object Tracking, Zheng Zhu, Qiang Wang, Bo Li, Wu Wei, Junjie Yan, Weiming Hu., The European Conference on Computer Vision (ECCV), 2018.
> [code](https://github.com/foolwood/DaSiamRPN)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/776bc8955e801f6965e85b35d8e2dd6f2f1498ad/6-Figure2-1.png)

DaSiamRPN considers that features used in most Siamese tracking approaches can only discriminate foreground from the non-semantic backgrounds, whereas the semantic backgrounds are always considered as distractors. To solve this problem, an effective sampling strategy is introduced in the training stage to address the imbalanced distribution of training data. During inference, a novel distractor-aware module is designed to perform incremental learning.

Sampling strategies:
- Diverse categories of positive pairs can promote the generalization ability, which can be achieved via data augmentation on dataset for object detection tasks.
- Semantic negative pairs can improve the discriminative ability. For existing algorithms, most negative samples are non-semantic (not real object, just background), and they can be easily classified. To address this, the constructed negative pairs consist of labelled targets both in the same cate- gories and different categories.
- Customizing effective data augmentation for visual tracking, e.g., motion blur.

During inference, the detection with the highest score is selected as the tracking result, whereas others detections with sufficient high score is used as hard negative samples, which are then used to incrementally update the tracker. For long term tracking, the faliure case (object out of scene) need to handled, where an iterative local-to-global search strategy is designed to re-detect the target.

## [DenseSiam](https://arxiv.org/abs/1809.02714)

> [code](http://www.votchallenge.net/vot2018/trackers.html)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nz4GTNMOhERibaEib3V2axibFJdWh3jPrrKMCxxHj2ZlwerRFDAqfYatedQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

DensSiam, a novel convolutional Siamese architecture, which uses the concept of dense layers (similar as DenseNet) and connects each dense layer to all layers in a feed-forward fashion with a similarity-learning function. DensSiam also includes a Self-Attention mechanism to force the network to pay more attention to the non-local features during offline training.

## [MBST](https://arxiv.org/abs/1808.07349)

> [code](https://github.com/zhenxili96/MBST)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nz7TPIjGWaibEvLmC9e05X5BQMBtq0Sh102gpM0zDCs8xPkwcTOrGO7Sg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

MBST learns multiple Siamese network and uses a selection module to select the network on the fly during inference.

## [Siam-BM](https://arxiv.org/abs/1809.01368)

> [project](https://77695.github.io/Siam-BM) [code](https://github.com/77695/Siam-BM)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzL6jyfD7l1Zde2zKOuufCmR1wsDhHW8ACQjXIoDeXNGdt5A9OR3xoHw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Siam-BM addresses the scale change and rotation during tracking by augment the query patches via rotation and scaling.

## [C-RPN](https://arxiv.org/abs/1812.06148)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nzdIXGbStFfLTFFRVFEW2fRYtDcg6ILGxr7RA1xTI7iadxbicebtjdk2Ag/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

C-RPN applies the cascading idea to SiamRPN.

## [SiamRPN++](https://arxiv.org/abs/1812.11703)
> SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks, Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan., IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

![](http://bo-li.info/SiamRPN++/img/SiamRPN_plus_plus.png)

This paper studies how to apply modern neural network backbone (e.g., ResNet) for siamese network based tracker. It is achieved via a simple yet effective spatial aware sampling strategy. Moreover, we propose a new model architecture to perform layer-wise and depth-wise aggregations, which not only further improves the accuracy but also reduces the model size.

When using deep networks for siamese network based tracker, the decrease in accuracy comes from the destroy of the strict translation invariance because of padding for convolution. To address this, spatial aware sampling strategy is introduced.

## [CIR](https://arxiv.org/abs/1901.01660)

> [code](https://gitlab.com/MSRA_NLPR/deeper_wider_siamese_trackers)

![](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oWvABcqBjd4Bh8SiaFaIo0nz3hEN3Q5xnmYWl4icNC0Lhf1zp7eEsG5aSXKd7TdCfEiciaOcdD5BFRVnA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

This paper discusses the problem of using modern super deep network, e.g., ResNet, for tracking problem. It is similar as SiamRPN++.

## [SiamMask](https://arxiv.org/abs/1812.05050)
> Fast Online Object Tracking and Segmentation: A Unifying Approach, Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr., IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
> [project](http://www.robots.ox.ac.uk/~qwang/SiamMask/) [code](http://www.robots.ox.ac.uk/~qwang/SiamMask/)

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
