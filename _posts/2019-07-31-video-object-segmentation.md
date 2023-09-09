---
layout: post
title: Video Object Segmentation
tags:  deep-learning feelvos premvos video-object-segmentation lucid-data-dreaming osvos
---

![](https://miro.medium.com/max/1400/1*qooRKoB2wPNKvGs-C_MI-A.png)

Compared with semantic segmentation, video object segmentation (VOS) solves a slightly different tasks:
- segmenting general, NON-semantic objects.
- a temporal component has been added: the task is to find the pixels corresponding to the object(s) of interest in each consecutive frame of a video.

There are three types of VOS:
- Semi-supervised video object segmentation
- Interactive video object segmentation
- Un-supervised video object segmentation

# Semi-supervised video object segmentation

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8wic9AoKYWNGDUX6Tfzhr1RibWOfu642TbC1I9K5DuG7t5qiaKOrZyt5foRAjqO6dr64v9iaYrZBT8tQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Semi-supervised video object segmentation is also referred as one-shot video object segmentation (OSVOS). In OSVOS, user typically provides the segmentation task for the first frame, the algorithm need to segment the other frames. Several works have been proposed to utilize the one-shot learning, which essentially fine-tune the network based on the input from the first frame.

## [Lucid data dreaming for object tracking](http://arxiv.org/abs/1703.09554)

> Convolutional networks reach top quality in pixel-level object tracking but require a large amount of training data (1k~10k) to deliver such results. We propose a new training strategy which achieves state-of-the-art results across three evaluation datasets while using 20x~100x less annotated data than competing methods. Our approach is suitable for both single and multiple object tracking. Instead of using large training sets hoping to generalize across domains, we generate in-domain training data using the provided annotation on the first frame of each video to synthesize ("lucid dream") plausible future video frames. In-domain per-video training data allows us to train high quality appearance- and motion-based models, as well as tune the post-processing stage. This approach allows to reach competitive results even when training from only a single annotated frame, without ImageNet pre-training. Our results indicate that using a larger training set is not automatically better, and that for the tracking task a smaller training set that is closer to the target domain is more effective. This changes the mindset regarding how many training samples and general "objectness" knowledge are required for the object tracking task.

## [One-Shot video object segmentation](http://arxiv.org/abs/1611.05198)

> This paper tackles the task of semi-supervised video object segmentation, i.e., the separation of an object from the background in a video, given the mask of the first frame. We present One-Shot Video Object Segmentation (OSVOS), based on a fully-convolutional neural network architecture that is able to successively transfer generic semantic information, learned on ImageNet, to the task of foreground segmentation, and finally to learning the appearance of a single annotated object of the test sequence (hence one-shot). Although all frames are processed independently, the results are temporally coherent and stable. We perform experiments on two annotated video segmentation databases, which show that OSVOS is fast and improves the state of the art by a significant margin (79.8% vs 68.0%).

## [PReMVOS: Proposal-generation, Refinement and Merging for the YouTube-VOS Challenge on Video Object Segmentation 2018](http://arxiv.org/abs/1807.09190)

> We address semi-supervised video object segmentation, the task of automatically generating accurate and consistent pixel masks for objects in a video sequence, given the first-frame ground truth annotations. Towards this goal, we present the PReMVOS algorithm (Proposal-generation, Refinement and Merging for Video Object Segmentation). Our method separates this problem into two steps, first generating a set of accurate object segmentation mask proposals for each video frame and then selecting and merging these proposals into accurate and temporally consistent pixel-wise object tracks over a video sequence in a way which is designed to specifically tackle the difficult challenges involved with segmenting multiple objects across a video sequence. Our approach surpasses all previous state-of-the-art results on the DAVIS 2017 video object segmentation benchmark with a J & F mean score of 71.6 on the test-dev dataset, and achieves first place in both the DAVIS 2018 Video Object Segmentation Challenge and the YouTube-VOS 1st Large-scale Video Object Segmentation Challenge.

However fine-tuning the network for each video could be too expensive, as a result several latest work doesn't require network fine-tuning.

## [FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation](http://arxiv.org/abs/1902.09513)

> Many of the recent successful methods for video object segmentation (VOS) are overly complicated, heavily rely on fine-tuning on the first frame, and/or are slow, and are hence of limited practical use. In this work, we propose FEELVOS as a simple and fast method which does not rely on fine-tuning. In order to segment a video, for each frame FEELVOS uses a semantic pixel-wise embedding together with a global and a local matching mechanism to transfer information from the first frame and from the previous frame of the video to the current frame. In contrast to previous work, our embedding is only used as an internal guidance of a convolutional network. Our novel dynamic segmentation head allows us to train the network, including the embedding, end-to-end for the multiple object segmentation task with a cross entropy loss. We achieve a new state of the art in video object segmentation without fine-tuning with a J&F measure of 71.5% on the DAVIS 2017 validation set. We make our code and models available at https://github.com/tensorflow/models/tree/master/research/feelvos.

## [Fast User-Guided Video Object Segmentation by Interaction-and-Propagation Networks](http://arxiv.org/abs/1904.09791)

> We present a deep learning method for the interactive video object segmentation. Our method is built upon two core operations, interaction and propagation, and each operation is conducted by Convolutional Neural Networks. The two networks are connected both internally and externally so that the networks are trained jointly and interact with each other to solve the complex video object segmentation problem. We propose a new multi-round training scheme for the interactive video object segmentation so that the networks can learn how to understand the user's intention and update incorrect estimations during the training. At the testing time, our method produces high-quality results and also runs fast enough to work with users interactively. We evaluated the proposed method quantitatively on the interactive track benchmark at the DAVIS Challenge 2018. We outperformed other competing methods by a significant margin in both the speed and the accuracy. We also demonstrated that our method works well with real user interactions.

# Interactive video object segmentation

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8wic9AoKYWNGDUX6Tfzhr1Rbh8nXT67H4BFh0eIugrfsiajjd13758BBfWRhWHdV8unDd3xicgpPONw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

For interactive video object segmentation, user provides input to any frame(s) to the video to guide the segmentation. The user input could be bounding box, scribble, corner points. It usually takes the following steps:

- user provide input for a frame;
- algorithm segment that frame based on user input;
- the segmentation result is propagated to other frames;
- user inspect the segmentation result and provide feedback on unsatistying result;
- update the segmentation result based on user input;
- repeat the steps above.

# Un-supervised video object segmentation

The unsupervised scenario assumes that the user does not interact with the algorithm to obtain the segmentation masks. Methods should provide a set of object candidates with no overlapping pixels that span through the whole video sequence. This set of objects should contain at least the objects that capture human attention when watching the whole video sequence i.e objects that are more likely to be followed by human gaze.

Most of existing methods relies on saliency module for this task.

# Metrics

The most widely used metric for VOS is Jaccard&F-measurement@60s and area under curve, which were proposed by [Davis Challenge on Video Object Segmentation](https://davischallenge.org/):
- Jaccard: Region Similarity is the intersection-over-union between mask M and ground truth G;
- F-measurement: Contour Accuracy is the F-measure for the contour based precision and recall.

