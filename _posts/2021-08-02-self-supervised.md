---
layout: post
title: Self Supervised Learning Reading Note
tags:  2021 tutorial simsiam deep-learning simpleclr byol obow self-supervise cvpr teacher-student bownet contrastive-learning dion
---
This is my reading note on CVPR 2021 tutorial on self supervised learning: [Leave Those Nets Alone:
Advances in Self-Supervised Learning](https://gidariss.github.io/self-supervised-learning-cvpr2021/) and [Data- and Label-Efficient Learning in An Imperfect World](https://vita-group.github.io/cvpr_2021_data_efficient_tutorial.html).

>  Over the last few years, deep learning-based methods have achieved impressive results on image understanding problems. However, real-world vision applications often require models that are able to learn with few or no annotated data. An important and active research approach for achieving this goal is self-supervised / unsupervised learning. Indeed, the last two years there has been a lot of exciting progress in this area, with many new self-supervised pre-training methods managing to match or even surpass the performance of supervised pre-training. 

# Generative vs Discriminative

Self supervised learning aims to learn a model from unlabled data. It could divided into two groups:

- generatitve modeling: generate or otherwise model pixels in the input space, examples Autoencoder, Generative Adversarial Networks

  ![image-20210730171548359](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_15_49_image-20210730171548359.png)

- Discriminative modeling: train networks to perform pretext tasks where both the inputs and labels are derived from an unlabeled dataset. Heuristic-based pretext tasks: colorization, relative patch location prediction, solving jigsaw puzzle, rotation prediction.

![image-20210730173232754](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_32_32_image-20210730173232754.png)

# Input Space vs Feature Space

## Input Reconstruction

Perturb an image and then train a network to reconstruct the original version. Intuition is to do that the network must recognize the visual concepts of the image . It is one of the earliest methods for self-supervised representation learning.

![image-20210802215632275](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_56_32_image-20210802215632275.png)

Limitations: input reconstruction is too hard and ambiguous and effort spent on “useless” details: exact color, good boundary, etc. Does not necessarily lead to features good for image understanding tasks.

## Feature Reconstruction

Instead of trying to reconstructing the input data, the other methods focus on reconstructing high-level visual concepts rid of “useless” image details.

![image-20210802224515605](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_45_15_image-20210802224515605.png)

# Teacher-Student Methods

The goal here is to distill the knowledge of a pre-trained teacher into a smaller student. Student is trained to predict the teacher target when given the same input image. It also referred as knowledge distillation. Teacher-student methods could be used for self-supervised learning.

![image-20210802220610400](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_06_10_image-20210802220610400.png)

However, where could we find the a good quality teacher?

## Bootstrap Your Own Latent (BYOL)

BYOL keeps update the teacher as exponential moving average of the student. Notes, the stop gradient for the teacher branch is critical--otherwise your feature will collapse. 

![image-20210802221612339](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_16_12_image-20210802221612339.png)

## DINO

The novelty of DINO is appyling centering by subtracting the mean feature to prevent collapsing to constant 1-hot targets, then sharpening by using low softmax temperature: prevents collapsing to a uniform target vector.

![image-20210802222716345](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_27_16_image-20210802222716345.png)

## SimSiam

Instead of updating the teacher from student. SimSiam simples make the teacher and student share the same network.

![image-20210802222417000](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_24_17_image-20210802222417000.png)

## BoWNet

BoWNet utilizes the bag of words ideas to align the feature space.

![image-20210802221144533](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_11_44_image-20210802221144533.png)

- Teacher: extract feature maps + convert them to Bag-of-Words (BoW) vectors
- Student: must predict the BoW of an image, given as input a perturbed version

The teacher networks is trained from predicting rotation of images. Students will be trained until convergence, then the teacher will be updated from student; repeat this process.

![image-20210802221358202](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_13_58_image-20210802221358202.png)

### OBoW

In OBoW, the teacher model is updated via exponential moving average of student.

![image-20210802223418610](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_34_18_image-20210802223418610.png)

# Contrastive Learning

The other group of methods utilize contrastive loss to avoid feature collapse during self supervised learning--for positive samples, pull their features together and for negative samples, push their features away. Constrast learning could be formulated as special type of teacher-student model--teacher and student share the same network like SimSiam.

![image-20210802215956390](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_59_56_image-20210802215956390.png)

One example is [SimpleCLR](https://arxiv.org/abs/2002.05709) : maximizing the agreement of representations under data transformation (e.g., random crop, coloring), using a contrastive loss in the latent/feature space. It reported 10% relative improvement over previous SOTA (cpc v2), outperforms AlexNet with 100X fewer labels.

![image-20210730173330433](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_33_30_image-20210730173330433.png)

## Limitation of Contrastive Learning

Many constrastive learning methods rely on the assumption that, applying the transform shouldn't change the target label and then the feature representation. This is especially true for ImageNet--there is only a single dominant object in the image.

It has been found, larger objects suppress the learning of smaller objects. The other limitation of constrast learning is that, it requires negative examples, to contrast with postive examples.

# Result

You may find self supervised method could even outperform the supervised methods in image classificaiton and object detection, across several different datasets. The best approach seems to be OBoW.

![image-20210802223631824](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_36_31_image-20210802223631824.png)

![image-20210802223642714](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_36_42_image-20210802223642714.png)

![image-20210802223656164](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_22_36_56_image-20210802223656164.png)

# How to Use Self-Supervised Learning

There are two ways of using self-supervised learning:

- pre-train the model with self-supervised learning, then fine tune it for your own task;

  ![image-20210730173049257](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_30_49_image-20210730173049257.png)

- use self-supervised learning as additional task/loss of your own task.

