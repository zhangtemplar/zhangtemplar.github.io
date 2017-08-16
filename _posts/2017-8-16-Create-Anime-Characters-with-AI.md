---
layout: posts
title: Create Anime Characters with A.I
---

There is a paper from Fudan University, Tongji University and CMU on proposing a method of generating AI character facial images via neural network.

# Abstract

Automatic generation of facial images has been well studied after the Generative Adversarial Network(GAN) came out. There exists some attempts applying the GAN model to the problem of generating facial images of anime characters, but none of the existing work gives a promising result. In this work, we explore the training of GAN models specialized on an anime facial image dataset. We address the issue from both the data and the model aspect, by collecting a more clean, well-suited dataset and leverage proper, empirical application of DRAGAN. With quantitative analysis and case studies we demonstrate that our efforts lead to a stable and highquality model. Moreover, to assist people with anime character design, we build a website1 with our pre-trained model available online, which makes the model easily accessible to general public.

# Examples

Some of the generated images are like:
![Some of the generated images](https://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2HgXFQchYyNx9NQmBJst2hFqGmxb4MpkB7cWCwtOGuZtgOU4TmDB4xTvJgWZD3GkzCD1CUwzdRZg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

On the website, you can change the settings to have different facial images:
![you can change the settings to have different facial images](https://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2HgXFQchYyNx9NQmBJst2h6dBVUs0icIFtNUtn8njHbCBoVKGT0ke3ibG70a6eOf5vo4XkNwCAy7hg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

# Technique Detail:

The author utilizes [DRAGAN](https://arxiv.org/pdf/1705.07215.pdf) to better convergence and training results. Generative network is similar to [SRResNet](https://arxiv.org/pdf/1609.04802.pdf)
![Generative network](https://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2HgXFQchYyNx9NQmBJst2hTKeor1FYvv0J8zyhqZE0nh7YwIjKyFCHLiaJlKFKbN9mibFTHac9TX7g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)
The discriminor network is also modified to use `sigmoid` instead of `softmax`:
![discriminator network](https://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2HgXFQchYyNx9NQmBJst2h5zKHyoliaGz2AQITLtXvKud8icqYdDRaIhYuLwMia3hrTOt32MMhMg6eg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)
# Links

  - [Demo website](MakeGirls.moe)
  - [Paper](https://makegirlsmoe.github.io/assets/pdf/technical_report.pdf)
  - [Github, front end only](make.girls.moe)
