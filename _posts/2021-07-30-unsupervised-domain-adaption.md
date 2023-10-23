---
layout: post
title: Unsupervised Domain Adaption
tags:  advserial normalization discrepancy cvpr self-supervise tutorial reconstruction deep-learning 2021 optimal-transport teacher-student domain-adaption
---

This is my reading note on [CVPR 2021 Tutorial: Data- and Label-Efficient Learning in An Imperfect World](https://vita-group.github.io/cvpr_2021_data_efficient_tutorial.html). The original [slides](https://utexas.box.com/s/6mdtvt1wj8hsojzen918xekjyh43zxtx) and [videos](https://utexas.box.com/s/ph3xebwa2hri404k8sml5kfovgi4ak1k) are available online. Unsupervised domain adaption methods could be divided into the following groups:

![image-20210730173609873](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_36_10_image-20210730173609873.png)

- reconstruction based methods: uses auxiliary reconstruction task to ensure domain-invariant features.

  - Deep Separation Networks

    ![image-20210730173840792](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_38_40_image-20210730173840792.png)

  - Deep Reconstruction Classification Network

    ![image-20210730173902762](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_39_02_image-20210730173902762.png)

- Discrepancy-based methods: align domain data representations with statistical measures

  - Deep Adaptation Networks: using MMD kernel for measures

    ![image-20210730174055918](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_40_56_image-20210730174055918.png)

  - Deep CORAL: Correlation Alignment: using 2nd-order measurement

    ![image-20210730174302503](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_43_02_image-20210730174302503.png)

  - HoMM: Higher-order Moment Matching: using even higher order. It is found higher order gives better result.

    ![image-20210730174345025](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_43_45_image-20210730174345025.png)

- Adversarial-based Methods: involve a domain discriminator to enforce domain confusion while learning representations

  - Feature based: doesn't try to reconstruct the images cross domains

    - Domain Adversarial Network: feature extractor is shared cross domains and use domain classifier to estimate the domain.

      ![image-20210730174443168](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_44_43_image-20210730174443168.png)

    - Adversarial Discriminative DA: feature extractor is seperated between domains.

      ![image-20210730174720055](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_47_20_image-20210730174720055.png)

    - Conditional domain adversarial network (CDAN)

      ![image-20210730174543093](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_45_43_image-20210730174543093.png)

  - Reconstruction based: use GAN to generate fake images cross domain

    - Pixel DA: Leverages GAN to generate fake target images from source images (conditional GAN)

      ![image-20210730174947265](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_49_47_image-20210730174947265.png)

    - Cycle-consistency: SBADA-GAN: use cycle-gan to generate images cross domain

      ![image-20210730175022689](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_50_22_image-20210730175022689.png)

    - Cycle-consistency: CyCADA: adds a semantic loss

      ![image-20210730175102904](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_51_03_image-20210730175102904.png)

    - Adversarial domain adaptation with domain mixup (DM-ADA): mixup samples obtained interpolating between source and target images and domain discriminator learns to output soft scores

      ![image-20210730175213918](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_52_14_image-20210730175213918.png)

- Normalization-based methods: embed in the network some domain alignment layers

  - DomaIn distribution Alignment Layers: each domain has their own parameter for batch norm layer, thus the data distribution is aligned.
  - Automatic DomaIn Alignment Layers - AutoDIAL
  - DomaIn Whitening Transform

- Optimal Transport-based: minimize domain discrepancy optimizing a measure based on Optimal Transport

  - DeepJDOT

    ![image-20210730175500227](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_55_00_image-20210730175500227.png)

  - Reliable Weighted Optimal Transport

    ![image-20210730175539296](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_07_30_17_55_39_image-20210730175539296.png)

- Teacher-Student-based: exploit the teacher-student paradigm for transferring knowledge on target data

# Results

Those methods have been evaluated on two tasks: image classification on office-31 dataset and digiti recognition (similar to image classification) on digitis dataset. There are methods outperforms fully supervised baseline on each of the dataset, however, there is no approach which works well on both datasets.

## Office-31

![image-20210802212848232](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_28_48_image-20210802212848232.png)

The supervised learning based method (full) reaches accuracy of 91.7%. The table shows that, some unsupervised domain adaption methods could output fully supervised learning methods, e.g.,

- Reliable Weighted Optimal Transport (RWOT): optimal transport method 
- Conditional domain adversarial network (CDAN): adversial based method (without fake data generation)

![image-20210802212736047](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_27_36_image-20210802212736047.png)

![image-20210802212723845](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_27_23_image-20210802212723845.png)

![image-20210802212557052](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_25_57_image-20210802212557052.png)

## Digits

Use supervised learning method on target domain, the performance is 96.5% on USPS, 96.7% on SVNH and 99.2% on MNIST. Some unsupervised domain adaption methods could output fully supervised learning methods on some domains, e.g., 

- Symmetric Bi-Directional Adaptive GAN (sbada-gan): outperforms the basline on MNIST-USPS and MINST-MNIST-M, but significantly lower on others;
- Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss (DWT): outperforms the basline on MNIST-USPS but slightly lower on others.

![image-20210802212709927](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_27_10_image-20210802212709927.png)

![image-20210802212530473](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2021_08_02_21_25_30_image-20210802212530473.png)


