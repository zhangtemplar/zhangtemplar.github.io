---
layout: post
title: Text Detection
tags:  textmountain deep-learning pixelink advancedeast text-detection textfield craft psenet pmtd east
---

In this post, we will introduce some of the most recent text detection methods. Text detection methods are highly related to object detection methods, thus could be categorized into one-stage methods and two-stage methods. Currently, two-stage methods could easily outperforms one-stage methods, e.g., using Mask-RCNN directly could give you 76% mAP on icdar2017mlt, while the current state of art methods psenet achieves 72%.

# One Stage

## [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)

![](https://github.com/huoyijie/AdvancedEAST/raw/master/image/East.network.png)

> Previous approaches for scene text detection have already achieved promising performances across various benchmarks. However, they usually fall short when dealing with challenging scenarios, even when equipped with deep neural network models, because the overall performance is determined by the interplay of multiple stages and components in the pipelines. In this work, we propose a simple yet powerful pipeline that yields fast and accurate text detection in natural scenes. The pipeline directly predicts words or text lines of arbitrary orientations and quadrilateral shapes in full images, eliminating unnecessary intermediate steps (e.g., candidate aggregation and word partitioning), with a single neural network. The simplicity of our pipeline allows concentrating efforts on designing loss functions and neural network architecture. Experiments on standard datasets including ICDAR 2015, COCO-Text and MSRA-TD500 demonstrate that the proposed algorithm significantly outperforms state-of-the-art methods in terms of both accuracy and efficiency. On the ICDAR 2015 dataset, the proposed algorithm achieves an F-score of 0.7820 at 13.2fps at 720p resolution.


## [AdvancedEAST](https://github.com/huoyijie/AdvancedEAST)

![](https://github.com/huoyijie/AdvancedEAST/raw/master/image/AdvancedEast.network.png)

> AdvancedEAST is an algorithm used for Scene image text detect, which is primarily based on EAST, and the significant improvement was also made, which make long text predictions more accurate.

## [PixelLink: Detecting Scene Text via Instance Segmentation](https://arxiv.org/abs/1801.01315)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/a4a88145718ec8eff1228267bf3fe9f380b9495f/3-Figure2-1.png)

> Most state-of-the-art scene text detection algorithms are deep learning based methods that depend on bounding box regression and perform at least two kinds of predictions: text/non-text classification and location regression. Regression plays a key role in the acquisition of bounding boxes in these methods, but it is not indispensable because text/non-text prediction can also be considered as a kind of semantic segmentation that contains full location information in itself. However, text instances in scene images often lie very close to each other, making them very difficult to separate via semantic segmentation. Therefore, instance segmentation is needed to address this problem. In this paper, PixelLink, a novel scene text detection algorithm based on instance segmentation, is proposed. Text instances are first segmented out by linking pixels within the same instance together. Text bounding boxes are then extracted directly from the segmentation result without location regression. Experiments show that, compared with regression-based methods, PixelLink can achieve better or comparable performance on several benchmarks, while requiring many fewer training iterations and less training data.


## [TextField: Learning A Deep Direction Field for Irregular Scene Text Detection](https://arxiv.org/abs/1812.01393)

![](https://d3i71xaburhd42.cloudfront.net/a80212fe263cad9760e83a34bacf7203f70816f8/4-Figure3-1.png)

> Scene text detection is an important step of scene text reading system. The main challenges lie on significantly varied sizes and aspect ratios, arbitrary orientations and shapes. Driven by recent progress in deep learning, impressive performances have been achieved for multi-oriented text detection. Yet, the performance drops dramatically in detecting curved texts due to the limited text representation (e.g., horizontal bounding boxes, rotated rectangles, or quadrilaterals). It is of great interest to detect curved texts, which are actually very common in natural scenes. In this paper, we present a novel text detector named TextField for detecting irregular scene texts. Specifically, we learn a direction field pointing away from the nearest text boundary to each text point. This direction field is represented by an image of two-dimensional vectors and learned via a fully convolutional neural network. It encodes both binary text mask and direction information used to separate adjacent text instances, which is challenging for classical segmentation-based approaches. Based on the learned direction field, we apply a simple yet effective morphological-based post-processing to achieve the final detection. Experimental results show that the proposed TextField outperforms the state-of-the-art methods by a large margin (28% and 8%) on two curved text datasets: Total-Text and CTW1500, respectively, and also achieves very competitive performance on multi-oriented datasets: ICDAR 2015 and MSRA-TD500. Furthermore, TextField is robust in generalizing to unseen datasets. The code is available at this https URL.


## [TextMountain: Accurate Scene Text Detection via Instance Segmentation](https://arxiv.org/abs/1811.12786)

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/848db2f0758c314ddbc0bc4030633501df3a9276/3-Figure2-1.png)

> In this paper, we propose a novel scene text detection method named TextMountain. The key idea of TextMountain is making full use of border-center information. Different from previous works that treat center-border as a binary classification problem, we predict text center-border probability (TCBP) and text center-direction (TCD). The TCBP is just like a mountain whose top is text center and foot is text border. The mountaintop can separate text instances which cannot be easily achieved using semantic segmentation map and its rising direction can plan a road to top for each pixel on mountain foot at the group stage. The TCD helps TCBP learning better. Our label rules will not lead to the ambiguous problem with the transformation of angle, so the proposed method is robust to multi-oriented text and can also handle well with curved text. In inference stage, each pixel at the mountain foot needs to search the path to the mountaintop and this process can be efficiently completed in parallel, yielding the efficiency of our method compared with others. The experiments on MLT, ICDAR2015, RCTW-17 and SCUT-CTW1500 databases demonstrate that the proposed method achieves better or comparable performance in terms of both accuracy and efficiency. It is worth mentioning our method achieves an F-measure of 76.85% on MLT which outperforms the previous methods by a large margin. Code will be made available.

## [Character Region Awareness for Text Detection](https://arxiv.org/abs/1904.01941)

![](https://storage.googleapis.com/groundai-web-prod/media/users/user_223603/project_348637/images/x3.png.344x181_q75_crop.png)

> Scene text detection methods based on neural networks have emerged recently and have shown promising results. Previous methods trained with rigid word-level bounding boxes exhibit limitations in representing the text region in an arbitrary shape. In this paper, we propose a new scene text detection method to effectively detect text area by exploring each character and affinity between characters. To overcome the lack of individual character level annotations, our proposed framework exploits both the given character-level annotations for synthetic images and the estimated character-level ground-truths for real images acquired by the learned interim model. In order to estimate affinity between characters, the network is trained with the newly proposed representation for affinity. Extensive experiments on six benchmarks, including the TotalText and CTW-1500 datasets which contain highly curved texts in natural images, demonstrate that our character-level text detection significantly outperforms the state-of-the-art detectors. According to the results, our proposed method guarantees high flexibility in detecting complicated scene text images, such as arbitrarily-oriented, curved, or deformed texts.


## [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1903.12473)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oXLvY90B9KF7UxCqyLSmK0uzRq8Erfeu7E8dteyvf7nnS8Ruic4YPXOqialanPKbkay6n4gyDlibtFLA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> Scene text detection has witnessed rapid progress especially with the recent development of convolutional neural networks. However, there still exists two challenges which prevent the algorithm into industry applications. On the one hand, most of the state-of-art algorithms require quadrangle bounding box which is in-accurate to locate the texts with arbitrary shape. On the other hand, two text instances which are close to each other may lead to a false detection which covers both instances. Traditionally, the segmentation-based approach can relieve the first problem but usually fail to solve the second challenge. To address these two challenges, in this paper, we propose a novel Progressive Scale Expansion Network (PSENet), which can precisely detect text instances with arbitrary shapes. More specifically, PSENet generates the different scale of kernels for each text instance, and gradually expands the minimal scale kernel to the text instance with the complete shape. Due to the fact that there are large geometrical margins among the minimal scale kernels, our method is effective to split the close text instances, making it easier to use segmentation-based methods to detect arbitrary-shaped text instances. Extensive experiments on CTW1500, Total-Text, ICDAR 2015 and ICDAR 2017 MLT validate the effectiveness of PSENet. Notably, on CTW1500, a dataset full of long curve texts, PSENet achieves a F-measure of 74.3% at 27 FPS, and our best F-measure (82.2%) outperforms state-of-art algorithms by 6.6%. The code will be released in the future.


# Two Stage

## [Omnidirectional Scene Text Detection with Sequential-free Box Discretization](https://arxiv.org/abs/1906.02371)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oXLvY90B9KF7UxCqyLSmK0umIfXQssI3XrPpicUBluo9GXEL4hqfapvuXzqgiaDeqibeEWvRSOu9ibYaw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> Scene text in the wild is commonly presented with high variant characteristics. Using quadrilateral bounding box to localize the text instance is nearly indispensable for detection methods. However, recent researches reveal that introducing quadrilateral bounding box for scene text detection will bring a label confusion issue which is easily overlooked, and this issue may significantly undermine the detection performance. To address this issue, in this paper, we propose a novel method called Sequential-free Box Discretization (SBD) by discretizing the bounding box into key edges (KE) which can further derive more effective methods to improve detection performance. Experiments showed that the proposed method can outperform state-of-the-art methods in many popular scene text benchmarks, including ICDAR 2015, MLT, and MSRA-TD500. Ablation study also showed that simply integrating the SBD into Mask R-CNN framework, the detection performance can be substantially improved. Furthermore, an experiment on the general object dataset HRSC2016 (multi-oriented ships) showed that our method can outperform recent state-of-the-art methods by a large margin, demonstrating its powerful generalization ability. Source code: this https URL.


## [Pyramid Mask Text Detector](https://arxiv.org/abs/1903.11800)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oXLvY90B9KF7UxCqyLSmK0u5zCWhj5zCwblN14lFTlNhVPk4ialOcH0ficnswr7yDNubQdNQ6dNZ91Q/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> Scene text detection, an essential step of scene text recognition system, is to locate text instances in natural scene images automatically. Some recent attempts benefiting from Mask R-CNN formulate scene text detection task as an instance segmentation problem and achieve remarkable performance. In this paper, we present a new Mask R-CNN based framework named Pyramid Mask Text Detector (PMTD) to handle the scene text detection. Instead of binary text mask generated by the existing Mask R-CNN based methods, our PMTD performs pixel-level regression under the guidance of location-aware supervision, yielding a more informative soft text mask for each text instance. As for the generation of text boxes, PMTD reinterprets the obtained 2D soft mask into 3D space and introduces a novel plane clustering algorithm to derive the optimal text box on the basis of 3D shape. Experiments on standard datasets demonstrate that the proposed PMTD brings consistent and noticeable gain and clearly outperforms state-of-the-art methods. Specifically, it achieves an F-measure of 80.13% on ICDAR 2017 MLT dataset.

