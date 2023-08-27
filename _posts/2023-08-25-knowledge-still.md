---
layout: post
title: Knowledge Distillation A Survey
tags:  deep-learning distill review
---
This is my reading note on [Knowledge Distillation: A Survey](http://arxiv.org/abs/2006.05525). As a representative type of model compression and acceleration, knowledge distillation effectively learns a small student model from a large teacher model [(p. 1)](zotero://open-pdf/library/items/8265TF5E?page=1&annotation=LDB2MZTV)

> In recent years, deep neural networks have been successful in both industry and academia, especially for computer vision tasks. The great success of deep learning is mainly due to its scalability to encode large-scale data and to maneuver billions of model parameters. However, it is a challenge to deploy these cumbersome deep models on devices with limited resources, e.g., mobile phones and embedded devices, not only because of the high computational complexity but also the large storage requirements. To this end, a variety of model compression and acceleration techniques have been developed. As a representative type of model compression and acceleration, knowledge distillation effectively learns a small student model from a large teacher model. It has received rapid increasing attention from the community. This paper provides a comprehensive survey of knowledge distillation from the perspectives of knowledge categories, training schemes, teacher-student architecture, distillation algorithms, performance comparison and applications. Furthermore, challenges in knowledge distillation are briefly reviewed and comments on future research are discussed and forwarded.

# Introduction
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-2-x29-y530.png) 

Applications of knowledge distillation:
1. Parameter pruning and sharing: These methods fo- cus on removing inessential parameters from deep neural networks without any significant effect on the performance. This category is further divided into model quantization (Wu et al., 2016), model bina- rization (Courbariaux et al., 2015), structural matri- ces (Sindhwani et al., 2015) and parameter sharing (Han et al., 2015; Wang et al., 2019f). 
2. Low-rank factorization: These methods identify re- dundant parameters of deep neural networks by em- ploying the matrix and tensor decomposition (Yu et al., 2017; Denton et al., 2014). 
3. Transferred compact convolutional filters: These meth- ods remove inessential parameters by transferring or compressing the convolutional filters (Zhai et al., 2016). 
4. Knowledge distillation (KD): These methods distill the knowledge from a larger deep neural network into a small network (Hinton et al., 2015). [(p. 2)](zotero://open-pdf/library/items/8265TF5E?page=2&annotation=KA6IKC7N)

Specifically, Urner et al. (2011) proved that the knowledge transfer from a teacher model to a student model using unlabeled data is PAC learnable [(p. 2)](zotero://open-pdf/library/items/8265TF5E?page=2&annotation=XVEK23V4)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-3-x32-y474.png) 

Successful distillation relies on data geometry, optimization bias of distillation objective and strong monotonicity of the student classifier [(p. 3)](zotero://open-pdf/library/items/8265TF5E?page=3&annotation=8IRM2ZQ6). Empirical results show that a larger model may not be a better teacher because of model capacity gap (Mirzadeh et al., 2020). Experiments also show that distillation adversely affects the student learning. [(p. 3)](zotero://open-pdf/library/items/8265TF5E?page=3&annotation=HRRWGJ6Q)
# Knowledge
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-4-x37-y480.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-5-x35-y636.png) 
## Response-Based Knowledge
Response-based knowledge usually refers to the neural response of the last output layer of the teacher model.  The main idea is to directly mimic the final prediction of the teacher model. The response-based knowledge distillation is simple yet effective for model compres- sion, and has been widely used in different tasks and applications. [(p. 5)](zotero://open-pdf/library/items/8265TF5E?page=5&annotation=X3W9Q8PL)

Generally, $$LR(p(z_t, T ), p(z_s, T ))$$ often employs Kullback- Leibler divergence loss [(p. 5)](zotero://open-pdf/library/items/8265TF5E?page=5&annotation=XJUYLVTL)
## Feature-Based Knowledge
From another perspective, the effectiveness of the soft targets is analogous to label smoothing (Kim and Kim, 2017) or regulariz- ers (Muller et al., 2019; Ding et al., 2019). However, the response-based knowledge usually relies on the output of the last layer, e.g., soft targets, and thus fails to address the intermediate-level supervision from the teacher model, which turns out to be very important for representation learning using very deep neural networks [(p. 5)](zotero://open-pdf/library/items/8265TF5E?page=5&annotation=99ZYP2IU)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-6-x40-y399.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-6-x292-y265.png) 

Though feature-based knowledge transfer provides favorable information for the learning of the student model, how to effectively choose the hint layers from the teacher model and the guided layers from the student model remains to be further investigated [(p. 7)](zotero://open-pdf/library/items/8265TF5E?page=7&annotation=HLYUV9D2)
## Relation-Based Knowledge
To explore the relationships between diﬀerent feature maps, Yim et al. (2017) proposed a ﬂow of solution process (FSP), which is defined by the Gram matrix between two layers. The FSP matrix summarizes the relations between pairs of feature maps. It is calculated using the inner products between features from two lay- ers. Using the correlations between feature maps as the distilled knowledge, knowledge distillation via singular value decomposition was proposed to extract key infor- mation in the feature maps [(p. 7)](zotero://open-pdf/library/items/8265TF5E?page=7&annotation=9QPQITTG)

The graph knowledge is the intra-data relations between any two feature maps via multi-head attention network. To explore the pairwise hint information, the student model also mimics the mutual information flow from pairs of hint layers of the teacher model [(p. 7)](zotero://open-pdf/library/items/8265TF5E?page=7&annotation=QULYTGDU)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-7-x291-y160.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-8-x38-y573.png) 

$L_{EM} (.)$, $L_H(.)$, $L_{AW} (.)$ and $\lVert\cdot\rVert_F$ are Earth Mover distance, Huber loss, Angle-wise loss and Frobenius norm, respectively. A [(p. 8)](zotero://open-pdf/library/items/8265TF5E?page=8&annotation=AYA8RC9M)
# Distillation Schemes
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-8-x288-y312.png) 
## Oﬄine Distillation
The offline distillation methods usually employ one- way knowledge transfer and two-phase training pro- cedure. However, the complex high-capacity teacher model with huge training time can not be avoided, while the training of the student model in offline distillation is usually efficient under the guidance of the teacher model. Moreover, the capacity gap between large teacher and small student always exists, and student often largely relies on teacher. [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=FNVDHHY5)
## Online Distillation
online distillation is proposed to further improve the performance of the student model, especially when a large-capacity high perfor- mance teacher model is not available [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=HIVJHL4K)

Specifically, in deep mutual learning (Zhang et al., 2018b), multiple neural networks work in a collab- orative way. Any one network can be the student model and other models can be the teacher during the training process [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=Q4JIECT6)
1. further introduced auxiliary peers and a group leader into deep mutual learning to form a diverse set of peer models [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=5NZLGFXL)
2. proposed a multi-branch architecture, in which each branch indicates a student model and different branches share the same backbone network [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=YKMGMAYK)

Co-distillation in parallel trains multiple models with the same architectures and any one model is trained by transferring the knowledge from the other models. Recently, an online adversarial knowledge distillation method is proposed to simultaneously train multiple networks by the discriminators using knowl- edge from both the class probabilities and a feature map [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=BX82WKTC)

Online distillation is a one-phase end-to-end train- ing scheme with efficient parallel computing. However, existing online methods (e.g., mutual learning) usually fails to address the high-capacity teacher in online settings, making it an interesting topic to further ex- plore the relationships between the teacher and student model in online settings. [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=QCBASY5N)
## Self-Distillation
Speciﬁcally, Zhang et al. (2019b) proposed a new self-distillation method, in which knowledge from the deeper sections of the network is distilled into its shallow sections [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=JEAU2DW5) The network utilizes the attention maps of its own layers as distillation targets for its lower layers [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=3IQS2F2W) knowledge in the earlier epochs of the network (teacher) is transferred into its later epochs (student) to sup- port a supervised training process within the same network. [(p. 9)](zotero://open-pdf/library/items/8265TF5E?page=9&annotation=96AQ8JKU)

To be specific, Yuan et al. proposed teacher-free knowledge distillation meth- ods based on the analysis of label smoothing regularization [(p. 10)](zotero://open-pdf/library/items/8265TF5E?page=10&annotation=3UZ9GYXG)
# Distillation Algorithms
## Adversarial Distillation
many adversarial knowledge distillation methods have been proposed to enable the teacher and stu- dent networks to have a better understanding of the true data distribution [(p. 11)](zotero://open-pdf/library/items/8265TF5E?page=11&annotation=E6CMY2BM)

In the first category, an adversarial genera- tor is trained to generate synthetic data, which is either directly used as the training dataset [(p. 11)](zotero://open-pdf/library/items/8265TF5E?page=11&annotation=4VPJ67MD). It utilized an adversarial generator to generate hard examples for knowledge transfer [(p. 11)](zotero://open-pdf/library/items/8265TF5E?page=11&annotation=QC722G5J)

$$L_{KD} = L_G(F_t(G(z)), F_s(G(z)))$$

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/gouKnowledgeDistillationSurvey2021-12-x40-y543.png) 

To make student well match teacher, a discriminator in the second category is introduced to distinguish the samples from the student and the teacher models [(p. 12)](zotero://open-pdf/library/items/8265TF5E?page=12&annotation=UTLR7QI5)

$$L_{GANKD} =L_{CE}(G(F_s(x)), y) + \alpha L_{KL}(G(F_s(x)), F_t(x)) + \beta L_{GAN}( F_s(x), F_t(x))$$
## Multi-Teacher Distillation
which one teacher trans- fers response-based knowledge to the student and the other teacher transfers feature-based knowledge to the student [(p. 13)](zotero://open-pdf/library/items/8265TF5E?page=13&annotation=QHCF2UQS). Another workrandomly selected one teacher from the pool of teacher networks at each iteration. [(p. 13)](zotero://open-pdf/library/items/8265TF5E?page=13&annotation=7N46JVTJ)
## Graph-Based Distillation
The core of attention transfer is to define the attention maps for feature embedding in the layers of a neural network. That is to say, knowledge about feature embedding is transferred using attention map functions. [(p. 15)](zotero://open-pdf/library/items/8265TF5E?page=15&annotation=8DUL8EH5)

