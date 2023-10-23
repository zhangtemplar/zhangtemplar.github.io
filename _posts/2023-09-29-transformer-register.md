---
layout: post
title: Vision Transformers Need Registers
tags:  attention clip vit self-supervised deep-learning transformer zero-shot-learning
---

This is my reading note for [Vision Transformers Need Registers](http://arxiv.org/abs/2309.16588). This paper analyzes the attention map of transformer and find too large scale transformer and trained after a long iteration, some token show exceptionally high norm. Those tokens usually correspond to patches in uniform background. Analysis indicates that those tokens are used to store global information. Thus at would heart dense prediction tasks like image segmentation. To tackle this, the paper proposes add additional tokens during trains and inference, but rejecting for outputs.

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-1-x104-y195.png) 

# Introduction
In this paper, we identify and characterize artifacts in feature maps of both supervised and self-supervised ViT networks. The artifacts correspond to high-norm tokens appearing during inference primarily in low-informative background areas of images, that are repurposed for internal computations. We propose a simple yet effective solution based on providing additional tokens to the input sequence of the Vision Transformer to fill that role. We show that this solution fixes that problem entirely for both supervised and self-supervised models, sets a new state of the art for self-supervised visual models on dense visual prediction tasks, enables object discovery methods with larger models, and most importantly leads to smoother feature maps and attention maps for downstream visual processing. [(p. 1)](zotero://open-pdf/library/items/U8QTMY6D?page=1&annotation=RC5C97U6)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-2-x96-y429.png) 

In particular, the DINO algorithm is shown to produce models that contain explicit information about the semantic layout of an image. Indeed, qualitative results show that the last attention layer naturally focuses on semantically consistent parts of images and often produces interpretable attention maps.  Exploiting these properties, object discovery algorithms such as LOST (Simeoni et al., 2021) build ´ on top of DINO. Such algorithms can detect objects without supervision by gathering information in attention maps. They are effectively unlocking a new frontier in computer vision. [(p. 2)](zotero://open-pdf/library/items/U8QTMY6D?page=2&annotation=N2WSTWA4)

When used to extract features, it delivers disappointing performance, only on par with supervised alternative backbones in this scenario. This suggests that DINOv2 behaves differently than DINO. The investigation described in this work notably exposes the presence of artefacts in the feature maps of DINOv2 that were not present in the first version of this model. These are observable qualitatively using straightforward methods. Also surprisingly, applying the same observations to supervised vision transformers exposes similar artifacts, as shown in Fig. 2. This suggests that DINO is, in fact, an exception, while DINOv2 models match the baseline behavior of vision transformers. [(p. 2)](zotero://open-pdf/library/items/U8QTMY6D?page=2&annotation=V8ECACGG)

We observe that they are tokens with roughly 10x higher norm at the output and correspond to a small fraction of the total sequence (around 2%). We also show that these tokens appear around the middle layers of the vision transformer, and that they only appear after a sufficiently long training of a sufficiently big transformer. In particular, we show that these outlier tokens appear in patches similar to their neighbors, meaning patches that convey little additional information. [(p. 2)](zotero://open-pdf/library/items/U8QTMY6D?page=2&annotation=BEEAMIQF)

As part of our investigation, we evaluate the outlier tokens with simple linear models to understand the information they contain. We observe that, compared to non-outlier tokens, they hold less information about their original position in the image or the original pixels in their patch. This observation suggests that the model discards the local information contained in these patches during inference. On the other hand, learning an image classifier on outlier patches yields significantly stronger accuracy than doing so on the other patches, suggesting that they contain global information about the image. We propose the following interpretation to these elements: the model learns to recognize patches containing little useful information, and recycle the corresponding tokens to aggregate global image information while discarding spatial information. [(p. 3)](zotero://open-pdf/library/items/U8QTMY6D?page=3&annotation=9XEAHCMC)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-3-x100-y570.png) 

This interpretation is consistent with an inner mechanism in transformer models that allows performing computations within a restricted set of tokens. In order to test this hypothesis, we append additional tokens that we call registers to the token sequence, independent of the input image. We train several models with and without this modification and observe that the outlier tokens disappear from the sequence entirely. As a result, the performance of the models increases in dense prediction tasks, and the resulting feature maps are significantly smoother. These smooth feature maps enable object discovery methods like LOST mentioned above with the updated models. [(p. 3)](zotero://open-pdf/library/items/U8QTMY6D?page=3&annotation=9S3B3ARI)

# Related Work
Extending the transformer sequence with special tokens was popularized in BERT (Devlin et al., 2019). However, most approaches add new tokens either to provide the network with new information as for example [SEP] tokens in BERT and tape tokens in AdaTape (Xue et al., 2023), or to gather information in these tokens, and use their output value as an output of the model: • for classification: as [CLS] tokens in BERT and ViT (Dosovitskiy et al., 2021) • for generative learning: as [MASK] in BERT and BEiT (Bao et al., 2021) • for detection: as object queries in DETR (Carion et al., 2020), detection tokens in YOLOS (Fang et al., 2021), and ViDT (Song et al., 2021) • for accumulating information from possibly multiple modalities before decoding, as latent token arrays in Perceivers (Jaegle et al., 2021; 2022). [(p. 9)](zotero://open-pdf/library/items/U8QTMY6D?page=9&annotation=D4W6AULF)

The Memory Transformer (Burtsev et al., 2020), closer to our work, presents a simple approach to improve transformer models using memory tokens added to the token sequence, improving translation performance. In follow-up work, Bulatov et al.  (2022) address complex copy-repeat-reverse tasks. Sandler et al. (2022) extend this line to the vision domain for fine-tuning but observe that such tokens do not transfer well across tasks. [(p. 9)](zotero://open-pdf/library/items/U8QTMY6D?page=9&annotation=BBMDK24E) 

# PROBLEM FORMULATION
## ARTIFACTS IN THE LOCAL FEATURES OF DINOV2
### Artifacts are high-norm outlier tokens
We clearly see that the norm of artifact patches is much higher than the norm of other patches. We also plot the distribution of feature norms over a small dataset of images in Fig. 3 (right), which is clearly bimodal, allowing us to choose a simple criterion for the rest of this section: tokens with norm higher than 150 will be considered as “high-norm” tokens, and we will study their properties relative to regular tokens. This hand-picked cutoff value can vary across models. [(p. 3)](zotero://open-pdf/library/items/U8QTMY6D?page=3&annotation=UMKNPT98)

### Outliers appear during the training of large models
First, these high-norm patches seem to differentiate themselves from other patches around layer 15 of this 40-layer ViT (Fig. 4a). Second, when looking at the distribution of norms along training of DINOv2, we see that these outliers only appear after one third of training (Fig. 4b). Finally, when analyzing more closely models of different size (Tiny, Small, Base, Large, Huge and giant), we see that only the three largest models exhibit outliers (Fig. 4c). [(p. 3)](zotero://open-pdf/library/items/U8QTMY6D?page=3&annotation=83GCGECU)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-4-x102-y552.png) 

### High-norm tokens appear where patch information is redundant
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-4-x105-y383.png) 

We observe that high-norm tokens appear on patches that are very similar to their neighbors.  This suggests that these patches contrain redundant information and that the model could discard their information without hurting the quality of the image representation. This matches qualitative observations (see Fig. 2) that they often appear in uniform, background areas. [(p. 4)](zotero://open-pdf/library/items/U8QTMY6D?page=4&annotation=K8MW79AP)

For each of these tasks, we train a linear model on top of the patch embeddings, and measure the performance of this model. We compare the performance achieved with high-norm tokens and with other tokens, to see if high-norm tokens contain different information than “normal” tokens. [(p. 4)](zotero://open-pdf/library/items/U8QTMY6D?page=4&annotation=ACDX7VH7)
1. **Position prediction**. We train a linear model to predict the position of each patch token in the image, and measure its accuracy. We note that this position information was injected in the tokens before the first ViT layer in the form of absolute position embeddings. We observe that high-norm tokens have much lower accuracy than the other tokens (Fig. 5b), suggesting they contain less information about their position in the image. [(p. 4)](zotero://open-pdf/library/items/U8QTMY6D?page=4&annotation=8LW6HRXC)
2. **Pixel reconstruction**. We train a linear model to predict the pixel values of the image from the patch embeddings, and measure the accuracy of this model. We observe again that high-norm tokens achieve much lower accuracy than other tokens (Fig. 5b). This suggests that high-norm tokens contain less information to reconstruct the image than the others. [(p. 4)](zotero://open-pdf/library/items/U8QTMY6D?page=4&annotation=TQ5EPIGJ)

### Artifacts hold global information
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-5-x101-y609.png) 

For each image in a classification dataset, we forward it through DINOv2-g and extract the patch embeddings. From those, we choose a single token at random, either high-norm or normal. This token is then considered as the image representation. We then train a logistic regression classifier to predict the image class from this representation, and measure the accuracy. We observe that the high-norm tokens have a much higher accuracy than the other tokens (Table 1). This suggests that outlier tokens contain more global information than other patch tokens. [(p. 5)](zotero://open-pdf/library/items/U8QTMY6D?page=5&annotation=ZNZBI8GC)

## HYPOTHESIS AND REMEDIATION
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-5-x105-y428.png) 

Having made these observations, we make the following hypothesis: large, sufficiently trained models learn to recognize redundant tokens, and to use them as places to store, process and retrieve global information. Furthermore, we posit that while this behavior is not bad in itself, the fact that it happens inside the patch tokens is undesirable. Indeed, it leads the model to discard local patch information (Tab. 5b), possibly incurring decreased performance on dense prediction tasks. [(p. 5)](zotero://open-pdf/library/items/U8QTMY6D?page=5&annotation=HGUSE9BL)

We therefore propose a simple fix to this issue: we explicitly add new tokens to the sequence, that the model can learn to use as registers. We add these tokens after the patch embedding layer, with a learnable value, similarly to the [CLS] token. At the end of the vision transformer, these tokens are discarded, and the [CLS] token and patch tokens are used as image representations, as usual. This mechanism was first proposed in Memory Transformers (Burtsev et al., 2020), improving translation tasks in NLP. Interestingly, we show here that this mechanism admits a natural justification for vision transformers, fixing an interpretability and performance issue that was present otherwise. [(p. 5)](zotero://open-pdf/library/items/U8QTMY6D?page=5&annotation=W2P7IBIU)

# EXPERIMENTS
## EVALUATION OF THE PROPOSED SOLUTION
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-6-x105-y598.png) 

We run linear probing on ImageNet classification, ADE20k Segmentation, and NYUd monocular depth estimation [(p. 6)](zotero://open-pdf/library/items/U8QTMY6D?page=6&annotation=XXPH6LRG)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-7-x99-y530.png) 

## Number of register tokens
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-7-x100-y303.png) 

There seems to be an optimal number of registers for dense tasks, and adding one brings most of the benefit. This optimum is likely explained by the disappearance of artifacts, leading to better local features. On ImageNet, however, performance improves when using more registers. In all our experiments, we kept 4 register tokens. [(p. 7)](zotero://open-pdf/library/items/U8QTMY6D?page=7&annotation=TJGDDT38)

## OBJECT DISCOVERY
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-8-x104-y554.png) 

For all models and on all datasets, adding registers for training improves the unsupervised object discovery performance. The performance of DINOv2 on VOC2007 still does not match that of DINO as reported in the work of Simeoni et al. (2021) ( ´ 61.9 corloc). However, the model with registers gets an improvement of 20.1 corloc (55.4 versus 35.3) [(p. 8)](zotero://open-pdf/library/items/U8QTMY6D?page=8&annotation=TQTAKVGT)

## QUALITATIVE EVALUATION OF REGISTERS
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/darcetVisionTransformersNeed2023-8-x99-y414.png) 

We see that registers do not have a completely aligned behavior.  Some selected registers exhibit interesting attention patterns, attending to the different objects in the scene. While nothing enforced this behavior, their activations had some natural diversity. [(p. 8)](zotero://open-pdf/library/items/U8QTMY6D?page=8&annotation=DPYLCUMW)

