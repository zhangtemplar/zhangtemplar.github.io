---
layout: post
title: The Impact of Depth and Width on Transformer Language Model Generalization
tags:  llm transformer deep-learning width depth
---

This is my reading note for [The Impact of Depth and Width on Transformer Language Model Generalization](http://arxiv.org/abs/2310.19956). This paper shows that deeper transformer is necessary to have a good performance. Usually 4 to 6 layers is a good choice.

# Introduction
We report three main conclusions:
1. after fine-tuning, deeper models generalize better out-of-distribution than shallower models do, but the relative benefit of additional layers diminishes rapidly;
2. within each family, deeper models show better language modeling performance, but returns are similarly diminishing;
3. the benefits of depth for compositional generalization cannot be attributed solely to better performance on language modeling or on in-distribution data. [(p. 1)](zotero://open-pdf/library/items/8P96VLXG?page=1&annotation=722VSVIY)

# METHODOLOGY
## CONSTRUCTING FAMILIES OF MODELS WITH EQUAL NUMBERS OF PARAMETERS
we can reduce the size of the feed-forward dimension 𝑑_ff, reduce the size of the residual stream (the embedding size) 𝑑_model, or reduce the size of the attention outputs 𝑑_attn (see Appendix B for a diagram of a transformer layer annotated with dimensionality labels). Vaswani et al. (2017) coupled these three variables at 𝑑_model = 𝑑_attn = 𝑑_ff/4. [(p. 2)](zotero://open-pdf/library/items/8P96VLXG?page=2&annotation=HF4M2U9H)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-13.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-3-x103-y515.png) 

# RESULTS
## LANGUAGE MODELING
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-5-x93-y566.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-5-x95-y359.png) 

1. While deeper models do, in general, perform better than shallower ones, the increase in performance that comes from adding layers diminishes rapidly as models become deeper (Figure 3a). [(p. 4)](zotero://open-pdf/library/items/8P96VLXG?page=4&annotation=VDGZFVB9)
 2. At the deeper end of our scale, adding layers is not only unhelpful for performance, but begins to harm it (see the right-hand sides of each size-class curve in Figure 3a). [(p. 5)](zotero://open-pdf/library/items/8P96VLXG?page=5&annotation=924S2EBK)
3. We find that smaller models are more sensitive to the particular value of the feed-forward ratio, and that for small models the standard ratio may not be optimal. This shows that larger models have more leeway to trade depth for width, becoming wider in proportion to their model dimension 𝑑model without incurring large penalties for their perplexity. It also shows that when 𝑑model/𝑑ff < 1 the feedforward ratio no longer serves as a predictor of relative perplexity independent of size. [(p. 5)](zotero://open-pdf/library/items/8P96VLXG?page=5&annotation=66JQWJHG)
## COMPOSITIONAL GENERALIZATION
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-6-x101-y551.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-7-x103-y568.png) 

4. On each of the datasets, deeper models tend to attain higher generalization accuracies than shallower models in the same size class. [(p. 6)](zotero://open-pdf/library/items/8P96VLXG?page=6&annotation=3L3NG8JT)
5. As with language modeling, most of the benefit of depth is gained by having only a few layers. This supports the hypothesis that the saturated effect of depth is due to the existence of easier subsets of the datasets, and shows that increasing depth alone does substantially improve the models’ ability to learn the correct inductive bias for these structural tasks [(p. 6)](zotero://open-pdf/library/items/8P96VLXG?page=6&annotation=PC5VEHJW)

## THE EFFECT OF DEPTH ON GENERALIZATION IS NOT SOLELY ATTRIBUTABLE TO BETTER PRETRAINING LOSS OR IN-DISTRIBUTION PERFORMANCE
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-8-x105-y497.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/pettyImpactDepthWidth2023-8-x107-y319.png)

6. Both of these observations are potential confounds for the interpretation of the previous section: perhaps depth does not directly improve generalization accuracy, but only does so indirectly by allowing models to either become better LMs or else to better learn the in-distribution fine-tuning data [(p. 7)](zotero://open-pdf/library/items/8P96VLXG?page=7&annotation=RNVT8SD3)

