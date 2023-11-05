---
layout: post
title: Grounding Visual Illusions in Language Do Vision-Language Models Perceive Illusions Like Humans?
tags:  multimodal deep-learning vision-illusion ofa unified-io llava instruction-blip vicuna
---

This is my reading note for [Grounding Visual Illusions in Language: Do Vision-Language Models Perceive Illusions Like Humans?](http://arxiv.org/abs/2311.00047). This paper shows that larger model though more powerful, also more vulnerable to vision illusion as human does.

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-1-x296-y446.png) 

# Introduction

Our findings have shown that although the overall alignment is low, larger models are closer to human perception and more susceptible to visual illusions. [(p. 1)](zotero://open-pdf/library/items/QEK2HEHB?page=1&annotation=8KCZ6RN8)

It’s well established that human perceptual systems are susceptible to visual illusions, which are defined as “consistent and persistent discrepancies between a physical state of affairs and its representation in consciousness” [(p. 1)](zotero://open-pdf/library/items/QEK2HEHB?page=1&annotation=L9Q4UHX5)

We specifically evaluated four state-of-the-art vision-language models: Unified-IO (Lu et al., 2022), OFA (Wang et al., 2022), LLaVA (Liu et al., 2023) and InstructBLIP (Dai et al., 2023). Our results have shown that these four models mostly do not align with human vision illusions, especially for QA-based tasks. However, for the RefLoc task, these models (especially ones with larger parameters) have demonstrated an impressive alignment with humans. [(p. 2)](zotero://open-pdf/library/items/QEK2HEHB?page=2&annotation=BZWLZ6CH)

# Related Work
## Human Visual Illusion 
Visual illusions in humans are instances where human subjective perceived properties, such as color or size, deviates from their true physical characteristics (Carbon, 2014). This underscores the fact that the human brain doesn’t perfectly replicate physical features; rather, it integrates contextual information and prior knowledge to form the perceptual experiences (Carbon, 2014). [(p. 2)](zotero://open-pdf/library/items/QEK2HEHB?page=2&annotation=JQL5CRNS)

## Machine Visual Illusion
previous works demonstrated that convolutional neural networks trained on ImageNet or low-level vision tasks can be misled by certain visual illusions, similar to human responses. These works have formed a foundation for scalable and reproducible research on machine illusions. [(p. 2)](zotero://open-pdf/library/items/QEK2HEHB?page=2&annotation=3JA92P2Y)

# The Grounding Visual Illusion in Language (GVIL) Benchmark
## Data
Each image consists of two objects which may look different to humans but are actually identical in their pixels. [(p. 3)](zotero://open-pdf/library/items/QEK2HEHB?page=3&annotation=6UZQN7QV)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-3-x302-y384.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-4-x66-y610.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-4-x304-y653.png) 

## Benchmark Tasks
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-4-x300-y382.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-5-x298-y396.png) 

# Experimental
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-6-x67-y517.png) 

First of all, we notice a large percentage of responses, across all models, fall under the N/A category. This suggests that these models often cannot even tell that the objects are identical in the illusion-free image, underscoring the need for improvement in standard vision-language reasoning capabilities beyond the scope of illusion contexts. [(p. 6)](zotero://open-pdf/library/items/QEK2HEHB?page=6&annotation=PY5VDABC)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-7-x65-y486.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-7-x63-y320.png) 

When examining cases where responses are applicable for testing illusion recognition, we observe that the majority of models are more likely to fail in recognizing illusions (35.4% on average) than producing humanlike responses (15.6% on average). This discrepancy is most pronounced in the case of InstructBLIP, where the model predominantly offers ’no-illusion’ answers. [(p. 7)](zotero://open-pdf/library/items/QEK2HEHB?page=7&annotation=ZS8XWIRV)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-7-x299-y487.png) 

This finding suggests a positive correlation between model scale and human-machine alignment under illusions. [(p. 7)](zotero://open-pdf/library/items/QEK2HEHB?page=7&annotation=DHQK4JJY)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-8-x71-y620.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/zhangGroundingVisualIllusions2023-8-x68-y363.png)
