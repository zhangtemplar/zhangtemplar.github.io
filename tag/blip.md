---
layout: tagpage
title: "Reading Note on Deep Learning"
tag: blip
---

Bootstrapping Language-Image Pre-training is a technique for pre-training vision-and-language models like CLIP. The key idea is to alternate between two pre-training objectives:
1. Masked language modeling (MLM) on text only 
2. Contrastive learning between text and images 

The two objectives are trained iteratively, with the image-text alignment objective bootstrapping off of the improvements in language modeling from the MLM objective. This allows the model to gradually build connections between textual concepts and visual concepts.

The bootstrapping approach allows the model to learn richer associations than training the two objectives jointly from the start. In experiments, models pre-trained with this bootstrapping approach achieved state-of-the-art results on vision-language tasks.
