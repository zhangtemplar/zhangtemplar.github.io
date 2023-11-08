---
layout: post
title: An Early Evaluation of GPT-4V(ision)
tags:  multimodal deep-learning dataset gpt-4v image-caption vqa
---

This is my reading note for [An Early Evaluation of GPT-4V(ision)](http://arxiv.org/abs/2310.16534). The highlights of our findings are as follows:
1. GPT-4V exhibits impressive performance on English visual-centric benchmarks but fails to recognize simple Chinese texts in the images;
2. GPT-4V shows inconsistent refusal behavior when answering questions related to sensitive traits such as gender, race, and age;
3. GPT-4V obtains worse results than GPT-4 (API) on language understanding tasks including general language understanding benchmarks and visual commonsense knowledge evaluation benchmarks;
4. Few-shot prompting can improve GPT-4V’s performance on both visual understanding and language understanding;
5. GPT-4V struggles to find the nuances between two similar images and solve the easy math picture puzzles; 
6. GPT-4V shows non-trivial performance on the tasks of similar modalities to image, such as video and thermal. O [(p. 1)](zotero://open-pdf/library/items/PXYRP5WJ?page=1&annotation=8UPXQSMZ)

The current version of GPT-4V does not support interleaved images and texts and can only accept a maximum of four images. These constraints limit the design space of prompts. [(p. 2)](zotero://open-pdf/library/items/PXYRP5WJ?page=2&annotation=5GK6CPWX)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-3-x64-y416.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-4-x74-y435.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-7-x66-y406.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-7-x77-y231.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-8-x140-y149.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-10-x73-y419.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-12-x78-y421.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-12-x78-y86.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-13-x64-y441.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/wuEarlyEvaluationGPT4V2023-13-x102-y113.png)
