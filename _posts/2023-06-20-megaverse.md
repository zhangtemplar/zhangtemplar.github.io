---
layout: post
title: MEGAVERSE Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks
tags:  llm multimodal deep-learning dataset metaverse benchmark multi-lingual marvl in22 XRiSAWOZ Belebele xm-3600 gpt palm llama
---

This is my reading note for [MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks](http://arxiv.org/abs/2311.07463). This paper proposes a new multilingual benchmark to test LLM and provides very limited dataset for multimodality. The language distribution is also strange which houses to much on south, Asia. Overall GPT and Palm get the best performance.

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-3-x54-y415.png) 

# Introduction
The benchmark comprises 22 datasets covering 81 languages, including low-resource African languages. We evaluate several state-of-the-art LLMs like GPT3.5-Turbo, GPT4, PaLM2, and Llama2 on the MEGAVERSE datasets. Additionally, we include two multimodal datasets in the benchmark and assess the performance of the LLaVav1.5 model. Our experiments suggest that GPT4 and PaLM2 outperform the Llama models on various tasks, notably on low-resource languages, with GPT4 outperforming PaLM2 on more datasets than vice versa. [(p. 1)](zotero://open-pdf/library/items/NYAI8CVQ?page=1&annotation=3R7MUGRA)

## Belebele
Belebele (Bandarkar et al., 2023) is a multiple choice machine reading comprehension (MRC) dataset is parallel across 122 languages. Each question is linked to a short passage from the FLORES200 dataset (Team et al., 2022). The questions were created by human annotators and the human annotation procedure was carefully curated to create questions that discriminate between different levels of language comprehension. This process was reinforced by extensive quality checks. [(p. 3)](zotero://open-pdf/library/items/NYAI8CVQ?page=3&annotation=8JUY3CCC)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-3-x301-y177.png) 

We perform zero-shot monolingual prompting for our experiments, as this dataset does not have a dev set. [(p. 3)](zotero://open-pdf/library/items/NYAI8CVQ?page=3&annotation=SAIQ5ZET)

## XRiSAWOZ
XRiSAWOZ (Moradshahi et al., 2023) is a (domain specific) task oriented dialogue modeling dataset. The dataset is a multilingual (English, Hindi, French, Korean) translation of RiSAWOZ dataset (Quan et al., 2020) which was Chinese.  XRiSAWOZ also includes an English-Hindi code mixed setting.  Each dialogue in XRiSAWOZ is confined to a narrow domain and the conversation agent must make use of structured knowledge available in the database to answer user queries. [(p. 4)](zotero://open-pdf/library/items/NYAI8CVQ?page=4&annotation=N4LFAVJP)

## IN22
IN22 (Gala et al., 2023) is a translation benchmark for all 22 scheduled Indic languages which is offered in two flavors, IN22-Gen and IN22-Conv [(p. 4)](zotero://open-pdf/library/items/NYAI8CVQ?page=4&annotation=ECTUC6CI)

## MaRVL
The concepts and images collected were entirely driven by native speakers and are representative of various cultures across the globe and span 5 languages, i.e., Indonesian, Chinese, Swahili, Tamil, Turkish. Each instance in the dataset consists of a pair of images (left image and right image) and a statement, and the task is to determine whether the statement is consistent with respect to the given pair of images. [(p. 4)](zotero://open-pdf/library/items/NYAI8CVQ?page=4&annotation=VWKTIC4H)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-5-x67-y660.png) 

## XM-3600
Crossmodal-3600 (Thapliyal et al., 2022) is a multilingual image captioning dataset consisting of 3600 geographically diverse images directly captioned in 36 different languages, avoiding any inconsistencies due to translations. [(p. 5)](zotero://open-pdf/library/items/NYAI8CVQ?page=5&annotation=LK589J2G)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-5-x63-y350.png) 

# Prompting strategies
We define five main components to define the prompts:
1. a test example x_test for which the predictions are to be made;
2. k few-shot exemplars ${(x_i, y_i)}^k _{i=1}$, that are used to provide in-context supervision to the model;
3. a task instruction I which describes the instruction in text for the task to LLM
4. a prompt template f_temp(x) which turns a dataset input example into a text format that can be used for prompting; 
5. and an answer verbalizer f_verb(y) that maps the label y to a textual representation. [(p. 6)](zotero://open-pdf/library/items/NYAI8CVQ?page=6&annotation=QL8P5AE7)

In our previous work, we show that the monolingual prompting variation outperforms the zero-shot cross-lingual prompting variation for most datasets, with the translate-test variation performing better than monolingual for a few low-resource languages.  We find that the gap between translate-test and monolingual prompting is minimal for models such as GPT4, and so for this work default to monolingual prompting except when specified otherwise. [(p. 6)](zotero://open-pdf/library/items/NYAI8CVQ?page=6&annotation=5RJ62HPY)

# Result
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-8-x73-y563.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-9-x73-y122.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-10-x73-y450.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-10-x69-y100.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-12-x65-y550.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-12-x73-y316.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-12-x63-y72.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-15-x64-y544.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-15-x73-y314.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-15-x70-y78.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-16-x62-y531.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-16-x60-y351.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-17-x59-y596.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/ahujaMEGAVERSEBenchmarkingLarge2023-17-x57-y408.png)
