---
layout: post
title: SeamlessM4T-Massively Multilingual & Multimodal Machine Translation
tags:  read llm self_supervised multimodal transformer audio review quantization response-ai language-identification segmentation whisper bert w2v-bert SeamlessM4T speech-recognition asr text-to-speech tts machine-translation nllb hubert laser sonar voice-activity-detection vad
---
This is my reading note 2/2 on [SeamlessM4T-Massively Multilingual & Multimodal Machine Translation](https://github.com/facebookresearch/seamless_communication). It is end to end multi language translation system supports multimodality (text and audio). This paper also provides a good review on machine translation. This note focus on data preparation part of the paper and please read [SeamlessM4T-data](https://zhangtemplar.github.io/SeamlessM4T-data/) for the other part.

# SeamlessM4T Models
The aforementioned approach alleviates the issue of cascaded error propagation and domain mismatch, while relying on an intermediate semantic representation to mitigate the problem of multi-modal source-target mapping. The vocoders for synthesizing speech are trained separately (see Section 4.3.1). Figure 4 provides an overview of the SeamlessM4T model, including its four building blocks: (1) SeamlessM4T-NLLB a massively multilingual T2TT model, (2) w2v-BERT 2.0, a speech representation learning model that leverages unlabeled speech audio data, (3) T2U, a text-to-unit sequence-to-sequence model, and (4) multilingual HiFi-GAN unit vocoder for synthesizing speech from units. [(p. 27)](zotero://open-pdf/library/items/5CXTS7WH?page=27&annotation=TESKY9HE)

The SeamlessM4T multitask UnitY model integrates components from the first three building blocks and is fine-tuned in three stages, starting from an X2T model (1,2) with English target only and ending with a full-fledged multitask UnitY (1,2,3) system capable of performing T2TT, S2TT and S2ST, as well as ASR [(p. 27)](zotero://open-pdf/library/items/5CXTS7WH?page=27&annotation=HFEJ3BYP)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-28-x84-y429.png) 

## Unsupervised Speech Pre-training
w2v-BERT 2.0 follows w2v-BERT [Chung et al., 2021] to combine contrastive learning and masked prediction learning, and improves w2v-BERT with additional codebooks in both learning objectives. The contrastive learning module is used to learn Gumbel vector quantization (GVQ) codebooks and contextualized representations that are fed into the subsequent masked prediction learning module. The latter refines the contextualized representations by a different learning task of predicting the GVQ codes directly instead of polarizing the prediction probability of correct and incorrect codes at the masked positions.  Instead of using a single GVQ codebook, w2v-BERT 2.0 follows Baevski et al. [2020] to use product quantization with two GVQ codebooks. Its contrastive learning loss Lc is the same as that in w2v-BERT, including a codebook diversity loss to encourage the uniform usage of codes. Following w2v-BERT, we use GVQ codebooks for masked prediction learning and denote the corresponding loss as LmGVQ . We also created an additional masked prediction task using random projection quantizers [Chiu et al., 2022] (RPQ), for which we denote the corresponding loss as LmRPQ . [(p. 29)](zotero://open-pdf/library/items/5CXTS7WH?page=29&annotation=2UJH4X3N)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-29-x83-y596.png) 

## X2T: Into-Text Translation and Transcription
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-29-x83-y124.png) 

In SeamlessM4T, we leveraged foundational models either pre-trained on unlabeled data (w2v-BERT 2.0 for speech encoder pre-training) or trained on supervised high-resource tasks (NLLB model for T2TT) to improve the quality of transfer tasks (speech-to-text and speech-to-speech). To fuse these pre-trained components and enable meaning transfer through multiple multimodal tasks, we trained an end-to-end model with (a) a speech encoder (w2v-BERT 2.0) postfixed with a length adapter, (b) text encoder (NLLB encoder), and (c) a text decoder (NLLB decoder). For the length adaptor, we used a modified version of M-adaptor [Zhao et al., 2022], where we replaced the 3 independent pooling modules for Q, K, and V with a shared pooling module to improve efficiency. [(p. 32)](zotero://open-pdf/library/items/5CXTS7WH?page=32&annotation=JGI939HN)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-32-x212-y232.png) 

We additionally optimize an auxiliary objective function in the form of token-level knowledge distillation (LKD), to further transfer knowledge from the strong MT model to the student speech translation task (S2TT) [(p. 32)](zotero://open-pdf/library/items/5CXTS7WH?page=32&annotation=YCRHZUIF)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-32-x174-y120.png) 

We trained our X2T model in two stages. Stage1 targeted training on supervised English ASR and into English S2TT data. We find that this step is necessary not only for improving the quality of X–eng translations but also eng–X translations. In fact, we hypothesized that allowing the model to focus on one target language while fine-tuning multilingual speech representations shields it from the interference that can propagate back from the target side. 
In Stage2, we add supervised eng–X S2TT and non-English ASR data to the mix. [(p. 33)](zotero://open-pdf/library/items/5CXTS7WH?page=33&annotation=RANVKNLM)

## Speech-to-Speech Translation
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-33-x81-y192.png) 

The key to our proposed speech-to-speech translation model is the use of self-supervised discrete acoustic units to represent target speech, thereby decomposing the S2ST problem into a speech-to-unit translation (S2UT) step and a unit-to-speech (U2S) conversion step. [(p. 33)](zotero://open-pdf/library/items/5CXTS7WH?page=33&annotation=8RWE99TZ)

Discrete acoustic units Recent works have achieved SOTA translation performance by using self-supervised discrete acoustic units as targets for building direct speech translation models [Tjandra et al., 2019; Lee et al., 2022a,b; Zhang et al., 2022; Chen et al., 2023c]. [(p. 34)](zotero://open-pdf/library/items/5CXTS7WH?page=34&annotation=LAYFRP58)

The mapping from XLS-R continuous representation space to discrete categories is required to map target speech into a sequence of discrete tokens.  We randomly selected and encoded 10K unlabeled audio samples from each language of the 35 supported target languages. We then applied a k-means algorithm on these representations to estimate K cluster centroids [Lakhotia et al., 2021; Polyak et al., 2021; Lee et al., 2022a]. 
These centroids resemble a codebook that is used to map a sequence of XLS-R speech representations into a sequence of centroid indices or acoustic units. [(p. 34)](zotero://open-pdf/library/items/5CXTS7WH?page=34&annotation=YEAZ8IQC)

Following Gong et al. [2023], we built the multilingual vocoder for speech synthesis from the learned units. The HiFi-GAN vocoder [Kong et al., 2020] is equipped with language embedding to model the languagespecific acoustic information. Moreover, to mitigate cross-lingual interference, language identification is used as an auxiliary loss in multilingual training. [(p. 34)](zotero://open-pdf/library/items/5CXTS7WH?page=34&annotation=A6VDZUN8)

The T2U model is a Transformer-based encoder-decoder model trained on aligned text units from ASR data. [(p. 35)](zotero://open-pdf/library/items/5CXTS7WH?page=35&annotation=IL9QGWMI)

# Experiment
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-38-x103-y517.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-38-x94-y288.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-39-x83-y301.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/communicationSeamlessM4TMassivelyMultilingualMultimodal2023-39-x88-y137.png)
