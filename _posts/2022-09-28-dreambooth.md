---
layout: post
title: DreamBooth Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
tags:  image-caption dreambooth deep-learning transformer diffusion text-image-synthesize
---
This is my reading note on [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242v1). Given as input just a few (3~5) images of a subject, DreamBooth fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, DreamBooth enables synthesizing the subject in diverse scenes, poses, views, and lighting conditions that do not appear in the reference images. (check Figure 1 as an example)

![image-20220928153058492](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_30_58_image-20220928153058492.png)

More formally, given a few images of a subject (∼3-5), our objective is to implant the subject into the output domain of the model such that it can be synthesized with a unique identifier. To that end, we propose techniques to represent a given subject with rare token identifiers and fine-tune a pre-trained, diffusion-based text-to-image framework that operates in two steps; 

1. generating a low-resolution image from text 
2. and subsequently applying super-resolution (SR) diffusion models. 

We first fine-tune the low-resolution text-to image model with the input images and text prompts containing a unique identifier followed by the class name of the subject (e.g., “A [V] dog”). In order to prevent overfitting and language drift [35, 40] that cause the model to associate the class name (e.g., “dog”) with the specific instance, we propose an autogenous, class-specific prior preservation loss, which leverages the semantic prior on the class that is embedded in the model, and encourages it to generate diverse instances of the same class as our subject. In the second step, we fine-tune the super-resolution component with pairs of low-resolution and high-resolution versions of the input images. This allows the model to maintain high fidelity to small (but important) details of the subject. We use the pre-trained Imagen model [56] as a base model in our experiments. Check Figure 3 for an overview of the DreamBooth.

![image-20220928153408802](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_35_39_2022_09_28_15_34_08_image-20220928153408802.png)

# Representing the Subject with a Rare-token Identifier

Our goal is to “implant” a new (key, value) pair into the diffusion model’s “dictionary” such that, given the key for our subject, we are able to generate fully-novel images of this specific subject with meaningful semantic modifications guided by a text prompt.

We opt for a simpler approach and label all input images of the subject “a [identifier] [class noun]”, where [identifier] is a unique identifier linked to the subject and [class noun] is a coarse class descriptor of the subject (e.g. cat, dog, watch, etc.). The class descriptor can be obtained using a classifier. Figure 11 shows a comparison of only using identifier vs wrong class noun vs proposed method.

![image-20220928154326310](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_43_26_image-20220928154326310.png)

How to get the identifier:

1. A naive way of constructing an identifier for our subject is to use an existing word. For example, using the words like “unique” or “special”. One problem is that existing English words tend to have a stronger prior due to occurrence in the training set of text-to-image diffusion models. We generally find increased training time and decreased performance when using such generic words to index our subject
2. A hazardous way of doing this is to select random characters in the English language and concatenate them to generate a rare identifier (e.g. “xxy5syt00”). In reality, the tokenizer might tokenize each letter separately, and the prior for the diffusion model is strong for these letters. Specifically, if we sample the model with such an identifier before fine-tuning we will get pictorial depictions of the letters or concepts that are linked to those letters. We often find that these tokens incur the same weaknesses as using common English words to index the subject.
3. **Proposed rare-token identifier**. In a nutshell, our approach is to find relatively rare tokens in the vocabulary, and then invert these rare tokens into text space. We observe that using uniform random sampling without replacement of tokens that correspond to 3 or fewer Unicode characters (without spaces) and using tokens in the T5-XXL tokenizer range of {5000, ..., 10000} works well.

# Class-specific Prior Preservation Loss

As mentioned above, DreamBooth fine-tune a pre-trained, diffusion-based text-to-image framework on a few images (3~5). This raises two issues (as shown in Figure 12):

![image-20220928154226782](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_46_59_2022_09_28_15_42_26_image-20220928154226782.png)

1. A key problem is that fine-tuning on a small set of images showing our subject is prone to overfitting on the given images. 
2. In addition, language drift [35, 40] is a common problem in language models, and manifests itself in text-to-image diffusion models as well: the model can forget how to generate other subjects of the same class, and lose the embedded knowledge on the diversity and natural variations of instances belonging to that class. 

For this, we present an autogenous class-specific prior preservation loss, where we alleviate overfitting and prevent language drift by encouraging the diffusion model to keep generating diverse instances of the same class as our subject. In essence, our method is to supervise the model with its own generated samples, in order for it to retain the prior once the few-shot fine-tuning begins. Figure 4 shows an overview of this finetune step.

![image-20220928154601147](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_46_01_image-20220928154601147.png)



# Personalized Instance-Specific Super-Resolution

We find that if SR networks are used without fine-tuning, the generated output can contain artifacts since the SR models might not be familiar with certain details or textures of the subject instance, or the subject instance might have hallucinated incorrect features, or missing details. Figure 14 shows a comparison for with/without this finetuning.

![image-20220928154130483](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_41_30_image-20220928154130483.png)

# Experiment Result

![image-20220928154913801](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_49_13_image-20220928154913801.png)

![image-20220928154929950](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_49_30_image-20220928154929950.png)

![image-20220928154941465](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_49_41_image-20220928154941465.png)

![image-20220928154959226](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_49_59_image-20220928154959226.png)

![image-20220928155010715](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_50_10_image-20220928155010715.png)

![image-20220928155024817](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_50_24_image-20220928155024817.png)

![image-20220928155056565](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/2022_09_28_15_50_56_image-20220928155056565.png)
