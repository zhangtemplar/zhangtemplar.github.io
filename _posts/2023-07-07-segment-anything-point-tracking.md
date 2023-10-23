---
layout: post
title: Segment Anything Meets Point Tracking
tags:  sam-track tracking tracking-anything raft xmem deoat persam dino segment-anything deep-learning transformer seg-gpt sam point-tracking
---

This is my reading note for [Segment Anything Meets Point Tracking](https://github.com/SysCV/sam-pt). This paper combines SAM with point tracker to perform object segment and tracking in video. To to that it use point tracker to track points through the frames.for points of each frame SAM generate masks from the points promote. After every 8 frames, new points will be sampled from the mask.for best performance, 8 positive points and l negative points is recommended.

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-1-x42-y338.png) 

# Introduction
These annotations are called “query points” and denote either the target object (positive points) or designate non-target segments (negative points). The points are tracked throughout the video using point trackers that propagate the query points to all video frames, producing predicted trajectories [(p. 1)](zotero://open-pdf/library/items/S8QB3TVJ?page=1&annotation=B45D6IQJ)

SAM-PT leverages robust and sparse point selection and propagation techniques for mask generation. Compared to traditional object-centric mask propagation strategies, we uniquely use point propagation to exploit local structure information that is agnostic to object semantics [(p. 1)](zotero://open-pdf/library/items/S8QB3TVJ?page=1&annotation=J2TEQT2B)

1. TAM [40] integrates SAM with the state-of-the-art memory-based mask tracker XMem [4]. Likewise, SAM-Track [6] combines SAM with DeAOT [41]. [(p. 2)](zotero://open-pdf/library/items/S8QB3TVJ?page=2&annotation=P9VFLQCH)
2. Other methods that do not leverage SAM, such as SegGPT [36], can successfully solve a number of segmentation problems using visual prompting, but still require mask annotation for the first video frame. [(p. 2)](zotero://open-pdf/library/items/S8QB3TVJ?page=2&annotation=Y48XSCZ6)

Instead of employing object-centric dense feature matching or mask propagation, we propose a point-driven approach that capitalizes on tracking points using rich local structure information embedded in videos. [(p. 2)](zotero://open-pdf/library/items/S8QB3TVJ?page=2&annotation=ZK7HTITG)

We identify that initializing points to track using K-Medoids cluster centers from a mask label was the strategy most compatible with prompting SAM. Tracking both positive and negative points enables the clear delineation of target objects from their background. To further refine the output masks, we propose multiple mask decoding passes that integrate both types of points. In addition, we devised a point re-initialization strategy that increases tracking accuracy over time. This approach involves discarding points that have become unreliable or occluded, and adding points from object parts or segments that become visible in later frames, such as when the object rotates. [(p. 2)](zotero://open-pdf/library/items/S8QB3TVJ?page=2&annotation=8BYE9MGE)

# Related Work
## Point Tracking for Video Segmentation.
their effectiveness is confined to a specific set of distinct interest points and they often struggle when applied to non-rigid, dynamic scenes. Flow-based methods, such as RAFT [30], excel in tracking dense points between successive frames. However, they stumble with deriving accurate long-range point trajectories. When chaining flow predictions over time, errors [(p. 2)](zotero://open-pdf/library/items/S8QB3TVJ?page=2&annotation=LD5IRFN6)

## Segment and Track Anything models
These methods employ SAM for mask initialization or correction and XMem/DeAOT for mask tracking and prediction. Using the pre-trained mask trackers recovers the indistribution performance, but hinders the performance in zero-shot settings. PerSAM [45] also demonstrates the ability to track multiple reference objects in a [(p. 3)](zotero://open-pdf/library/items/S8QB3TVJ?page=3&annotation=X7YEKTMF)

## Zero-shot VOS / VIS
In the semi-supervised video object segmentation, they take a reference mask as input and perform frame-by-frame feature matching, which propagates the reference mask across the entirety of the video [(p. 3)](zotero://open-pdf/library/items/S8QB3TVJ?page=3&annotation=E67X79UI)

## Segment Anything Model
SAM comprises of three main components: an image encoder, a flexible prompt encoder, and a fast mask decoder. The image encoder is a Vision Transformer (ViT) backbone and processes high-resolution 1024 × 1024 images to generate an image embedding of 64 × 64 spatial size. The prompt encoder takes sparse prompts as input, including points, boxes, and text, or dense prompts such as masks, and translates these prompts into c-dimensional tokens. The lightweight mask decoder then integrates the image and prompt embeddings to predict segmentation masks in real-time, allowing SAM to adapt to diverse prompts with minimal computational overhead. [(p. 3)](zotero://open-pdf/library/items/S8QB3TVJ?page=3&annotation=DAG375DC)

# Method
SAM-PT is illustrated in Fig. 2 and is primarily composed of four steps: 1) selecting query points for the first frame; 2) propagating these points to all video frames using point trackers; 3) using SAM to generate per-frame segmentation masks based on the propagated points; 4) optionally reinitializing the process by sampling query points from the predicted masks. We next elaborate on these four steps. [(p. 3)](zotero://open-pdf/library/items/S8QB3TVJ?page=3&annotation=NWX4FFVQ)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-4-x48-y440.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-5-x38-y450.png) 

## Segmentation
we prompt SAM exclusively with positive points to define the object’s initial localization. Subsequently, in the second pass, we prompt SAM with both positive and negative points along with the previous mask prediction. Negative points provide a more nuanced distinction between the object and the background and help by removing wrongly segmented areas. [(p. 5)](zotero://open-pdf/library/items/S8QB3TVJ?page=5&annotation=8WRK3WGA)

Lastly, we execute a variable number of mask refinement iterations by repeating the second pass. This utilizes SAM’s capacity to refine vague masks into more precise ones [(p. 5)](zotero://open-pdf/library/items/S8QB3TVJ?page=5&annotation=YKSTCVJ7)

## Point Tracking Reinitialization
We optionally execute a reinitialization of the query points using the predicted masks once a prediction horizon of h = 8 frames is reached, and denote the variant as SAM-PT-reinit. Upon reaching this horizon, we have h predicted masks and will take the last predicted mask to sample new points. At this stage, all previous points are discarded and substituted with the newly sampled points [(p. 5)](zotero://open-pdf/library/items/S8QB3TVJ?page=5&annotation=J5ES2ZZC)

The steps are iteratively executed until the entire video is processed. The reinitialization process serves to enhance tracking accuracy over time by discarding points that have become unreliable or occluded, while incorporating points from object segments that become visible later in the video. [(p. 5)](zotero://open-pdf/library/items/S8QB3TVJ?page=5&annotation=NZFSWGBE)

## SAM-PT vs. Object-centric Mask Propagation
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-5-x46-y243.png) 

First, point propagation exploits local structure context that is agnostic to global object semantics. This enhances our model’s capability for zero-shot generalization, [(p. 5)](zotero://open-pdf/library/items/S8QB3TVJ?page=5&annotation=PFG2WBAR)

SAM-PT allows for a more compact object representation with sparse points, capturing enough information to characterize the object’s segments/parts effectively. Finally, the use of points is naturally compatible with SAM, an image segmentation foundation model trained to operate on sparse point prompts, offering an integrated solution that aligns well with the intrinsic capacities of the underlying model. [(p. 5)](zotero://open-pdf/library/items/S8QB3TVJ?page=5&annotation=23IIH4I4)

## Implementation Details
PIPS is trained exclusively on a synthetic dataset, FlyingThings++ [11], derived from the FlyingThings [24] optical flow dataset. [(p. 7)](zotero://open-pdf/library/items/S8QB3TVJ?page=7&annotation=MKKD9D3J)

we found that using iterative refinement negatively impacted both SAM-PT and SAM-PT-reinit on the MOSE dataset, and likewise hindered SAM-PT-reinit on the YouTube-VOS dataset [(p. 7)](zotero://open-pdf/library/items/S8QB3TVJ?page=7&annotation=FS4NDW7W)

# Experiment
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-6-x43-y470.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-10-x43-y98.png) 

# Ablation Study
## Query Point Sampling
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-8-x44-y538.png) 

## Point Tracking
![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-8-x69-y672.png) 

TapNet’s limitations stem from its lack of effective time consistency and its training on 256x256 images, which hampered its performance with higher-resolution images. SuperGlue, while proficient in matching sparse features across rigid scenes, grapples with effectively matching points from the reference frame in dynamic scenes, particularly under object deformations. 
RAFT, being an optical flow model, faced difficulties handling occlusions. [(p. 8)](zotero://open-pdf/library/items/S8QB3TVJ?page=8&annotation=IX9X67HN)

## Negative Points
Tab. 2b highlights that incorporating negative points had a favorable impact, particularly in reducing segmentation errors when points deviated from the target object. The addition of negative points empowered SAM to better handle the point trackers’ failure cases, leading to improved segmentation and a 1.8-point enhancement over the non-use of negative points. [(p. 8)](zotero://open-pdf/library/items/S8QB3TVJ?page=8&annotation=N685GB6V)

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-8-x303-y383.png) 

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/rajicSegmentAnythingMeets2023-9-x41-y499.png) 

In summary, our best-performing SAM-PT model employs K-Medoids for point selection with 8 points per mask, PIPS for point tracking, a single negative point per mask, and employs 12 iterations for iterative refinement without patch similarity filtering. Meanwhile, using reinitialization achieved optimum performance with 12 refinement iterations and 72 negative points per mask. [(p. 9)](zotero://open-pdf/library/items/S8QB3TVJ?page=9&annotation=WRYMKBAG)

# Limitation
Despite the competitive zero-shot performance, certain limitations persist, primarily due to the limitations of our point tracker in handling occlusion, small objects, motion blur, and re-identification. In such scenarios, the point tracker’s errors propagate into future video frames [(p. 9)](zotero://open-pdf/library/items/S8QB3TVJ?page=9&annotation=AUPMWMGF)

