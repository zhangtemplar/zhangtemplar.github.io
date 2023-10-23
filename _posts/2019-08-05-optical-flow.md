---
layout: post
title: Optical Flow
tags:  klt flownet deep-learning optical-flow lucas-kanade
---

![](https://developer.nvidia.com/sites/default/files/akamai/designworks/opticalflow/OF_SDK_000.png)

> Optical flow or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene. Optical flow can also be defined as the distribution of apparent velocities of movement of brightness pattern in an image.

In this post, we will introduce some optical flow algorithms, from oldest one to latest one.

# Lucas-Kanade

Lucas-Kanade is most classical methods for computing the optical flow. Its assumes that, the illumination of the same object in the same frame doesn't change. That is:

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+I%28x%2Cy%2Ct%29%3DI%28x%2Bu%2Cy%2Bv%2Ct%2B%E2%88%86_t%29+%5Ctag%7B3-1-1%7D+%5Cend%7Bequation%7D+)

By Taylor expansion, we have:

![](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bequation%7D+I%28x%2Bu%2Cy%2Bv%2Ct%2B%E2%88%86_t%29%3DI%28x%2Cy%2Ct%29%2BI_x%5E%E2%80%B2+u%2BI_y%5E%E2%80%B2+v%2BI_t%5E%E2%80%B2+%E2%88%86_t+%5Ctag%7B3-1-2%7D+%5Cend%7Bequation%7D+)
![](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bequation%7D+I%28x%2Cy%2Ct%29+%3D+I%28x%2Cy%2Ct%29%2BI_x%5E%E2%80%B2+u%2BI_y%5E%E2%80%B2+v%2BI_t%5E%E2%80%B2+%E2%88%86_t+%5C%5C+I_x%5E%E2%80%B2+u%2BI_y%5E%E2%80%B2+v%2BI_t%5E%E2%80%B2+%E2%88%86_t%3D0+%5Ctag%7B3-1-3%7D+%5Cend%7Bequation%7D+)
![](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bequation%7D+%5Cbegin%7Bbmatrix%7D+I_x%27%2C+%5Cspace+I_y%27++%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7Du+%5C%5C+v%5Cend%7Bbmatrix%7D%3D-I_t%5E%E2%80%B2+%E2%88%86_t+%5Ctag%7B3-1-4%7D%5Cend%7Bequation%7D+)

where $I_x$ and $I_y$ is the partial gradient in x and y direction, $I_t^'\Delta_t$ is the difference of illumination between two frames.

However, there are infinite number of solutions to the problem above. To resolve this problem, it further assumes the optical flow is consistent for a local region. Thus we have:

![](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bequation%7D+%5Cbegin%7Bbmatrix%7D++I_x%27%5E%7B%281%29%7D%2C+%5Cspace+I_y%27%5E%7B%281%29%7D+%5C%5C+I_x%27%5E%7B%282%29%7D%2C+%5Cspace+I_y%27%5E%7B%282%29%7D+%5C%5C+%5Ccdots%5C%5C+I_x%27%5E%7B%28n%29%7D%2C+%5Cspace+I_y%27%5E%7B%28n%29%7D+%5Cend%7Bbmatrix%7D+%5Cbegin%7Bbmatrix%7Du+%5C+v%5Cend%7Bbmatrix%7D%3D+%5Cbegin%7Bbmatrix%7D+-%E2%88%86I_t%5E%7B%281%29%7D+%5C%5C+-%E2%88%86I_t%5E%7B%282%29%7D+%5C%5C++%5Ccdots+%5C%5C+-%E2%88%86I_t%5E%7B%28n%29%7D+%5C+%5Cend%7Bbmatrix%7D+%5Ctag%7B3-1-6%7D+%5Cend%7Bequation%7D+)

# [FlowNet](https://arxiv.org/abs/1504.06852)

![](https://pic4.zhimg.com/80/v2-909349624534aa61fe3421bf3f717ff3_hd.jpg)

> Convolutional neural networks (CNNs) have recently been very successful in a variety of computer vision tasks, especially on those linked to recognition. Optical flow estimation has not been among the tasks where CNNs were successful. In this paper we construct appropriate CNNs which are capable of solving the optical flow estimation problem as a supervised learning task. We propose and compare two architectures: a generic architecture and another one including a layer that correlates feature vectors at different image locations. 
Since existing ground truth data sets are not sufficiently large to train a CNN, we generate a synthetic Flying Chairs dataset. We show that networks trained on this unrealistic data still generalize very well to existing datasets such as Sintel and KITTI, achieving competitive accuracy at frame rates of 5 to 10 fps.

FlowNet is the first end-to-end neural network model for computing optical flow. To overcome the limits of ground truth optical flow data, it uses synthetical data. Two types of models, both based on encoder-decode, were proposed:

- combine two adjacent frames in channels
![](https://pic2.zhimg.com/80/v2-a10cce2c9829b0a251c486fc2b2d90d9_hd.jpg)
- perform feature extraction in two adjacent frames in independent branches the combined together via a correlation operation (block matching) in 21x21 neighbor
![](https://pic3.zhimg.com/80/v2-43e03ca60a3cd39e2774e7375de32cb6_hd.jpg)

The refinement module is shown below:

![](https://pic1.zhimg.com/v2-796f6a0dbc92bc7b8a8bb7f4cb3c9020_r.jpg)
![](https://pic1.zhimg.com/80/v2-796f6a0dbc92bc7b8a8bb7f4cb3c9020_hd.jpg)

# [FlowNet2.0](https://arxiv.org/abs/1612.01925)

![](https://pic1.zhimg.com/80/v2-3a332b9167da3940ad4573b5130322b4_hd.jpg)

> The FlowNet demonstrated that optical flow estimation can be cast as a learning problem. However, the state of the art with regard to the quality of the flow has still been defined by traditional methods. Particularly on small displacements and real-world data, FlowNet cannot compete with variational methods. In this paper, we advance the concept of end-to-end learning of optical flow and make it work really well. The large improvements in quality and speed are caused by three major contributions: first, we focus on the training data and show that the schedule of presenting data during training is very important. Second, we develop a stacked architecture that includes warping of the second image with intermediate optical flow. Third, we elaborate on small displacements by introducing a sub-network specializing on small motions. FlowNet 2.0 is only marginally slower than the original FlowNet but decreases the estimation error by more than 50%. It performs on par with state-of-the-art methods, while running at interactive frame rates. Moreover, we present faster variants that allow optical flow computation at up to 140fps with accuracy matching the original FlowNet.

FlowNet2.0 improves the FlowNet in the following two aspects:
- higher speed via coarse-to-fine;
- reducing the small displacement.

# [NVIDIA Optical Flow SDK](https://developer.nvidia.com/opticalflow-sdk)

![](https://devblogs.nvidia.com/wp-content/uploads/2019/02/Football-1024x288.png)

NVIDIA Optical Flow SDK exposes a new set of APIs for this hardware functionality:

- C-API – Windows and Linux
  - Windows – CUDA and DirectX
  - Linux – CUDA
- Granularity: 4x4 vectors at ¼ pixel resolution
  - Raw vectors – directly from hardware
  - Pre-/post-processed vectors via algorithms to improve accuracy
- Accuracy: low average EPE (End-Point-Error)
- Robust to intensity changes
