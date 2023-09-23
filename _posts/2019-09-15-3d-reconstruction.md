---
layout: post
title: 3D Reconstruction
tags:  3d-reconstruction deep-learning depth mesh-rcnn 3d-r2n2 pixel2mesh
---

# [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tZCFBZYib9jR6Z3stKjtWdSnhYz6AibCqwgnLeyG2ZGswFYEUb91OJK6A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> Predicting depth is an essential component in understanding the 3D geometry of a scene. While for stereo images local correspondence suffices for estimation, finding depth relations from a single image is less straightforward, requiring integration of both global and local information from various cues. Moreover, the task is inherently ambiguous, with a large source of uncertainty coming from the overall scale. In this paper, we present a new method that addresses this task by employing two deep network stacks: one that makes a coarse global prediction based on the entire image, and another that refines this prediction locally. We also apply a scale-invariant error to help measure depth relations rather than scale. By leveraging the raw datasets as large sources of training data, our method achieves state-of-the-art results on both NYU Depth and KITTI, and matches detailed depth boundaries without the need for superpixelation.

It is one of the earliest paper for depth estimation, which proposed scale-invariant error to address global scale uncertainty:

![$$\ell(y,\hat{y})=\sum_{i,j}{log(\frac{y_i}{y_j})-\log{(\frac{\hat{y}_i}{-\hat{y}_j})}}$$](https://latex.codecogs.com/gif.latex?%5Cell%28y%2C%5Chat%7By%7D%29%3D%5Csum_%7Bi%2Cj%7D%7Blog%28%5Cfrac%7By_i%7D%7By_j%7D%29-%5Clog%7B%28%5Cfrac%7B%5Chat%7By%7D_i%7D%7B-%5Chat%7By%7D_j%7D%29%7D%7D)

# [3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/abs/1604.00449)

> Inspired by the recent success of methods that employ shape priors to achieve robust 3D reconstructions, we propose a novel recurrent neural network architecture that we call the 3D Recurrent Reconstruction Neural Network (3D-R2N2). The network learns a mapping from images of objects to their underlying 3D shapes from a large collection of synthetic data. Our network takes in one or more images of an object instance from arbitrary viewpoints and outputs a reconstruction of the object in the form of a 3D occupancy grid. Unlike most of the previous works, our network does not require any image annotations or object class labels for training or testing. Our extensive experimental analysis shows that our reconstruction framework i) outperforms the state-of-the-art methods for single view reconstruction, and ii) enables the 3D reconstruction of objects in situations when traditional SFM/SLAM methods fail (because of lack of texture and/or wide baseline).

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tNdNe4yjwsme9qDu1phx0giaoL9zr4z4fDialOImNjBBa91TGNXrfGJMQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tgvFZ4ppBvaVOeO1Tk4xhPKm1nCU6Be3wfspczyyEdGyIj5D11ZuLog/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

This paper proposed to use LSTM to generate 3D scene from multiple view of the scene. A CNN is used to compute feature from 2D image, LSTM is then used to encode the multiple view information from computed feature and its output is sent to the other CNN to decode the depth information.

# [A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/1612.00603)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tuoErHhicXDDVhoe0gNSSdqoDuERAF9evBpyBicIo9w96ibof92Uu95jkw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> Generation of 3D data by deep neural network has been attracting increasing attention in the research community. The majority of extant works resort to regular representations such as volumetric grids or collection of images; however, these representations obscure the natural invariance of 3D shapes under geometric transformations and also suffer from a number of other issues. In this paper we address the problem of 3D reconstruction from a single image, generating a straight-forward form of output -- point cloud coordinates. Along with this problem arises a unique and interesting issue, that the groundtruth shape for an input image may be ambiguous. Driven by this unorthodox output form and the inherent ambiguity in groundtruth, we design architecture, loss function and learning paradigm that are novel and effective. Our final solution is a conditional shape sampler, capable of predicting multiple plausible 3D point clouds from an input image. In experiments not only can our system outperform state-of-the-art methods on single image based 3d reconstruction benchmarks; but it also shows a strong performance for 3d shape completion and promising ability in making multiple plausible predictions.

This is the first paper generating point cloud from a 2D image. To handle the ambiguity of 3D model reconstructed from 2D image, Min-of-N loss was proprosed, which by introducing disturb to the network with multiple trials, there should be one good reconstruction.

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tNK7IpSBiayyqRcLKkniaPvA9C3SrpTOu4jSxzKYzicn8aHBVO9C8Aux2w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

# [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](https://arxiv.org/abs/1804.01654)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tAbU1bT0GaPiaCQUL60ibPdWdwadUHP4dnrPD5Ne3jHibuRmIxASZGOB4g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tjGySQzZcT1lqJM29IjkB07AVos8Qo0XHuDAbLROlKyamLeLhSWy55A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> We propose an end-to-end deep learning architecture that produces a 3D shape in triangular mesh from a single color image. Limited by the nature of deep neural network, previous methods usually represent a 3D shape in volume or point cloud, and it is non-trivial to convert them to the more ready-to-use mesh model. Unlike the existing methods, our network represents 3D mesh in a graph-based convolutional neural network and produces correct geometry by progressively deforming an ellipsoid, leveraging perceptual features extracted from the input image. We adopt a coarse-to-fine strategy to make the whole deformation procedure stable, and define various of mesh related losses to capture properties of different levels to guarantee visually appealing and physically accurate 3D geometry. Extensive experiments show that our method not only qualitatively produces mesh model with better details, but also achieves higher 3D shape estimation accuracy compared to the state-of-the-art.

Obviously, neither depth nor point cloud nor voxel is the optimal way of representing 3D models. In facts, mesh is like the standard way to do that. This proposed to use Graph CNN to generate 3D mesh from a 2D image:
- CNN extract feature from image
- Graph-CNN extract feature around each vertex according to its location in image
- combined with the vertex feature itself, graph-based ResNet predicts new vertex location and vertex feature.

The 3D model is intialized as a ellipsoid and refined to the final model, during which more vertices are added to the mesh.

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tAwS8ZC3GeRWyuJUWztKblMTciaibGyAufQ1XTTBTggv71Fzz28MPodaQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tX7gPC2FN7SicnlXibc7mr4icWfhPbLbnhquvusSSaupo2Qvwb31xibaSkQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

# [Mesh R-CNN](https://arxiv.org/abs/1906.02739)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tniaWIRJxVewq9Ho8ziaoiarInnEWkYOg5XO6eRLIZ05Xh5NNXXTBoiczWw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> Rapid advances in 2D perception have led to systems that accurately detect objects in real-world images. However, these systems make predictions in 2D, ignoring the 3D structure of the world. Concurrently, advances in 3D shape prediction have mostly focused on synthetic benchmarks and isolated objects. We unify advances in these two areas. We propose a system that detects objects in real-world images and produces a triangle mesh giving the full 3D shape of each detected object. Our system, called Mesh R-CNN, augments Mask R-CNN with a mesh prediction branch that outputs meshes with varying topological structure by first predicting coarse voxel representations which are converted to meshes and refined with a graph convolution network operating over the mesh's vertices and edges. We validate our mesh prediction branch on ShapeNet, where we outperform prior work on single-image shape prediction. We then deploy our full Mesh R-CNN system on Pix3D, where we jointly detect objects and predict their 3D shapes.

# [Conditional Single-view Shape Generation for Multi-view Stereo Reconstruction](https://arxiv.org/abs/1904.06699)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX8HdKQiaQRBCVvDergu7X2tgiaHrGV4Xs0uRpfu5m74AOvsqdV5Z9R5POKNNSHQm8dOqHniaXC5iadvA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> In this paper, we present a new perspective towards image-based shape generation. Most existing deep learning based shape reconstruction methods employ a single-view deterministic model which is sometimes insufficient to determine a single groundtruth shape because the back part is occluded. In this work, we first introduce a conditional generative network to model the uncertainty for single-view reconstruction. Then, we formulate the task of multi-view reconstruction as taking the intersection of the predicted shape spaces on each single image. We design new differentiable guidance including the front constraint, the diversity constraint, and the consistency loss to enable effective single-view conditional generation and multi-view synthesis. Experimental results and ablation studies show that our proposed approach outperforms state-of-the-art methods on 3D reconstruction test error and demonstrate its generalization ability on real world data.

This paper considers the inherent uncertainty of 3D reconstruction from 2d image--occlusion. The basic idea is that, by introducing a random disturbe to the network, multiple 3D models will be generated from a single 3D image; if there are images of multiple view available, take majority voting will leads to the final 3D model. A frontal constraint is used to enforce the frontal view of the generated 3D models is compatible with the input images.
