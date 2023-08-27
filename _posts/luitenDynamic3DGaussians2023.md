---
layout: post
title: Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis
tags:  3d nerf read computer-science---computer-vision-and-pattern-recognition
---

> We present a method that simultaneously addresses the tasks of dynamic scene novel-view synthesis and six degree-of-freedom (6-DOF) tracking of all dense scene elements. We follow an analysis-by-synthesis framework, inspired by recent work that models scenes as a collection of 3D Gaussians which are optimized to reconstruct input images via differentiable rendering. To model dynamic scenes, we allow Gaussians to move and rotate over time while enforcing that they have persistent color, opacity, and size. By regularizing Gaussians' motion and rotation with local-rigidity constraints, we show that our Dynamic 3D Gaussians correctly model the same area of physical space over time, including the rotation of that space. Dense 6-DOF tracking and dynamic reconstruction emerges naturally from persistent dynamic view synthesis, without requiring any correspondence or flow as input. We demonstrate a large number of downstream applications enabled by our representation, including first-person view synthesis, dynamic compositional scene synthesis, and 4D video editing.

# Notes
%% begin annotations %%

![](https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/luitenDynamic3DGaussians2023-1-x44-y333.png) 

Each scene is parameterized by 200-300k Dynamic 3D Gaussians which move over time. [(p. 1)](zotero://open-pdf/library/items/SN5GH4GY?page=1&annotation=6YF86EJD)

We follow an analysis-by-synthesis framework, inspired by recent work that models scenes as a collection of 3D Gaus- sians which are optimized to reconstruct input images via differentiable rendering. To model dynamic scenes, we al- low Gaussians to move and rotate over time while enforcing that they have persistent color, opacity, and size. By regu- larizing Gaussians’ motion and rotation with local-rigidity constraints, we show that our Dynamic 3D Gaussians cor- rectly model the same area of physical space over time, in- cluding the rotation of that space. [(p. 1)](zotero://open-pdf/library/items/SN5GH4GY?page=1&annotation=SGCWPY7R)
%%This paper proposes a track algorem by modelly 3D dynamic scene as James of 3D Gaussian.in the cases each frame is represented by set of3d
Gaussian, whose locations could by estimated via differential rendering. A rigid constraint is applied%%

It rep- resents complex scenes as a combination of a large number of coloured 3D Gaussians which are rendered into camera views via splatting-based rasterization. The positions, sizes, rotations, colours and opacities of these Gaussians can then be adjusted via differentiable rendering and gradient-based optimization such that they represent the 3D scene given by a set of input images [(p. 2)](zotero://open-pdf/library/items/SN5GH4GY?page=2&annotation=WAZRT9RB)

Our key insight is that we restrict all attributes of the Gaus- sians (such as their number, color, opacity, and size) to be the same over time, but let their position and orienta- tion vary. This allows our Gaussians to be thought of as a particle-based physical model of the world, where oriented particles undergo rigid-body transformations over time. [(p. 2)](zotero://open-pdf/library/items/SN5GH4GY?page=2&annotation=2XQW6SVR)

Crucially, particles allow us to operationalize physical pri- ors over their movement that act as regularizers for the opti- mization: a local rigidity prior, a local rotational-similarity prior, and a long-term local isometry prior. These priors ensure that local neighborhoods of particles move approxi- mately rigidly between timesteps, and that nearby particles remain closeby over all timesteps. [(p. 2)](zotero://open-pdf/library/items/SN5GH4GY?page=2&annotation=2G42SNXL)

Previous approaches to neural reconstruction of dynamic scenes can be seen as either Eulerian representations that keep track of scene motion at fixed grid locations [5, 10, 36] or Lagrangian representations where an observer follows a particular particle through space and time. We fall in the latter category, but in contrast to prior point-based repre- sentations [1, 45], we make use of oriented particles that allow for richer physical priors (as above) and that directly reconstruct the 6-DOF motion of all 3D points, enabling a variety of downstream applications (see Fig. 3 and 7). [(p. 2)](zotero://open-pdf/library/items/SN5GH4GY?page=2&annotation=RFRACCVA)

An remarkable feature of our approach is that tracking arises exclusively from the process of rendering per-frame images [(p. 2)](zotero://open-pdf/library/items/SN5GH4GY?page=2&annotation=J5A747YK)

(c) Methods that represent the 3D scene in a canonical timestep and use a deformation field to warp this to the rest of the timesteps [(p. 2)](zotero://open-pdf/library/items/SN5GH4GY?page=2&annotation=299TR26M)

(d) Template guided methods [13, 20, 42], which model dynamic scenes in restricted environments where the mo- tion can be modelled by a predefined template e.g. a set of human-pose skeleton transformations [(p. 3)](zotero://open-pdf/library/items/SN5GH4GY?page=3&annotation=I46GZ796)

(e) Point-based methods [1, 45], which compared to all of the above categories, hold the most promise for representing dynamic scenes in a way where accurate correspondence over time can emerge due to their natural Lagrangian repre- sentation. [(p. 3)](zotero://open-pdf/library/items/SN5GH4GY?page=3&annotation=ENLGBTVB)

Other than rendering accuracy and speed, modeling the dy- namic world with Gaussians has a distinct advantage over points as Gaussian’s have a notion of ‘rotation’ so we can use them to model the full 6 degree-of-freedom (DOF) mo- tion of a scene at every point and can use this to construct physically-plausible local rigidity losses. [(p. 3)](zotero://open-pdf/library/items/SN5GH4GY?page=3&annotation=PYC9NCJP)

The most similar method to ours is OmniMotion [40] which also fits a dynamic radiance field representation using test-time optimization for the purpose of long-term track- ing. They focus on monocular video while we focus on multi-camera capture and as such we can reconstruct tracks in metric-3D while they produce a ‘pseudo-3D’ representa- tion [(p. 3)](zotero://open-pdf/library/items/SN5GH4GY?page=3&annotation=GD85V86I)

The reconstruction is performed temporally online, i.e., one timestep of the scene is reconstructed at a time with each one being initialized using the previous timestep’s repre- sentation. The first timestep acts as an initialization for our scene where we optimize all properties, and then fix all for the subsequent timesteps except those defining the motion of the scene. Each timestep is trained via gradient based op- timization using a differentiable renderer (R) to render the scene at each timestep into each of the training cameras. 
Iˆ t,c = R(St, Kc, Et,c) [(p. 3)](zotero://open-pdf/library/items/SN5GH4GY?page=3&annotation=QTHSCMFM)

3D Gaussians. Our dynamic scene representa- tion (S) is parameterized by a set of Dynamic 3D Gaus- sians, each of which has the following parameters: 1) a 3D center for each timestep (xt, yt, zt). 
2) a 3D rotation for each timestep parameterized by a quaternion (qwt, qxt, qyt, qzt). 
3) a 3D size in standard deviations (consistent over all timesteps) (sx, sy, sz) 4) a color (consistent over all timesteps) (r, g, b) 5) an opacity logit (consistent over all timesteps) (o) 6) a background logit (consistent over timesteps) (bg) [(p. 4)](zotero://open-pdf/library/items/SN5GH4GY?page=4&annotation=B5GEDSQL)

In our experiments, scenes are represented by between 200- 300k Gaussians, of which only 30-100k usually are not part of the static background. While the code contains the ability to represent view-dependent color using spherical harmon- ics, we turn this off in our experiments for simplicity. [(p. 4)](zotero://open-pdf/library/items/SN5GH4GY?page=4&annotation=XU6CTGV4)

The softness of this Gaussian representation also means that Gaussians typically need to significantly overlap in order to represent a physically solid object. [(p. 4)](zotero://open-pdf/library/items/SN5GH4GY?page=4&annotation=ZR2R9CRY)

The center of the Gaussian is splatted using the standard point rendering formula: µ2D = K ((Eµ)/(Eµ)z) where the 3D Gaussian center µ is projected into a 2D im- age by multiplication with the world-to-camera extrinsic matrix E, z-normalization, and multiplication by the intrin- sic projection matrix K [(p. 4)](zotero://open-pdf/library/items/SN5GH4GY?page=4&annotation=3GTD776F)

The influence of all Gaussians on this pixel can be combined by sorting the Gaussians in depth order and performing front-to-back volume rendering using the Max [24] volume rendering formula (the same as is used in NeRF [25]): Cpix = X i∈S cif 2D i,pix i Y−1 j=1 (1 − f 2D j,pix) [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=B66GGRF8)

We find that just fixing the color, opacity and size of Gaussians is not enough on its own to generate long-term persistent tracks, especially across ar- eas of the scene where there is a large area of near uni- form colour. In such situation the Gaussians move freely around the area of similar colour as there is no restriction on them doing so [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=VPAPEXFG)

We introduce three regulariza- tion losses, short-term local-rigidity Lrigid and local-rotation similarity Lrot losses and a long-term local-isometry loss. 
The most important of these is the local-rigidity loss Lrigid , defined as: Lrigid i,j = wi,j (µj,t−1 − µi,t−1) − Ri,t−1R−1 i,t (µj,t − µi,t) 2 [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=TMF4TS34)

Lrigid = 1 k|S| X i∈S X j∈knni;k Lrigid i,j This states that, for each Gaussian i, nearby Gaussians j should move in a way that follows the rigid-body transform of the coordinate system of i between timesteps. See [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=GW6R3XE5)

We restrict the set of Gaussians j to be the k-nearest- neighbours of i (k=20), and weight the loss by the a weight- ing factor for the Gaussian pair: wi,j = exp  −λw ∥µj,0 − µi,0∥2 2 [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=Z6MQN7GZ)

however we found better convergence if we explicitly force neighbouring Gaussians to have the same rotation over time: Lrot = 1 k|S| X i∈S X j∈knni;k wi,j qˆj,tqˆ−1 j,t−1 − qˆi,tqˆ−1 i,t−1 2 [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=6RQMEMJU)

We apply Lrigid and Lrot only between the current timestep and the directly preceding timestep, thus only enforcing these losses over short-time horizons. Which sometimes causes elements of the scene to drift apart, thus we apply a third loss, the isometry loss, over the long-term: Liso= 1 k|S| X i∈S X j∈knni;k wi,j   ∥µj,0 − µi,0∥2− ∥µj,t − µi,t∥2 [(p. 5)](zotero://open-pdf/library/items/SN5GH4GY?page=5&annotation=TVEECJH8)

forcing the positions between two Gaussians to be the same it only enforces the distances between them to be the same. [(p. 6)](zotero://open-pdf/library/items/SN5GH4GY?page=6&annotation=SZJKCBHG)

[17], in the first timestep we initialize the scene using a coarse point cloud that could be obtained from run- ning colmap, but instead we use available sparse samples from depth cameras. Note that these depth values are only used for initializing a sparse point cloud in the first timestep and are not used at all during optimization. 
We use the densification from [17] in the first timestep in or- der to increase the density of Gaussians and achieve a high quality reconstruction [(p. 6)](zotero://open-pdf/library/items/SN5GH4GY?page=6&annotation=64PIIBT7)

We noticed that often the shirt was being mis-tracked as it was confused with the back- ground, while more contrastive elements like pants and hair were being tracked correctly. [(p. 6)](zotero://open-pdf/library/items/SN5GH4GY?page=6&annotation=VYHDK6DI)

To determine the correspondence of any point in 3D space p across timesteps, we can linearize the motion- space by simply taking the point’s location in the coordinate system of the Gaussian that has the most influence f(p) over this point (or the static background coordinate system if f(p) < 0.5 for all Gaussians) [(p. 6)](zotero://open-pdf/library/items/SN5GH4GY?page=6&annotation=SX462D5U) %% end annotations %%

%% Import Date: 2023-08-26T23:08:53.819-07:00 %%
