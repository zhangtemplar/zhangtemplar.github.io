---
layout: tagpage
title: "Reading Note on Neural Radiance Field"
tag: nerf
---

Neural Radiance Field (NeRF), you may have heard words many times for the past few months. Yes, this is the latest progress of neutral work and computer graphics. NeRF represents a scene with learned, continuous volumetric radiance field $F_{\theta}$ defined over a bounded 3D volume. In Nerf, $F_{\theta}$ is a multilayer perceptron (MLP) that takes as input a 3D position $x=(x,y,z)$ and unit-norm viewing direction $d=(d_x,d_y,d_z)$, and produces as output a density $\sigma$ and color $c=(r,g,b)$. By enumerating all most position and direction for a bounded 3D volumne, we could obtain the 3D scene.
