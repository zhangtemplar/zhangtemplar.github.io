---
layout: post
title: Effcient Deep Neural Network
tags:  flattened-network deep-learning xception-network mixnet mobilenet squeezenet factorized-network
---

In this post, we will introduce some neural networks which are suitable for running on mobile devices.

# SqueezeNet

> SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

SqueezeNet is one of the first work to reduce the network size and computation cost. It utilizes the following techniques:
- Replace 3x3 filters with 1x1 filters.
- Decrease the number of input channels to 3x3 filters.
- Downsample late in the network so that convolution layers have large activation maps.

It proposes `fire module` which is comprised of a squeeze convolution layer (which has only 1x1 filters), feeding into an expand layer that has a mix of 1x1 and 3x3 convolution. Bypass is applied cross layers.

![](https://cdn-images-1.medium.com/max/1600/1*xji5NAhX6m3Nk7BmR_9GFw.png)

# Flattened Network

> Flattened Convolutional Neural Networks for Feedforward Acceleration

Flattened network identifies the redundancy of parameter of convolution filters and address it via low rank approximation. The convolution can be represented via matrix product as $F \times X$, if the filter can be low rank approximated as $F = A \times B$, then $F \times X = A \times (B \times X))$. Flattened network uses 1D filter. Flattened network uses rank-one filter. The saving computation cost for a filter of size $K_x \times K_y \times M \times N$ for flattened network over standard one would be

$$\frac{K_x \times N + K_y \times N + M \times N}{K_x \times K_y \times M \times N} = \frac{1}{K_y \times M} + \frac{1}{K_x \times M} + \frac{1}{K_y \times K_x}$$

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/aaedce40b1f83d2156c603b2702c63ee7864c3a6/4-Figure2-1.png)

# XCeption

> Xception: Deep Learning with Depthwise Separable Convolutions

Similar to the idea of MobileNet but working on Inception.

![](https://cdn-images-1.medium.com/max/1600/1*SRBSbojkg48DTUMcP5VVHg.jpeg)

# MobileNet

> MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

MobileNet mainly utilizes the separable convolution to reduce the parameter numbers and computational cost. It factorizes the convolution to a pair of depthwise convolution (whose input channel and output channel is one) and pointwise convolution (whose kernel size is 1x1). Depthwise convolution extract spatial information and pointwise convolution combines feature cross channels. For convolution with size $K_x \times K_y \times N$ on input $P \times Q \times M$, the improvement of parameter size and cost for separable convoltuion over standard convolution would be:
$$\frac{K_x \times K_y \times M + N \times M}{K_x \times K_y \times M \times N}=\frac{1}{N} + \frac{1}{K_x \times K_y}\\ \frac{P \times Q \times K_x \times K_y \times M + P \times Q \times N \times M}{P \times Q \times K_x \times K_y \times M \times N}=\frac{1}{N} + \frac{1}{K_x \times K_y}$$

![](https://cdn-images-1.medium.com/max/1600/1*L97mX8J7dBNPtRwb5VwqUw.png)

# MobileNetV2

> [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

![](https://camo.githubusercontent.com/b5134a2b9100ca83833437ed61aa4325dbab322f/68747470733a2f2f6873746f2e6f72672f776562742f776c2f796f2f737a2f776c796f737a716e77733538697464346f6a743163717437736e672e706e67)

The difference of MobileNet V2 to V1 is the inverted residual with linear bottleneck. This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution.

# Factorized Network

> Factorized convolutional neural networks

Factorized network is Similar as MobileNet, but also has the idea of residual network.

# [MixNet](https://arxiv.org/abs/1907.09595)

![](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oW1eynFBETaFUuCYOIMtaVZubEoAXggyyy3iaZySRJWZoTCibERrjiaH2OP8uO2Y8WsO1bjicg6zHqJaQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

MixNet finds larger kernel size (up to 9x9) tends to improve the performance on image classification and object detection compared with 3x3 kernel size used in MobileNet V3. As a result, MixNet proposes to have convolution with different kernel size in parallel and combined via concatenation, namely MdConv. MdConv is similar to group wise convolution, where each group has different kernel size.

```
def mdconv(x, filters, **args):
    G = len(filters)
    y = []
    for xi, fi in zip(tf.split(x, G, axis=-1), filters):
        y.append(tf.nn.depthwise_conv2d(xi, fi, **args))
    return tf.concat(y, axis=-1)
```

Implementations are already available at [Tensorflow](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) and [PyTorch](https://github.com/rwightman/pytorch-image-models)

