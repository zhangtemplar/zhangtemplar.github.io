---
layout: post
title: An Overview of Normalization Methods in Deep Learning
tags: deep-learning normlization batch-normalization group-normalization
---

![](https://i1.wp.com/mlexplained.com/wp-content/uploads/2018/11/Screen-Shot-2018-11-28-at-4.56.06-PM.png?w=522)

Batch normalization is one of the reasons why deep learning has made such outstanding progress in recent years. Batch normalization enables the use of higher learning rates, greatly accelerating the learning process. It also enabled the training of deep neural networks with sigmoid activations that were previously deemed too difficult to train due to the vanishing gradient problem.

This is based on study of [An Overview of Normalization Methods in Deep Learning
](http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/) and [An Intuitive Explanation of Why Batch Normalization Really Works ](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)

# Batch Normalization

Batch normalization is a normalization method that normalizes activations in a network across the mini-batch. For each feature, batch normalization computes the mean and variance of that feature in the mini-batch. It then subtracts the mean and divides the feature by its mini-batch standard deviation.

$$f(x, y, c, i) = \frac{f(x, y, c, i) - \mu_{c}(f(x, y, c, i))}{\sigma_{c}(f(x, y, c, i)) + \epsilon}$$

So the normalization is done feature by feature.

Batch normalization can also adds two additional learnable parameters: the mean and magnitude of the activations.

Note for inference, we need to compute the mean and batch based on the training data.

If many complex network, the batch size is usually small, e.g., $1$, or $2$. Small batch makes the normalization very noisy. So alternative solutions are needed.

# Weight Normalization

In weight normalization, instead of normalizing the activations directly, we normalize the weights of the layer. This has a similar effect to dividing the inputs by the standard deviation in batch normalization.

# Layer Normalization

[Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) is a method developed by Geoffery Hinton. Batch normalization normalizes the input features across the batch dimension. The key feature of layer normalization is that it normalizes the inputs across the features.

$$f(x, y, c, i) = \frac{f(x, y, c, i) - \mu_{i}(f(x, y, c, i))}{\sigma_{i}(f(x, y, c, i)) + \epsilon}$$

So the normalization is done image by image.

![](https://i1.wp.com/mlexplained.com/wp-content/uploads/2018/01/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2018-01-11-11.48.12.png?resize=1024%2C598)

Experimental results show that layer normalization performs well on RNNs.

# Instance Normalization 
[Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf) is similar to layer normalization but goes one step further: it computes the mean/standard deviation and normalize across each channel in each training example.

Originally devised for style transfer, the problem instance normalization tries to address is that the network should be agnostic to the contrast of the original image. Therefore, it is specific to images and not trivially extendable to RNNs.

$$f(x, y, c, i) = \frac{f(x, y, c, i) - \mu_{c, i}(f(x, y, c, i))}{\sigma_{c, i}(f(x, y, c, i)) + \epsilon}$$

# Group Normalization

[Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) computes the mean and standard deviation over groups of channels for each training example. In a way, group normalization is a combination of layer normalization and instance normalization.

Though layer normalization and instance normalization were both effective on RNNs and style transfer respectively, they were still inferior to batch normalization for image recognition tasks. Group normalization was able to achieve much closer performance to batch normalization with a batch size of 32 on ImageNet and outperformed it on smaller batch sizes.

One of the implicit assumptions that layer normalization makes is that all channels are “equally important” when computing the mean. This assumption is not always true in convolution layers. 

# Batch Renormalization

[Batch Renormalization](https://arxiv.org/pdf/1702.03275.pdf) is another interesting approach for applying batch normalization to small batch sizes. The basic idea behind batch renormalization comes from the fact that we do not use the individual mini-batch statistics for batch normalization during inference. Instead, we use a moving average of the mini batch statistics. This is because a moving average provides a better estimate of the true mean and variance compared to individual mini-batches.

# Batch-Instance Normalization

[Batch-Instance Normalization](https://arxiv.org/pdf/1805.07925.pdf) is an extension of instance normalization that attempts to account for differences in contrast and style in images. The problem with instance normalization is that it completely erases style information. It is simply an interpolation between batch normalization and instance normalization.

$$y=(\rho\hat{X}_B + (1-\rho)\hat{X}_I)\gamma + \beta$$

where $\hat{X}_B$ is batch normalization result and $\hat{X}_I$ is the instance normalization result. $\rho$ is a parameter to be learned.

Batch-instance normalization outperformed batch normalization on CIFAR-10/100, ImageNet, domain adaptation, and style transfer. In image classification tasks, the value of \rho  tended to be close to 0 or 1, meaning many layers used either instance or batch normalization almost exclusively. In addition, layers tended to use batch normalization more than instance normalization.

# Spectral Normalization

[Spectral Normalization](https://arxiv.org/pdf/1802.05957.pdf) the spectral normalization normalized the weight matrix by their spectral norm or $\ell_2$ norm: $\hat{W} = \frac{W}{\lvert W\rvert_2}$. Experimental results show that spectral normalization improves the training of GANs with minimal additional tuning.

$$\lvert W\rvert_2 = \max_h{\frac{\lvert Wh\rvert_2}{\lvert h\rvert_2}} = \max_{h\sim \lvert h \rvert_2 = 1}{\frac{\lvert Wh\rvert_2}{\lvert h\rvert_2}} = \sigma_1(W)$$

Note $W\in\mathbb{R}^{M\times (NWH)}$ is 2D representation of the weight tensor $W\in\mathbb{R}^{M\times N\times W\times H)}$ where $M$ is the number of output channels and $N$ is the number of input channels. 

# Additional Pitfalls of Batch Normalization

## Dependence of the loss between samples in a minibatch

When we introduce batch normalization, the loss value for each sample in a minibatch becomes dependent on other samples in the minibatch.

This isn’t much of a problem when training a model on a single machine, but when we start to conduct distributed training, things can get ugly. As mentioned in (https://arxiv.org/pdf/1706.02677.pdf), we need to take extra care in choosing the batch size and learning rate in the presence of batch normalization when doing distributed training. If two different machines use different batch sizes, they will indirectly be optimizing different loss functions: this means that the value of $\gamma$  that worked for one machine is unlikely to work for another machine. This is why the authors stressed that the batch size for each worker must be kept constant across all machines.

## Fine tuning

For fine tuning, it might be better to use the statistics of the original dataset instead, even the batch size is different.
