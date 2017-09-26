---
layout: post
title: Deep Learning Framework for Mobile
---

In this post, I will introduce several deep learning package which can be delpoyed to mobile platforms.

# Tensorflow

[TensorFlow](https://www.tensorflow.org) is the most popular deep learning frameworks, which supports both `convolution neural network` and `recurrent neural network`. This means you could run analyze image, text and audio with tensorflow. It is developed and maintained with Google, thus it is quality is assured.

`TensorFlow` has provided [official](https://www.tensorflow.org/mobile/) mobile support for Android, iOS and Raspberry Pi, where you can access the examples in [Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/), [iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios/) and [Raspberry Pi](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/pi_examples/). Google has been continuing improve its performance on Mobile platform, e.g., reduce the code footprint, and supporting quantization and lower precision arithmetic that reduce model size.

![](https://lh3.googleusercontent.com/FVCMMqnerBqM9p86oLays5jA16uR7EDTZU-1EipudR9k2lmo77OLQ4ww0mLxaBGuCHRBztN8FO92oZuHDUDDZ_lF0xDbTgkU=s688)

# MXNet

[MXNet](https://mxnet.incubator.apache.org/) is yet anthor popular deep learning framework, which has gained recently offical support from Amazon AWS. It also provides support for Mobile platform, including iOS and Android, where you can find the example for [Android](https://github.com/Leliana/WhatsThis) and [iOS](https://github.com/pppoe/WhatsThis-iOS).

```
Thanks to [Jack Deng](https://github.com/jdeng), MXNet provides an [amalgamation](https://github.com/dmlc/mxnet/tree/master/amalgamation) script that compiles all code needed for prediction based on trained DL models into a single `.cc`` file, containing approximately 30K lines of code. This code only depends on the BLAS library. Moreover, we’ve also created an even more minimal version, with the BLAS dependency removed. You can compile the single file into JavaScript by using [emscripten](https://github.com/kripken/emscripten)
```

However, it doesn't seems to apply speical optimizations to mobile platform.

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/apk/subinception.png)

# Mobile-deep-learning（MDL）

Baidu has open sourced their mobile deep learning framework on [Github](https://github.com/baidu/mobile-deep-learning). This research aims at simply deploying CNN on mobile devices, with low complexity and high speed. It supports calculation on iOS GPU, and is already adopted by Baidu APP.

* Size: 340k+ (on arm v7)
* Speed: 40ms (for iOS Metal GPU Mobilenet) or 30 ms (for Squeezenet)

![Android Demo](android_showcase.gif)

![iOS QR code](https://gss0.baidu.com/9rkZbzqaKgQUohGko9WTAnF6hhy/mms-res/graph/mobile-deep-learning/iOS/qrcode_ios.png)

[Android-Googlenet](http://gss0.baidu.com/9rkZbzqaKgQUohGko9WTAnF6hhy/mms-res/graph/mobile-deep-learning/Android/mdl_demo.4493eea5.apk):

![iOS QR code](http://gss0.baidu.com/9rkZbzqaKgQUohGko9WTAnF6hhy/mms-res/graph/mobile-deep-learning/Android/qrcode_android.33e91161.png)

# DeepLearningKit

DeepLearningKit is an Open Source – with Apache 2.0 Licence – Deep Learning Framework for Apple’s iOS, OS X and tvOS available at [github.com/DeepLearningKit/DeepLearningKit](github.com/DeepLearningKit/DeepLearningKit).

![](http://deeplearningkit.org/wp-content/uploads/2015/12/deeplearningkitoverview.png)

# iOS Core ML

[Core ML](https://developer.apple.com/machine-learning/) is the Apple's offical machine learning library which lets you integrate a broad variety of machine learning model types into your app. In addition to supporting extensive deep learning with over 30 layer types, it also supports standard models such as tree ensembles, SVMs, and generalized linear models. Because it’s built on top of low level technologies like Metal and Accelerate, Core ML seamlessly takes advantage of the CPU and GPU to provide maximum performance and efficiency. You can run machine learning models on the device so data doesn't need to leave the device to be analyzed.

![](https://developer.apple.com/assets/elements/icons/core-ml/core-ml-128x128_2x.png)

# Snapdragon Neural Processing Engine

Qualcomm has also provided a deep learning library optimized for its Snapdragon platforms, namely, [Snapdragon Neural Processing Engine](https://developer.qualcomm.com/software/snapdragon-neural-processing-engine). The Qualcomm® Snapdragon™ Neural Processing Engine (NPE) SDK is designed to help developers run one or more neural network models trained in Caffe/Caffe2 or TensorFlow on Snapdragon mobile platforms, whether that is the CPU, GPU or DSP.

It only supports convolutional neural network.

  - Android and Linux runtimes for neural network model execution
  - Acceleration support for Qualcomm® Hexagon™ DSPs, Qualcomm® Adreno™ GPUs and Qualcomm® Kryo™, CPUs1
  - Support for models in Caffe, Caffe2 and TensorFlow formats2
  - APIs for controlling loading, execution and scheduling on the runtimes
  - Desktop tools for model conversion
  - Performance benchmark for bottleneck identification

![](https://developer.qualcomm.com/sites/default/files/attachments/npe-landing-page2_0.png)
