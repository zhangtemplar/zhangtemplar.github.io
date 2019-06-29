---
layout: post
title: FPGA-based Neuromorphic Accelerator board recognizes objects 7x more efficiently than GPUs on GoogleNet, AlexNet
---

BrainChip Holdings has just announced the [BrainChip Accelerator](http://www.brainchipinc.com/products/civil-surveillance-solutions/brainchip-accelerator), a PCIe server-accelerator card that simultaneously processes 16 channels of video in a variety of video formats using spiking neural networks rather than convolutional neural networks (CNNs). The BrainChip Accelerator card is based on a 6-core implementation BrainChip’s Spiking Neural Network (SNN) processor instantiated in an on-board Xilinx Kintex UltraScale FPGA.
 
Here’s a photo of the BrainChip Accelerator card:

![](https://xlnx.i.lithium.com/t5/image/serverpage/image-id/35582i8E5F4B2C5D15E584/image-size/original?v=1.0&px=-1)

Note, this implementation is based on spike-model, which is quite DIFFERENT from convolutional neural network, such as GoogleNet, Alexnet. It is also LESS popular. Qualcomm and IBM are used to be the active researcheres in spike-models.

The following image shows a comparison of the spike model and convolution neural networks:

![](https://xlnx.i.lithium.com/t5/image/serverpage/image-id/35584i3B263FFBC0468314/image-size/original?v=1.0&px=-1)

This is the speed comparison:

![](https://xlnx.i.lithium.com/t5/image/serverpage/image-id/35583i0CD1935F53503389/image-size/original?v=1.0&px=-1)
