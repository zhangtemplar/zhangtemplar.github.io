---
layout: post
title: Open-source GUINNESS makes FPGA-accelerated, binarized neural networks easy to pour right from the SDSoC tap
---

A new open-source tool named GUINNESS makes it easy for you to develop binarized (2-valued) neural networks (BNNs) for [Zynq SoCs](https://www.xilinx.com/products/silicon-devices/soc/zynq-7000.html) and [Zynq UltraScale+ MPSoCs](https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html) using the [SDSoC Development Environment](https://www.xilinx.com/products/design-tools/software-zone/sdsoc.html). GUINNESS is a GUI-based tool that uses the Chainer deep-learning framework to train a binarized CNN. In a paper titled [On-Chip Memory Based Binarized Convolutional Deep Neural Network Applying Batch Normalization Free Technique on an FPGA](http://ieeexplore.ieee.org/document/7965031/) presented at the recent 2017 IEEE International Parallel and Distributed Processing Symposium Workshops, authors Haruyoshi Yonekawa and Hiroki Nakahara describe a system they developed to implement a binarized CNN for the VGG-16 benchmark on the [Xilinx ZCU102 Eval Kit](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html), which is based on a Zynq UltraScale+ ZU9EG MPSoC. Nakahara presented the GUINNESS tool again this week at [FPL2017](https://www.fpl2017.org/) in Ghent, Belgium.</p>

According to the IEEE paper, the Zynq-based BNN is 136.8x faster and 44.7x more power efficient than the same CNN running on an ARM Cortex-A57 processor. Compared to the same CNN running on an Nvidia Maxwell GPU, the Zynq-based BNN is 4.9x faster and 3.8x more power efficient.

GUINNESS is now available on [GitHub](https://github.com/HirokiNakahara/GUINNESS)

![](https://xlnx.i.lithium.com/t5/image/serverpage/image-id/35461i72FBA10CC2C0F60C/image-size/original?v=1.0&amp;px=-1)
