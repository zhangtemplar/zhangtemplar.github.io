---
layout: post
title: Interfacing with an FPGA from Linux on ZYNQ
---

[Mitchell Orsucci](https://www.hackster.io/morsucci) has proposed method to control an FPGA from the Linux OS, where you are dynamically reconfigure the FPGA from Linux user-space. You can read more [here](https://www.hackster.io/morsucci/interfacing-with-an-fpga-from-linux-on-zynq-90ea3e)

![](https://hackster.imgix.net/uploads/attachments/346596/photo_sep_05_8_12_16_am_6v8Kkds7lH.jpg?auto=compress%2Cformat&w=900&h=675&fit=min)

This project represents the control of an FPGA from Linux user-space. This project utilizes a Digilent PmodOLED_RGB and a Digilent PmodCDC1, as well as the available inputs and outputs on the ArtyZ7-20 board.

The ArtyZ7-20 contains a Xilinx Zynq chip which contains a 650Mhz ARM dual-core processor as well as some FPGA fabric. An FPGA design can be instantiated using Xilinx Vivado. Additionally, using Xilinx Petalinux, a Linux kernel and root file-system can be obtained for the ARM processor. This allows us to run the Linux operating system, but still use the functionality of the FPGA. FPGA configurations can be loaded and changed dynamically without interrupting or crashing the running kernel.

![](https://hackster.imgix.net/uploads/attachments/346620/artydemobd_goKFeGnlyx.png?auto=compress%2Cformat&w=680&h=510&fit=max)

The code is available on [Github](https://github.com/mitchellorsucci/ArtyZ720)
