---
layout: post
title: Bluetooth Audio
tags:  sbc aac bluetooth audio aptx ldac
---

> [Bluetooth](https://en.wikipedia.org/wiki/Bluetooth) is a wireless technology standard for exchanging data between fixed and mobile devices over short distances using short-wavelength UHF radio waves in the industrial, scientific and medical radio bands, from 2.400 to 2.485 GHz, and building personal area networks (PANs). It was originally conceived as a wireless alternative to RS-232 data cables.

> Bluetooth is managed by the Bluetooth Special Interest Group (SIG), which has more than 30,000 member companies in the areas of telecommunication, computing, networking, and consumer electronics

The speed of bluebooth in different generations can be found in the following table:
![](http://img.expreview.com/news/2019/05/23/BT_04.png)

# [Sub-band coding(SBC)](https://en.wikipedia.org/wiki/Sub-band_coding)

The mandatory standard for bluetooth audio. The audio quality is slightly worse than MP3.

# [Advanced Audio Coding(AAC)](https://en.wikipedia.org/wiki/Advanced_Audio_Coding)

> Advanced Audio Coding (AAC) is an audio coding standard for lossy digital audio compression. Designed to be the successor of the MP3 format, AAC generally achieves better sound quality than MP3 at the same bit rate.

However, AAC on Bluetooth cannot deliver the same quality as AAC.

# [aptX](https://en.wikipedia.org/wiki/AptX)

aptX (formerly apt-X) is a family of proprietary audio codec compression algorithms owned by Qualcomm. It has several variants:
- aptX: The aptX audio codec is used for consumer and automotive wireless audio applications, notably the real-time streaming of lossy stereo audio over the Bluetooth A2DP connection
- Enhanced aptX: Enhanced aptX provides coding at 4:1 compression ratios for professional audio broadcast applications. Enhanced aptX can handle up to 4 stereo pairs of AES3 audio and compress to 1 AES3 stream for transmit. Enhanced aptX supports bit-depths of 16, 20 or 24 bit. For audio sampled at 48 kHz, the bit-rate for E-aptX is 384 kbit/s/channel. Its lowest bit-rate is 60(?) kbit/s for mono audio sampled at 16 kHz, offering about 7.5 kHz frequency response just below that of wideband telephony codecs (which usually operate at 16 kHz sampling rate).
- aptX Live: aptX Live is a low-complexity audio codec that is specifically designed to maximise digital wireless microphone channel density in bandwidth-constrained scenarios, such as live performance. aptX Live offers up to 8:1 compression of 24-bit resolution digital audio streams and ensuring latency of around 1.8 ms at 48 kHz sampling rates.
- aptX-HD: (also known as aptX Lossless) has bit-rate of 576 kbit/s. It supports high-definition audio up to 48 kHz sampling rates and sample resolutions up to 24 bits. Unlike what the name suggests, the codec is still considered lossy.
- aptX Low Latency: aptX Low Latency is intended for video and gaming applications requiring comfortable audio-video synchronization. The technology offers an end-to-end latency of 32 ms over Bluetooth.

# [LDAC](https://en.wikipedia.org/wiki/LDAC_(codec))

LDAC is an audio coding technology developed by Sony, which allows streaming audio over Bluetooth connections up to 990 kbit/s at 24 bit/96 kHz (also called high-resolution audio). It is used by various Sony products, including headphones, smartphones, portable media players, active speakers and home theaters. LDAC is a lossy codec. 

Starting from Android 8.0 "Oreo", LDAC is part of the Android Open Source Project, enabling every OEM to integrate this standard into their own Android devices freely.
