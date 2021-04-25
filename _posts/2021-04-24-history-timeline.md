---
layout: post
title: World History Timeline with Map
tags: world history timeline altas map
---

Note this website is based on work from [The World History and Atlas](http://x768.com/w/twha.ja). There is a version with 3D map in [历史时间线](http://gonnavis.com/timeline/twha/).

<iframe src="https://nifty-hypatia-aa8f48.netlify.app/"  height="600" width="600"></iframe>

The data of the history and altas is provided as javascript file in [src/twha/js/region.js](https://github.com/gonnavis/Timeline/blob/master/src/twha/js/regions.js). It is an array, where each element has the following format:

0. label
1. year start
2. year end
3. 国名：如果长度大于3

   0. year start
   1. year end
   2. 图标 路径是twha/sym/图标.png
   3. 日语区域名
   4. 英文区域名
   5. 中文区域名，@/$表示中文和英文是一样的
   6. 日语区域缩写
   7. 英文区域缩写
   8. 中文区域缩写，@/$表示中文和英文是一样的
   9. x坐标
   10. y坐标
   11. display level
4. 统治者称呼：如果长度==3

   0. 日语称呼
   1. 英语称呼
   2. 中文称呼，@/$表示中文和英文是一样的
5. 统治者：接下来如果长度大于3

   0. year start
   1. year end
   2. 图标
   3. 日语区域名
   4. 英文区域名
   5. 中文区域名，@/$表示中文和英文是一样的
