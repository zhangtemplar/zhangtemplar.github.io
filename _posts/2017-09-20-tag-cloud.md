---
layout: post
title: How to create tag clouds
---

# wordcloud

The most simple way is to use a Python package `wordcloud`. Then you can generate a word cloud with the following code:

```
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud()
text = open(r"222.txt",encoding='utf-8').read()
wc.generate(text)
plt.imshow(wc)
```

This is the example:

![](https://pic4.zhimg.com/v2-31b792b309182643b327ed6cd31d7583_b.jpg)

# Django service

There is a free service in [词云助手](http://www.huabandata.com/tools/wordcloud/) let you generate a tag cloud and you can even specify the image for the cloud. For example, with this as background:

![](https://pic2.zhimg.com/v2-0bdf6249f5661410c7cc65150f579ee5_b.jpg)

We can get

![](https://pic2.zhimg.com/v2-d7e8b9f26b73fc8eb06206c822d51d39_b.jpg)
