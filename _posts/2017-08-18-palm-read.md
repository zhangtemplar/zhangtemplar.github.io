---
layout: post
title: palm read via neural network
---

There is a Taiwan scholar invented [Handbot](https://www.facebook.com/handbot2017) — the first palm reading chatbot using deep learning (CNN) to analyze one’s palm to predict his/her character traits, health, career, relationship and many more aspects.

![The example interface](https://cdn-images-1.medium.com/max/2000/1*hYzJLZMJ8I2yFWieF-XjbA.png)

Method: 

```we use VGG-16(16-layer network) in Keras with Tensorflow backend as our CNN model. Using GPU to accelerate, we have trained 10 epoches and 1000 steps per epoch. The MSE of the model is 1.3066. Besides, the valuation MSE is 1.1721. Overall, the result is robust enough to predict a new palm.```

# More readings

  - [medium](https://medium.com/towards-data-science/when-ai-meets-3000-year-old-chinese-palmistry-a767b7f3defb)
  - [code](https://github.com/kkshyu/palm-read)
