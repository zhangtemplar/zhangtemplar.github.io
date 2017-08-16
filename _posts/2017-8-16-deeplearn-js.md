---
layout: posts
title: deeplearn.js
---

Google has open sourced `deeplearn.js` to have you run deep learning in your browser.

# Introduction

`deeplearn.js` is an open source hardware-accelerated JavaScript library for machine intelligence. `deeplearn.js` brings performant machine learning building blocks to the web, allowing you to train neural networks in a browser or run pre-trained models in inference mode.

`deeplearn.js` has two APIs, an immediate execution model (think NumPy) and a deferred execution model mirroring the TensorFlow API. 

`deeplearn.js` was originally developed by the Google Brain PAIR team to build powerful interactive machine learning tools for the browser, but it can be used for everything from education, to model understanding, to art projects.

# Usage

From `JavaScript`: `Typescript` is the preferred language of choice for `deeplearn.js`, however you can use it with plain `JavaScript`.

For this use case, you can load the latest version of the library directly from Google CDN:

```
<script src="https://storage.googleapis.com/learnjs-data/deeplearn.js"></script>
```

You can use it build a model, as example [here](https://pair-code.github.io/deeplearnjs/demos/model-builder/model-builder-demo.html); or play with camera with `imagenet` [here](https://pair-code.github.io/deeplearnjs/demos/imagenet/imagenet-demo.html).

# Supported environments

`deeplearn.js` targets `WebGL 1.0` devices with the `OES_texture_float` extension and also targets WebGL 2.0 devices. For platforms without WebGL, we provide CPU fallbacks.

However, currently our demos do **NOT** support Mobile, Firefox, and Safari. Please view them on desktop Chrome for now. We are working to support more devices. Check back soon!

# Links

[Github](https://pair-code.github.io/deeplearnjs/)
