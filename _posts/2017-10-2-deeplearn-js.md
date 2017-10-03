---
layout: post
title: deeplearn.js--A hardware-accelerated deep learning library for the web.
---

# Getting started

**deeplearn.js** is an open source hardware-accelerated JavaScript library for
machine intelligence. **deeplearn.js** brings performant machine learning
building blocks to the web, allowing you to train neural networks in a browser
or run pre-trained models in inference mode.

It provides two APIs, an immediate execution model (think NumPy) and a deferred
execution model mirroring the TensorFlow API.
**deeplearn.js** was originally developed by the Google Brain PAIR team to build
powerful interactive machine learning tools for the browser, but it can be used
for everything from education, to model understanding, to art projects.

## Usage

#### Typescript / ES6 JavaScript

```
npm install deeplearn
```

A simple example that sums an array with a scalar (broadcasted):

```ts
import {Array1D, NDArrayMathGPU, Scalar} from 'deeplearn';

const math = new NDArrayMathGPU();
const a = Array1D.new([1, 2, 3]);
const b = Scalar.new(2);
math.scope(() => {
  const result = math.add(a, b);
  console.log(result.getValues());  // Float32Array([3, 4, 5])
});
```

#### ES3/ES5 JavaScript

You can also use **deeplearn.js** with plain JavaScript. Load the latest version
of the library directly from the Google CDN:

```html
<script src="https://storage.googleapis.com/learnjs-data/deeplearn-latest.js"></script>
```

To use a specific version, replace `latest` with a version number
(e.g. `deeplearn-0.1.0.js`), which you can find in the
[releases](https://github.com/PAIR-code/deeplearnjs/releases) page on GitHub.
After importing the library, the API will be available as `deeplearn` in the
global namespace:

```js
var math = new deeplearn.NDArrayMathGPU();
var a = deeplearn.Array1D.new([1, 2, 3]);
var b = deeplearn.Scalar.new(2);
math.scope(function() {
  var result = math.add(a, b);
  console.log(result.getValues());  // Float32Array([3, 4, 5])
});
```

![](https://github.com/reiinakano/fast-style-transfer-deeplearnjs/raw/master/demo_screen.png)
