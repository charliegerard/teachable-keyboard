# Teachable keyboard using Tensorflow.js

This prototype is using the [Teachable Machine Boilerplate](https://github.com/googlecreativelab/teachable-machine-boilerplate).

It uses [tensorflow.js](https://github.com/tensorflow/tfjs-models) with a KNN classifier that is trained live in the browser using images from the webcam.

In the background, it uses an activation of [MobileNet](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet) and is using a technique called [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning).


## Install

In your terminal, run:

```
npm install
```

## Try

In your terminal, run:

```
npm start
```

Open a new tab in your browser and enter [`localhost:9966`](http://localhost:9966).

If you haven't already, allow permission to your webcam, and start adding examples by clicking on the buttons.

To start the predictions, click on the "Start prediction" button.
