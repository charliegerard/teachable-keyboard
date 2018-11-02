// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { expectPromiseToFail } from "@tensorflow/tfjs-core/dist/test_util";

// Number of classes to classify
const NUM_CLASSES = 4;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

const classes = ['Right', 'Left', 'Down', 'Neutral'];
let letterIndex = 0;

let testPrediction = false;
let startPrediction = false;
let training = true;

const trainingSection = document.getElementsByClassName('training-section')[0];
const buttonsSection = document.createElement('section');
buttonsSection.classList.add('buttons-section');

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.classList.add('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    // Add video element to DOM
    trainingSection.appendChild(this.video);

    // Create training buttons and info texts
    for (let i = 0; i < NUM_CLASSES; i++) {
      const buttonBlock = document.createElement('div');
      buttonBlock.classList.add('button-block');


      // Create training button
      const button = document.createElement('button');
      button.innerText = classes[i];
      buttonBlock.appendChild(button);
      buttonsSection.appendChild(buttonBlock);

      const div = document.createElement('div');
      div.classList.add('examples-text');
      buttonBlock.appendChild(div);
      div.style.marginBottom = '10px';

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('touchstart', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);
      button.addEventListener('touchend', () => this.training = -1);

      // // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }
    trainingSection.appendChild(buttonsSection);


    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  controlKeyboard(command) {
    const keys = document.getElementsByClassName('letter');

    if(command === "Right"){
      if(this.timer % 15 === 0 && letterIndex < keys.length){
        if(letterIndex !== keys.length - 1){
          letterIndex++;
        }
        document.getElementsByClassName("selected")[0].classList.remove("selected");
        document.getElementsByClassName("letter")[letterIndex].classList.add("selected")
      }
    } else if(command === "Left"){
      if(this.timer % 15 === 0 && letterIndex >= 0){
        if(letterIndex !== 0){
          letterIndex--;
        }
        document.getElementsByClassName("selected")[0].classList.remove("selected");
        document.getElementsByClassName("letter")[letterIndex].classList.add("selected");
      }
    } else if(command === "Down"){
      if(this.timer % 15 === 0){
        const selected = document.getElementsByClassName("selected")[0].textContent;

        if(selected === "Space"){
          document.getElementsByClassName('message-input')[0].value += " ";
        } else if(selected === "Delete"){
          if(document.getElementsByClassName('message-input')[0].value.length > 0){
            document.getElementsByClassName('message-input')[0].value = document.getElementsByClassName('message-input')[0].value.slice(0, -1);
          }
        } else {
          document.getElementsByClassName('message-input')[0].value += selected;
        }
      }
    }

  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();

      //start prediction
      if(testPrediction){
        training = false;
        if (numClasses > 0) {

          // If classes have been added run predict
          logits = infer();
          const res = await this.knn.predictClass(logits, TOPK);

          for (let i = 0; i < NUM_CLASSES; i++) {

            // The number of examples for each class
            const exampleCount = this.knn.getClassExampleCount();

            // Make the predicted class bold
            if (res.classIndex == i) {
              this.infoTexts[i].style.fontWeight = 'bold';
              if(startPrediction){
                this.controlKeyboard(classes[res.classIndex])
              }

            } else {
              this.infoTexts[i].style.fontWeight = 'normal';
            }

            // Update info text
            if (exampleCount[i] > 0) {
              this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
            }
          }
        }
      }


      if(training){
        // The number of examples for each class
        const exampleCount = this.knn.getClassExampleCount();

        for (let i = 0; i < NUM_CLASSES; i++) {
          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`
          }
        }
      }


      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}


if(window.location.pathname === "/teachable-keyboard/training.html" || window.location.pathname === "/training.html"){
  window.addEventListener('load', () => new Main());
  document.getElementsByClassName('test-predictions')[0].addEventListener('click', function(){
    testPrediction = true;
  })

  document.getElementsByClassName('start-prediction')[0].addEventListener('click', function(){
    if (!testPrediction) testPrediction = true;
    startPrediction = true;

    if(startPrediction){
      document.getElementsByClassName('training-section')[0].classList.add('no-display');
      document.getElementsByClassName('predictions-buttons')[0].classList.add('no-display');
      document.getElementsByClassName('interaction-block')[0].classList.add('display');
    }
  })
}

