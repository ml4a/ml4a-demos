// modified from https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js

const NUM_SLIDERS = 1;

import * as tf from '@tensorflow/tfjs';
import {ControllerDataset} from './controller_dataset';
import {Webcam} from './webcam';

const trainStatusElement = document.getElementById('train-status');

// Set hyper params from UI values.
const learningRateElement = document.getElementById('learningRate');
const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs');
const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById('dense-units');
const getDenseUnits = () => +denseUnitsElement.value;
const statusElement = document.getElementById('status');

function outputSliders(output) {
  var lerpRate = 1.0-parseFloat(document.getElementById('lerp').value);
  var sliderValues = [];
  for (var i=0; i<NUM_SLIDERS; i++) {
    var predValue = output[i];
    var currentValue = parseFloat(document.getElementById('slider'+i).value);
    var nextValue = (1.0-lerpRate)*currentValue + lerpRate*predValue;
    sliderValues.push(nextValue);
    document.getElementById('slider'+i).value = nextValue;
  }
  // send to p5
  mySketch.predict(sliderValues);
}

function ui_isPredicting() {
  statusElement.style.visibility = 'visible';
}
function donePredicting() {
  statusElement.style.visibility = 'hidden';
}
function trainStatus(status) {
  trainStatusElement.innerText = status;
}

let addExampleHandler;
function setExampleHandler(handler) {
  addExampleHandler = handler;
}
let mouseDown = false;

const totals = Array(NUM_SLIDERS).fill(0);;

const thumbDisplayed = {};
  
async function handler(label) {
  mouseDown = true;
  const sliderName = "slider"+label;//CONTROLS[label];
  const button  = document.getElementById(sliderName);
  const total = document.getElementById('class-total');
  while (mouseDown) {
    addExampleHandler(label);
    document.body.setAttribute('data-active', 'slider'+label);
    total.innerText = totals[label]++;
    await tf.nextFrame();
  }
  document.body.removeAttribute('data-active');
}

function drawThumb(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById('slider'+label+'-thumb');
    draw(img, thumbCanvas);
  }
}

function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}


// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(1);

let mobilenet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our regression model.
async function loadMobilenet() {
  const mobilenet = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}


// This function will add a slider for recording to each output
function addSlider(idx){
    var main = document.createElement('div');
    var mainS = document.createElement('p');
    mainS.innerHTML = '<input type="range" min="0.0" max="1.0" step="0.01" value="0.5" class="slider" id="slider'+idx+'">';
    main.className = 'thumb-box'
    main.appendChild(mainS);    
    document.getElementById('sliders').appendChild(main);
};

// Add the buttons
for (var c=0; c<NUM_SLIDERS; c++) {
  addSlider(c);
}


var mainOuter = document.createElement('div');
mainOuter.className = 'thumb-box-outer'

var mainP = document.createElement('p');
mainP.innerHTML = '<b>all sliders</b>: <span id="class-total">0</span> examples';
    
var mainButton = document.createElement('button');
mainButton.className = "record-button"
mainButton.setAttribute("id", name)
mainButton.innerHTML = "<span>Add Sample</span>"

mainOuter.appendChild(mainButton);
mainOuter.appendChild(mainP);
document.getElementById('sliders').appendChild(mainOuter)

// add mouse handlers
mainButton.addEventListener('mousedown', () => handler(0));
mainButton.addEventListener('mouseup', () => mouseDown = false);


// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the slider values
setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    var output = [...Array(NUM_SLIDERS).keys()].map(i => parseFloat(document.getElementById('slider'+i).value));
    controllerDataset.addExample(mobilenet.predict(img), output);

    // Draw the preview thumbnail.
    //drawThumb(img, label);
  });
});

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of outputs we want to predict.
      tf.layers.dense({
        units: NUM_SLIDERS,
        useBias: false,
        kernelInitializer: 'Zeros',
        activation: 'linear'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(getLearningRate());
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'meanSquaredError'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainStatus('Loss: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui_isPredicting();
  while (isPredicting) {
    const predictedValues = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = model.predict(activation);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      //return predictions.as1D().argMax();
      return predictions.as1D();
    });

    const prediction = (await predictedValues.data());
    var output = [...Array(NUM_SLIDERS).keys()].map(i => prediction[i]);

    predictedValues.dispose();
    outputSliders(output);
    await tf.nextFrame();
  }
  donePredicting();
}

document.getElementById('train').addEventListener('click', async () => {
  trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});

document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  predict();
});

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL              
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  // Once webcam and mobilenet loaded, display the app
  document.getElementById('controller').style.display = '';
  document.getElementById('no-webcam').style.display = 'none';  
  statusElement.style.display = 'none';
}

// Initialize the application.
init();
