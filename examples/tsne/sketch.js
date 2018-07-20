var img;
var tensorImage;
var featureExtractor;
var feature;

let images = [];
let featureVectors = [];

let animals = ['bat-0001.jpg', 'bear-0001.jpg', 'bonsai-101-0001.jpg',
 'butterfly-0001.jpg', 'cactus-0001.jpg', 'camel-0001.jpg', 
 'centipede-0001.jpg', 'chimp-0001.jpg' ]


function preload() {
  animals.forEach(function (a){
    images.push(loadImage('data/animals3/'+a));
  });  
}

function setup() {
  noCanvas();

  //do_pca();
  featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
}

function modelReady() {
  images.forEach(function (img){
    img.loadPixels();
    tensorImage = imgToTensor(img.imageData, [224, 224]);
    features = featureExtractor.mobilenetFeatures.predict(tensorImage);
    //console.log('Tensor with features until conv_pw_13_relu:', features);
    features.data().then(function(data) {
      //console.log(result);
      //featureVectors.push(data.slice(0, 100));
      //featureVectors.push(Array.prototype.slice.call(data.slice(0, 100)));
      featureVectors.push(Array.prototype.slice.call(data));
      console.log("Vec"+str(featureVectors.length))
      if (featureVectors.length == images.length) {
        console.log("GO PCA!")
        do_pca();
      }
    });
  });
}

var eigenvectors;

var projectedValues;
var z2, z4;

function do_pca() {
  var nf = featureVectors[0].length; //1200;  // num features
  var k = 200;
  var projection = ml5.tf.randomUniform([nf, k])
  var tFeatures = ml5.tf.tensor2d(featureVectors);
  projectedValues = tFeatures.dot(projection);

  /*
  var data = featureVectors;//[[5,6,5,7], [19,-15,-5,0], [4,5,5,6], [23,-10,-7, 1], [9,8,4,8]];
  eigenvectors = PCA.getEigenVectors(featureVectors);
  var adData = PCA.computeAdjustedData(data, ...eigenvectors.slice(0, 5))

  //console.log("done!!!PCA")
  //console.log(vectors)
  console.log(data)
  console.log("===")
  console.log(adData.adjustedData)
  */
}

function imgToTensor(input, size = null) {
  return ml5.tf.tidy(() => {
    let img = ml5.tf.fromPixels(input);
    if (size) {
      img = ml5.tf.image.resizeBilinear(img, size);
    }
    const croppedImage = ml5.cropImage(img);
    const batchedImage = croppedImage.expandDims(0);
    return batchedImage.toFloat().div(ml5.tf.scalar(127)).sub(ml5.tf.scalar(1));
  });
}


/*
let featureExtractor;
let classifier;
let video;
let loss;
//let img;
let numImages;
let samples;
let label;
let buttons;


//let animals = ['bat-0001.jpg', 'bear-0001.jpg', 'bonsai-101-0001.jpg', 'butterfly-0001.jpg', 'cactus-0001.jpg', 'camel-0001.jpg', 'centipede-0001.jpg', 'chimp-0001.jpg', 'cockroach-0001.jpg', 'conch-0001.jpg', 'crab-101-0001.jpg', 'dog-0001.jpg', 'dolphin-101-0001.jpg', 'duck-0001.jpg', 'elephant-101-0001.jpg', 'elk-0001.jpg', 'fern-0001.jpg', 'frog-0001.jpg', 'giraffe-0001.jpg', 'goat-0001.jpg', 'goose-0001.jpg', 'grasshopper-0001.jpg', 'greyhound-0001.jpg', 'hawksbill-101-0001.jpg', 'hibiscus-0001.jpg', 'house-fly-0001.jpg', 'hummingbird-0001.jpg', 'iguana-0001.jpg', 'iris-0001.jpg', 'killer-whale-0001.jpg', 'llama-101-0001.jpg', 'mushroom-0001.jpg', 'ostrich-0001.jpg', 'owl-0001.jpg', 'penguin-0001.jpg', 'porcupine-0001.jpg', 'praying-mantis-0001.jpg', 'raccoon-0001.jpg', 'scorpion-101-0001.jpg', 'skunk-0001.jpg', 'snail-0001.jpg', 'spider-0001.jpg', 'starfish-101-0001.jpg', 'sunflower-101-0001.jpg', 'swan-0001.jpg', 'toad-0001.jpg', 'triceratops-0001.jpg', 'trilobite-101-0001.jpg', 'unicorn-0001.jpg', 'zebra-0001.jpg']

let animals = ['bat-0001.jpg', 'bear-0001.jpg', 'bonsai-101-0001.jpg', 'butterfly-0001.jpg', 'cactus-0001.jpg', 'camel-0001.jpg', 'centipede-0001.jpg', 'chimp-0001.jpg', 'cockroach-0001.jpg', 'conch-0001.jpg', 'crab-101-0001.jpg', 'dog-0001.jpg', 'dolphin-101-0001.jpg', 'duck-0001.jpg', 'elephant-101-0001.jpg', 'elk-0001.jpg', 'fern-0001.jpg', 'frog-0001.jpg', 'giraffe-0001.jpg', 'goat-0001.jpg', 'goose-0001.jpg', 'grasshopper-0001.jpg', 'greyhound-0001.jpg', 'hawksbill-101-0001.jpg', 'hibiscus-0001.jpg', 'house-fly-0001.jpg', 'hummingbird-0001.jpg', 'iguana-0001.jpg', 'iris-0001.jpg', 'killer-whale-0001.jpg', 'llama-101-0001.jpg', 'mushroom-0001.jpg', 'ostrich-0001.jpg', 'owl-0001.jpg', 'penguin-0001.jpg', 'porcupine-0001.jpg', 'praying-mantis-0001.jpg', 'raccoon-0001.jpg', 'scorpion-101-0001.jpg', 'skunk-0001.jpg', 'snail-0001.jpg', 'spider-0001.jpg', 'starfish-101-0001.jpg', 'sunflower-101-0001.jpg', 'swan-0001.jpg', 'toad-0001.jpg', 'triceratops-0001.jpg', 'trilobite-101-0001.jpg', 'unicorn-0001.jpg', 'zebra-0001.jpg', 'bat-0002.jpg', 'bear-0002.jpg', 'bonsai-101-0002.jpg', 'butterfly-0002.jpg', 'cactus-0002.jpg', 'camel-0002.jpg', 'centipede-0002.jpg', 'chimp-0002.jpg', 'cockroach-0002.jpg', 'conch-0002.jpg', 'crab-101-0002.jpg', 'dog-0002.jpg', 'dolphin-101-0002.jpg', 'duck-0002.jpg', 'elephant-101-0002.jpg', 'elk-0002.jpg', 'fern-0002.jpg', 'frog-0002.jpg', 'giraffe-0002.jpg', 'goat-0002.jpg', 'goose-0002.jpg', 'grasshopper-0002.jpg', 'greyhound-0002.jpg', 'hawksbill-101-0002.jpg', 'hibiscus-0002.jpg', 'house-fly-0002.jpg', 'hummingbird-0002.jpg', 'iguana-0002.jpg', 'iris-0002.jpg', 'killer-whale-0002.jpg', 'llama-101-0002.jpg', 'mushroom-0002.jpg', 'ostrich-0002.jpg', 'owl-0002.jpg', 'penguin-0002.jpg', 'porcupine-0002.jpg', 'praying-mantis-0002.jpg', 'raccoon-0002.jpg', 'scorpion-101-0002.jpg', 'skunk-0002.jpg', 'snail-0002.jpg', 'spider-0002.jpg', 'starfish-101-0002.jpg', 'sunflower-101-0002.jpg', 'swan-0002.jpg', 'toad-0002.jpg', 'triceratops-0002.jpg', 'trilobite-101-0002.jpg', 'unicorn-0002.jpg', 'zebra-0002.jpg', 'bat-0003.jpg', 'bear-0003.jpg', 'bonsai-101-0003.jpg', 'butterfly-0003.jpg', 'cactus-0003.jpg', 'camel-0003.jpg', 'centipede-0003.jpg', 'chimp-0003.jpg', 'cockroach-0003.jpg', 'conch-0003.jpg', 'crab-101-0003.jpg', 'dog-0003.jpg', 'dolphin-101-0003.jpg', 'duck-0003.jpg', 'elephant-101-0003.jpg', 'elk-0003.jpg', 'fern-0003.jpg', 'frog-0003.jpg', 'giraffe-0003.jpg', 'goat-0003.jpg', 'goose-0003.jpg', 'grasshopper-0003.jpg', 'greyhound-0003.jpg', 'hawksbill-101-0003.jpg', 'hibiscus-0003.jpg', 'house-fly-0003.jpg', 'hummingbird-0003.jpg', 'iguana-0003.jpg', 'iris-0003.jpg', 'killer-whale-0003.jpg', 'llama-101-0003.jpg', 'mushroom-0003.jpg', 'ostrich-0003.jpg', 'owl-0003.jpg', 'penguin-0003.jpg', 'porcupine-0003.jpg', 'praying-mantis-0003.jpg', 'raccoon-0003.jpg', 'scorpion-101-0003.jpg', 'skunk-0003.jpg', 'snail-0003.jpg', 'spider-0003.jpg', 'starfish-101-0003.jpg', 'sunflower-101-0003.jpg', 'swan-0003.jpg', 'toad-0003.jpg', 'triceratops-0003.jpg', 'trilobite-101-0003.jpg', 'unicorn-0003.jpg', 'zebra-0003.jpg', 'bat-0004.jpg', 'bear-0004.jpg', 'bonsai-101-0004.jpg', 'butterfly-0004.jpg', 'cactus-0004.jpg', 'camel-0004.jpg', 'centipede-0004.jpg', 'chimp-0004.jpg', 'cockroach-0004.jpg', 'conch-0004.jpg', 'crab-101-0004.jpg', 'dog-0004.jpg', 'dolphin-101-0004.jpg', 'duck-0004.jpg', 'elephant-101-0004.jpg', 'elk-0004.jpg', 'fern-0004.jpg', 'frog-0004.jpg', 'giraffe-0004.jpg', 'goat-0004.jpg', 'goose-0004.jpg', 'grasshopper-0004.jpg', 'greyhound-0004.jpg', 'hawksbill-101-0004.jpg', 'hibiscus-0004.jpg', 'house-fly-0004.jpg', 'hummingbird-0004.jpg', 'iguana-0004.jpg', 'iris-0004.jpg', 'killer-whale-0004.jpg', 'llama-101-0004.jpg', 'mushroom-0004.jpg', 'ostrich-0004.jpg', 'owl-0004.jpg', 'penguin-0004.jpg', 'porcupine-0004.jpg', 'praying-mantis-0004.jpg', 'raccoon-0004.jpg', 'scorpion-101-0004.jpg', 'skunk-0004.jpg', 'snail-0004.jpg', 'spider-0004.jpg', 'starfish-101-0004.jpg', 'sunflower-101-0004.jpg', 'swan-0004.jpg', 'toad-0004.jpg', 'triceratops-0004.jpg', 'trilobite-101-0004.jpg', 'unicorn-0004.jpg', 'zebra-0004.jpg']

let images = [];


let img78;
let img77;

function imgToTensor(input, size = null) {
  return ml5.tf.tidy(() => {
    console.log("do 1")
    let img = ml5.tf.fromPixels(input);
    console.log("do 2")
    if (size) {
      img = ml5.tf.image.resizeBilinear(img, size);
    }
    console.log("do 3")
    const croppedImage = ml5.cropImage(img);
    const batchedImage = croppedImage.expandDims(0);
    return batchedImage.toFloat().div(ml5.tf.scalar(127)).sub(ml5.tf.scalar(1));
  });
}


let img3;

let max_dim = 300;

function preload() {

  images.push(loadImage('data/animals3/bat-0001.jpg'));

  img3 = loadImage('data/animals3/bat-0001.jpg');
  //img78 = createImg('data/animals3/bat-0001.jpg');



  console.log("GO!!!!!")
  console.log("image",images[0])


  img77 = select('#theImage');

  // load all the images
  animals.forEach(function (a){
   // images.push(loadImage('data/animals3/'+a));
  });  

  // analyze all the images

}

function do_pca() {
//const PCA = require('ml-pca');
//const dataset = require('ml-dataset-iris').getNumbers();
// dataset is a two-dimensional array where rows represent the samples and columns the features


  var data = [[5,6,5,7], [19,-15,-5,0], [4,5,5,6], [23,-10,-7, 1], [9,8,4,8]];
  var vectors = PCA.getEigenVectors(data);

  var adData = PCA.computeAdjustedData(data,vectors[0], vectors[1])

  //console.log("done!!!PCA")
  //console.log(vectors)
  console.log(data)
  console.log("===")
  console.log(adData.adjustedData)

}

function do_pca2(){

var dataset = [[5,6,19,-1], [1,0,-3,2], [4,16,5,9], [2,6,7,8]];

const pca = new PCA(dataset);
console.log(pca.getExplainedVariance());

const newPoints = [[4.9, 3.2, 1.2, 0.4], [5.4, 3.3, 1.4, 0.9]];
console.log(pca.predict(newPoints)); // project new points into the PCA space

}


function setup() {
  console.log("set up")
  createCanvas(1024, 800);

  //do_pca();

  doThis();


}

function doThis() {
// Grab any img
// = document.getElementById('theImage')




// Convert the image to the size the network expects
var tensorImage = imgToTensor(img77.elt, [224, 224]);
// Create a Feature Extractor

//var featureExtractor = ml5.featureExtractor('MobileNet', () => {});
// Get the features until conv_pw_13_relu
//var feature = featureExtractor.mobilenetFeatures.predict(tensorImage);



//  img3.loadPixels();
  


  console.log("====")
  //console.log("image222",img3)
  

  //let img4 = ml5.tf.fromPixels(img3.imageData);
  //console.log(img4);

//  var img5 = ml5.tf.image.resizeBilinear(img5, [224, 224]);
 // console.log(img5);

  //var tensorImage = imgToTensor(img3.imageData, [224, 224]);
  //console.log("got tensor")
  //console.log(tensorImage)






//  do_pca();

  background(255, 0, 0);

  images.forEach(function (img){


    var rs = max(img.width/max_dim, img.height/max_dim);
    var w = int(img.width/rs);
    var h = int(img.height/rs);

    var x = random(1000);
    var y = random(800);

    image(img, x, y, w, h);
  });  

  //numImages = Array(numClasses).fill(0);

  //console.log("lets make a canvas")
  //featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
  //featureExtractor.numClasses = numClasses;
  //classifier = featureExtractor.classification(video);
  

  //img = loadImage('data/animals/bat-0001.jpg', gotImage);


  //createButtons();

}

function gotImage() {
  console.log('got image')
}

function test2() {
  const data = tf.randomUniform([2000,10]);



    // Get a tsne optimizer
    const tsneOpt = tsne.tsne(data);
    console.log("do it")
    // Compute a T-SNE embedding, returns a promise.
    // Runs for 1000 iterations be default.
    tsneOpt.compute().then(() => {
      // tsne.coordinate returns a *tensor* with x, y coordinates of
      // the embedded data.
      const coordinates = tsneOpt.coordinates();
      coordinates.print();
      console.log("done!")
    }) ;



}

function modelReady() {
  select('#loading').html('Base Model (MobileNet) loaded!');
  // begin analysis
}

function addImage(label) {
//  classifier.addImage(label);
}

function classify() {
//  classifier.classify(gotResults);
}


function gotResults(nextLabel) {
  // select('#result').html(nextLabel);
  // if (label != nextLabel) {
  //   label = nextLabel;
  //   samples[int(label)].play();  
  // }
//  classify();
}

function draw() {
  
  
  //image(img, 50, 50);

}
*/