var tf = ml5.tf;


var tensorImage;
var featureExtractor;
var feature;

let images = [];
let featureVectors = [];

var projectedValues;

let animals = ['bat-0001.jpg', 'bear-0001.jpg', 'bonsai-101-0001.jpg',
 'butterfly-0001.jpg', 'cactus-0001.jpg', 'camel-0001.jpg', 
 'centipede-0001.jpg', 'chimp-0001.jpg' ]


function preload() {
  animals.forEach(function (a){
    //images.push(loadImage('data/animals3/'+a));
  });  
}

function setup() {
  noCanvas();

  //do_pca();
  
  //featureExtractor = ml5.featureExtractor('MobileNet', modelReady);


  run_tsne();
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
      


      // how to do this right?  callback/promise?
      if (featureVectors.length == images.length) {
        console.log("GO PCA!")
        //run_pca();
      }

    });
  });
}


function run_tsne() {
  const data2 = tf.randomUniform([2000,10]);
  const tsneOpt = tsne.tsne(data2);
  tsneOpt.compute().then(() => {
    console.log("success")
  });
}


function run_pca() {
  var nf = featureVectors[0].length; //1200;  // num features
  var k = 200;
  var projection = tf.randomUniform([nf, k])
  var tFeatures = tf.tensor2d(featureVectors);
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
  //run_tsne();
}

function imgToTensor(input, size = null) {
  return tf.tidy(() => {
    let img = tf.fromPixels(input);
    if (size) {
      img = tf.image.resizeBilinear(img, size);
    }
    const croppedImage = ml5.cropImage(img);
    const batchedImage = croppedImage.expandDims(0);
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  });
}


/*

function draw() {

  images.forEach(function (img){


    var rs = max(img.width/max_dim, img.height/max_dim);
    var w = int(img.width/rs);
    var h = int(img.height/rs);

    var x = random(1000);
    var y = random(800);

    image(img, x, y, w, h);
  });  

}
*/