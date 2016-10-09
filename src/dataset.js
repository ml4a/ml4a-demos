function dataset(datasetName) 
{
	this.get_dim = function(){return sw;}
	this.get_channels = function(){return channels;}
	this.get_samples_per_batch = function(){return samplesPerBatch;}
	this.get_batch_idx = function(){return idxBatch;}
	this.get_sample_idx = function(){return idxSample;}
	this.get_classes = function(){return classes;}

	this.load_MNIST = function() {
		sw = 28;
		sh = 28;
		channels = 1;
		samplesPerBatch = 3000;
		nBatches = 21;
		batchPath = "/datasets/mnist/mnist";
		batch.width = sw * sh;
		batch.height = samplesPerBatch;
		classes = ["0","1","2","3","4","5","6","7","8","9"];
		labelsFile = "../datasets/mnist/mnist_labels.js";
		labelsLoaded = false;
		idxBatch = -1;
		idxSample = -1;
	}

	this.load_CIFAR = function() {
		sw = 32;
		sh = 32;
		channels = 3;
		samplesPerBatch = 1000;
		nBatches = 51;
		batchPath = "/datasets/cifar/cifar10";
		batch.width = sw * sh;
		batch.height = samplesPerBatch;   
		classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];
		labelsFile = "../datasets/cifar/cifar10_labels.js";
		labelsLoaded = false;
		idxBatch = -1;
		idxSample = -1;
	};

	this.load_labels = function(callback) { 
		$.getScript(labelsFile, function(){
			labelsLoaded = true;
			callback();
		});
	};

	this.load_next_batch = function(callback) {
		load_batch(idxBatch+1, callback);
	};

	var load_batch = function(idxBatch_, callback) {
		idxBatch = idxBatch_;
		batchImg.onload = function() {
			batchCtx.drawImage(batchImg, 0, 0);
			batchImgData = batchCtx.getImageData(0, 0, batch.width, batch.height).data;
			console.log("loaded batch "+idxBatch);
			callback();
		};
		batchImg.src = batchPath+"_batch_"+idxBatch+".png";
	};

	var get_batch_sample = function(k) {
  		var W = sw * sh;
  		var y = labels[idxBatch * samplesPerBatch + k];
  		var x = new convnetjs.Vol(sw, sh, channels, 0.0);
		for(var dc=0; dc<channels; dc++) {
			var idx=0;
		    for(var xc=0; xc<sw; xc++) {
		    	for(var yc=0; yc<sh; yc++) {
		        	var ix = ((W * k) + idx) * 4 + dc;
		        	x.set(yc, xc, dc, batchImgData[ix]/255.0 - 0.5);
		        	idx++;
		      	}
		    }
		}
		return {x:x, y:y};
	};

	this.get_next_sample = function(idx, callback) {
		var returnSample = function(){
			var k = idxSample % samplesPerBatch;
			var sample = get_batch_sample(k);
			callback(sample);
		};
		idxSample += 1;
		if (idxSample >= samplesPerBatch * (idxBatch + 1)) {
			this.load_next_batch(returnSample);
		} else {
			returnSample();
		}
	};

	this.draw_current_sample = function(ctx, x, y, scale, grid_thickness) {
		this.draw_sample(ctx, idxSample, x, y, scale, grid_thickness);
	};

	this.draw_sample = function(ctx, idx, x, y, scale, grid_thickness) {
		var g = (grid_thickness === undefined) ? 0 : grid_thickness;
		var sampleImg = batchCtx.getImageData(0, idx, sw*sh, 1);
		var newImg = ctx.createImageData(sw * (scale + g), sh * (scale + g));
		for (var j=0; j<sh; j++) {
		 	for (var i=0; i<sw; i++) {
		    	var idxS = (j * sw + i) * 4;
		    	for (var sj=0; sj<scale+g; sj++) {
		      		for (var si=0; si<scale+g; si++) {
		      			var idxN = ((j * (scale + g) + sj) * sw * (scale + g) + (i * (scale + g) + si)) * 4;
		      			if (si < scale && sj < scale) {
			        		newImg.data[idxN]   = sampleImg.data[idxS];
			        		newImg.data[idxN+1] = sampleImg.data[idxS+1];
			        		newImg.data[idxN+2] = sampleImg.data[idxS+2];
			        		newImg.data[idxN+3] = sampleImg.data[idxS+3];                
			        	} else {
			        		newImg.data[idxN] = 127;
			        		newImg.data[idxN+1] = 127;
			        		newImg.data[idxN+2] = 127;
			        		newImg.data[idxN+3] = 255;
			        	}
		      		}
		    	}
		  	}
		}
		ctx.putImageData(newImg, x, y);
	};

	this.draw_sample_grid = function(ctx, rows, cols, scale, margin, label) {
		var draw_next_sample = function(n, idx, label) {
		  	if (labels[samplesPerBatch * idxBatch + idx] == label || label == null) {
		    	var y = margin + (sh * scale + margin) * Math.floor(n / cols);
		    	var x = margin + (sw * scale + margin) * (n % cols);
		    	self.draw_sample(ctx, idx, x, y, scale);
		    	n++;
		    	if (n==rows*cols) return;
		  	}
		  	if (idx+1 < samplesPerBatch) {
		    	draw_next_sample(n, idx+1, label);
		  	} 
		  	else if (idxBatch+1 < nBatches) {
		    	load_batch(idxBatch+1, function() {           
		    		draw_next_sample(n, 0, label);
		    	}); 
		  	}
		};
		draw_next_sample(0, 0, label);
	};

	// initialize
	var self = this;
	var batchPath, idxBatch;
  	var sw, sh, channels, samplesPerBatch, nBatches;
  	var labelsFile, labelsLoaded, classes;
  	var idxSample;

	// setup canvases
	var batch = document.createElement('canvas');
	var batchImg = new Image();
	var batchImgData;
	var batchCtx = batch.getContext('2d');
  
  	// set dataset
  	if (datasetName == 'MNIST') {
		this.load_MNIST();
	} else if (datasetName == 'CIFAR') {
		this.load_CIFAR();
	}
};
