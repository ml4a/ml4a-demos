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









function dataset3() {
	var root_dir;

	var batch_idx = 1;

	var self = this;

	this.loadMNIST = function() {
		root_dir = '/datasets/mnist/mnist';
	};

	this.load_batch = function() {
		var batch_path = root_dir+"_batch_"+batch_idx+".png";
		
		var img = new Image();
		  
		  img.onload = function() { 
		    // var data_canvas = document.createElement('myCanvas');
		    var data_canvas = document.getElementById('myCanvas');
		    data_canvas.width = img.width;
		    data_canvas.height = img.height;
		    console.log(img.height);
		    var data_ctx = data_canvas.getContext("2d");
		    data_ctx.drawImage(img, 0, 0); // copy it over... bit wasteful :(
		    self.img_data = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
		    //loaded[batch_num] = true;
		    //if(batch_num < test_batch) { loaded_train_batches.push(batch_num); }
		    console.log('finished loading data batch ' + batch_idx);
		  };

		  img.src = batch_path;
	}


	/*
		this.load_batch = function(batch_idx, callback) {
		var batch_path = root_dir+"_batch_"+batch_idx+".png";
		loadImage(batch_path, function(img) {
			var w = dim;
			var nc = channels;
			var n = rows_per_batch;
			img.loadPixels();
			for (var r=0; r<n; r++) {
		    	var b_label = labels[n * batch_idx + r];
		    	var b_vol = new convnetjs.Vol(w, w, nc, 0.0);
		    	var W = w * w;
		    	for (var i=0; i<W; i++) {
		     		var ix = ((W * r) + i) * 4;
			      for (var c=0; c<nc; c++) {
				     	b_vol.w[nc*i+c] = img.pixels[ix+c] / 255.0; 
						}	
		    	}
		    	if (test_batches.indexOf(batch_idx) == -1) {
			    	data.train.push({idx:data.train.length, vol:b_vol, label:b_label});
			    }
			    else {
			    	data.test.push({idx:data.test.length, vol:b_vol, label:b_label});	
			    }
			}
			console.log("loaded batch "+batch_idx+". size: {training set:"+data.train.length+", test set:"+data.test.length+")");
			fully_loaded = ((data.train.length + data.test.length) == (test_batch_only?test_batches.length:num_batches) * rows_per_batch);	
			loading_batch = false;
			if (callback != null) {
				callback();
			}
			if (callback_batch != null) {
				callback_batch();
			}
			if (callback_main != null && fully_loaded) {
				callback_main();
			}
		});
	};
	*/

}














function dataset2() 
{
	var dim;
	var channels;
	var classes;
	var rows_per_batch;
	var num_batches;	
	var fully_loaded;
	var loading_batch;
	var batch_idx, t_batch_idx;
	var test_batches;
	var test_batch_only;
	var sample_idx = {train:0, test:0};
	var data = {train:[], test:[]};
	var callback_main = null;
	var callback_batch = null;
	
	this.get_dim = function() {
		return dim;
	};

	this.get_channels = function() {
		return channels;
	};

	this.get_classes = function() {
		return classes;
	};

	this.get_training_size = function() {
		return data.train.length;
	};

	this.get_test_size = function() {
		return data.test.length;
	};

	this.get_training_size_all = function() {
		return (num_batches - test_batches.length) * rows_per_batch;
	};

	this.get_test_size_all = function() {
		return (num_batches - test_batches.length) * rows_per_batch;
	};

	this.is_loading = function() {
		return loading_batch;
	};

	this.is_fully_loaded = function() {
		return fully_loaded;
	};

	this.finished_testing = function() {
		return sample_idx.test >= test_batches.length * rows_per_batch;
	};

	this.get_training_sample = function(t) {
		return data.train[t];
	};

	this.get_test_sample = function(t) {
		return data.test[t];
	};

	this.get_sample_index = function() {
		return sample_idx;
	};

	this.get_next_training_sample = function() {
		if (sample_idx.train < data.train.length) {
			sample_idx.train += 1;
			return data.train[sample_idx.train-1];
		}
		else {
			return null;
		}
	};

	this.get_next_test_sample = function() {
		if (sample_idx.test < data.test.length) {
			sample_idx.test += 1;
			return data.test[sample_idx.test-1];
		}
		else {
			return null;
		}
	};

	this.set_train_index = function(idx) {
		sample_idx.train = idx;
	};

	this.set_test_index = function(idx) {
		sample_idx.test = idx;
	};

	this.initialize = function() {
		loading_batch = false;
		fully_loaded = false;
		sample_idx.train = 0;
		sample_idx.test = 0;
		t_batch_idx = 0;
		batch_idx = test_batch_only ? test_batches[t_batch_idx] : 0;
		this.load_batch(batch_idx);
	};

	this.loadMNIST = function(test_batch_only_, callback_main_, callback_batch_) {
		if (callback_main_ != null) {
			callback_main = callback_main_;
		}
		if (callback_batch_ != null) {
			callback_batch = callback_batch_;
		}
		test_batch_only = test_batch_only_ || false;
		root_dir = '/datasets/mnist/mnist';
		dim = 28;
		channels = 1;
		rows_per_batch = 3000;
		num_batches = 21;
		test_batches = [20]; //[0,1,2];//[18,19,20];
		classes = ["0","1","2","3","4","5","6","7","8","9"];
		this.initialize();
	};

	this.loadCIFAR = function(test_batch_only_, callback_main_, callback_batch_) {
		if (callback_main_ != null) {
			callback_main = callback_main_;
		}
		if (callback_batch_ != null) {
			callback_batch = callback_batch_;
		}
		test_batch_only = test_batch_only_ || false;
		root_dir = '/datasets/cifar/cifar10';
		dim = 32;
		channels = 3;
		rows_per_batch = 1000;
		num_batches = 51;
		test_batches = [50];//[0,1,2,3,4,5,6,7,8,9]; //[41,42,43,44,45,46,47,48,49,50];
		classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];
		this.initialize();
	};
	
	this.load_batch = function(batch_idx, callback) {
		var batch_path = root_dir+"_batch_"+batch_idx+".png";
		loadImage(batch_path, function(img) {
			var w = dim;
			var nc = channels;
			var n = rows_per_batch;
			img.loadPixels();
			for (var r=0; r<n; r++) {
		    	var b_label = labels[n * batch_idx + r];
		    	var b_vol = new convnetjs.Vol(w, w, nc, 0.0);
		    	var W = w * w;
		    	for (var i=0; i<W; i++) {
		     		var ix = ((W * r) + i) * 4;
			      for (var c=0; c<nc; c++) {
				     	b_vol.w[nc*i+c] = img.pixels[ix+c] / 255.0; 
						}	
		    	}
		    	if (test_batches.indexOf(batch_idx) == -1) {
			    	data.train.push({idx:data.train.length, vol:b_vol, label:b_label});
			    }
			    else {
			    	data.test.push({idx:data.test.length, vol:b_vol, label:b_label});	
			    }
			}
			console.log("loaded batch "+batch_idx+". size: {training set:"+data.train.length+", test set:"+data.test.length+")");
			fully_loaded = ((data.train.length + data.test.length) == (test_batch_only?test_batches.length:num_batches) * rows_per_batch);	
			loading_batch = false;
			if (callback != null) {
				callback();
			}
			if (callback_batch != null) {
				callback_batch();
			}
			if (callback_main != null && fully_loaded) {
				callback_main();
			}
		});
	};

	this.request_next_batch = function(callback){
		console.log("loading "+loading_batch);
		if (!loading_batch) {
			if (test_batch_only && t_batch_idx < test_batches.length-1) {
				t_batch_idx += 1;
				batch_idx = test_batches[t_batch_idx];
				loading_batch = true;
				this.load_batch(batch_idx, callback);
			}
			else if (batch_idx < num_batches-1) {
				batch_idx += 1;
				loading_batch = true;
				this.load_batch(batch_idx, callback);
			}
		}
	};

	this.get_image = function(sample) {
		var nc = channels;
		var img = createImage(dim, dim);
		img.loadPixels();
		for (var i=0; i<dim*dim; i++) {
			for (var j=0; j<3; j++) {	
				var iw = nc * i + (nc == 1 ? 0 : j);
				img.pixels[4*i+j] = 255 * sample.vol.w[iw];
			}
			img.pixels[4*i+3] = 255;
		}
		img.updatePixels();
		return img;
	};

	this.get_current_training_sample_image = function() {
		return this.get_image(data.train[min(data.train.length-1,sample_idx.train)]);
	};









};


/*
function dataset() 
{
	var dim;
	var channels;
	var classes;
	var rows_per_batch;
	var num_batches;	
	var fully_loaded;
	var loading_batch;
	var batch_idx, t_batch_idx;
	var test_batches;
	var test_batch_only;
	var sample_idx = {train:0, test:0};
	var data = {train:[], test:[]};
	var callback_main = null;
	var callback_batch = null;
	
	this.get_dim = function() {
		return dim;
	};

	this.get_channels = function() {
		return channels;
	};

	this.get_classes = function() {
		return classes;
	};

	this.get_training_size = function() {
		return data.train.length;
	};

	this.get_test_size = function() {
		return data.test.length;
	};

	this.get_training_size_all = function() {
		return (num_batches - test_batches.length) * rows_per_batch;
	};

	this.get_test_size_all = function() {
		return (num_batches - test_batches.length) * rows_per_batch;
	};

	this.is_loading = function() {
		return loading_batch;
	};

	this.is_fully_loaded = function() {
		return fully_loaded;
	};

	this.finished_testing = function() {
		return sample_idx.test >= test_batches.length * rows_per_batch;
	};

	this.get_training_sample = function(t) {
		return data.train[t];
	};

	this.get_test_sample = function(t) {
		return data.test[t];
	};

	this.get_sample_index = function() {
		return sample_idx;
	};

	this.get_next_training_sample = function() {
		if (sample_idx.train < data.train.length) {
			sample_idx.train += 1;
			return data.train[sample_idx.train-1];
		}
		else {
			return null;
		}
	};

	this.get_next_test_sample = function() {
		if (sample_idx.test < data.test.length) {
			sample_idx.test += 1;
			return data.test[sample_idx.test-1];
		}
		else {
			return null;
		}
	};

	this.set_train_index = function(idx) {
		sample_idx.train = idx;
	};

	this.set_test_index = function(idx) {
		sample_idx.test = idx;
	};

	this.initialize = function() {
		loading_batch = false;
		fully_loaded = false;
		sample_idx.train = 0;
		sample_idx.test = 0;
		t_batch_idx = 0;
		batch_idx = test_batch_only ? test_batches[t_batch_idx] : 0;
		this.load_batch(batch_idx);
	};

	this.loadMNIST = function(test_batch_only_, callback_main_, callback_batch_) {
		if (callback_main_ != null) {
			callback_main = callback_main_;
		}
		if (callback_batch_ != null) {
			callback_batch = callback_batch_;
		}
		test_batch_only = test_batch_only_ || false;
		root_dir = '/datasets/mnist/mnist';
		dim = 28;
		channels = 1;
		rows_per_batch = 3000;
		num_batches = 21;
		test_batches = [20]; //[0,1,2];//[18,19,20];
		classes = ["0","1","2","3","4","5","6","7","8","9"];
		this.initialize();
	};

	this.loadCIFAR = function(test_batch_only_, callback_main_, callback_batch_) {
		if (callback_main_ != null) {
			callback_main = callback_main_;
		}
		if (callback_batch_ != null) {
			callback_batch = callback_batch_;
		}
		test_batch_only = test_batch_only_ || false;
		root_dir = '/datasets/cifar/cifar10';
		dim = 32;
		channels = 3;
		rows_per_batch = 1000;
		num_batches = 51;
		test_batches = [50];//[0,1,2,3,4,5,6,7,8,9]; //[41,42,43,44,45,46,47,48,49,50];
		classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];
		this.initialize();
	};
	
	this.load_batch = function(batch_idx, callback) {
		var batch_path = root_dir+"_batch_"+batch_idx+".png";
		loadImage(batch_path, function(img) {
			var w = dim;
			var nc = channels;
			var n = rows_per_batch;
			img.loadPixels();
			for (var r=0; r<n; r++) {
		    	var b_label = labels[n * batch_idx + r];
		    	var b_vol = new convnetjs.Vol(w, w, nc, 0.0);
		    	var W = w * w;
		    	for (var i=0; i<W; i++) {
		     		var ix = ((W * r) + i) * 4;
			      for (var c=0; c<nc; c++) {
				     	b_vol.w[nc*i+c] = img.pixels[ix+c] / 255.0; 
						}	
		    	}
		    	if (test_batches.indexOf(batch_idx) == -1) {
			    	data.train.push({idx:data.train.length, vol:b_vol, label:b_label});
			    }
			    else {
			    	data.test.push({idx:data.test.length, vol:b_vol, label:b_label});	
			    }
			}
			console.log("loaded batch "+batch_idx+". size: {training set:"+data.train.length+", test set:"+data.test.length+")");
			fully_loaded = ((data.train.length + data.test.length) == (test_batch_only?test_batches.length:num_batches) * rows_per_batch);	
			loading_batch = false;
			if (callback != null) {
				callback();
			}
			if (callback_batch != null) {
				callback_batch();
			}
			if (callback_main != null && fully_loaded) {
				callback_main();
			}
		});
	};

	this.request_next_batch = function(callback){
		console.log("loading "+loading_batch);
		if (!loading_batch) {
			if (test_batch_only && t_batch_idx < test_batches.length-1) {
				t_batch_idx += 1;
				batch_idx = test_batches[t_batch_idx];
				loading_batch = true;
				this.load_batch(batch_idx, callback);
			}
			else if (batch_idx < num_batches-1) {
				batch_idx += 1;
				loading_batch = true;
				this.load_batch(batch_idx, callback);
			}
		}
	};

	this.get_image = function(sample) {
		var nc = channels;
		var img = createImage(dim, dim);
		img.loadPixels();
		for (var i=0; i<dim*dim; i++) {
			for (var j=0; j<3; j++) {	
				var iw = nc * i + (nc == 1 ? 0 : j);
				img.pixels[4*i+j] = 255 * sample.vol.w[iw];
			}
			img.pixels[4*i+3] = 255;
		}
		img.updatePixels();
		return img;
	};

	this.get_current_training_sample_image = function() {
		return this.get_image(data.train[min(data.train.length-1,sample_idx.train)]);
	};
};
*/