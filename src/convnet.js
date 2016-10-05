// TODO
//  - augmentation


function convnet(dataset_) 
{
    this.get_summary = function(){return summary;}

    this.add_layer = function(layer_def) {
        layer_defs.push(layer_def);
    };

    this.setup_trainer = function(trainer_params) {
        net = new convnetjs.Net();
        net.makeLayers(layer_defs);
        trainer = new convnetjs.SGDTrainer(net, trainer_params);
    };

    var check_if_ready = function(callback) {
        if (!dataset.labelsLoaded) {
            dataset.load_labels(function(){
                callback(0);
            });
        } else {
            callback(0);
        }  
    };

    var test_sample = function(sample) {
        var out_prob = net.forward(sample.x).w;
        var top_prob = Math.max.apply(Math, out_prob);
        var p = out_prob.indexOf(top_prob); 
        var a = sample.y;   
        summary.confusion[a][p] += 1;
        summary.actuals[a] += 1;
        summary.predictions[p] += 1;
        summary.total += 1;
        summary.correct += (a==p?1:0);    
        var inserted = false;
        for (var i=0; i<summary.tops[a][p].length; i++) {
            if (top_prob > summary.tops[a][p][i].prob) {
                summary.tops[a][p].splice(i, 0, {idx:-1, prob:top_prob});  // <----- idx: test_idx???
                inserted = true;
                break;
            }
        }
        if (!inserted && summary.tops[a][p].length < max_tops) {
            summary.tops[a][p].push({idx:-1, prob:top_prob});
        }
        return {predicted:p, actual:a, prob:out_prob};
    };

    this.train = function(numTrain, callback) {
        var train_next_sample = function(t){            
            dataset.get_next_sample(t, function(sample){
                trainer.train(sample.x, sample.y);
                if (t+1 < numTrain) {
                    train_next_sample(t+1);
                } else {
                    callback();
                }
            });
        };
        check_if_ready(train_next_sample);
    };

    this.test = function(numTest, callback) {
        var results = [];
        var test_next_sample = function(t){
            dataset.get_next_sample(t, function(sample){
                result = test_sample(sample);
                results.push(result);
                if (t+1 < numTest) {
                    test_next_sample(t+1);
                } else {
                    callback(results);
                }
            });
        };
        check_if_ready(test_next_sample);
    };

    var initialize = function() {
        var n = dataset.get_classes().length;
        summary = {
            confusion: [...Array(n).keys()].map(i => [...Array(n).keys()].map(i => 0)), 
            tops: [...Array(n).keys()].map(i => [...Array(n).keys()].map(i => [])), 
            actuals: [...Array(n).keys()].map(i => 0), 
            predictions: [...Array(n).keys()].map(i => 0), 
            correct: 0, total: 0
        };
        layer_defs = [];
        layer_defs.push({
            type:'input', 
            out_sx:dataset.get_dim(), 
            out_sy:dataset.get_dim(), 
            out_depth:dataset.get_channels()
        });
        idxSample = -1;
    };
    
    // initialize    
    var self = this;
    var maxmin = cnnutil.maxmin;
    var f2t = cnnutil.f2t;
    var dataset = dataset_;
    var max_tops = 64;
    var layer_defs;
    var trainer;
    var net;
    var summary;
    
    initialize();    
};


function convnet2(dataset_) 
{
  var self = this;
  var maxmin = cnnutil.maxmin;
  var f2t = cnnutil.f2t;
  var dataset = dataset_;

  var layer_defs = [];
  var trainer;
  var net;

  var results;
  var max_tops = 32;

  var test_idx;
  var out_prob;
  var a, p;
  
  this.add_layer = function(layer_def) {
    layer_defs.push(layer_def);
  };

  this.get_layer_defs = function() {
    return layer_defs;
  };
  
  this.get_prob = function() {
    return out_prob;
  };

  this.get_actual_label = function() {
    return a;
  };

  this.get_predicted_label = function() {
    return p;
  };

  this.get_in_acts = function(l) {
    return net.layers[l].in_act.w;
  };
  
  this.get_out_acts = function(l) {
    return net.layers[l].out_act.w;
  };

  this.get_net = function() {
    return net;
  };

  this.get_dataset = function() {
    return dataset;
  };

  this.get_results = function() {
    return results;
  };

  this.get_maxmin = function(a) {
    return maxmin(a);
  }

  this.load_from_json = function(json) {
    net = new convnetjs.Net();
    net.fromJSON(json);
  };

  this.setup_trainer = function(trainer_params) {
    net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    trainer = new convnetjs.SGDTrainer(net, trainer_params);
  };

  var setup_results = function() {
    results = {confusion:[], tops:[], actuals:[], predictions:[], correct:0, total:0};
    var n = dataset.get_classes().length;
    for (var i=0; i<n; i++) {
      var crow = [];
      var trow = [];
      for (var j=0; j<n; j++) {
        crow.push(0);
        trow.push([]);
      }
      results.confusion.push(crow);
      results.tops.push(trow);
      results.predictions.push(0);
      results.actuals.push(0);
    }
  };

  this.train_next = function() {
    var sample = dataset.get_next_training_sample();
    if (sample == null) { 
      if (!dataset.is_fully_loaded()) {
        dataset.request_next_batch(function() {
          //self.train_all(callback);
        });
      }
      return false;
    }
    else {
      trainer.train(sample.vol, sample.label);
      return true;
    }
  };

  this.train_all = function(callback) {
    var is_training = true;
    while (is_training) {
      is_training = this.train_next();
      if (!is_training) {
        console.log(is_training);
        if (!dataset.is_fully_loaded()) {
          dataset.request_next_batch(function() {
            self.train_all(callback);
          });
        }
        else if (callback != null) {
          callback();
        }
      }
    }
  };

  this.test_all = function(callback) {  
    var is_testing = true;
    while (is_testing) {
      is_testing = this.test_next();
      if (!is_testing) {
        if (!dataset.finished_testing()) {
          dataset.request_next_batch(function() {
            self.test_all(callback);
          });
        }
        else if (callback != null) {
          callback();
        }
      }
    }
  };

  this.test_next = function() {
    var sample = dataset.get_next_test_sample();
    if (sample == null) {
      if (!dataset.finished_testing()) {
        dataset.request_next_batch(function (){
          return self.test_next();
        });
      }
      return false;
    }    
    test_idx = sample.idx;

    // problems in firefox?
    /*
    out_prob = net.forward(sample.vol).w;
    var prob = Math.max.apply(Math, out_prob);
    a = sample.label;
    p = out_prob.indexOf(prob);    
    */

    // this is poor form, but works...
    out_prob = net.forward(sample.vol).w;
    var prob = 0;
    p = 0;
    for (var i=0; i<out_prob.length; i++) {
      if (out_prob[i] > prob) {
        p = i;
        prob = out_prob[i];
      }
    }
    a = sample.label;

    results.confusion[a][p] += 1;
    results.actuals[a] += 1;
    results.predictions[p] += 1;
    results.total += 1;
    results.correct += (a==p?1:0);    
    var inserted = false;
    for (var i=0; i<results.tops[a][p].length; i++) {
      if (prob > results.tops[a][p][i].prob) {
        results.tops[a][p].splice(i, 0, {idx:test_idx, prob:prob});
        inserted = true;
        break;
      }
    }
    if (!inserted && results.tops[a][p].length < max_tops) {
      results.tops[a][p].push({idx:test_idx, prob:prob});
    }
    return true;
  };

  this.start_over = function() {
    setup_results();
    dataset.set_test_index(0);
  };

  this.save_dataset = function() {
    var data = JSON.stringify(net.toJSON());
    var url = 'data:text/json;charset=utf8,' + encodeURIComponent(data);
    window.open(url, '_blank');
    window.focus();
  };

  this.vol_to_image = function(V, idx, scale, is_weight) {
    var nx = V.sx;
    var ny = V.sy;
    var nz = V.depth;
    var nc = dataset.get_channels();
    if (nx * ny == 1) {
      nz = dataset.get_channels();
      nx = sqrt(V.depth / nz);
      ny = nx;
    }
    var W = scale * nx;
    var H = scale * ny;
    var mm = maxmin(V.w);
    var img = createImage(W, H);
    img.loadPixels();    
    for(var x=0;x<nx;x++) {
      for(var y=0;y<ny;y++) {
        var z = nz * (y * nx + x) + (is_weight ? 0 : idx);
        var val = [];
        for(var dz=0;dz<nc;dz++) {
          val.push(Math.floor(255 * (V.w[z+dz] - mm.minv) / mm.dv));
        }
        for(var dx=0;dx<scale;dx++) {
          for(var dy=0;dy<scale;dy++) {
            var idx_ = ((W * (y*scale+dy)) + (dx + x*scale)) * 4;
            for (var dz=0;dz<3;dz++) {
              img.pixels[idx_+dz] = val[dz % nc];
            }
            img.pixels[idx_+3] = 255;
          }
        }
      }
    }
    img.updatePixels();
    return img;
  };

  this.get_weights_image = function(layer, idx, scale) {
    var V = net.layers[layer].filters[idx];
    return this.vol_to_image(V, idx, scale, true);
  };

  this.get_activations_image = function(layer, idx, scale) {
    var V = net.layers[layer].out_act;
    return this.vol_to_image(V, idx, scale, false);
  };

  this.get_num_activations = function(layer) {
    return net.layers[1].out_act.depth;
  };
  
  this.get_num_weights = function(layer) {
    return net.layers[1].filters.length;
  };
  
  this.get_training_sample_image = function(t) {
    var sample = dataset.get_training_sample(t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };

  this.get_test_sample_image = function(t) {
    var sample = dataset.get_test_sample(t == null ? test_idx : t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };

  // initialize
  layer_defs.push({type:'input', out_sx:dataset.get_dim().w, out_sy:dataset.get_dim().h, out_depth:dataset.get_channels()});
  setup_results();
};



/*
function convnet(dataset_) 
{
  var self = this;
  var maxmin = cnnutil.maxmin;
  var f2t = cnnutil.f2t;
  var dataset = dataset_;

  var layer_defs = [];
  var trainer;
  var net;

  var results;
  var max_tops = 32;

  var test_idx;
  var out_prob;
  var a, p;
  
  this.add_layer = function(layer_def) {
    layer_defs.push(layer_def);
  };

  this.get_layer_defs = function() {
    return layer_defs;
  };
  
  this.get_prob = function() {
    return out_prob;
  };

  this.get_actual_label = function() {
    return a;
  };

  this.get_predicted_label = function() {
    return p;
  };

  this.get_in_acts = function(l) {
    return net.layers[l].in_act.w;
  };
  
  this.get_out_acts = function(l) {
    return net.layers[l].out_act.w;
  };

  this.get_net = function() {
    return net;
  };

  this.get_dataset = function() {
    return dataset;
  };

  this.get_results = function() {
    return results;
  };

  this.get_maxmin = function(a) {
    return maxmin(a);
  }

  this.load_from_json = function(json) {
    net = new convnetjs.Net();
    net.fromJSON(json);
  };

  this.setup_trainer = function(trainer_params) {
    net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    trainer = new convnetjs.SGDTrainer(net, trainer_params);
  };

  var setup_results = function() {
    results = {confusion:[], tops:[], actuals:[], predictions:[], correct:0, total:0};
    var n = dataset.get_classes().length;
    for (var i=0; i<n; i++) {
      var crow = [];
      var trow = [];
      for (var j=0; j<n; j++) {
        crow.push(0);
        trow.push([]);
      }
      results.confusion.push(crow);
      results.tops.push(trow);
      results.predictions.push(0);
      results.actuals.push(0);
    }
  };

  this.train_next = function() {
    var sample = dataset.get_next_training_sample();
    if (sample == null) { 
      if (!dataset.is_fully_loaded()) {
        dataset.request_next_batch(function() {
          //self.train_all(callback);
        });
      }
      return false;
    }
    else {
      trainer.train(sample.vol, sample.label);
      return true;
    }
  };

  this.train_all = function(callback) {
    var is_training = true;
    while (is_training) {
      is_training = this.train_next();
      if (!is_training) {
        console.log(is_training);
        if (!dataset.is_fully_loaded()) {
          dataset.request_next_batch(function() {
            self.train_all(callback);
          });
        }
        else if (callback != null) {
          callback();
        }
      }
    }
  };

  this.test_all = function(callback) {  
    var is_testing = true;
    while (is_testing) {
      is_testing = this.test_next();
      if (!is_testing) {
        if (!dataset.finished_testing()) {
          dataset.request_next_batch(function() {
            self.test_all(callback);
          });
        }
        else if (callback != null) {
          callback();
        }
      }
    }
  };

  this.test_next = function() {
    var sample = dataset.get_next_test_sample();
    if (sample == null) {
      if (!dataset.finished_testing()) {
        dataset.request_next_batch(function (){
          return self.test_next();
        });
      }
      return false;
    }    
    test_idx = sample.idx;

    // problems in firefox?
    /*
    out_prob = net.forward(sample.vol).w;
    var prob = Math.max.apply(Math, out_prob);
    a = sample.label;
    p = out_prob.indexOf(prob);    
    */
/*
    // this is poor form, but works...
    out_prob = net.forward(sample.vol).w;
    var prob = 0;
    p = 0;
    for (var i=0; i<out_prob.length; i++) {
      if (out_prob[i] > prob) {
        p = i;
        prob = out_prob[i];
      }
    }
    a = sample.label;

    results.confusion[a][p] += 1;
    results.actuals[a] += 1;
    results.predictions[p] += 1;
    results.total += 1;
    results.correct += (a==p?1:0);    
    var inserted = false;
    for (var i=0; i<results.tops[a][p].length; i++) {
      if (prob > results.tops[a][p][i].prob) {
        results.tops[a][p].splice(i, 0, {idx:test_idx, prob:prob});
        inserted = true;
        break;
      }
    }
    if (!inserted && results.tops[a][p].length < max_tops) {
      results.tops[a][p].push({idx:test_idx, prob:prob});
    }
    return true;
  };

  this.start_over = function() {
    setup_results();
    dataset.set_test_index(0);
  };

  this.save_dataset = function() {
    var data = JSON.stringify(net.toJSON());
    var url = 'data:text/json;charset=utf8,' + encodeURIComponent(data);
    window.open(url, '_blank');
    window.focus();
  };

  this.vol_to_image = function(V, idx, scale, is_weight) {
    var nx = V.sx;
    var ny = V.sy;
    var nz = V.depth;
    var nc = dataset.get_channels();
    if (nx * ny == 1) {
      nz = dataset.get_channels();
      nx = sqrt(V.depth / nz);
      ny = nx;
    }
    var W = scale * nx;
    var H = scale * ny;
    var mm = maxmin(V.w);
    var img = createImage(W, H);
    img.loadPixels();    
    for(var x=0;x<nx;x++) {
      for(var y=0;y<ny;y++) {
        var z = nz * (y * nx + x) + (is_weight ? 0 : idx);
        var val = [];
        for(var dz=0;dz<nc;dz++) {
          val.push(Math.floor(255 * (V.w[z+dz] - mm.minv) / mm.dv));
        }
        for(var dx=0;dx<scale;dx++) {
          for(var dy=0;dy<scale;dy++) {
            var idx_ = ((W * (y*scale+dy)) + (dx + x*scale)) * 4;
            for (var dz=0;dz<3;dz++) {
              img.pixels[idx_+dz] = val[dz % nc];
            }
            img.pixels[idx_+3] = 255;
          }
        }
      }
    }
    img.updatePixels();
    return img;
  };

  this.get_weights_image = function(layer, idx, scale) {
    var V = net.layers[layer].filters[idx];
    return this.vol_to_image(V, idx, scale, true);
  };

  this.get_activations_image = function(layer, idx, scale) {
    var V = net.layers[layer].out_act;
    return this.vol_to_image(V, idx, scale, false);
  };

  this.get_num_activations = function(layer) {
    return net.layers[1].out_act.depth;
  };
  
  this.get_num_weights = function(layer) {
    return net.layers[1].filters.length;
  };
  
  this.get_training_sample_image = function(t) {
    var sample = dataset.get_training_sample(t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };

  this.get_test_sample_image = function(t) {
    var sample = dataset.get_test_sample(t == null ? test_idx : t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };

  // initialize
  layer_defs.push({type:'input', out_sx:dataset.get_dim(), out_sy:dataset.get_dim(), out_depth:dataset.get_channels()});
  setup_results();
};*/