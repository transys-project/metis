# Lightweight Deployment

In this case, we provide the original implementation of Pensieve [SIGCOMM'17] and AuTO [SIGCOMM'18] and the implementation provided by TranSys.


## Steps to Implement Pensieve Model (`./pensieve_browser/`)
We implement Pensieve into JavaScript with [Tensorflow.js](https://js.tensorflow.org/). 


### Prerequisites
You need to first install node.js and grunt. In the `./dash.js/` directory, 
```
sudo apt install npm nodejs
npm install
npm install -g grunt-cli
grunt
```

### Original Implementations
- First, you need to obtain the frozen graph model `.pb` of the DNN. You can utilize the `freeze_graph` to convert the `.ckpt` checkpoint to `.pb` frozen graph.
- Put the frozen graph into `./pensieve_browser/video_server/`. We provide two examples (one hidden layer and five hidden layers) in the folder. 

### Lightweightified Implementations

- Convert the `DecisionTreeClassifier` into JavaScript codes with sklearn-porter (https://github.com/nok/sklearn-porter).
- Replace the JS codes in `dash.js/src/streaming/controllers/ViperDecisionTree.js` with the converted JS codes.
- Run the Gruntfile under `dash.js/` to generate a new `dash.all.min.js`. Note that you may need to add the `--force` option to ignore spelling warnings. 
- The `dash.all.min.js` could be found at `dash.js/dist/`. Put the `dash.all.min.js` to `./video_server`.
- Move the `./video_server/` to `/var/www/html/`.
- Visit the `http://localhost/myindex_XX.html` (XX should be the name of the ABR). The memory and latency statistics will be displayed on the web page.

## Steps to Implement AuTO Model
We adopt the original codes provided by the authors of AuTO at [https://bitbucket.org/JustinasLingys/auto_sigcomm2018/].
The codes for AuTO is still under refactoring. 