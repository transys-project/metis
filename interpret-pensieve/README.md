# Explain Pensieve (SIGCOMM'17)

## Prerequisites
### Dependencies
Install required dependencies (tested under Python 3.7.4):

``
pip install numpy==1.17.2 tensorflow==1.14.0 tflearn==0.3.2 scikit-learn==0.21.3 pydotplus==2.0.2
``

### Traces
Unzip the cooked traces:

``
unzip cooked_traces.zip
``

Traces are originally from the `cooked_test_traces` at https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja, which was compiled by the authors of Pensieve.

### Video
We provide the video information in `video/`, where each file records the chunk size at different (six) bitrates. Put your video manifests there.

### DNN Model
We provide the pre-trained DNN model at `models/pretrain_linear_reward.ckpt`. You can follow the instructions in the original Pensieve repository to train your own DNN model.

## Generate a decision tree with given data traces 

```
mkdir decision_tree
python main.py 100
```

The decision tree could be found in `decision_tree` in the format of `.svg`.