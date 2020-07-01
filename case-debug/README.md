# Troubleshooting Pensieve

## Dataset Preparation

Same as the requirements in ``interpret-pensieve``, the training traces and finetuned models are needed to be put into relevant directories. Specificly,
- mkdir models. Put the pretrained model into models, and change the `model_path` in `main.py`.
- mkdir traces. Put the cooked traces into the directory, and change the `TRAIN_TRACES` in `main.py`.
- Put the video size files into the same directory. Change the `parameters[VIDEO_SIZE_FILE]` in `main.py`.

## Results Reproducing

For Figure 12(a) and 12(b), the frequencies can be obtained by viewing the `.svg` visualization of the decision tree, and read at the root node. Such as: 

![](./doc/root.png)

For Figure 12(c), we provide the Mahimahi traces for different bandwidth in `./mahimahi_traces/`. We also provide the trace generator (`mahimahi_gen.py`) with the usage of:
```
python mahimahi_gen.py --bw 3000
```
indicating generate traces with a constant bandwidth of 3000kbps.

## Oversampling Fix

```
python main.py 100 10 50
```
- The first parameter (100) indicates the number of leaf nodes. 
- The second parameter (10) indicates the oversampling ratio of the samples of 1250kbps.
- The third parameter (50) indicates the oversampling ratio of the samples of 2850kbps.

Finally, result decision trees could be found in `decision_tree/oversample`. We provide an oversampled decision tree in this repository.

