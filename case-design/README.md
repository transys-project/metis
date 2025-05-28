# Modification on Pensieve (SIGCOMM'17)

Metis finds that Pensieve significantly relies on the last chunk bitrate (r^t ) when making decisions.
Based on this observation, we modify the DNN structure of Pensieve to enlarge the influence of chunk bitrate on the output result.

## Modification

The modified a3c neural network is defined in `./a3c.py`, we modify the network structure in function `create_actor_network` and `create_critic_network` by directly concatenating the r^t to the output layer, as shown in the snippet below.

``` python
merge_net = tflearn.merge([split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')
dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
merge_net_1 = tflearn.merge([split_0, dense_net_0], 'concat')
out = tflearn.fully_connected(merge_net_1, self.a_dim, activation='softmax')
```

## Pretrained model

A pretrained modified model is given as `./models/nn_model_mod_800000.ckpt`, the model is trained with 800k epochs.
To make a comparison, a pretrained original model with 800k epochs is also given as `./models/nn_model_origin_800000.ckpt`.

## Running

To test this modification model, run

``` bash
cd pensieve
cp -r test test_mod
cd test_mod/
python get_video_sizes.py
cp ../../*.py .
cp ../../models/* ./models/
python rl_no_training.py
```

The result will be stored at `pensieve/test_mod/results` with `sim_mod_rl` scheme.
To plot the QoE line, run `python plot_results.py`.
To compare with origin model, user can also store the origin results in `pensieve/test_mod/results`, modify the `SCHEMES` field in `plot_results.py` and run `python plot_results.py`.
