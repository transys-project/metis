# Modification on Pensieve (SIGCOMM'17)

Metis finds that Pensieve significantly relies on the last chunk bitrate (r t ) when making decisions.
Based on this observation, we modify the DNN structure of Pensieve to enlarge the influence of chunk bitrate on the output result.

## Modification

The modified a3c neural network is defined in `./a3c.py`.

## Pretrained model

A pretrained model with modified network structure is given in `./models`.
